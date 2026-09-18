# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from tt_bfp_quant import capture_linear_inputs, gptq_search, search_linear
from tt_bfp_quant.checkpoint import export_checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def recipe(tmp_path, method="max-minus-one", bits=4, **extras):
    return write_json(
        tmp_path / "recipe.json",
        {
            "schema_version": 1,
            "rules": [{"match": "model.layers.*.mlp.*_proj.weight", "method": method, "bits": bits, **extras}],
        },
    )


def source_checkpoint(tmp_path, sharded=False):
    model = tmp_path / "original"
    model.mkdir()
    torch.manual_seed(718)
    weights = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(40, 17).bfloat16(),
        "model.layers.0.mlp.up_proj.weight": torch.randn(40, 17).bfloat16(),
        "norm.weight": torch.randn(17),
    }
    write_json(model / "config.json", {"model_type": "llama", "tie_word_embeddings": False})
    write_json(model / "tokenizer_config.json", {"test": "preserve this exactly"})
    (model / "chat_templates").mkdir()
    (model / "chat_templates" / "default.jinja").write_text("{{ messages }}")
    if sharded:
        first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
        save_file(
            {k: v for k, v in weights.items() if k != "norm.weight"}, str(model / first), metadata={"format": "pt"}
        )
        save_file({"norm.weight": weights["norm.weight"]}, str(model / second), metadata={"format": "pt"})
        write_json(
            model / "model.safetensors.index.json",
            {
                "metadata": {"total_size": sum(w.nbytes for w in weights.values())},
                "weight_map": {k: second if k == "norm.weight" else first for k in weights},
            },
        )
    else:
        save_file(weights, str(model / "model.safetensors"), metadata={"format": "pt"})
    return model, weights


def digest_files(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("method,bits", [("round", 4), ("max-minus-one", 4), ("max-minus-one", 8), ("gptq-search", 4)])
def test_checkpoint_export_matches_api_and_preserves_source(tmp_path, sharded, method, bits):
    model, weights = source_checkpoint(tmp_path, sharded)
    before = digest_files(model)
    r = recipe(tmp_path, method, bits, output_splits=[20, 20])
    output = tmp_path / "prepared"
    h = torch.eye(17)
    torch.save(h, tmp_path / "shared.pt")
    hp = write_json(tmp_path / "hessians.json", {k: "shared.pt" for k in weights if k != "norm.weight"})
    report = export_checkpoint(model, output, r, hessians=hp, backend="numpy", threads=2)
    actual = {}
    for file in output.glob("*.safetensors"):
        actual.update(load_file(file))
    assert set(actual) == set(weights)
    for name, weight in weights.items():
        assert actual[name].dtype == weight.dtype
        if name == "norm.weight":
            assert torch.equal(weight, actual[name])
        else:
            q, _ = (
                gptq_search(weight, h, output_splits=[20, 20], backend="numpy")
                if method == "gptq-search"
                else search_linear(
                    weight, bits, (0,) if method == "round" else (0, -1), output_splits=[20, 20], backend="numpy"
                )
            )
            assert torch.equal(actual[name].float(), q)
            assert report["weights"][name]["validation"]["numerical_repacking_exact"]
    assert report["factor_cache_hits"] == (1 if method == "gptq-search" else 0)
    assert before == digest_files(model)
    for name in ("config.json", "tokenizer_config.json", "chat_templates/default.jinja"):
        assert (model / name).read_bytes() == (output / name).read_bytes()
    if sharded:
        assert (model / "model.safetensors.index.json").read_bytes() == (
            output / "model.safetensors.index.json"
        ).read_bytes()
        untouched = "model-00002-of-00002.safetensors"
        assert (model / untouched).read_bytes() == (output / untouched).read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):  # allow-pytest.raises: CPU-only tests.
        export_checkpoint(model, output, r)


def test_export_cli_dry_run_and_complete_export(tmp_path):
    model, _ = source_checkpoint(tmp_path)
    r = recipe(tmp_path)
    output = tmp_path / "cli-output"
    command = [
        sys.executable,
        "-m",
        "tt_bfp_quant.cli",
        "export",
        "--model",
        str(model),
        "--output",
        str(output),
        "--recipe",
        str(r),
        "--backend",
        "numpy",
        "--threads",
        "2",
    ]
    result = subprocess.run(command + ["--dry-run"], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["selected_count"] == 2 and not output.exists()
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert Path(json.loads(result.stdout)["report"]).is_file()


def test_gptq_dry_run_without_statistics(tmp_path):
    model, _ = source_checkpoint(tmp_path)
    plan = export_checkpoint(model, tmp_path / "out", recipe(tmp_path, "gptq-search"), dry_run=True)
    assert len(plan["hessians_required"]) == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "problem", ["unmatched", "overlap", "gptq8", "split", "missing-hessian", "quantized", "tied", "unknown-field"]
)
def test_bad_recipe_or_source_rejected_before_writing(tmp_path, problem):
    model, _ = source_checkpoint(tmp_path)
    r = recipe(tmp_path)
    data = json.loads(r.read_text())
    rule = data["rules"][0]
    if problem == "unmatched":
        rule["match"] = "wrong.*"
    elif problem == "overlap":
        data["rules"].append(rule.copy())
    elif problem == "gptq8":
        rule.update(method="gptq-search", bits=8)
    elif problem == "split":
        rule["output_splits"] = [16, 16]
    elif problem == "missing-hessian":
        rule["method"] = "gptq-search"
    elif problem == "quantized":
        write_json(model / "config.json", {"text_config": {"quantization_config": {"quant_method": "fp8"}}})
    elif problem == "tied":
        weights = load_file(model / "model.safetensors")
        weights["lm_head.weight"] = torch.randn(40, 17)
        save_file(weights, str(model / "model.safetensors"))
        write_json(model / "config.json", {"tie_word_embeddings": True})
        rule["match"] = "lm_head.weight"
    elif problem == "unknown-field":
        rule["output_split"] = [40]
    write_json(r, data)
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        export_checkpoint(model, tmp_path / "out", r)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_failure_cleans_staging_and_restores_threads(tmp_path):
    model, _ = source_checkpoint(tmp_path)
    r = recipe(tmp_path, "gptq-search")
    torch.save(torch.eye(9), tmp_path / "wrong.pt")
    hp = write_json(
        tmp_path / "hessians.json", {f"model.layers.0.mlp.{p}_proj.weight": "wrong.pt" for p in ("gate", "up")}
    )
    threads = torch.get_num_threads()
    with pytest.raises(ValueError, match="Hessian"):  # allow-pytest.raises: CPU-only tests.
        export_checkpoint(model, tmp_path / "out", r, hessians=hp, threads=1)
    assert torch.get_num_threads() == threads
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob(".out.tmp-*"))


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("method", ["max-minus-one", "gptq-search"])
def test_unchanged_transformers_loader_and_forward(tmp_path, sharded, method):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.manual_seed(914)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=97,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                tie_word_embeddings=False,
            )
        )
        .to(torch.bfloat16)
        .eval()
    )
    source, output = tmp_path / "tiny-llama", tmp_path / "exported-tiny-llama"
    model.save_pretrained(source, safe_serialization=True, max_shard_size="12KB" if sharded else "1GB")
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(source)
    r = recipe(tmp_path, method)
    data = json.loads(r.read_text())
    data["rules"][0]["match"] = ["model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"]
    write_json(r, data)
    module_names = [name for name, _ in model.named_modules() if name.endswith(("mlp.gate_proj", "mlp.up_proj"))]
    hpath = None
    if method == "gptq-search":
        with capture_linear_inputs(model, module_names) as stats, torch.inference_mode():
            model(input_ids=torch.randint(0, 97, (2, 16)), use_cache=False)
        paths = {}
        for i, (name, accumulator) in enumerate(stats.items()):
            filename = f"H-{i}.pt"
            torch.save(accumulator.value(), tmp_path / filename)
            paths[name + ".weight"] = filename
        hpath = write_json(tmp_path / "hessians.json", paths)
    report = export_checkpoint(source, output, r, hessians=hpath, backend="numpy", threads=2)
    # These are the ordinary public loader calls: no package hooks or patches.
    loaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True, torch_dtype="auto").eval()
    loaded_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert loaded_tokenizer("hello world")["input_ids"] == [1, 2]
    original_state, loaded_state = model.state_dict(), loaded.state_dict()
    assert set(original_state) == set(loaded_state)
    disk = {}
    for file in output.glob("*.safetensors"):
        disk.update(load_file(file))
    for name, original in original_state.items():
        assert original.shape == loaded_state[name].shape
        assert torch.equal(loaded_state[name], disk[name])
        if name not in report["selected"]:
            assert torch.equal(original, loaded_state[name])
    with torch.inference_mode():
        logits = loaded(input_ids=torch.randint(0, 97, (1, 8))).logits
    assert logits.shape == (1, 8, 97) and torch.isfinite(logits).all()
