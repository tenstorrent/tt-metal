# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only HF text/cache preflight; never loads weights or opens a TT mesh."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from readiness_check.reference_cache import ReferenceCache, model_revision, prompt_hash, tokenizer_fingerprint
from transformers import AutoTokenizer, GenerationConfig

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output artifact")
    acceptance = Path(__file__).with_name("run_full_model_acceptance.py")
    names = {"require_cached_autoregressive", "pinned_stopping_config"}
    nodes = [
        node
        for node in ast.parse(acceptance.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == 2
    namespace = dict(
        torch=torch, MODEL_ID=MODEL_ID, AutoTokenizer=AutoTokenizer, GenerationConfig=GenerationConfig, json=json
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(acceptance), "exec"), namespace
    )
    aime = namespace["require_cached_autoregressive"](ROOT, ROOT / "doc/full_model/hf_reference/prompt.txt")
    stop = namespace["pinned_stopping_config"](ROOT)
    quality = ROOT / "doc/full_model/qualitative"
    metadata = json.loads((quality / "qualitative_prompt_format.json").read_text())
    prompts = json.loads((quality / "prompts.json").read_text())
    outputs = json.loads((quality / "hf_outputs.json").read_text())
    assert json.loads((quality / "hf_status.json").read_text())["status"] == "complete"
    assert model_revision(MODEL_ID) == f"hub:{metadata['revision']}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=metadata["revision"], local_files_only=True)
    assert tokenizer_fingerprint(MODEL_ID, tokenizer) == metadata["tokenizer_fingerprint"]
    assert metadata["sampled_recipe"] == dict(temperature=1.0, top_k=20, top_p=0.95, repetition_penalty=1.0)
    assert metadata["seeds"] == list(range(10)) and metadata["max_new_tokens"] == 128
    assert len(prompts) == 6 and len(outputs) == 66
    by_id = {prompt["id"]: prompt for prompt in prompts}
    for prompt in prompts:
        assert (
            tokenizer.apply_chat_template(prompt["messages"], tokenize=False, add_generation_prompt=True)
            == prompt["rendered"]
        )
        encoded = tokenizer(prompt["rendered"], add_special_tokens=False)["input_ids"]
        assert encoded == prompt["token_ids"]
    cache = ReferenceCache.for_model_dir(ROOT)
    assert cache.enabled
    assert {(row["prompt_id"], row["seed"]) for row in outputs} == {
        (prompt["id"], seed) for prompt in prompts for seed in [None, *range(10)]
    }
    keys = []
    for row in outputs:
        key = cache.make_key(
            kind="hf_qualitative",
            hf_model_id=MODEL_ID,
            revision=model_revision(MODEL_ID),
            tokenizer=tokenizer_fingerprint(MODEL_ID, tokenizer),
            prompt=prompt_hash(by_id[row["prompt_id"]]["token_ids"]),
            settings=row["settings"],
        )
        assert cache.get_json(key) == row["raw_generated_token_ids"], (row["prompt_id"], row["seed"])
        assert all(row["stopping"][field] == stop[field] for field in ("eos_token_ids", "source", "max_new_tokens"))
        keys.append(key)
    ref = torch.load(ROOT / "readiness_aime24_chat.refpt", weights_only=True, map_location="cpu")
    assert ref["hf_model_id"] == MODEL_ID and ref["k"] == 100 and len(ref["entries"]) == 1
    entry = ref["entries"][0]
    assert entry["generated_tokens"].numel() == 100 and tuple(entry["topk_tokens"].shape) == (100, 100)
    result = dict(
        passed=True,
        scope="CPU exact cache/tokenizer preflight only; no model quality claim",
        autoregressive=aime,
        stopping=stop,
        qualitative_exact_cache_hits=len(keys),
        qualitative_prompt_lengths=[len(p["token_ids"]) for p in prompts],
        reference_prompt_tokens=entry["prompt_tokens"].numel(),
        reference_generated_tokens=100,
        acceptance_sha256=hashlib.sha256(acceptance.read_bytes()).hexdigest(),
        controls_sha256=hashlib.sha256((quality / "run_controls.py").read_bytes()).hexdigest(),
        preflight_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
