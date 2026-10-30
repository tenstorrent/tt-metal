# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate independent CPU/Hugging Face final-token goldens, 2K before 4K.

Run on a compute host with the checkpoint and enough RAM for FP32 weights.
Outputs contain two small, portable golden JSON files plus private run timing,
checkpoint location, CPU affinity and full-vocabulary final-row logits. Only the
small golden JSON files belong beside the checked-in book fixture.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path


# Stream large checkpoint files without keeping another copy of the weights.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Replace each report atomically so a completed 2K result survives a later failure.
def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


# Hash token values independently of whitespace in the checked-in JSON file.
def token_digest(ids):
    return hashlib.sha256((json.dumps(ids, separators=(",", ":")) + "\n").encode()).hexdigest()


# Keep parsing import-light so --help works without torch or a device runtime.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/model/fixtures/book",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    fixture = json.loads((args.fixture_dir / "provenance.json").read_text())
    pinned = fixture["input"]
    for file_key, hash_key in (("text_file", "text_sha256"), ("token_ids_file", "token_ids_sha256")):
        assert sha256(args.fixture_dir / pinned[file_key]) == pinned[hash_key], file_key
    ids = json.loads((args.fixture_dir / pinned["token_ids_file"]).read_text())
    assert len(ids) == 4096 and ids[0] == 128000 and ids.count(128000) == 1
    assert all(type(token) is int and 0 <= token < 128256 for token in ids)
    for name, digest in fixture["checkpoint_files_sha256"].items():
        assert sha256(args.checkpoint / name) == digest, f"Checkpoint metadata changed: {name}"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    # Read bytes to preserve the pinned CRLF text exactly on every platform.
    book_text = (args.fixture_dir / pinned["text_file"]).read_bytes().decode("utf-8")
    assert [tokenizer.bos_token_id] + tokenizer.encode(book_text, add_special_tokens=False) == ids
    index_path = args.checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_files = sorted(set(index["weight_map"].values()))
    checkpoint_hashes = dict(fixture["checkpoint_files_sha256"])
    checkpoint_hashes[index_path.name] = sha256(index_path)
    for name in weight_files:
        assert Path(name).name == name, "Checkpoint shards must be local filenames"
        checkpoint_hashes[name] = sha256(args.checkpoint / name)
    run = {
        "model_id": fixture["model_id"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "host": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "dtype": "float32",
        "device": "cpu",
        "attention_implementation": "sdpa",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "checkpoint_files_sha256": checkpoint_hashes,
        "generator_sha256": sha256(__file__),
        "fixture": fixture,
        "preparation_seconds": time.perf_counter() - started,
        "results": [],
    }
    save_json(args.output_dir / "run.json", run)
    print(json.dumps({"event": "loading", "preparation_seconds": run["preparation_seconds"]}), flush=True)
    load_started = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .to("cpu")
    )
    run["load_seconds"] = time.perf_counter() - load_started
    assert len(model.model.layers) == 32 and model.config.vocab_size == 128256
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in model.parameters())
    save_json(args.output_dir / "run.json", run)
    print(json.dumps({"event": "loaded", "load_seconds": run["load_seconds"]}), flush=True)
    with torch.inference_mode():
        for length in (2048, 4096):
            input_ids = torch.tensor([ids[:length]], dtype=torch.long)
            forward_started = time.perf_counter()
            # HF's body includes all 32 layers and final RMSNorm. Project only the
            # final valid row: no [length, vocab_size] logits or KV-cache snapshot.
            hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
            logits = model.lm_head(hidden[:, -1:, :]).float().reshape(-1)
            forward_seconds = time.perf_counter() - forward_started
            assert logits.shape == (128256,) and torch.isfinite(logits).all()
            values, indices = logits.topk(5)
            result = {
                "context_length": length,
                "final_position": length - 1,
                "input_ids_sha256": token_digest(ids[:length]),
                "reference_top1_id": int(indices[0]),
                "reference_top5_ids": indices.tolist(),
                "reference_top5_logits": values.tolist(),
                "reference_top5_tokens": [tokenizer.decode([int(token)]) for token in indices],
                "forward_seconds": forward_seconds,
                "total_seconds": time.perf_counter() - started,
            }
            torch.save(logits.cpu(), args.output_dir / f"logits_{length}.pt")
            # This portable result is the only generated payload needed by pytest.
            golden = {key: value for key, value in result.items() if key not in ("forward_seconds", "total_seconds")}
            golden.update(
                {
                    "model_id": fixture["model_id"],
                    "num_layers": 32,
                    "vocab_size": 128256,
                    "reference": "transformers.AutoModelForCausalLM.model + final-row lm_head",
                    "dtype": "float32",
                    "attention_implementation": "sdpa",
                    "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "checkpoint_files_sha256": checkpoint_hashes,
                }
            )
            save_json(args.output_dir / f"golden_{length}.json", golden)
            run["results"].append(result)
            save_json(args.output_dir / "run.json", run)
            print(json.dumps({"event": "golden_saved", **result}), flush=True)
            del hidden, logits, input_ids
    run["complete"] = True
    run["total_seconds"] = time.perf_counter() - started
    save_json(args.output_dir / "run.json", run)


if __name__ == "__main__":
    main()
