# SPDX-License-Identifier: Apache-2.0
"""Shared chat-template suite, identical pinned token inputs on HF and TT."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..tt.checkpoint import MODEL_ID, REVISION, SNAPSHOT


def run_suite(model, backend, *, max_new_tokens=128):
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ.get("FULL_ARTIFACT_DIR", root / "doc/full_model"))
    prompts = json.loads((out / "qualitative_prompts.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    results = []
    for item in prompts:
        start = time.monotonic()
        prompt = item["token_ids"]
        if backend == "hf":
            generated = model.generate(torch.tensor([prompt]), max_new_tokens=max_new_tokens)[0, len(prompt) :].tolist()
        else:
            generated = model.generate(prompt, max_new_tokens)
        result = dict(
            id=item["id"],
            prompt=item["prompt"],
            prompt_ids=prompt,
            generated_ids=generated,
            completion=tokenizer.decode(generated, skip_special_tokens=False),
            seconds=time.monotonic() - start,
            backend=backend,
            model_id=MODEL_ID,
            revision=REVISION,
        )
        if backend == "tt":
            result["metrics"] = model.last_generation_metrics
        results.append(result)
        (out / f"qualitative_{backend}.json").write_text(json.dumps(results, indent=2) + "\n")
        print("QUALITATIVE_RESULT", json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "tt"], required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.backend == "hf":
        from .hf_model import Kolibri1ForCausalLM

        model = Kolibri1ForCausalLM.from_pretrained(SNAPSHOT).eval()
        run_suite(model, "hf", max_new_tokens=args.tokens)
    else:
        import ttnn

        from ..tt.generator import build_generator

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = None
        try:
            model = build_generator(Path(__file__).resolve().parents[1], mesh)
            run_suite(model, "tt", max_new_tokens=args.tokens)
        finally:
            if model:
                model.close()
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
