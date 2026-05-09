# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

# Make `import models.*` work when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


def _default_output_path(snapshot_dir: Path, *, num_layers: int, max_new_tokens: int) -> Path:
    snap_name = Path(snapshot_dir).name
    root = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/golden"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"golden_tokens_{snap_name}_layers{num_layers}_new{max_new_tokens}.json"


def _load_model_for_num_layers(snapshot_dir: Path, *, num_layers: int):
    from transformers import AutoConfig
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM

    config = AutoConfig.from_pretrained(snapshot_dir, local_files_only=True)
    config.num_hidden_layers = int(num_layers)
    if hasattr(config, "mlp_layer_types") and isinstance(config.mlp_layer_types, list):
        config.mlp_layer_types = list(config.mlp_layer_types[: int(num_layers)])

    # Prefer bf16 to keep memory bounded for MoE expert weights.
    try:
        return Glm4MoeLiteForCausalLM.from_pretrained(
            snapshot_dir,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        # Some environments may not support low_cpu_mem_usage (accelerate missing).
        return Glm4MoeLiteForCausalLM.from_pretrained(
            snapshot_dir,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Generate offline golden greedy tokens for a truncated GLM-4.7-Flash model.")
    p.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    p.add_argument("--snapshot-dir", type=Path, default=None)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-prompts", type=int, default=8)
    args = p.parse_args()

    if args.snapshot_dir is None:
        snapshot_dir = resolve_best_effort_snapshot_dir(args.model_id)
    else:
        snapshot_dir = Path(args.snapshot_dir)

    missing = find_missing_shards(snapshot_dir)
    if missing:
        print(f"WARNING: snapshot is missing {len(missing)} shards (example: {missing[0]}). Golden generation may fail.")

    out_path = Path(args.output) if args.output is not None else _default_output_path(
        snapshot_dir, num_layers=int(args.num_layers), max_new_tokens=int(args.max_new_tokens)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True, use_fast=True)
    model = _load_model_for_num_layers(snapshot_dir, num_layers=int(args.num_layers))
    model.eval()

    prompts = [
        "Hello.",
        "Write a haiku about chips.",
        "Explain what KV cache is in one sentence.",
        "List three colors.",
        "2+2=",
        "The capital of France is",
        "Once upon a time,",
        "Say the word 'pong' and nothing else.",
    ][: max(0, int(args.max_prompts))]

    torch.manual_seed(0)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prompt in prompts:
            enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", None)

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                use_cache=True,
            )
            gen_ids = out[0, input_ids.shape[1] :].to(dtype=torch.int32).tolist()
            records.append(
                {
                    "prompt": prompt,
                    "prompt_input_ids": input_ids[0].to(dtype=torch.int32).tolist(),
                    "generated_ids": gen_ids,
                }
            )

    payload = {
        "model_id": args.model_id,
        "snapshot_dir": str(snapshot_dir),
        "num_layers": int(args.num_layers),
        "max_new_tokens": int(args.max_new_tokens),
        "prompts": prompts,
        "records": records,
        "torch_version": torch.__version__,
    }
    try:
        import transformers

        payload["transformers_version"] = transformers.__version__
    except Exception:
        pass

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote golden tokens: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
