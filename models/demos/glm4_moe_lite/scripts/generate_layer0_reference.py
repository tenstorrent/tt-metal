# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import torch

# Make `import models.*` work when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from models.demos.glm4_moe_lite.tt.reference_layer0 import run_layer0_reference
from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


def _default_output_path(snapshot_dir: Path) -> Path:
    snap_name = Path(snapshot_dir).name
    root = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/reference"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"layer0_reference_{snap_name}.pt"


def main() -> int:
    p = argparse.ArgumentParser(description="Generate GLM-4.7-Flash Layer-0 CPU reference artifacts (offline).")
    p.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    p.add_argument("--snapshot-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-prompts", type=int, default=8)
    args = p.parse_args()

    if args.snapshot_dir is None:
        snapshot_dir = resolve_best_effort_snapshot_dir(args.model_id)
    else:
        snapshot_dir = Path(args.snapshot_dir)

    missing = find_missing_shards(snapshot_dir)
    if missing:
        # We can still generate layer0 refs because the needed weights are in shard 1,
        # but print a loud warning because users will inevitably hit this later.
        print(f"WARNING: snapshot is missing {len(missing)} shards (example: {missing[0]}).")

    out_path = Path(args.output) if args.output is not None else _default_output_path(snapshot_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = [
        "Hello.",
        "Write a haiku about chips.",
        "Explain what KV cache is in one sentence.",
        "List three colors.",
        "2+2=",
        "The capital of France is",
        "Once upon a time,",
        "Say the word 'pong' and nothing else.",
    ][: max(0, args.max_prompts)]

    records: list[dict[str, Any]] = []
    for prompt in prompts:
        r = run_layer0_reference(snapshot_dir, prompt)
        records.append(
            {
                "prompt": prompt,
                "input_ids": r.input_ids,
                "x_embed": r.x_embed,
                "x_attn_out": r.x_attn_out,
                "x_mlp_out": r.x_mlp_out,
            }
        )

    payload = {
        "model_id": args.model_id,
        "snapshot_dir": str(snapshot_dir),
        "prompts": prompts,
        "records": records,
        "torch_version": torch.__version__,
    }
    torch.save(payload, out_path)
    print(f"Wrote {len(records)} prompts to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
