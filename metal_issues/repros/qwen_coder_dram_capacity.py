# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Repro — B4: long-context DRAM per-bank capacity shortfall (Qwen2.5-Coder-32B, TP4).

NOT a tt-metal framework bug: the prefill terminal RMSNorm output (DRAM-interleaved BF16,
hidden 5120, after a terminal Ring all-gather) genuinely exceeds physical DRAM at long
context; the allocator OOM report is correct (fragmentation ruled out via cold-mesh repro).
Two rounding boundaries interact: KV-cache pages padded to 128 tokens, prompt activations
tile in 32-token steps. Remedy is a MODEL-CONFIG change: lower advertised max_context from
32,768 to the measured ceiling (10,720; analytic contract 10,496). Included because it was
a stage-05 timeout contributor and the boundary is exactly reproducible.

Requires the ported model + weights (run on branch mvasiljevic/model/qwen-qwen2.5-coder-32b-instruct).
NOT standalone.

Boundary: prompt 10,720 fits; prompt 10,721 (-> 10,752 physical rows) is the first OOM.

Expected at 10721:
  "Out of Memory: Not enough space to allocate 3513057280 B DRAM buffer across 8 banks,
   where each bank needs to store 439132160 B, ... largest free block: 367502336 B"
Expected at 10720: passes.

Run:
  TT_METAL_HOME=$METAL PYTHONPATH=$METAL python3 repros/qwen_coder_dram_capacity.py \
      --max-context 10721 --prompt-length 10721 --expect oom
  # ... and --max-context 10720 --prompt-length 10720 --expect pass
"""
from __future__ import annotations

import argparse
import re

import torch
import ttnn
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.generator import build_generator
from models.common.readiness_check.mesh_device import (
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

MODEL_DIR = "models/autoports/qwen_qwen2_5_coder_32b_instruct"
_OOM_FIELDS = {
    "requested_allocation_bytes": r"allocate (\d+) B",
    "requested_bytes_per_bank": r"each bank needs to store (\d+) B",
    "free_bytes_per_bank": r"free: (\d+) B",
    "largest_free_block_bytes_per_bank": r"largest free block: (\d+) B",
}


def _oom_fields(msg: str) -> dict:
    return {k: int(m.group(1)) for k, p in _OOM_FIELDS.items() if (m := re.search(p, msg))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-context", type=int, required=True)
    ap.add_argument("--prompt-length", type=int, required=True)
    ap.add_argument("--expect", choices=["pass", "oom"], required=True)
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = open_readiness_mesh_device("P300", "FABRIC_1D_RING")
    try:
        generator = build_generator(MODEL_DIR, mesh, max_seq_len=args.max_context)
        cache = generator._ensure_kv_cache()
        prompt = (list(range(32)) * ((args.prompt_length // 32) + 1))[: args.prompt_length]
        try:
            generator.prefill_forward(torch.tensor([prompt]), kv_cache=cache,
                                      prompt_lens=[args.prompt_length], sampling_mode="device")
            tile_padded = ((args.prompt_length + 31) // 32) * 32
            page_padded = ((args.max_context + 127) // 128) * 128
            print(f"PASS: prefill at max_context={args.max_context} prompt={args.prompt_length} "
                  f"(tile_padded_rows={tile_padded}, page_padded_cache={page_padded})")
            return 0 if args.expect == "pass" else 1
        except RuntimeError as e:
            if "Out of Memory" not in str(e):
                raise
            print(f"REPRODUCED B4 DRAM OOM (terminal RMSNorm output): {_oom_fields(str(e))}")
            return 0 if args.expect == "oom" else 1
    finally:
        close_readiness_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
