# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Repro — B3: batch-32 L1 circular-buffer clash with a persistent L1 buffer (Llama-3.1-70B, TP4).

NOT a tt-metal framework bug: the allocator's `validate_circular_buffer_region` correctly
detects that the packed gate/up decode matmul's static CB region overlaps a persistent
(inter-layer) L1 buffer, because the ported decode path is MISSING the explicit inter-layer
last-use deallocations that prefill / production Llama already have. Separately the
output-projection 8-tile K block genuinely exceeds per-core L1 at fused batch-32
(needs 1,979,136 B vs 1,572,864 B). Included here because it was a stage-05 timeout
contributor and the boundary is exactly reproducible.

Requires the ported model + weights (run on branch mvasiljevic/model/meta-llama-llama-3.1-70b-instruct).
NOT standalone.

Trigger: fused batch-32 traced decode with >= 2 decoder layers (1 layer passes; 2 reproduce).

Expected on the un-fixed decode path:
  "Statically allocated circular buffers in program N clash with L1 buffers on core range
   [0-0 - 7-9]. L1 buffer allocated at 927488 and static circular buffer region ends at 1032960"
Expected after the model-side fix (ordered last-use deallocs + stack-owned residual +
K-block shrink to 4 tiles, or 2 for batch-32 DRAM-sharded gate/up): passes.

Run:
  TT_METAL_HOME=$METAL PYTHONPATH=$METAL \
    python3 repros/llama70b_l1_cb_collision.py --layers 2 --batch 32
"""
from __future__ import annotations

import argparse

import ttnn
from models.autoports.meta_llama_llama_3_1_70b_instruct.tt.generator import build_generator
from models.common.readiness_check.mesh_device import (
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

MODEL_DIR = "models/autoports/meta_llama_llama_3_1_70b_instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2, help=">=2 reproduces the clash; 1 passes")
    ap.add_argument("--batch", type=int, default=32, help="fused batch (32 -> per_core_M=16 -> 8-tile K overflow)")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = open_readiness_mesh_device("P300", "FABRIC_1D_RING")
    try:
        # num_hidden_layers override drives the "2-layer reduced" isolation of the inter-layer
        # L1 liveness clash; batch-32 fused drives the 8-tile K-block L1 overflow.
        generator = build_generator(
            MODEL_DIR, mesh, model_path=args.model_path,
            max_batch_size=args.batch, max_context_len=256, num_blocks=128,
            num_hidden_layers=args.layers,
        )
        prompt = [list(range(16))] * args.batch
        cache = generator._ensure_kv_cache()
        generator.prefill_forward(prompt, kv_cache=cache, prompt_lens=[16] * args.batch,
                                  sampling_mode="device")
        # the clash fires when layer 1 enters gate/up while layer 0's output is still live:
        generator.decode_forward(enable_trace=True)     # capture
        generator.decode_forward(enable_trace=True)      # replay
        print(f"PASS: {args.layers}-layer batch-{args.batch} decode completed (no CB clash).")
        return 0
    except RuntimeError as e:
        if "clash with L1 buffers" in str(e) or "validate_circular_buffer_region" in str(e):
            print(f"REPRODUCED B3 CB clash:\n{e}")
            return 3
        raise
    finally:
        close_readiness_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
