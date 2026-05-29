# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr vision transformer block (composite).

Profiles :class:`TtVisionBlock` -- the pre-norm residual composite
(TtVisionRMSNorm x2 + TtVisionAttention + TtVisionMLP + two residual adds) in
isolation under metal trace so the CSV reflects device-kernel time rather than
host dispatch. Uses the production-representative shapes from the seed-0 golden:
seq_length=256, cu_seqlens=[0,96,256], 12 heads, head_dim 128, embed_dim 1536,
intermediate 4224.

The leaf modules (vision_attention -23.8%, vision_mlp -13.8%) were already
optimized with L1 memory-config pins; vision_block inherits those wins because
it composes the exact same modules. This harness exists to check the composite
boundaries: the two residual ttnn.add ops and the norm->attn / attn->mlp
reshards.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_vision_block.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("dots_tt_vision_block_profile", os.path.join(_TT_DIR, "vision_block.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionBlock = _mod.TtVisionBlock

# Production-representative shapes from the seed-0 golden.
SEQ = 256
DIM = 1536
NUM_HEADS = 12
HEAD_DIM = 128
INTERMEDIATE = 4224
EPS = 1e-5
CU_SEQLENS = torch.tensor([0, 96, 256], dtype=torch.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        norm1_weight = torch.randn(DIM, dtype=torch.float32)
        qkv_weight = torch.randn(3 * DIM, DIM, dtype=torch.float32) * 0.02
        proj_weight = torch.randn(DIM, DIM, dtype=torch.float32) * 0.02
        norm2_weight = torch.randn(DIM, dtype=torch.float32)
        fc1_weight = torch.randn(INTERMEDIATE, DIM, dtype=torch.float32) * 0.02
        fc3_weight = torch.randn(INTERMEDIATE, DIM, dtype=torch.float32) * 0.02
        fc2_weight = torch.randn(DIM, INTERMEDIATE, dtype=torch.float32) * 0.02
        rotary_pos_emb = torch.randn(SEQ, HEAD_DIM // 2, dtype=torch.float32)

        block = TtVisionBlock(
            device=device,
            norm1_weight=norm1_weight,
            qkv_weight=qkv_weight,
            proj_weight=proj_weight,
            norm2_weight=norm2_weight,
            fc1_weight=fc1_weight,
            fc3_weight=fc3_weight,
            fc2_weight=fc2_weight,
            rotary_pos_emb=rotary_pos_emb,
            cu_seqlens=CU_SEQLENS,
            seq_length=SEQ,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            eps=EPS,
        )

        host_in = torch.randn(SEQ, DIM, dtype=torch.float32)
        x = ttnn.from_torch(
            host_in,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernels into the program cache.
        for _ in range(3):
            out = block(x)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = block(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = block(x)
            ttnn.synchronize_device(device)

        print("profile_vision_block done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
