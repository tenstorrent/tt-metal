# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr Qwen2 LM rotary-embedding block.

Profiles :class:`TtRoPE` (the rotary cos/sin table generator: a broadcast
multiply of positions x inv_freq, a concat of the two halves, then elementwise
cos / sin -- all fp32, no matmul) in isolation under metal trace so the CSV
reflects device-kernel time rather than host dispatch. The language model
builds these tables per forward over the full prefill positions, so a
production-representative sequence length is used for the shape.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \\
      models/demos/rednote_hilab_dots.ocr/tt/profile_rope.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv
(and a cpp_device_perf_report.csv under generated/profiler/.logs/).
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("dots_tt_rope_profile", os.path.join(_TT_DIR, "rope.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtRoPE = _mod.TtRoPE

# Production-representative shape: the LM builds the cos/sin tables over the full
# prefill positions. The reduced PCC golden is seq_len 128; a production prefill
# pushes many more positions through the same table generation, so a 1024-length
# sequence is a representative tile.
HEAD_DIM = 128
ROPE_THETA = 1000000.0
SEQ_LEN = 1024
BATCH = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        rope = TtRoPE(device=device, head_dim=HEAD_DIM, rope_theta=ROPE_THETA)

        # Positions as [batch, 1, seq, 1] float so they broadcast against inv_freq.
        pos = torch.arange(SEQ_LEN, dtype=torch.float32).reshape(BATCH, 1, SEQ_LEN, 1)
        tt_pos = ttnn.from_torch(
            pos,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernels into the program cache.
        for _ in range(3):
            cos, sin = rope(tt_pos)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            cos, sin = rope(tt_pos)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            # One profiled replay.
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            cos, sin = rope(tt_pos)
            ttnn.synchronize_device(device)

        print("profile_rope done; cos shape", tuple(cos.shape), "sin shape", tuple(sin.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
