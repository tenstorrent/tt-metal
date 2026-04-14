# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for compute_auto_shard_spec across unary, binary, and ternary ops.

Tests combinations of:
- Op types: unary (relu, sigmoid, exp), binary (add, mul, sub), ternary (where, addcmul, addcdiv)
- Shapes: medium (1,1,1024,1024), tall (1,1,4096,256), wide (1,1,256,4096)
- Shard strategies: DRAM interleaved (baseline), L1 HEIGHT_SHARDED, L1 WIDTH_SHARDED, L1 BLOCK_SHARDED

Usage:
    python3 -m tracy -v -r -p tests/ttnn/unit_tests/operations/eltwise/test_auto_shard_perf.py

The CSV with DEVICE KERNEL DURATION [ns] will be at:
    $TT_METAL_HOME/generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
"""

import sys
import torch
import ttnn


SHAPES = {
    "1kx1k": [1, 1, 1024, 1024],
    "4kx256": [1, 1, 4096, 256],
    "256x4k": [1, 1, 256, 4096],
}

MEMORY_CONFIGS = {
    "DRAM": ttnn.DRAM_MEMORY_CONFIG,
    "L1_HEIGHT": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    "L1_WIDTH": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    "L1_BLOCK": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
}


def make_tensor(device, shape):
    """Create a bfloat16 tensor on device in DRAM interleaved."""
    return ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def run_op(label, fn):
    """Run a single op, catch and report errors."""
    sys.stdout.flush()
    try:
        fn()
        print(f"  [OK]   {label}", flush=True)
        return True
    except Exception as e:
        msg = str(e).split("\n")[0][:100]
        print(f"  [FAIL] {label}: {msg}", flush=True)
        return False


def main():
    device = ttnn.open_device(device_id=0)
    ok = 0
    fail = 0

    for shape_name, shape in SHAPES.items():
        for mem_name, mem_cfg in MEMORY_CONFIGS.items():
            print(f"\n--- Shape: {shape_name} | Memory: {mem_name} ---", flush=True)

            # Unary ops
            for op_name, op_fn in [("relu", ttnn.relu), ("sigmoid", ttnn.sigmoid), ("exp", ttnn.exp)]:
                label = f"unary.{op_name} {shape_name} {mem_name}"

                def do_unary(op=op_fn, mc=mem_cfg, s=shape):
                    x = make_tensor(device, s)
                    y = op(x, memory_config=mc)
                    ttnn.synchronize_device(device)
                    del y, x

                if run_op(label, do_unary):
                    ok += 1
                else:
                    fail += 1

            # Binary ops
            for op_name, op_fn in [("add", ttnn.add), ("mul", ttnn.mul), ("sub", ttnn.sub)]:
                label = f"binary.{op_name} {shape_name} {mem_name}"

                def do_binary(op=op_fn, mc=mem_cfg, s=shape):
                    a = make_tensor(device, s)
                    b = make_tensor(device, s)
                    y = op(a, b, memory_config=mc)
                    ttnn.synchronize_device(device)
                    del y, a, b

                if run_op(label, do_binary):
                    ok += 1
                else:
                    fail += 1

            # Ternary ops
            for op_name, op_fn in [
                ("where", lambda a, b, c, mc: ttnn.where(a > 0.5, b, c, memory_config=mc)),
                ("addcmul", lambda a, b, c, mc: ttnn.addcmul(a, b, c, value=0.5, memory_config=mc)),
                ("addcdiv", lambda a, b, c, mc: ttnn.addcdiv(a, b, c + 1.0, value=0.5, memory_config=mc)),
            ]:
                label = f"ternary.{op_name} {shape_name} {mem_name}"

                def do_ternary(op=op_fn, mc=mem_cfg, s=shape):
                    a = make_tensor(device, s)
                    b = make_tensor(device, s)
                    c = make_tensor(device, s)
                    y = op(a, b, c, mc)
                    ttnn.synchronize_device(device)
                    del y, a, b, c

                if run_op(label, do_ternary):
                    ok += 1
                else:
                    fail += 1

    print(f"\n{'='*60}", flush=True)
    print(f"TOTAL: {ok + fail} | PASSED: {ok} | FAILED: {fail}", flush=True)
    print(f"{'='*60}", flush=True)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
