# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for compute_auto_shard_spec across unary, binary, and ternary ops.

Tests all combinations of:
- Op types: unary (relu, sigmoid, exp), binary (add, mul, sub), ternary (where, addcmul, addcdiv)
- Shapes: small (1,1,32,32), medium (1,1,1024,1024), large (1,1,2048,2048), tall (1,1,8192,256), wide (1,1,256,8192)
- Shard strategies: DRAM interleaved, L1 interleaved, L1 HEIGHT_SHARDED, L1 WIDTH_SHARDED, L1 BLOCK_SHARDED

Usage:
    python3 -m tracy -v -r -p tests/ttnn/unit_tests/operations/eltwise/test_auto_shard_perf.py
"""

import torch
import ttnn
import time


SHAPES = {
    "small_32x32": [1, 1, 32, 32],
    "medium_1kx1k": [1, 1, 1024, 1024],
    "large_2kx2k": [1, 1, 2048, 2048],
    "tall_8kx256": [1, 1, 8192, 256],
    "wide_256x8k": [1, 1, 256, 8192],
}

MEMORY_CONFIGS = {
    "DRAM_INTERLEAVED": ttnn.DRAM_MEMORY_CONFIG,
    "L1_INTERLEAVED": ttnn.L1_MEMORY_CONFIG,
    "L1_HEIGHT_SHARDED": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    "L1_WIDTH_SHARDED": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    "L1_BLOCK_SHARDED": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
}


def create_input_tensor(device, shape, memory_config=None):
    """Create a bfloat16 tensor on device with optional memory config."""
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    if memory_config is not None and memory_config != ttnn.DRAM_MEMORY_CONFIG:
        t = ttnn.to_memory_config(t, memory_config)
    return t


def run_unary_ops(device, shape, mem_config, results):
    """Run unary ops: relu, sigmoid, exp."""
    shape_name = [k for k, v in SHAPES.items() if v == shape][0]
    mem_name = [k for k, v in MEMORY_CONFIGS.items() if v == mem_config][0]

    unary_ops = {
        "relu": ttnn.relu,
        "sigmoid": ttnn.sigmoid,
        "exp": ttnn.exp,
    }

    for op_name, op_fn in unary_ops.items():
        try:
            input_t = create_input_tensor(device, shape)
            output = op_fn(input_t, memory_config=mem_config)
            ttnn.synchronize_device(device)
            results.append({
                "op": op_name,
                "type": "unary",
                "shape": shape_name,
                "memory": mem_name,
                "status": "OK",
            })
            del output, input_t
        except Exception as e:
            results.append({
                "op": op_name,
                "type": "unary",
                "shape": shape_name,
                "memory": mem_name,
                "status": f"FAIL: {str(e)[:80]}",
            })


def run_binary_ops(device, shape, mem_config, results):
    """Run binary ops: add, mul, sub."""
    shape_name = [k for k, v in SHAPES.items() if v == shape][0]
    mem_name = [k for k, v in MEMORY_CONFIGS.items() if v == mem_config][0]

    binary_ops = {
        "add": ttnn.add,
        "mul": ttnn.mul,
        "sub": ttnn.sub,
    }

    for op_name, op_fn in binary_ops.items():
        try:
            input_a = create_input_tensor(device, shape)
            input_b = create_input_tensor(device, shape)
            output = op_fn(input_a, input_b, memory_config=mem_config)
            ttnn.synchronize_device(device)
            results.append({
                "op": op_name,
                "type": "binary",
                "shape": shape_name,
                "memory": mem_name,
                "status": "OK",
            })
            del output, input_a, input_b
        except Exception as e:
            results.append({
                "op": op_name,
                "type": "binary",
                "shape": shape_name,
                "memory": mem_name,
                "status": f"FAIL: {str(e)[:80]}",
            })


def run_ternary_ops(device, shape, mem_config, results):
    """Run ternary ops: where, addcmul, addcdiv."""
    shape_name = [k for k, v in SHAPES.items() if v == shape][0]
    mem_name = [k for k, v in MEMORY_CONFIGS.items() if v == mem_config][0]

    ternary_ops = {
        "where": lambda a, b, c, **kw: ttnn.where(a > 0.5, b, c, **kw),
        "addcmul": lambda a, b, c, **kw: ttnn.addcmul(a, b, c, 0.5, **kw),
        "addcdiv": lambda a, b, c, **kw: ttnn.addcdiv(a, b, c + 1.0, 0.5, **kw),
    }

    for op_name, op_fn in ternary_ops.items():
        try:
            input_a = create_input_tensor(device, shape)
            input_b = create_input_tensor(device, shape)
            input_c = create_input_tensor(device, shape)
            output = op_fn(input_a, input_b, input_c, memory_config=mem_config)
            ttnn.synchronize_device(device)
            results.append({
                "op": op_name,
                "type": "ternary",
                "shape": shape_name,
                "memory": mem_name,
                "status": "OK",
            })
            del output, input_a, input_b, input_c
        except Exception as e:
            results.append({
                "op": op_name,
                "type": "ternary",
                "shape": shape_name,
                "memory": mem_name,
                "status": f"FAIL: {str(e)[:80]}",
            })


def main():
    device = ttnn.open_device(device_id=0)

    results = []

    for shape_name, shape in SHAPES.items():
        for mem_name, mem_config in MEMORY_CONFIGS.items():
            print(f"\n{'='*70}")
            print(f"Shape: {shape_name} {shape} | Memory: {mem_name}")
            print(f"{'='*70}")

            # Unary ops
            run_unary_ops(device, shape, mem_config, results)

            # Binary ops
            run_binary_ops(device, shape, mem_config, results)

            # Ternary ops
            run_ternary_ops(device, shape, mem_config, results)

    # Print summary table
    print(f"\n\n{'='*100}")
    print(f"{'PERFORMANCE BENCHMARK SUMMARY':^100}")
    print(f"{'='*100}")
    print(f"{'Op':<12} {'Type':<8} {'Shape':<16} {'Memory':<22} {'Status':<40}")
    print(f"{'-'*100}")

    ok_count = 0
    fail_count = 0
    for r in results:
        status_str = r["status"]
        if status_str == "OK":
            ok_count += 1
        else:
            fail_count += 1
        print(f"{r['op']:<12} {r['type']:<8} {r['shape']:<16} {r['memory']:<22} {status_str:<40}")

    print(f"\n{'='*100}")
    print(f"Total: {len(results)} | Passed: {ok_count} | Failed: {fail_count}")
    print(f"{'='*100}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
