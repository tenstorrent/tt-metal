# Verifier perf probe: device kernel time of g6_clone on the perf-relevant shapes (> 32 tiles).
# Run explicitly under the profiler:
#   scripts/run_safe_pytest.sh --profile --no-py-zones \
#       tests/ttnn/unit_tests/operations/g6_clone/probes/perf_baseline_probe.py
# Each shape runs one warm-up call and REPEATS timed calls; read DEVICE KERNEL DURATION [ns] per
# input shape from the printed ops_perf_results CSV.
#
# SUPPORTED is widened in-process to TARGET so the bfloat16 shapes run on the Phase 0 kernels
# (probe_006 showed every refused cell passes bit-exact with no kernel change). This is a
# measurement probe, not a support claim: the op file's SUPPORTED is unchanged.

import importlib

import pytest
import torch
import ttnn

op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")

REPEATS = 5

SHAPES = [
    pytest.param((16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, id="p1_f32"),
    pytest.param((8, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, id="p2_f32"),
    pytest.param((1, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, id="q13_f32"),
    pytest.param((1, 512, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, id="p3_bf16"),
    pytest.param((1, 1, 32, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT, id="p4_bf16"),
    pytest.param((1, 1, 1, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT, id="partial_62080_bf16"),
    pytest.param((1, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, id="partial_5120_bf16"),
    pytest.param((16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="rm_16x3x2560_bf16"),
]


@pytest.mark.parametrize("shape,dtype,layout", SHAPES)
def test_perf_probe(device, shape, dtype, layout, monkeypatch):
    monkeypatch.setitem(op.SUPPORTED, "dtype", [ttnn.float32, ttnn.bfloat16, ttnn.int32, ttnn.uint32])
    monkeypatch.setitem(op.SUPPORTED, "layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
    torch.manual_seed(0)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = op.g6_clone(x)
    for _ in range(REPEATS):
        out = op.g6_clone(x)
    ttnn.synchronize_device(device)
    plan = op.g6_clone_plan(x)
    print(
        f"PLAN {shape} {dtype} {layout} pages={plan['tensor_pages']} page_bytes={plan['aligned_page_bytes']} "
        f"cores={plan['num_cores']} block_pages={plan['block_pages']} depth={plan['cb_depth_blocks']}"
    )
    assert torch.equal(
        ttnn.to_torch(out).view(torch.int16 if dtype == ttnn.bfloat16 else torch.int32),
        ttnn.to_torch(x).view(torch.int16 if dtype == ttnn.bfloat16 else torch.int32),
    )
