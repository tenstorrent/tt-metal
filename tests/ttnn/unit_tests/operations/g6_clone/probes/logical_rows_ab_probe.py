# Refinement 1 perf probe: logical_rows_partial_tile vs flat_page_range, device kernel time.
# Run explicitly under the profiler:
#   scripts/run_safe_pytest.sh --profile --no-py-zones \
#       tests/ttnn/unit_tests/operations/g6_clone/probes/logical_rows_ab_probe.py
# Each case runs 1 + REPEATS g6_clone calls, in parametrize order. Variants:
#   auto  - the shipped plan;
#   flat  - flat_page_range forced on a partial-tile shape (partial_tile_walk returns None);
#   merge - PIECE_MERGE_GAP_BYTES large: one span per partial page through the face gap;
#   g32   - pieces rounded to 32 B instead of the 64 B DRAM alignment (sub-64 B transfers).
# Read DEVICE KERNEL DURATION [ns] of the generic ops in CSV order, 1 + REPEATS per case.

import importlib

import pytest
import torch
import ttnn

op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
pd = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")

REPEATS = 5

PARTIAL = [((1, 1, 1, 62080), ttnn.bfloat16, "q8_bf16"), ((1, 5120), ttnn.bfloat16, "f13_bf16")]
# Wider face gaps than bf16's 448 B: f32 one-row tiles (960 B), and 17x16 tiles whose second piece
# is face 2 (a whole skipped face, 512 B bf16 / 1024 B f32).
GAP_SWEEP = [
    ((1, 1, 1, 62080), ttnn.float32, "q8_f32"),
    ((1, 1940, 17, 16), ttnn.bfloat16, "r17c16_bf16"),
    ((1, 1940, 17, 16), ttnn.float32, "r17c16_f32"),
]
FULL = [
    ((16, 12, 128, 128), ttnn.float32, "p1_f32"),
    ((8, 12, 128, 128), ttnn.float32, "p2_f32"),
    ((1, 1, 32, 62080), ttnn.bfloat16, "p4_bf16"),
]
CASES = (
    [
        pytest.param(shape, dtype, variant, id=f"{name}-{variant}")
        for shape, dtype, name in PARTIAL
        for variant in ("auto", "flat", "merge", "g32")
    ]
    + [
        pytest.param(shape, dtype, variant, id=f"{name}-{variant}")
        for shape, dtype, name in GAP_SWEEP
        for variant in ("auto", "merge")
    ]
    + [pytest.param(shape, dtype, "auto", id=f"{name}-auto") for shape, dtype, name in FULL]
)


@pytest.mark.parametrize("shape,dtype,variant", CASES)
def test_logical_rows_ab(device, shape, dtype, variant, monkeypatch):
    monkeypatch.setattr(op, "_DESCRIPTOR_CACHE", {})
    if variant == "flat":
        monkeypatch.setattr(pd, "partial_tile_walk", lambda *args: None)
    elif variant == "merge":
        monkeypatch.setattr(pd, "PIECE_MERGE_GAP_BYTES", 1 << 20)
    elif variant == "split":
        monkeypatch.setattr(pd, "PIECE_MERGE_GAP_BYTES", 0)
    elif variant == "g32":
        monkeypatch.setattr(ttnn, "get_dram_alignment", lambda: 32)
    torch.manual_seed(0)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.randn(shape, dtype=torch_dtype)
    x = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    out = op.g6_clone(x)
    for _ in range(REPEATS):
        out = op.g6_clone(x)
    ttnn.synchronize_device(device)
    plan = op.g6_clone_plan(x)
    print(
        f"PLAN {variant} {shape} {dtype} regime={plan['regime']} pages={plan['tensor_pages']} "
        f"page_bytes={plan['aligned_page_bytes']} cores={plan['num_cores']} block_pages={plan['block_pages']} "
        f"kinds={plan['partial_tile_walk'] and plan['partial_tile_walk']['kinds']}"
    )
    bits = torch.int16 if dtype == ttnn.bfloat16 else torch.int32
    assert torch.equal(ttnn.to_torch(out).view(bits), torch_input.view(bits))
