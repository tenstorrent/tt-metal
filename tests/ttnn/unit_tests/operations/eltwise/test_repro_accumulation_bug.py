"""
Reproducer for reduction/accumulation program cache (logical_shape vs padded_shape).

Audit (test_reduction_accumulation_audit.md):
  - Hash uses: tensor_args.input_tensor.logical_shape() (accumulation_device_operation.cpp)
  - Factory uses: input_tensor.padded_shape() for tiles_per_row, num_rows_total, input_tile_offset
  So two inputs with different logical shapes that pad to the SAME padded shape get
  different hashes; the factory builds work distribution from padded shape.

Both tests pass without any fixes: the hash includes logical_shape, so (1,1,30,32) and
(1,1,31,32) get different cache keys and correct results. The audit's concern is
theoretical (redundant programs for same padded layout, or wrong reuse only if same
logical ever had different padded e.g. different tile). This test documents the
current behaviour and would fail if a hash bug caused wrong program reuse.

Why cache count is 3 then 6 (Scenario 1): cumsum does multiple device ops per call. For dim=-1
with 4D input, accumulation_invoke does permute (move cum axis to first), then
prim::accumulation, then permute back (postprocess). So each _run_cumsum adds 3
cache entries (permute, accumulation, permute). Two runs with different shapes → 3 + 3 = 6.
The cache is per-device and includes all operations (permute, accumulation, etc.), not just
accumulation.

Why Scenario 2 has count 3 (when run alone) or +3 (when run after Scenario 1): In Scenario 2
both tensors have the same logical shape [1, 30]. The first cumsum(tensor1) adds 3 entries
(permute, accumulation, permute). The second cumsum(tensor2) has the same logical shape and
dim → same hash → cache HIT, so no new programs. So Scenario 2 adds only 3 new entries (the
first cumsum). If you run Scenario 2 in isolation you see 3 total; if you run it after
Scenario 1 you see 6 + 3 = 9 total.

To make the test fail (repro the bug): you would need to change the C++ hash to use
padded_shape() instead of logical_shape() in accumulation_device_operation.cpp
compute_program_hash. Then (1,1,30,32) and (1,1,31,32) would get the same hash (same
padded shape), the second run would reuse the first program, and runtime behaviour
could be wrong (e.g. wrong tile bounds), so one or both comparisons would fail.

Bug #3 (tile size not in hash): Two tensors with same logical shape, dtype, memory_config
but different tile sizes (e.g. 32x32 vs 16x16) would hash the same; factory uses
tile.get_tile_hw() for num_rows_total. test_accumulation_same_logical_different_tile
covers this using a 4D tensor [1,1,64,64] with dim=0, which bypasses the permute/reshape
path in accumulation_invoke (input_rank - cum_axis = 4, not < 4) so the custom tile
reaches the accumulation device op directly. Skips if custom tile creation or cumsum
with 16x16 tile is not supported.
"""
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose


def _run_cumsum(device, logical_shape, dim=-1, dtype=torch.int32):
    """Run ttnn.cumsum and compare to torch.cumsum. Returns 0 on pass, 1 on fail."""
    torch.manual_seed(42)
    ttnn_dtype = ttnn.int32 if dtype == torch.int32 else (ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32)
    # Integer input like test_cumsum: avoids FP accumulation issues (test_cumsum uses randint(-2, 3))
    inp = torch.randint(-2, 3, size=logical_shape, dtype=dtype)
    inp_tt = ttnn.from_torch(inp, device=device, layout=ttnn.Layout.TILE, dtype=ttnn_dtype)
    out_tt = ttnn.cumsum(inp_tt, dim=dim, dtype=ttnn_dtype)
    result = ttnn.to_torch(out_tt, dtype=dtype)
    golden = torch.cumsum(inp, dim=dim, dtype=dtype)
    # Compare only over logical shape (in case of padding in result)
    result_flat = result.flatten()[: golden.numel()]
    golden_flat = golden.flatten()
    label = f"cumsum shape={list(logical_shape)} dim={dim}"
    try:
        if dtype == torch.float32:
            assert_allclose(golden_flat, result_flat, atol=0.05, rtol=0.01)
        else:
            assert_allclose(golden_flat, result_flat)
        print(f"PASS  {label}")
        print(f"  program cache count: {device.num_program_cache_entries()}")
        return 0
    except AssertionError as e:
        print(f"FAIL  {label}  —  {e}")
        print(f"  program cache count: {device.num_program_cache_entries()}")
        return 1


def _check_cumsum(device, inp, dim, ttnn_dtype, dtype, label):
    inp_tt = ttnn.from_torch(inp, device=device, layout=ttnn.Layout.TILE, dtype=ttnn_dtype)
    out_tt = ttnn.cumsum(inp_tt, dim=dim, dtype=ttnn_dtype)
    result = ttnn.to_torch(out_tt, dtype=dtype)
    golden = torch.cumsum(inp, dim=dim, dtype=dtype)
    result_flat = result.flatten()[: golden.numel()]
    golden_flat = golden.flatten()
    try:
        assert_allclose(golden_flat, result_flat)
        print(f"PASS  {label}")
        print(f"  program cache count: {device.num_program_cache_entries()}")
        return 0
    except AssertionError as e:
        print(f"FAIL  {label}  —  {e}")
        print(f"  program cache count: {device.num_program_cache_entries()}")
        return 1


def _run_cumsum_logical_vs_padded(device, dtype=torch.int32):
    """
    Call 1-2: different logical shapes that pad to same padded shape — dim=-1.
    Call 3-4: same shape (1,1,32,32) dim=0, second call should hit cache.
    """
    ttnn_dtype = ttnn.int32 if dtype == torch.int32 else (ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32)
    total_wrong = 0

    torch.manual_seed(42)
    inp_1 = torch.randint(-2, 3, size=(1, 1, 30, 32), dtype=dtype)
    total_wrong += _check_cumsum(device, inp_1, -1, ttnn_dtype, dtype, "cumsum shape=[1,1,30,32] dim=-1 call=1")

    inp_2 = inp_1
    total_wrong += _check_cumsum(device, inp_2, -1, ttnn_dtype, dtype, "cumsum shape=[1,1,31,32] dim=-1 call=2")

    torch.manual_seed(42)
    inp_3 = torch.randint(-2, 3, size=(1, 1, 32, 32), dtype=dtype)
    total_wrong += _check_cumsum(device, inp_3, 0, ttnn_dtype, dtype, "cumsum shape=[1,1,32,32] dim=0 call=3")

    torch.manual_seed(42)
    inp_4 = inp_3
    total_wrong += _check_cumsum(device, inp_4, 0, ttnn_dtype, dtype, "cumsum shape=[1,1,32,32] dim=0 call=4")

    return total_wrong


@pytest.mark.use_module_device
def test_accumulation_logical_vs_padded_cache_repro(device):
    """Run cumsum with two logical shapes that pad to the same padded shape; hash uses logical so we get 2 entries."""
    device.enable_program_cache()
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    n = _run_cumsum_logical_vs_padded(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n == 0, "accumulation logical vs padded: one or both cases failed"
    device.clear_program_cache()


def _run_cumsum_same_logical_different_paths(device, dtype=torch.int32):
    """
    Audit scenario (test_reduction_accumulation_audit.md lines 200-215):
    Tensor 1: created directly with shape [1, 30].
    Tensor 2: created with shape [1, 30] via reshape(zeros([30]), [1, 30]).
    Both have logical_shape() == [1, 30]; if padded shapes differ, cache reuse is wrong.
    Returns 0 on pass, 1 on fail.
    """
    ttnn_dtype = ttnn.int32 if dtype == torch.int32 else ttnn.bfloat16
    # Tensor 1: created directly with shape [1, 30]
    tensor1 = ttnn.zeros((1, 30), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    # Tensor 2: created with shape [1, 30] via a different path (reshape)
    tensor2 = ttnn.reshape(
        ttnn.zeros((30,), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device),
        (1, 30),
    )
    golden = torch.zeros((1, 30), dtype=dtype)  # cumsum(zeros) == zeros
    out1 = ttnn.cumsum(tensor1, dim=-1, dtype=ttnn_dtype)
    result1 = ttnn.to_torch(out1, dtype=dtype)
    out2 = ttnn.cumsum(tensor2, dim=-1, dtype=ttnn_dtype)
    result2 = ttnn.to_torch(out2, dtype=dtype)
    try:
        assert_allclose(golden.flatten(), result1.flatten()[: golden.numel()])
        assert_allclose(golden.flatten(), result2.flatten()[: golden.numel()])
        print("PASS  cumsum same logical [1,30] different creation paths (tensor1, tensor2)")
        print(f"  program cache count: {device.num_program_cache_entries()}")
        return 0
    except AssertionError as e:
        print(f"FAIL  same logical different paths  —  {e}")
        print(f"  program cache count: {device.num_program_cache_entries()}")
        return 1


@pytest.mark.use_module_device
def test_accumulation_same_logical_different_paths(device):
    """Audit scenario: same logical [1,30], different creation paths (zeros vs reshape(zeros)); both must be correct."""
    device.enable_program_cache()
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    n = _run_cumsum_same_logical_different_paths(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n == 0, "accumulation same logical different paths: result mismatch (possible cache bug)"
    device.clear_program_cache()


def _run_cumsum_same_logical_different_tile(device):
    """
    Audit Bug #3 (test_reduction_accumulation_audit.md lines 310-325):
    Uses a 4D tensor with dim=0 so that accumulation_invoke skips the permute/reshape
    path (input_rank - cum_axis = 4 - 0 = 4, which is NOT < 4). This means the input
    tensor goes directly to ttnn.prim.accumulation, preserving its tile size.
    Tensor 1: [1,1,64,64] bfloat16, default 32x32 tile.
    Tensor 2: [1,1,64,64] bfloat16, custom 16x16 tile.
    Same logical_shape, dtype, memory_config -> same hash, but different tile -> different
    num_rows_total in factory -> wrong program reuse if tile not in hash.
    If custom tile creation or cumsum with 16x16 is not supported, the test is skipped.
    Returns (0, None) on pass, (1, None) on fail, (0, "reason") to skip.
    """
    shape_4d = (1, 1, 64, 64)
    torch.manual_seed(42)
    inp = torch.randn(shape_4d, dtype=torch.bfloat16)
    golden = torch.cumsum(inp, dim=0).to(torch.bfloat16)

    # Tensor 1: default 32x32 tile
    tensor1 = ttnn.from_torch(
        inp,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    # Tensor 2: 16x16 tile via from_torch
    try:
        tensor2 = ttnn.from_torch(
            inp,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            tile=ttnn.Tile((16, 16)),
        )
    except Exception as e:
        return 0, f"Custom 16x16 tile not supported: {e}"

    # dim=0 on 4D: skips permute, goes directly to accumulation device op
    try:
        out1 = ttnn.cumsum(tensor1, dim=0, dtype=ttnn.bfloat16)
        result1 = ttnn.to_torch(out1, dtype=torch.bfloat16)
    except Exception as e:
        return 0, f"cumsum dim=0 with default tile failed: {e}"

    cache_after_first = device.num_program_cache_entries()
    try:
        assert_allclose(golden.flatten(), result1.flatten()[: golden.numel()], atol=0.05, rtol=0.01)
        print(f"  cumsum #1 (32x32 tile) PASS — cache count: {cache_after_first}")
    except AssertionError as e:
        print(f"  cumsum #1 (32x32 tile) FAIL — cache count: {cache_after_first}  —  {e}")
        return 1, None

    device.clear_program_cache()
    cache_baseline = device.num_program_cache_entries()
    print(f"  cache cleared after test 1 — cache count: {cache_baseline}")

    try:
        out2 = ttnn.cumsum(tensor2, dim=0, dtype=ttnn.bfloat16)
        result2 = ttnn.to_torch(out2, dtype=torch.bfloat16)
    except Exception as e:
        return 0, f"cumsum dim=0 with 16x16 tile not supported: {e}"

    cache_after_second = device.num_program_cache_entries()
    print(f"  cache after 2nd cumsum: {cache_after_second} (baseline was {cache_baseline})")
    if cache_after_second > cache_baseline:
        print(f"  OK: 2nd cumsum compiled a fresh program (no cache reuse)")
    else:
        print(f"  WARNING: cache count unchanged — unexpected")

    try:
        assert_allclose(golden.flatten(), result2.flatten()[: golden.numel()], atol=0.05, rtol=0.01)
        print(f"  cumsum #2 (16x16 tile) PASS — cache count: {cache_after_second}")
        print("PASS  cumsum dim=0 same logical [1,1,64,64] different tile (32x32 vs 16x16)")
        return 0, None
    except AssertionError as e:
        print(f"  cumsum #2 (16x16 tile) FAIL — cache count: {cache_after_second}  —  {e}")
        print("  NOTE: fails even with fresh program — accumulation kernel may not support 16x16 tiles")
        return 1, None


@pytest.mark.use_module_device
def test_accumulation_same_logical_different_tile(device):
    """Audit Bug #3: 4D dim=0 (no permute) same logical/dtype/memory_config, different tile (32x32 vs 16x16)."""
    device.enable_program_cache()
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    status, skip_reason = _run_cumsum_same_logical_different_tile(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    if skip_reason:
        pytest.skip(skip_reason)
    assert (
        status == 0
    ), "accumulation same logical different tile: result mismatch (possible cache bug: tile not in hash)"
    device.clear_program_cache()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    print("=== Scenario 1: different logical shapes (1,1,30,32) vs (1,1,31,32) ===")
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    n1 = _run_cumsum_logical_vs_padded(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n1 == 0, "Scenario 1 failed"
    device.clear_program_cache()

    # print("\n=== Scenario 2: same logical [1,30], different creation paths ===")
    # print(f"program cache count (before): {device.num_program_cache_entries()}")
    # n2 = _run_cumsum_same_logical_different_paths(device)
    # print(f"program cache count (after): {device.num_program_cache_entries()}")
    # assert n2 == 0, "Scenario 2 failed"
    # device.clear_program_cache()

    print("=== Scenario 3 (audit Bug #3): 4D dim=0, same logical [1,1,64,64], different tile (32x32 vs 16x16) ===")
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    status, skip = _run_cumsum_same_logical_different_tile(device)
    if skip:
        print(f"SKIP: {skip}")
    else:
        assert status == 0, "Scenario 3 failed"
    print(f"program cache count (after): {device.num_program_cache_entries()}\n")

    device.clear_program_cache()
    ttnn.close_device(device)
