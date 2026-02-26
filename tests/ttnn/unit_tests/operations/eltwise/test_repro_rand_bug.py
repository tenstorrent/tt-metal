"""
Reproducer for rand program cache — seed vs from/to handling.

Hash behaviour (rand_device_operation.cpp):
  - compute_program_hash copies operation_attributes, zeros out seed, then hashes.
  - So seed is excluded from the hash (it's a runtime arg updated in override_runtime_arguments).
  - from and to ARE in the hash (they remain in the copied attributes).

Program factory (rand_program_factory.cpp):
  - create(): seed, from, to are all passed as runtime args to the compute kernel.
  - override_runtime_arguments(): only seed (and output address) are updated on cache hit.
    from and to are NOT updated on cache hit.

Consequence: if from/to were removed from the hash, a cache hit with different from/to
would reuse the stale from/to values baked into the first program's runtime args.
Currently this is safe because from/to stay in the hash, so different from/to → cache miss → fresh program.

Test 1 (same from/to, different seed): two calls with identical shape/dtype/layout/memory_config
and same from/to but different seeds. Hash is the same → cache hit → override_runtime_arguments
updates the seed → both calls produce valid random output in the correct range, but with
different values (different seed).

Test 2 (different from/to): two calls with identical shape/dtype/layout/memory_config but
different from/to ranges. Hash differs → cache miss → each gets its own program with the
correct from/to baked in.
"""
import pytest
import torch
import ttnn


def _check_range(tensor, low, high, label):
    """Check that all values in tensor are within [low, high). Returns number of out-of-range elements."""
    t = ttnn.to_torch(tensor).float()
    eps = 1e-3
    out_of_range = ((t < low - eps) | (t >= high + eps)).sum().item()
    if out_of_range == 0:
        print(f"PASS  {label}  — all values in [{low}, {high})")
    else:
        print(f"FAIL  {label}  — {out_of_range}/{t.numel()} values out of [{low}, {high})")
    return out_of_range


def _run_rand_same_range_different_seed(device, dtype=ttnn.bfloat16):
    """
    Two rand calls: same shape, dtype, layout, memory_config, same from/to, different seed.
    Hash is the same (seed is zeroed in hash) → cache hit on second call.
    override_runtime_arguments updates seed → both should produce correct range but different values.
    """
    shape = (1, 1, 64, 64)
    low, high = 0.0, 1.0
    seed_1, seed_2 = 42, 99

    out_1 = ttnn.rand(shape, device, dtype=dtype, low=low, high=high, seed=seed_1)
    cache_after_1 = device.num_program_cache_entries()
    print(f"  call 1 (seed={seed_1}): cache count = {cache_after_1}")

    out_2 = ttnn.rand(shape, device, dtype=dtype, low=low, high=high, seed=seed_2)
    cache_after_2 = device.num_program_cache_entries()
    print(f"  call 2 (seed={seed_2}): cache count = {cache_after_2}")

    n_wrong = 0
    n_wrong += _check_range(out_1, low, high, f"rand seed={seed_1}")
    n_wrong += _check_range(out_2, low, high, f"rand seed={seed_2}")

    t1 = ttnn.to_torch(out_1).flatten()
    t2 = ttnn.to_torch(out_2).flatten()
    if torch.equal(t1, t2):
        print("  WARNING: both outputs are identical — seed may not have been updated")
    else:
        print("  OK: outputs differ (different seeds produced different values)")

    return n_wrong


def _run_rand_different_range(device, dtype=ttnn.bfloat16):
    """
    Two rand calls: same shape, dtype, layout, memory_config, but different from/to.
    Hash should differ (from/to are in the hash) → cache miss on second call.
    Each call gets its own program with the correct from/to.
    """
    shape = (1, 1, 64, 64)
    seed = 42

    low_1, high_1 = 0.0, 1.0
    low_2, high_2 = 5.0, 10.0

    out_1 = ttnn.rand(shape, device, dtype=dtype, low=low_1, high=high_1, seed=seed)
    cache_after_1 = device.num_program_cache_entries()
    print(f"  call 1 (range=[{low_1}, {high_1})): cache count = {cache_after_1}")

    out_2 = ttnn.rand(shape, device, dtype=dtype, low=low_2, high=high_2, seed=seed)
    cache_after_2 = device.num_program_cache_entries()
    print(f"  call 2 (range=[{low_2}, {high_2})): cache count = {cache_after_2}")

    if cache_after_2 > cache_after_1:
        print("  OK: second call caused cache miss (different from/to → different hash)")
    else:
        print("  WARNING: cache count unchanged — from/to may not be in the hash")

    n_wrong = 0
    n_wrong += _check_range(out_1, low_1, high_1, f"rand range=[{low_1}, {high_1})")
    n_wrong += _check_range(out_2, low_2, high_2, f"rand range=[{low_2}, {high_2})")

    return n_wrong


@pytest.mark.use_module_device
def test_rand_same_range_different_seed(device):
    """Same from/to, different seed → cache hit, override_runtime_arguments updates seed."""
    device.enable_program_cache()
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    n = _run_rand_same_range_different_seed(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n == 0, "rand same range different seed: values out of range"
    device.clear_program_cache()


@pytest.mark.use_module_device
def test_rand_different_range(device):
    """Different from/to → cache miss, each call gets correct range."""
    device.enable_program_cache()
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    n = _run_rand_different_range(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n == 0, "rand different range: values out of expected range (possible cache collision on from/to)"
    device.clear_program_cache()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    print("=== Test 1: same from/to, different seed (cache hit expected) ===")
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    _run_rand_same_range_different_seed(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")

    device.clear_program_cache()

    print("\n=== Test 2: different from/to (cache miss expected) ===")
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    _run_rand_different_range(device)
    print(f"program cache count (after): {device.num_program_cache_entries()}")

    device.clear_program_cache()
    ttnn.close_device(device)
