# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ternary (where / addcmul) program-cache behavior after the migration off the legacy
``get_dynamic_runtime_args`` cache-hit hook onto the top-level ``override_runtime_arguments`` hook.

TernaryDeviceOperation::compute_program_hash() deliberately EXCLUDES two axes that vary per dispatch:
  * buffer base addresses of every input/output tensor (never part of any hash), and
  * scalar_input_a / scalar_input_b (addcmul/addcdiv value, where-scalar operand).

On a cache HIT the descriptor is NOT rebuilt, so both of those must be re-derived and re-applied to the
cached program by override_runtime_arguments(); otherwise the hit reuses the first-miss buffer addresses
and scalar and silently corrupts the result. These tests pin that contract:

  * addresses re-applied  -> fresh allocations on each hit still produce the correct result,
  * in-place alias correct -> output_tensor aliasing an input writes the output slot from the output
    tensor (the resolve_bindings std::find first-occurrence bug class),
  * scalar re-applied      -> a different scalar changes the output without growing the cache,
  * one shared entry       -> asserts a HIT (not a rebuild masking a frozen-arg bug).

Mirrors tests/.../test_binary_ng_program_cache.py.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def _to_tt(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


# =============================================================================
# where (TTT): buffer-address re-application on the cache-hit path
# =============================================================================


def test_where_cache_hit_readdresses_interleaved(device, isolate_program_cache):
    """where(predicate, true, false) over INTERLEAVED tensors, repeated at the SAME shape but with
    freshly-allocated operands kept alive, so each cache HIT sees DIFFERENT buffer addresses. The
    three input addresses ride in the reader runtime args and the output address in the writer
    runtime args; override_runtime_arguments must re-apply all of them from the current tensors on the
    hit (no rebuild) or the result is stale. Regression guard for the address axis that resolve_bindings
    used to own."""
    shape = [1, 1, 96, 64]
    keep_alive = []  # hold refs so each iteration's tensors get fresh (different) addresses
    for i in range(4):
        torch.manual_seed(i)
        pred = torch.rand(shape) > 0.5
        t = torch.rand(shape, dtype=torch.bfloat16)
        f = torch.rand(shape, dtype=torch.bfloat16)

        tt_pred = _to_tt(pred.to(torch.bfloat16), device)
        tt_t = _to_tt(t, device)
        tt_f = _to_tt(f, device)
        tt_out = ttnn.where(tt_pred, tt_t, tt_f)
        keep_alive += [tt_pred, tt_t, tt_f, tt_out]

        ref = torch.where(pred, t, f)
        assert_with_pcc(ref, ttnn.to_torch(tt_out), 0.999)

    # One shared program reused across all four differently-addressed hits.
    assert device.num_program_cache_entries() == 1


def test_where_cache_hit_same_entry_different_outer_dims(device, isolate_program_cache):
    """Two shapes with the SAME per-tensor volume AND the same last-two dims (H, W) but different outer
    dims hit the SAME cache entry (ternary keys volume + H,W, so these collide). override must re-derive
    every per-core arg (D/N/C, strides, start_id, tile counts) for the current tensors — proving the
    shape-derived args are re-applied on the hit, not frozen at first miss."""

    def run(shape, seed):
        torch.manual_seed(seed)
        pred = torch.rand(shape) > 0.5
        t = torch.rand(shape, dtype=torch.bfloat16)
        f = torch.rand(shape, dtype=torch.bfloat16)
        tt_out = ttnn.where(_to_tt(pred.to(torch.bfloat16), device), _to_tt(t, device), _to_tt(f, device))
        return torch.where(pred, t, f), ttnn.to_torch(tt_out)

    ref1, out1 = run([1, 2, 32, 32], 0)  # miss
    assert_with_pcc(ref1, out1, 0.999)

    ref2, out2 = run([2, 1, 32, 32], 1)  # HIT on the same entry (same volume + H,W)
    assert_with_pcc(ref2, out2, 0.999)

    assert device.num_program_cache_entries() == 1  # proves it was a hit, not a rebuild


def test_where_cache_correctness_repeated(device, isolate_program_cache):
    """Repeated where at the same config with different data — all results correct, one cache entry."""
    shape = [1, 1, 64, 128]
    for i in range(5):
        torch.manual_seed(i)
        pred = torch.rand(shape) > 0.5
        t = torch.rand(shape, dtype=torch.bfloat16)
        f = torch.rand(shape, dtype=torch.bfloat16)
        tt_out = ttnn.where(_to_tt(pred.to(torch.bfloat16), device), _to_tt(t, device), _to_tt(f, device))
        assert_with_pcc(torch.where(pred, t, f), ttnn.to_torch(tt_out), 0.999)
    assert device.num_program_cache_entries() == 1


# =============================================================================
# addcmul (TTT + scalar): in-place alias + scalar re-application on the cache-hit path
# =============================================================================


def _addcmul_ref(a, b, c, value):
    return a + value * (b * c)


def test_addcmul_inplace_cache_reuse_readdresses(device, isolate_program_cache):
    """In-place addcmul (output_tensor aliases input_a) over INTERLEAVED tensors, repeated with
    freshly-allocated operands kept alive so each cache HIT sees different addresses. The output slot
    must be written from the output tensor (== input_a here) on every hit. override re-derives every
    rt-arg address for the actual current tensors, so the aliased in-place case can't be mis-patched
    the way resolve_bindings' first-occurrence std::find could."""
    shape = [1, 1, 64, 64]
    keep_alive = []
    for i in range(4):
        torch.manual_seed(i)
        a = torch.rand(shape, dtype=torch.bfloat16)
        b = torch.rand(shape, dtype=torch.bfloat16)
        c = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = _to_tt(a, device)
        tt_b = _to_tt(b, device)
        tt_c = _to_tt(c, device)
        tt_out = ttnn.addcmul(tt_a, tt_b, tt_c, value=2.0, output_tensor=tt_a)  # in-place onto a
        # Prove it is ACTUALLY in-place: otherwise PCC + single-entry would pass without testing the alias.
        assert tt_out.buffer_address() == tt_a.buffer_address()
        keep_alive += [tt_a, tt_b, tt_c, tt_out]
        assert_with_pcc(_addcmul_ref(a, b, c, 2.0), ttnn.to_torch(tt_out), 0.999)

    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize("first_inplace", [True, False], ids=["inplace_first", "outofplace_first"])
def test_addcmul_mixed_inplace_outofplace_interleaved(device, isolate_program_cache, first_inplace):
    """One cached INTERLEAVED program reused across a MIX of in-place (output aliases input_a) and
    out-of-place calls sharing a single cache entry, in both orders. The legacy resolve_bindings maps
    an aliased buffer to its FIRST occurrence, so a program built under one aliasing pattern and reused
    under another would patch the writer's output address from the wrong slot. override re-derives every
    address for the current tensors, so both orders must stay correct."""

    def do(seed, inplace):
        torch.manual_seed(seed)
        a = torch.rand([1, 1, 32, 64], dtype=torch.bfloat16)
        b = torch.rand([1, 1, 32, 64], dtype=torch.bfloat16)
        c = torch.rand([1, 1, 32, 64], dtype=torch.bfloat16)
        tt_a = _to_tt(a, device)
        tt_b = _to_tt(b, device)
        tt_c = _to_tt(c, device)
        out = tt_a if inplace else None
        tt_out = ttnn.addcmul(tt_a, tt_b, tt_c, value=3.0, output_tensor=out)
        if inplace:
            assert tt_out.buffer_address() == tt_a.buffer_address()
        return _addcmul_ref(a, b, c, 3.0), ttnn.to_torch(tt_out)

    for i, inplace in enumerate([first_inplace, not first_inplace, first_inplace, not first_inplace]):
        ref, out = do(i, inplace)
        assert_with_pcc(ref, out, 0.999)
    assert device.num_program_cache_entries() == 1


def test_addcmul_scalar_reapplied_on_hit(device, isolate_program_cache):
    """The addcmul scalar (scalar_input_a) is EXCLUDED from the hash. A different value must be a cache
    HIT (no new entry) yet change the output (scalar re-applied by override), and the same value must
    reproduce the output. This is the axis get_dynamic_runtime_args used to own; it now flows through
    override_runtime_arguments."""
    shape = [1, 1, 128, 128]
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)
    c = torch.rand(shape, dtype=torch.bfloat16)
    tt_a = _to_tt(a, device)
    tt_b = _to_tt(b, device)
    tt_c = _to_tt(c, device)

    out_v1 = ttnn.to_torch(ttnn.addcmul(tt_a, tt_b, tt_c, value=2.0)).float().clone()
    assert device.num_program_cache_entries() == 1

    out_v1_again = ttnn.to_torch(ttnn.addcmul(tt_a, tt_b, tt_c, value=2.0)).float().clone()
    assert device.num_program_cache_entries() == 1, "same scalar must reuse the cached program"
    assert torch.equal(out_v1, out_v1_again), "same scalar must reproduce identical output"

    out_v2 = ttnn.to_torch(ttnn.addcmul(tt_a, tt_b, tt_c, value=7.0)).float().clone()
    assert device.num_program_cache_entries() == 1, (
        "a different scalar must NOT create a new cache entry -- the scalar is dynamic, re-applied by "
        "override_runtime_arguments, not part of the program hash"
    )
    assert not torch.equal(out_v1, out_v2), (
        "a different scalar must change the output (scalar re-applied on the cache hit). Identical "
        "output here means the hit reused the first call's frozen scalar runtime arg"
    )


# =============================================================================
# where (TTT) SHARDED: tensor-backed CB base-address re-application by CBIndex
# =============================================================================


def test_where_sharded_cache_hit_readdresses(device, isolate_program_cache):
    """where over HEIGHT-SHARDED tensors, repeated at the SAME shard config (sharded ternary keys
    shard_volume, so a different shape would MISS) with freshly-allocated operands kept alive so each
    cache HIT sees a DIFFERENT buffer address. For sharded ternary the input/output addresses ride on
    tensor-backed (globally-allocated) circular-buffer base addresses; override must re-point every one
    by CBIndex from the current tensors on the hit (no rebuild) or the result is stale. get_dynamic
    could never touch CBs at all — this is exactly the axis it could not cover."""
    shape = [1, 1, 256, 256]
    mem = ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=8, x=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    keep_alive = []
    for i in range(4):
        torch.manual_seed(i)
        pred = torch.rand(shape) > 0.5
        t = torch.rand(shape, dtype=torch.bfloat16)
        f = torch.rand(shape, dtype=torch.bfloat16)
        tt_pred = _to_tt(pred.to(torch.bfloat16), device, memory_config=mem)
        tt_t = _to_tt(t, device, memory_config=mem)
        tt_f = _to_tt(f, device, memory_config=mem)
        tt_out = ttnn.where(tt_pred, tt_t, tt_f, memory_config=mem)
        keep_alive += [tt_pred, tt_t, tt_f, tt_out]
        assert_with_pcc(torch.where(pred, t, f), ttnn.to_torch(tt_out), 0.999)

    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize("first_inplace", [True, False], ids=["inplace_first", "outofplace_first"])
def test_where_mixed_inplace_outofplace_sharded(device, isolate_program_cache, first_inplace):
    """One cached SHARDED where program reused across a MIX of in-place (output aliases the value_true
    input) and out-of-place calls sharing a single cache entry, in both orders. For sharded ops the
    input/output addresses ride on tensor-backed CB base addresses; the legacy path resolved these by
    first-occurrence (and get_dynamic could not touch CBs), so a program built under one aliasing
    pattern and reused under another mis-resolved the output CB. override re-applies the CB addresses by
    CBIndex from the current tensors, so both orders must stay correct."""
    shape = [1, 1, 256, 256]
    mem = ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=8, x=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    keep_alive = []

    def do(seed, inplace):
        torch.manual_seed(seed)
        pred = torch.rand(shape) > 0.5
        t = torch.rand(shape, dtype=torch.bfloat16)
        f = torch.rand(shape, dtype=torch.bfloat16)
        tt_pred = _to_tt(pred.to(torch.bfloat16), device, memory_config=mem)
        tt_t = _to_tt(t, device, memory_config=mem)
        tt_f = _to_tt(f, device, memory_config=mem)
        out = tt_t if inplace else None  # in-place onto value_true (output dtype == value_true dtype)
        tt_out = ttnn.where(tt_pred, tt_t, tt_f, memory_config=mem, output_tensor=out)
        if inplace:
            assert tt_out.buffer_address() == tt_t.buffer_address()
        keep_alive.extend([tt_pred, tt_t, tt_f, tt_out])
        return torch.where(pred, t, f), ttnn.to_torch(tt_out)

    for i, inplace in enumerate([first_inplace, not first_inplace, first_inplace, not first_inplace]):
        ref, out = do(i, inplace)
        assert_with_pcc(ref, out, 0.999)
    assert device.num_program_cache_entries() == 1
