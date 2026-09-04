# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_ulp, assert_equal


@pytest.mark.parametrize(
    "c_shape, t_shape, f_shape",
    [
        ((8, 4, 1), (8, 4, 768), (1, 1, 1)),  # Ccol, Tfull, Fscalar
        ((8, 1, 768), (8, 4, 768), (1, 1, 1)),  # Crow, Tfull, Fscalar
        ((8, 4, 768), (8, 4, 1), (8, 1, 768)),  # Cfull, Tcol, Frow
        ((8, 4, 768), (8, 1, 768), (8, 4, 1)),  # Cfull, Trow, Fcol
        ((8, 4, 1), (8, 1, 768), (8, 4, 768)),  # Ccol, Trow, Ffull
        ((8, 1, 768), (8, 4, 1), (8, 4, 768)),  # Crow, Tcol, Ffull
        ((8, 1, 768), (8, 4, 768), (8, 4, 1)),  # Crow, Tfull, Fcol
        ((8, 4, 1), (8, 4, 768), (8, 1, 768)),  # Ccol, Tfull, Frow
        ((1, 1, 1), (8, 4, 1), (8, 1, 768)),  # Cscalar, Tcol, Frow
        ((1, 1, 1), (8, 1, 768), (8, 4, 1)),  # Cscalar, Trow, Fcol
        ((8, 1, 768), (1, 1, 1), (8, 4, 1)),  # Crow, Tscalar, Fcol
        ((8, 4, 1), (1, 1, 1), (8, 1, 768)),  # Ccol, Tscalar, Frow
    ],
)
def test_ttnn_where_row_col_mixed_bcast(c_shape, t_shape, f_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape).to(torch.bfloat16)
    T = torch.randn(t_shape, dtype=torch.bfloat16)
    F = torch.ones(f_shape, dtype=torch.bfloat16) * 10
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


@pytest.mark.parametrize(
    "c_shape, t_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Trow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Tcol
    ],
)
def test_ttnn_where_row_col_mixed_bcast_tts(c_shape, t_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape).to(torch.bfloat16)
    T = torch.randn(t_shape, dtype=torch.bfloat16)
    F = 10.0
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


@pytest.mark.parametrize(
    "c_shape, f_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Frow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Fcol
    ],
)
def test_ttnn_where_row_col_mixed_bcast_tst(c_shape, f_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape).to(torch.bfloat16)
    T = 10.0
    F = torch.randn(f_shape, dtype=torch.bfloat16)
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        ((8, 4, 768), (8, 4, 768), (8, 4, 768)),  # full, full, full
        ((8, 4, 1), (8, 4, 768), (8, 4, 768)),  # acol, bfull, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 768)),  # arow, bfull, cfull
        ((8, 4, 768), (8, 4, 1), (8, 1, 768)),  # afull, bcol, crow
        ((8, 4, 768), (8, 1, 768), (8, 4, 1)),  # afull, brow, ccol
        ((8, 4, 1), (8, 1, 768), (8, 4, 768)),  # acol, brow, cfull
        ((8, 1, 768), (8, 4, 1), (8, 4, 768)),  # arow, bcol, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 1)),  # arow, bfull, ccol
        ((8, 4, 1), (8, 4, 768), (8, 1, 768)),  # acol, bfull, crow
        ((1, 1, 1), (8, 4, 1), (8, 1, 768)),  # ascalar, bcol, crow
        ((1, 1, 1), (8, 1, 768), (8, 4, 1)),  # ascalar, brow, ccol
        ((8, 1, 768), (1, 1, 1), (8, 4, 1)),  # arow, bscalar, ccol
        ((8, 4, 1), (1, 1, 1), (8, 1, 768)),  # acol, bscalar, crow
    ],
)
@pytest.mark.parametrize("value", [1.5, 0.5, -0.25])
@pytest.mark.parametrize("ttnn_op", [ttnn.addcmul, ttnn.addcdiv])
def test_ttnn_addc_ops_row_col_mixed_bcast(a_shape, b_shape, c_shape, value, ttnn_op, device):
    torch.manual_seed(0)
    in_data1 = torch.empty(a_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data2 = torch.empty(b_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data3 = torch.empty(c_shape, dtype=torch.bfloat16).uniform_(-100, 100)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    golden = golden_fn(in_data1, in_data2, in_data3, value=value)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn_op(input_tensor1, input_tensor2, input_tensor3, value=value)
    result = ttnn.to_torch(ttnn_result)

    # For addcdiv: if denominator (in_data3) is zero, golden is nan but ttnn may return inf; normalize to nan
    if ttnn_op is ttnn.addcdiv:
        zero_mask = in_data3 == 0
        golden_nan_mask = torch.isnan(golden)
        result_inf_mask = torch.isinf(result)
        normalize_mask = zero_mask & golden_nan_mask & result_inf_mask
        if normalize_mask.any():
            result = torch.where(
                normalize_mask,
                torch.tensor(float("nan"), dtype=result.dtype, device=result.device),
                result,
            )

    assert_with_ulp(golden, result, ulp_threshold=10, allow_nonfinite=True)


@pytest.mark.parametrize(
    "c_shape, t_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Trow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Tcol
    ],
)
@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ttnn_lerp_tts_scalar_weight(c_shape, t_shape, weight, device):
    torch.manual_seed(0)
    in_data1 = torch.empty(c_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data2 = torch.empty(t_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    golden = torch.lerp(in_data1, in_data2, weight)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.lerp(input_tensor1, input_tensor2, weight)
    result = ttnn.to_torch(ttnn_result)

    assert_with_ulp(golden, result, ulp_threshold=10)


@pytest.mark.parametrize(
    "input_shape, end_shape, weight_shape",
    [
        ((8, 4, 768), (8, 4, 768), (8, 4, 768)),  # full, full, full
        ((8, 4, 1), (8, 4, 768), (8, 4, 768)),  # acol, bfull, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 768)),  # arow, bfull, cfull
        ((8, 4, 768), (8, 4, 1), (8, 1, 768)),  # afull, bcol, crow
        ((8, 4, 768), (8, 1, 768), (8, 4, 1)),  # afull, brow, ccol
        ((8, 4, 1), (8, 1, 768), (8, 4, 768)),  # acol, brow, cfull
        ((8, 1, 768), (8, 4, 1), (8, 4, 768)),  # arow, bcol, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 1)),  # arow, bfull, ccol
        ((8, 4, 1), (8, 4, 768), (8, 1, 768)),  # acol, bfull, crow
        ((1, 1, 1), (8, 4, 1), (8, 1, 768)),  # ascalar, bcol, crow
        ((1, 1, 1), (8, 1, 768), (8, 4, 1)),  # ascalar, brow, ccol
        ((8, 1, 768), (1, 1, 1), (8, 4, 1)),  # arow, bscalar, ccol
        ((8, 4, 1), (1, 1, 1), (8, 1, 768)),  # acol, bscalar, crow
    ],
)
def test_ttnn_lerp_ttt_row_col_mixed_bcast(input_shape, end_shape, weight_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data2 = torch.empty(end_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    # Weight in [0, 1] for lerp
    in_data3 = torch.empty(weight_shape, dtype=torch.bfloat16).uniform_(0, 1)
    golden = torch.lerp(in_data1, in_data2, in_data3)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.lerp(input_tensor1, input_tensor2, input_tensor3)
    result = ttnn.to_torch(ttnn_result)

    assert_with_ulp(golden, result, ulp_threshold=10)


# --------------------------------------------------------------------------------------------------
# Tests to verify cache collision due to different outer dims
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def isolate_program_cache(device):
    """Start each case with an empty cache, so the first dispatch is a genuine MISS.
    Without this the parametrized cases leak into each other -- e.g. the two 5D cases hash
    identically, so the second one's first dispatch would already be a hit."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def _where_variant(device, variant, pred, true_t, false_t, scalar_true=1.5, scalar_false=-2.5):
    """Dispatch `variant` and return (ttnn result, torch golden). TTS/TST replace one operand
    with a scalar, which exercises the packed-scalar compute arg and the TTS/TST reader layout
    (buffers at slots 0/1, slot 2 a plain 0) rather than TTT's three-buffer layout."""
    pred_tt = ttnn.from_torch(pred, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    if variant == "TTT":
        true_tt = ttnn.from_torch(true_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        false_tt = ttnn.from_torch(false_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.to_torch(ttnn.where(pred_tt, true_tt, false_tt)), torch.where(pred.bool(), true_t, false_t)
    if variant == "TTS":
        true_tt = ttnn.from_torch(true_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        golden = torch.where(pred.bool(), true_t, torch.full_like(true_t, scalar_false))
        return ttnn.to_torch(ttnn.where(pred_tt, true_tt, scalar_false)), golden
    # TST
    false_tt = ttnn.from_torch(false_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    golden = torch.where(pred.bool(), torch.full_like(false_t, scalar_true), false_t)
    return ttnn.to_torch(ttnn.where(pred_tt, scalar_true, false_tt)), golden


@pytest.mark.parametrize("variant", ["TTT", "TTS", "TST"])
@pytest.mark.parametrize(
    "pred_shape_1, pred_shape_2",
    [
        ((4, 1, 32, 32), (1, 4, 32, 32)),  # N/C swap, equal volume
        ((2, 2, 1, 32, 32), (2, 1, 2, 32, 32)),  # 5D: D/N/C permutation
        ((2, 2, 1, 32, 32), (1, 2, 2, 32, 32)),
    ],
)
def test_where_program_cache_same_volume_different_leading_dims(
    device, isolate_program_cache, variant, pred_shape_1, pred_shape_2
):
    """Predicates that hash identically (same padded volume, same H/W) but need different strides."""
    torch.manual_seed(42)
    # The true/false tensors are the broadcast target: same shape for both calls, so the ONLY
    # difference between the two dispatches is the predicate's dim factorization.
    out_shape = [max(a, b) for a, b in zip(pred_shape_1, pred_shape_2)]

    results = []
    deltas = []
    for pred_shape in (pred_shape_1, pred_shape_2):
        pred = torch.randint(0, 2, pred_shape).to(torch.bfloat16)
        true_t = torch.randn(out_shape, dtype=torch.bfloat16)
        false_t = torch.randn(out_shape, dtype=torch.bfloat16)

        device.cache_entries_counter.reset()
        with device.cache_entries_counter.measure():
            result, expected = _where_variant(device, variant, pred, true_t, false_t)
        deltas.append(device.cache_entries_counter.total)
        results.append((result, expected))

    # Both halves matter. Asserting only "the second call added nothing" is vacuous: it also holds
    # when the program cache is off and NOTHING was ever cached, so the test would go green without
    # exercising a cache hit at all.
    assert deltas[0] == 1, (
        f"first dispatch {pred_shape_1} added {deltas[0]} cache entries, expected exactly 1 "
        "(program cache disabled, or the op did not compile a program?)"
    )
    # And the second must REUSE it: if the hash is ever widened so these shapes stop colliding, both
    # calls become independent, both pass, and the coverage silently disappears.
    assert deltas[1] == 0, (
        f"{pred_shape_1} and {pred_shape_2} no longer collide in the program cache "
        f"(second dispatch added {deltas[1]} entries); this test no longer covers issue 54235"
    )

    for result, expected in results:
        assert_equal(expected, result)


def test_where_program_cache_same_input_volume_different_output_volume(device, isolate_program_cache):
    """
    Equal per-tensor input volumes and equal H/W hash identically, but the OUTPUT shape is the
    per-dim broadcast maximum of the three inputs and is absent from the key entirely -- so the
    work-split (per-core tile counts and start ids) differs between these two dispatches.

    Order matters: small output first, then large, then small again. The large-after-small step is
    the one that promotes a core from noop to work, so the cached zero-filled noop row has to be
    replaced with real args and buffer addresses -- a path the reverse order never reaches.
    """
    torch.manual_seed(42)
    # Every call: predicate volume 4096, true volume 4096, false volume 1024, H=W=32 -> one key.
    cases = [
        ((4, 1, 32, 32), (4, 1, 32, 32), (1, 1, 32, 32)),  # out [4,1,32,32]  (small)
        ((4, 1, 32, 32), (1, 4, 32, 32), (1, 1, 32, 32)),  # out [4,4,32,32]  (large: noop -> work)
        ((4, 1, 32, 32), (4, 1, 32, 32), (1, 1, 32, 32)),  # out [4,1,32,32]  (back to small)
    ]

    deltas = []
    for pred_shape, true_shape, false_shape in cases:
        pred = torch.randint(0, 2, pred_shape).to(torch.bfloat16)
        true_t = torch.randn(true_shape, dtype=torch.bfloat16)
        false_t = torch.randn(false_shape, dtype=torch.bfloat16)
        expected = torch.where(pred.bool(), true_t, false_t)

        pred_tt = ttnn.from_torch(pred, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        true_tt = ttnn.from_torch(true_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        false_tt = ttnn.from_torch(false_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        device.cache_entries_counter.reset()
        with device.cache_entries_counter.measure():
            result = ttnn.to_torch(ttnn.where(pred_tt, true_tt, false_tt))
        deltas.append(device.cache_entries_counter.total)

        assert_equal(expected, result)

    # As above, both halves: the first call must actually cache something, and the rest must reuse it.
    assert deltas[0] == 1, (
        f"first dispatch added {deltas[0]} cache entries, expected exactly 1 "
        "(program cache disabled, or the op did not compile a program?)"
    )
    assert deltas[1:] == [0, 0], (
        "the output-volume cases no longer collide in the program cache "
        f"(dispatches 2,3 added {deltas[1:]} entries); this test no longer covers issue 54235"
    )


@pytest.mark.parametrize("variant", ["TTT", "TTS", "TST"])
def test_where_sharded_program_cache_hit_repoints_cbs(device, isolate_program_cache, variant):
    """
    A sharded cache HIT must re-point the tensor-backed circular buffers at the CURRENT buffers.

    For sharded tensors the buffer base address rides on the globally-allocated CB, not on a runtime
    arg, so re-applying runtime args alone would leave the second dispatch reading and writing the
    FIRST dispatch's buffers. Both calls use the same shape and shard spec, so they share one cache
    entry, but they use different allocations: both sets of tensors (and both outputs) are kept alive
    so the second set necessarily lands at different addresses.

    Coverage note, established by disabling the CB re-point in override_runtime_arguments and
    re-running: only TTT actually fails without it. TTS/TST stay correct on these shapes -- their
    kernels reach the operands through the reader/writer address args rather than the CB backing --
    so they are extra correctness coverage here, not a guard on the re-point path. TTT is the case
    that regresses if the re-point is dropped.
    """
    shape = (1, 1, 32, 128)
    mem_cfg = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=4, y=1),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    def make(seed):
        torch.manual_seed(seed)
        pred = torch.randint(0, 2, shape).to(torch.bfloat16)
        true_t = torch.randn(shape, dtype=torch.bfloat16)
        false_t = torch.randn(shape, dtype=torch.bfloat16)
        to_dev = lambda x: ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg
        )
        return (pred, true_t, false_t), (to_dev(pred), to_dev(true_t), to_dev(false_t))

    # Both sets stay alive, so set 2 cannot reuse set 1's addresses.
    host_1, dev_1 = make(1)
    host_2, dev_2 = make(2)

    # If the allocator ever handed out the same addresses, the CB re-point would be a no-op and this
    # test would pass without exercising anything.
    assert (
        dev_1[0].buffer_address() != dev_2[0].buffer_address()
    ), "both dispatches got the same predicate buffer address; this test cannot detect a stale CB"

    deltas, results = [], []
    outs = []  # keep the device outputs alive too, so dispatch 2's OUTPUT cannot reuse dispatch 1's
    for (pred, true_t, false_t), (pred_tt, true_tt, false_tt) in ((host_1, dev_1), (host_2, dev_2)):
        if variant == "TTT":
            args, golden = (pred_tt, true_tt, false_tt), torch.where(pred.bool(), true_t, false_t)
        elif variant == "TTS":
            args, golden = (pred_tt, true_tt, -2.5), torch.where(pred.bool(), true_t, torch.full_like(true_t, -2.5))
        else:  # TST
            args, golden = (pred_tt, 1.5, false_tt), torch.where(pred.bool(), torch.full_like(false_t, 1.5), false_t)

        device.cache_entries_counter.reset()
        with device.cache_entries_counter.measure():
            out = ttnn.where(*args, memory_config=mem_cfg)
        outs.append(out)
        deltas.append(device.cache_entries_counter.total)
        results.append((ttnn.to_torch(out), golden))

    # Without this the output CB could go stale undetected: if dispatch 1's output were freed first,
    # dispatch 2 would reuse its address and a stale c_3 would still land on the right buffer.
    assert (
        outs[0].buffer_address() != outs[1].buffer_address()
    ), "both dispatches got the same output buffer address; this test cannot detect a stale output CB"

    assert deltas[0] == 1, (
        f"first sharded dispatch added {deltas[0]} cache entries, expected exactly 1 "
        "(program cache disabled, or the op did not compile a program?)"
    )
    assert deltas[1] == 0, (
        f"the second sharded dispatch added {deltas[1]} entries instead of reusing the cached "
        "program; this test no longer exercises the sharded cache-hit CB re-point path"
    )

    # The second result is the one that catches a stale CB address: it would return dispatch 1's data.
    for result, golden in results:
        assert_equal(golden, result)
