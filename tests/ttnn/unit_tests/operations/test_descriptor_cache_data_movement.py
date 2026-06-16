# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Cache-correctness sweep for data_movement ops migrated to the descriptor framework (#46506).

Companion to test_descriptor_cache.py — same pattern, different op family. For each migrated op we
run its SIMPLEST existing test invocation (copied verbatim from the op's own test file / from
test_descriptor_no_rebuild.py — not invented), but under an ENABLED program cache and across the
value the op varies per call (data / addresses / scalar / shape). Two things must hold:

  * not stale  : every call's result matches the op's golden, even on a program-cache HIT
                 (a frozen per-call rt-arg would make a later call wrong).
  * not over-caching : the op does not create a new program-cache entry for a value it should
                 re-apply (e.g. an address-derived count) — that is the "cache too restrictive"
                 failure that silently rebuilds/recompiles every call.

Ops covered (data_movement family): bcast, move, slice, transpose, untilize_with_unpadding.
(concat and permute are covered in test_descriptor_cache.py — skipped here.)
"""

import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def cache_device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# data_movement / bcast. Same ttnn.bcast(a, b, math_op=ADD, dim=...) invocation as
# test_descriptor_no_rebuild.py (test_bcast_h/_w/_hw), with the torch reference taken from the real
# bcast unit test (tests/tt_eager/.../misc/test_bcast.py): the second operand has size 1 on the
# bcast axis and the golden is torch.add with natural broadcasting. math_op and dim live in
# operation_attributes -> default program hash; all remaining rt-args are shape/geometry-derived
# (in the hash) and addresses are CB/Buffer*-bound; BcastDeviceOperation::get_dynamic_runtime_args
# returns {} (fast path). Varying input DATA (fresh address) must not go stale; one entry per dim.
# PREDICTION: OK.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dim, b_shape",
    [
        (ttnn.BcastOpDim.H, (1, 1, 1, 64)),
        (ttnn.BcastOpDim.W, (1, 1, 64, 1)),
        (ttnn.BcastOpDim.HW, (1, 1, 1, 1)),
    ],
)
def test_bcast_cache(cache_device, dim, b_shape):
    torch.manual_seed(0)
    a_shape = (1, 1, 64, 64)

    def _run():
        ta = torch.rand(a_shape, dtype=torch.bfloat16)
        tb = torch.rand(b_shape, dtype=torch.bfloat16)
        a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)
        out = ttnn.to_torch(ttnn.bcast(a, b, math_op=ttnn.BcastOpMath.ADD, dim=dim)).float()
        ref = torch.add(ta.float(), tb.float()).float()  # ADD, second operand broadcasts over the bcast dim
        assert torch.allclose(out, ref, atol=0.1), f"bcast stale (dim={dim})"

    # WARM-UP then assert ZERO growth across fresh allocations (dim held FIXED).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"bcast: cache grew past {base} across fresh allocations (dim={dim})"


# ---------------------------------------------------------------------------
# data_movement / move (sharded in-place). Invocation copied from test_descriptor_no_rebuild.py
# (test_move_sharded_no_rebuild): HEIGHT-sharded TILE input, ttnn.move(x).
# This is the class-2 case: MoveShardedProgramFactory bakes address-derived reader args
# (num_chunks, move_chunk_size_bytes, remainder) = f(output_addr - input_addr). Each ttnn.move
# deallocates its input and allocates a fresh output, so those addresses change across cache hits;
# MoveDeviceOperation::get_dynamic_runtime_args re-applies the three slots each hit. If it were
# frozen (the #46506 bug) a later call would read the wrong chunk geometry -> stale/garbage.
# PREDICTION: OK after fix (was the bug class; re-applied via get_dynamic_runtime_args).
# ---------------------------------------------------------------------------
def test_move_sharded_cache(cache_device):
    torch.manual_seed(0)
    shape = (1, 1, 512, 64)
    mem = ttnn.create_sharded_memory_config(
        shape, ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )

    def _run():
        ta = torch.randn(shape, dtype=torch.bfloat16)
        x = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=cache_device, dtype=ttnn.bfloat16, memory_config=mem)
        out = ttnn.move(x)
        assert torch.allclose(ttnn.to_torch(out).float(), ta.float(), atol=0.05), "move stale (sharded)"

    # WARM-UP then assert ZERO growth across fresh allocations (addr-derived args must be patched).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"move: cache grew past {base} across fresh allocations (addr-derived not patched)"


# ---------------------------------------------------------------------------
# data_movement / slice. Invocation copied from test_descriptor_no_rebuild.py
# (test_slice_tile_no_rebuild) for the tiled path and from test_slice.py run_slice_rm for the
# row-major path. slice_start/slice_end/step live in operation_attributes -> default hash, so a
# DIFFERENT slice window legitimately re-hashes (a new program). To probe the cache HIT path we
# hold the window FIXED and vary input DATA (fresh address). The row-major path
# (SliceRmProgramFactory) bakes reader_arg0 = src_addr + begins_bytes - misalignment; that
# address-derived arg is re-applied each hit via SliceDeviceOperation::get_dynamic_runtime_args.
# If frozen, a fresh-address call would read the wrong base -> stale. The tiled path binds
# addresses as Buffer* (framework-patched). PREDICTION: OK (tiled bound; rm re-applied via fix).
# ---------------------------------------------------------------------------
def test_slice_tile_cache(cache_device):
    torch.manual_seed(0)
    shape = (1, 1, 128, 128)

    def _run():
        ta = torch.randn(shape, dtype=torch.bfloat16)
        a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)
        out = ttnn.to_torch(ttnn.slice(a, [0, 0, 0, 0], [1, 1, 64, 64])).float()
        ref = ta[:, :, :64, :64].float()
        assert torch.allclose(out, ref, atol=0.1), "slice stale (tiled)"

    # WARM-UP then assert ZERO growth across fresh allocations (window held FIXED).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"slice(tile): cache grew past {base} across fresh allocations"


def test_slice_rm_cache(cache_device):
    # Row-major interleaved path -> SliceRmProgramFactory, the only slice factory that bakes an
    # address-derived reader arg (start_addr + begins_bytes). Fixed window, fresh data each call.
    torch.manual_seed(0)
    shape = (1, 1, 128, 128)
    begins = [0, 0, 0, 0]
    ends = [1, 1, 64, 128]

    def _run():
        ta = torch.rand(shape, dtype=torch.bfloat16)
        a = ttnn.from_torch(ta, layout=ttnn.ROW_MAJOR_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)
        out = ttnn.to_torch(ttnn.slice(a, begins, ends)).float()
        ref = ta[begins[0] : ends[0], begins[1] : ends[1], begins[2] : ends[2], begins[3] : ends[3]].float()
        assert torch.allclose(out, ref, atol=0.1), "slice stale (row-major)"

    # WARM-UP then assert ZERO growth across fresh allocations (window held FIXED).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"slice(rm): cache grew past {base} across fresh allocations (addr not patched)"


# data_movement / slice -- DIFFERENT windows that program-cache COLLIDE.
# Regression for #46506: SliceRmProgramFactory folds padded_shape / CB sizing (not the absolute
# slice start) into compute_program_hash(), so two slices with a different start can hit the SAME
# cached program. On that hit the reader's start stick id + misalignment (reader args 5/6) and the
# writer's per-core stick offset are per-call values that must be re-applied; the original fix
# re-applied only the buffer address (arg 0), leaving the start frozen -> the hit read the previous
# window's region (PCC garbage). This is the user-visible failure exercised by ttnn.split's
# slice-based path (test_split.py[shape=(32, 64, 4096) ... ROW_MAJOR], chunksize=1). Drive it via the
# same path: split a wide row-major last dim into width-1 chunks; later windows cache-HIT earlier
# ones and every chunk must still match. Assert a hit actually happened (else the test wouldn't probe
# the bug). PREDICTION: OK (every per-core arg re-applied on each hit).
def test_slice_rm_split_window_cache(cache_device):
    torch.manual_seed(0)
    shape = (32, 64, 4096)  # last dim wide enough that distinct windows collide in the cache
    ta = torch.rand(shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, layout=ttnn.ROW_MAJOR_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)

    base = cache_device.num_program_cache_entries()
    outs = ttnn.split(a, 1, dim=2)  # width-1 chunks -> sequence of slices over distinct windows
    grew = cache_device.num_program_cache_entries() - base

    assert len(outs) == shape[2]
    for i, o in enumerate(outs):
        got = ttnn.to_torch(o).float()
        ref = ta[:, :, i : i + 1].float()
        assert torch.allclose(got, ref, atol=0.1), f"slice(rm) window {i} stale on a program-cache hit"
    # Some windows must have re-used a cached program, or this test never exercised the hit path.
    assert grew < len(outs), f"expected a cross-window program-cache hit (grew {grew} for {len(outs)} slices)"


# ---------------------------------------------------------------------------
# data_movement / transpose. Invocation copied from test_descriptor_no_rebuild.py
# (test_transpose_hc_no_rebuild): ttnn.transpose(a, 1, 2) on a TILE tensor (HC tiled factory).
# dim + pad_value live in operation_attributes -> default hash. All reader/writer rt-args are pure
# shape/geometry (Wt, H, Ct, HW_bytes, num_tiles_read, ...) -> fully determined by the hash;
# addresses are bound as Buffer* (arg 0) and patched on a hit. No per-call scalar/index escapes the
# hash, so no get_dynamic_runtime_args is needed. Varying input DATA (fresh address) must not go
# stale; one entry. PREDICTION: OK.
# ---------------------------------------------------------------------------
def test_transpose_hc_cache(cache_device):
    torch.manual_seed(0)
    shape = (1, 4, 64, 64)

    def _run():
        ta = torch.randn(shape, dtype=torch.bfloat16)
        a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)
        out = ttnn.to_torch(ttnn.transpose(a, 1, 2)).float()
        ref = ta.transpose(1, 2).float()
        assert torch.allclose(out, ref, atol=0.1), "transpose stale (HC)"

    # WARM-UP then assert ZERO growth across fresh allocations.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"transpose: cache grew past {base} across fresh allocations"


# ---------------------------------------------------------------------------
# data_movement / untilize_with_unpadding. Invocation copied from test_descriptor_no_rebuild.py
# (test_untilize_with_unpadding_no_rebuild): ttnn.untilize_with_unpadding(a, [0,0,47,63]).
# output_tensor_end lives in operation_attributes -> default hash, AND the output shape is derived
# from it, so a DIFFERENT end legitimately produces a new program/cache entry. To probe the cache
# HIT path we hold output_tensor_end FIXED and vary input DATA. The unpadded row/geometry rt-args
# are output-shape-derived (in the hash); addresses are bound as Buffer*. No custom hash, no
# get_dynamic_runtime_args. PREDICTION: OK (varying the end is correct over-caching, not a bug).
# ---------------------------------------------------------------------------
def test_untilize_with_unpadding_cache(cache_device):
    torch.manual_seed(0)
    shape = (1, 1, 64, 64)
    output_end = [0, 0, 47, 63]

    def _run():
        ta = torch.randn(shape, dtype=torch.bfloat16)
        a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)
        out = ttnn.to_torch(ttnn.untilize_with_unpadding(a, output_end)).float()
        ref = ta[:, :, :48, :64].float()
        assert torch.allclose(out, ref, atol=0.1), "untilize_with_unpadding stale"

    # WARM-UP then assert ZERO growth across fresh allocations (output_tensor_end held FIXED).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"untilize_with_unpadding: cache grew past {base} across fresh allocations"
