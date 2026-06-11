# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the descriptor-framework slow-path rebuild perf bug (issue #46506).

A ProgramDescriptor op that bakes a tensor buffer base address as a raw uint32 into a kernel's
runtime args (instead of registering it via emplace_runtime_args(Buffer*)) leaves no patchable
binding, so the mesh-device-operation adapter re-runs create_descriptor() on EVERY program-cache
hit. That per-dispatch host cost is invisible to trace (captured once), which is why ResNet50's
non-trace path regressed ~20x while trace stayed fine.

This test sets TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1 (read by the adapter) so that ANY
op falling to the slow-path rebuild raises instead of silently rebuilding. Each op is run 3x with
the program cache enabled:
  - 1st call: cache miss -> resolve_bindings() validates that every declared Buffer* binding sits at
              a runtime-arg slot actually holding that buffer's address (TT_FATAL otherwise), so a
              MISPLACED binding is caught here.
  - 2nd/3rd call: cache hit -> the guard fires (TT_FATAL) if the op rebuilds, so a SLOW-PATH op is
              caught here.
Numerical correctness is checked against a torch reference for the ops that have a trivial one.

Must run with the env var set; the module sets it before importing ttnn.
"""

import os

# Must be set before ttnn first dispatches (the adapter reads it via getenv on first slow-path hit).
os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _tile(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


def _run_thrice(dev, fn):
    """Run fn 3x. Raises if the op rebuilds its descriptor on a cache hit (guard) or binds a
    Buffer* at a wrong runtime-arg slot (resolve_bindings validation on the first/miss call)."""
    out = None
    for _ in range(3):
        out = fn()
        ttnn.synchronize_device(dev)
    return out


# ---------------------------------------------------------------------------
# eltwise
# ---------------------------------------------------------------------------
def test_unary_relu_no_rebuild(device):
    ta = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    a = _tile(ta, device)
    out = _run_thrice(device, lambda: ttnn.relu(a))
    assert torch.allclose(ttnn.to_torch(out).float(), torch.relu(ta).float(), atol=0.1)


def test_binary_ng_add_no_rebuild(device):
    ta = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    a, b = _tile(ta, device), _tile(tb, device)
    out = _run_thrice(device, lambda: ttnn.add(a, b))
    assert torch.allclose(ttnn.to_torch(out).float(), (ta + tb).float(), atol=0.1)


def test_binary_ng_add_inplace_no_rebuild(device):
    # In-place: output aliases input A. Exercises both the in-place resolver alias path and the
    # output Buffer* binding.
    a = _tile(torch.randn(1, 1, 256, 256, dtype=torch.bfloat16), device)
    b = _tile(torch.randn(1, 1, 256, 256, dtype=torch.bfloat16), device)
    _run_thrice(device, lambda: ttnn.add_(a, b))


# ---------------------------------------------------------------------------
# data_movement
# ---------------------------------------------------------------------------
def test_transpose_hc_no_rebuild(device):
    ta = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    a = _tile(ta, device)
    out = _run_thrice(device, lambda: ttnn.transpose(a, 1, 2))
    assert torch.allclose(ttnn.to_torch(out).float(), ta.transpose(1, 2).float(), atol=0.1)


def test_slice_tile_no_rebuild(device):
    ta = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
    a = _tile(ta, device)
    out = _run_thrice(device, lambda: ttnn.slice(a, [0, 0, 0, 0], [1, 1, 64, 64]))
    assert torch.allclose(ttnn.to_torch(out).float(), ta[:, :, :64, :64].float(), atol=0.1)


def test_permute_tiled_no_rebuild(device):
    ta = torch.randn(1, 2, 64, 96, dtype=torch.bfloat16)
    a = _tile(ta, device)
    out = _run_thrice(device, lambda: ttnn.permute(a, (0, 1, 3, 2)))
    assert torch.allclose(ttnn.to_torch(out).float(), ta.permute(0, 1, 3, 2).float(), atol=0.1)


def test_untilize_with_unpadding_no_rebuild(device):
    ta = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    a = _tile(ta, device)
    out = _run_thrice(device, lambda: ttnn.untilize_with_unpadding(a, [0, 0, 47, 63]))
    assert torch.allclose(ttnn.to_torch(out).float(), ta[:, :, :48, :64].float(), atol=0.1)


# ---------------------------------------------------------------------------
# moreh
# ---------------------------------------------------------------------------
def test_moreh_dot_no_rebuild(device):
    ta = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16)
    a, b = _tile(ta, device), _tile(tb, device)
    _run_thrice(device, lambda: ttnn.moreh_dot(a, b))


def test_tanh_bw_no_rebuild(device):
    grad = _tile(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), device)
    inp = _tile(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), device)
    _run_thrice(device, lambda: ttnn.tanh_bw(grad, inp))


def test_bcast_h_no_rebuild(device):
    # H-broadcast: second operand has H=1; exercises bcast_multi_core_h.
    a = _tile(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), device)
    b = _tile(torch.randn(1, 1, 32, 64, dtype=torch.bfloat16), device)
    _run_thrice(device, lambda: ttnn.bcast(a, b, math_op=ttnn.BcastOpMath.ADD, dim=ttnn.BcastOpDim.H))


def test_bcast_w_no_rebuild(device):
    a = _tile(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), device)
    b = _tile(torch.randn(1, 1, 64, 32, dtype=torch.bfloat16), device)
    _run_thrice(device, lambda: ttnn.bcast(a, b, math_op=ttnn.BcastOpMath.ADD, dim=ttnn.BcastOpDim.W))


def test_bcast_hw_no_rebuild(device):
    a = _tile(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), device)
    b = _tile(torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), device)
    _run_thrice(device, lambda: ttnn.bcast(a, b, math_op=ttnn.BcastOpMath.ADD, dim=ttnn.BcastOpDim.HW))


def test_concat_no_rebuild(device):
    ta = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    a, b = _tile(ta, device), _tile(tb, device)
    out = _run_thrice(device, lambda: ttnn.concat([a, b], dim=2))
    assert torch.allclose(ttnn.to_torch(out).float(), torch.cat([ta, tb], dim=2).float(), atol=0.1)


def test_clone_no_rebuild(device):
    ta = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    a = _tile(ta, device)
    out = _run_thrice(device, lambda: ttnn.clone(a))
    assert torch.allclose(ttnn.to_torch(out).float(), ta.float(), atol=0.01)  # clone == identity


def test_upsample_no_rebuild(device):
    # NHWC row-major input for upsample.
    a = ttnn.from_torch(torch.randn(1, 8, 8, 32, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16)
    _run_thrice(device, lambda: ttnn.upsample(a, 2))


# ---------------------------------------------------------------------------
# class 2 — address-derived / CB-bound ops that fast-path via get_dynamic_runtime_args
# ---------------------------------------------------------------------------
def test_move_sharded_no_rebuild(device):
    # Sharded in-place move: reader args (num_chunks/chunk_size/remainder) derive from
    # output_addr - input_addr, recomputed each hit via MoveDeviceOperation::get_dynamic_runtime_args.
    # Each ttnn.move deallocates its input and allocates a fresh output, so addresses change across
    # the cache hits — exercising the recompute. Guard ON asserts no descriptor rebuild.
    shape = (1, 1, 512, 64)
    mem = ttnn.create_sharded_memory_config(
        shape, ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    for _ in range(4):
        ta = torch.randn(shape, dtype=torch.bfloat16)
        x = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=mem)
        out = ttnn.move(x)
        ttnn.synchronize_device(device)
        assert torch.allclose(ttnn.to_torch(out).float(), ta.float(), atol=0.05)


# ---------------------------------------------------------------------------
# normalization (verified: eps/momentum/scale ARE in each op's program hash,
# so binding the tensor addresses cannot freeze a stale scalar on a cache hit)
# ---------------------------------------------------------------------------
def test_layer_norm_no_rebuild(device):
    ta = torch.randn(1, 1, 64, 256, dtype=torch.bfloat16)
    a = _tile(ta, device)
    _run_thrice(device, lambda: ttnn.layer_norm(a, epsilon=1e-5))


def test_softmax_no_rebuild(device):
    ta = torch.randn(1, 1, 64, 256, dtype=torch.bfloat16)
    a = _tile(ta, device)
    out = _run_thrice(device, lambda: ttnn.softmax(a, dim=-1))
    assert torch.allclose(ttnn.to_torch(out).float(), torch.softmax(ta.float(), dim=-1), atol=0.05)


def test_group_norm_no_rebuild(device):
    # group_norm requires row-major, non-inplace input.
    a = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    _run_thrice(device, lambda: ttnn.group_norm(a, num_groups=2, epsilon=1e-5, inplace=False))


def test_batch_norm_no_rebuild(device):
    a = _tile(torch.randn(1, 4, 32, 32, dtype=torch.bfloat16), device)
    rmean = _tile(torch.zeros(1, 4, 1, 1, dtype=torch.bfloat16), device)
    rvar = _tile(torch.ones(1, 4, 1, 1, dtype=torch.bfloat16), device)
    _run_thrice(
        device,
        lambda: ttnn.batch_norm(a, running_mean=rmean, running_var=rvar, training=False, eps=1e-5),
    )
