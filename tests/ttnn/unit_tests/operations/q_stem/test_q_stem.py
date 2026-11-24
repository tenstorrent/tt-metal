# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for q_stem (Phase 0: bf16 everywhere, TILE in / TILE out, fp32 DEST, NoPE, latent in DRAM).

IMMUTABLE SPEC — the implementer must not modify this file.

Every test runs on the focus group: G = 4 devices along cluster_axis 1 under FABRIC_2D_TORUS_XY (a Galaxy keeps
its system mesh and rings along its TP axis; a QuietBox 2 opens its snake ring as 1x4). The mesh comes from the
root `mesh_device` fixture (indirectly parametrized); no device is opened here.

Reference (per group of G devices, group position p holds q_b[p]):
    q[p]     = x @ q_b[p]                                       (B, 1, S, Hl * (Dn + Dr))
    heads[p] = q[p].view(B, S, Hl, Dn + Dr).permute(0, 2, 1, 3)
    device q' receives row quarter q' of every source p, heads concatenated in source order (head p * Hl + j)
    out[q']  = cat([nope[q'] @ wkv_b1, rope[q']], dim=-1)       (B, H, S/G, N + Dr)
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.q_stem.helpers import TOLERANCES
from eval.golden_tests.q_stem.topology import FOCUS_CLUSTER_AXIS, focus_mesh_shape
from eval.metrics import check_output, default_tolerance
from ttnn.operations.q_stem import q_stem

CLUSTER_AXIS = FOCUS_CLUSTER_AXIS


def _focus_shape():
    try:
        shape, _ = focus_mesh_shape()
    except Exception:  # noqa: BLE001 — no cluster descriptor: the tests skip below
        shape = None
    return shape


_MESH = _focus_shape()
_MESH_PARAM = [
    pytest.param(
        _MESH if _MESH is not None else (1, 4),
        marks=[]
        if _MESH is not None
        else [pytest.mark.skip(reason="this cluster cannot close the 4-device ring along cluster_axis 1")],
    )
]
_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}]


# --- reference ---------------------------------------------------------------------------------------------------


def _groups(rows, cols):
    """Collective groups along cluster_axis 1: each mesh row, in group-position order."""
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def reference(x, q_b, wkv_b1, G):
    """x (R, C, B, 1, S, Q), q_b (R, C, 1, 1, Q, Hl*(Dn+Dr)), wkv_b1 (1, H, Dn, N) -> (R, C, B, H, S/G, N+Dr), fp32."""
    rows, cols = x.shape[0], x.shape[1]
    B, _, S, _ = x.shape[2:]
    _, H, Dn, N = wkv_b1.shape
    Hl = H // G
    Dr = q_b.shape[-1] // Hl - Dn
    w = wkv_b1.float()
    out = torch.empty((rows, cols, B, H, S // G, N + Dr), dtype=torch.float32)
    for members in _groups(rows, cols):
        nope, rope = [], []
        for r, c in members:
            q = torch.matmul(x[r, c].float(), q_b[r, c].float())
            heads = q.reshape(B, S, Hl, Dn + Dr).permute(0, 2, 1, 3)
            nope.append(torch.chunk(heads[..., :Dn], G, dim=2))
            rope.append(torch.chunk(heads[..., Dn:], G, dim=2))
        for qpos, (r, c) in enumerate(members):
            nope_q = torch.cat([nope[p][qpos] for p in range(G)], dim=1)
            rope_q = torch.cat([rope[p][qpos] for p in range(G)], dim=1)
            out[r, c] = torch.cat([torch.matmul(nope_q, w), rope_q], dim=-1)
    return out


# --- tensors -------------------------------------------------------------------------------------------------------


def _to_mesh(stacked, mesh_device, dtype=ttnn.bfloat16, memory_config=None):
    """stacked (R, C, *shape): device (r, c) receives stacked[r, c]."""
    rows, cols = tuple(mesh_device.shape)
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _read_back(t, mesh_device):
    rows, cols = tuple(mesh_device.shape)
    devs = ttnn.get_device_tensors(t)
    return [devs[i] for i in range(rows * cols)]


def _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N, seed=42):
    """Latent identical within a group; q_b column slice per group position; wkv_b1 replicated. torch.randn, seed 42."""
    torch.manual_seed(seed)
    rows, cols = tuple(mesh_device.shape)
    G = cols
    H = G * Hl
    Cq = Hl * (Dn + Dr)
    x = torch.empty((rows, cols, B, 1, S, Q))
    q_b = torch.empty((rows, cols, 1, 1, Q, Cq))
    full_qb = torch.randn((Q, G * Cq)) / Q**0.5
    for members in _groups(rows, cols):
        lat = torch.randn((B, 1, S, Q))
        for p, (r, c) in enumerate(members):
            x[r, c] = lat
            q_b[r, c] = full_qb[:, p * Cq : (p + 1) * Cq].reshape(1, 1, Q, Cq)
    wkv = torch.randn((1, H, Dn, N)) / Dn**0.5
    x, q_b, wkv = x.to(torch.bfloat16), q_b.to(torch.bfloat16), wkv.to(torch.bfloat16)
    wkv_stacked = wkv.reshape(1, 1, 1, H, Dn, N).expand(rows, cols, 1, H, Dn, N).contiguous()
    return x, q_b, wkv, _to_mesh(x, mesh_device), _to_mesh(q_b, mesh_device), _to_mesh(wkv_stacked, mesh_device)


def _tolerance(dtype, fp32_dest_acc_en=True):
    for key in ((dtype, fp32_dest_acc_en), dtype):
        if key in TOLERANCES:
            return TOLERANCES[key]
    return default_tolerance(dtype, fp32_dest_acc_en)


def _check(out, expected, mesh_device, out_shape):
    tol = _tolerance(ttnn.bfloat16, True)
    rows, cols = tuple(mesh_device.shape)
    for idx, dev in enumerate(_read_back(out, mesh_device)):
        r, c = divmod(idx, cols)
        check_output(
            dev,
            expected[r, c],
            shape=out_shape,
            dtype=ttnn.bfloat16,
            expected_layout=ttnn.TILE_LAYOUT,
            tolerance=tol,
            fp32_dest_acc_en=True,
        )


# --- tests ---------------------------------------------------------------------------------------------------------

# (B, S, Q, Hl, Dn, Dr, N): one tile everywhere / multi-tile / non-square / multi-batch / the Kimi K3.1 model call /
# a wide wkv_b1 output (N-streamed stage-3 regime) / a deep q_b_proj reduction (K-blocked stage 1).
SHAPES = [
    pytest.param((1, 128, 64, 2, 32, 32, 32), id="single_tile"),
    pytest.param((1, 256, 256, 4, 64, 32, 128), id="multi_tile"),
    pytest.param((1, 128, 512, 4, 128, 128, 512), id="non_square_wide_rope"),
    pytest.param((2, 128, 512, 8, 128, 64, 512), id="multi_batch"),
    pytest.param((1, 640, 1536, 24, 128, 64, 512), id="kimi_k31_640"),
    pytest.param((1, 128, 512, 4, 128, 64, 4096), id="wide_n_streamed_wkv"),
    pytest.param((1, 128, 4096, 4, 128, 64, 512), id="deep_q_k_blocked"),
]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", _MESH_PARAM, indirect=True)
@pytest.mark.parametrize("dims", SHAPES)
def test_q_stem(mesh_device, dims):
    B, S, Q, Hl, Dn, Dr, N = dims
    G = tuple(mesh_device.shape)[CLUSTER_AXIS]
    x, q_b, wkv, tx, tq, tw = _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N)
    expected = reference(x, q_b, wkv, G)
    out = q_stem(tx, tq, tw, cluster_axis=CLUSTER_AXIS)
    _check(out, expected, mesh_device, (B, G * Hl, S // G, N + Dr))


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", _MESH_PARAM, indirect=True)
def test_q_stem_back_to_back_deterministic(mesh_device):
    """Three calls with no host sync in between: each correct, all bit-identical (counters re-armed per call)."""
    B, S, Q, Hl, Dn, Dr, N = (1, 256, 256, 4, 64, 32, 128)
    G = tuple(mesh_device.shape)[CLUSTER_AXIS]
    x, q_b, wkv, tx, tq, tw = _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N)
    expected = reference(x, q_b, wkv, G)
    outs = [q_stem(tx, tq, tw, cluster_axis=CLUSTER_AXIS) for _ in range(3)]
    first = [ttnn.to_torch(d) for d in _read_back(outs[0], mesh_device)]
    for out in outs:
        _check(out, expected, mesh_device, (B, G * Hl, S // G, N + Dr))
        for a, b in zip(first, [ttnn.to_torch(d) for d in _read_back(out, mesh_device)]):
            assert torch.equal(a, b), "q_stem is not deterministic across back-to-back calls"


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", _MESH_PARAM, indirect=True)
def test_q_stem_caller_errors(mesh_device, expect_error):
    """Caller errors raise ValueError / RuntimeError (not support refusals): every registry axis below is in the
    Phase 0 SUPPORTED set, only the shapes are inconsistent."""
    B, S, Q, Hl, Dn, Dr, N = (1, 128, 64, 2, 32, 32, 32)
    rows, cols = tuple(mesh_device.shape)
    _, _, _, tx, tq, _ = _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N)
    # wkv_b1 head count (6) not divisible by the group size G = 4.
    bad_wkv = _to_mesh(torch.randn((rows, cols, 1, 6, Dn, N)).to(torch.bfloat16), mesh_device)
    with expect_error((ValueError, RuntimeError), r".*"):
        q_stem(tx, tq, bad_wkv, cluster_axis=CLUSTER_AXIS)
    # q_b_proj whose Q (128) differs from the latent's (64).
    bad_qb = _to_mesh(torch.randn((rows, cols, 1, 1, 2 * Q, Hl * (Dn + Dr))).to(torch.bfloat16), mesh_device)
    good_wkv = _to_mesh(torch.randn((rows, cols, 1, cols * Hl, Dn, N)).to(torch.bfloat16), mesh_device)
    with expect_error((ValueError, RuntimeError), r".*"):
        q_stem(tx, bad_qb, good_wkv, cluster_axis=CLUSTER_AXIS)
