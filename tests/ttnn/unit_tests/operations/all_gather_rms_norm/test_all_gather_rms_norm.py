# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for all_gather_rms_norm (immutable spec — do not edit to make it pass).

Every device (r, c) of a mesh holds its own x[r, c] of shape (..., M, W_local). Along `cluster_axis` the G devices
of a group concatenate their slices (group-position order) into the full row of W = G * W_local and every member
returns torch rms_norm / layer_norm of that full row. Checked per device and per group-position slice against a
pure-torch reference, with the golden suite's tolerances (eval/golden_tests/all_gather_rms_norm/helpers.TOLERANCES),
gating PCC and relative RMS together.

Meshes are opened by the root `mesh_device` fixture under FABRIC_2D_TORUS_XY with the production router payload:
a 1x4 ring (cluster_axis 1; a QuietBox 2 is physically a 4-cycle, a Galaxy rings along its TP axis) and a 2x2 mesh
(cluster_axis 0 and 1, Linear).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.all_gather_rms_norm.helpers import (
    TOLERANCES,
    build_memory_config,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    quantize_like_device,
)
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

EPS = 1e-6
ROW_OFFSET_SCALE = 2.0  # LayerNorm cells: per-row mean offsets so LN != RMS and a per-slice mean would be caught

_TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def _router():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router()}]


# --- reference / data -----------------------------------------------------------------------------------------


def torch_all_gather_norm(x, cluster_axis, gamma, beta, norm_type, eps):
    """x: stacked (R, C, ..., M, W_local). Returns stacked (R, C, ..., M, G * W_local) in fp32."""
    x = x.to(torch.float32)
    R, C = x.shape[0], x.shape[1]
    G = (R, C)[cluster_axis]
    w = None if gamma is None else gamma.to(torch.float32)
    b = None if beta is None else beta.to(torch.float32)
    out = torch.empty((R, C, *x.shape[2:-1], G * x.shape[-1]), dtype=torch.float32)
    for other in range(C if cluster_axis == 0 else R):
        members = [(r, other) for r in range(R)] if cluster_axis == 0 else [(other, c) for c in range(C)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        W = full.shape[-1]
        if norm_type == "layer_norm":
            normed = torch.nn.functional.layer_norm(full, (W,), weight=w, bias=b, eps=eps)
        else:
            normed = torch.nn.functional.rms_norm(full, (W,), weight=w, eps=eps)
        for r, c in members:
            out[r, c] = normed
    return out


def make_input(mesh_shape, shape, dtype, *, seed, row_offset_scale=0.0, offset=0.0):
    torch.manual_seed(seed)
    R, C = mesh_shape
    x = torch.randn((R, C, *shape), dtype=torch.float32) + offset
    if row_offset_scale:
        x = x + torch.randn((R, C, *shape[:-1], 1), dtype=torch.float32) * row_offset_scale
    return quantize_like_device(x.to(_TORCH[dtype]), dtype)


def make_weight(W, dtype, seed):
    torch.manual_seed(seed)
    return torch.randn((W,), dtype=torch.float32).to(_TORCH[dtype])


def _pcc_rel_rms(a, e):
    a, e = a.flatten().to(torch.float64), e.flatten().to(torch.float64)
    ac, ec = a - a.mean(), e - e.mean()
    den = ac.norm() * ec.norm()
    pcc = 1.0 if den < 1e-30 else float((ac * ec).sum() / den)
    rms = float(((a - e) ** 2).mean().sqrt() / max(float((e**2).mean().sqrt()), 1e-30))
    return pcc, rms


def check(out, expected, *, out_shape, dtype, fp32_dest_acc_en, G):
    assert list(out.shape) == list(out_shape), f"shape {list(out.shape)} != {list(out_shape)}"
    assert out.dtype == dtype, f"dtype {out.dtype} != {dtype}"
    assert out.layout == ttnn.TILE_LAYOUT
    pcc_t, rms_t = TOLERANCES[(dtype, fp32_dest_acc_en)]
    tensors = ttnn.get_device_tensors(out)
    R, C = expected.shape[0], expected.shape[1]
    assert len(tensors) == R * C
    w = out_shape[-1] // G
    for idx, t in enumerate(tensors):
        r, c = divmod(idx, C)
        a = ttnn.to_torch(t).to(torch.float32)
        e = expected[r, c]
        assert not torch.isnan(a).any() and not torch.isinf(a).any(), f"device ({r},{c}): nan/inf"
        spans = [("row", a, e)] + [
            (f"slice{p}", a[..., p * w : (p + 1) * w], e[..., p * w : (p + 1) * w]) for p in range(G)
        ]
        for name, aa, ee in spans:
            pcc, rms = _pcc_rel_rms(aa, ee)
            assert (
                pcc >= pcc_t and rms <= rms_t
            ), f"device ({r},{c}) {name}: pcc={pcc:.6f} (>= {pcc_t}) rel_rms={rms:.6f} (<= {rms_t})"


def run_case(
    mesh_device,
    *,
    cluster_axis,
    shape,
    dtype=ttnn.bfloat16,
    norm_type="rms_norm",
    gamma_layout="none",
    gamma_dtype=ttnn.bfloat16,
    with_beta=False,
    topology=ttnn.Topology.Ring,
    num_links=1,
    fp32_dest_acc_en=True,
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    offset=0.0,
    eps=EPS,
    seed=42,
):
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    out_shape = (*shape[:-1], G * shape[-1])
    W = out_shape[-1]
    layer_norm = norm_type == "layer_norm"
    x = make_input(
        mesh_shape, shape, dtype, seed=seed, row_offset_scale=ROW_OFFSET_SCALE if layer_norm else 0.0, offset=offset
    )
    gamma = beta = tg = tb = None
    if gamma_layout != "none":
        gamma = make_weight(W, gamma_dtype, seed + 1)
        tg = create_ttnn_gamma(gamma, mesh_device, dtype=gamma_dtype, layout=gamma_layout)
        if with_beta:
            beta = make_weight(W, gamma_dtype, seed + 2)
            tb = create_ttnn_gamma(beta, mesh_device, dtype=gamma_dtype, layout=gamma_layout)
    expected = torch_all_gather_norm(x, cluster_axis, gamma, beta, norm_type, eps)

    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_layout=memory_layout)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    kwargs = {}
    if layer_norm:
        kwargs["norm_type"] = norm_type
    if tb is not None:
        kwargs["beta"] = tb
    if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        kwargs["memory_config"] = build_memory_config(
            memory_layout, ttnn.TILE_LAYOUT, out_shape, mesh_device, dtype=dtype
        )
    out = all_gather_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        gamma=tg,
        epsilon=eps,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=cfg,
        **kwargs,
    )
    check(out, expected, out_shape=out_shape, dtype=dtype, fp32_dest_acc_en=fp32_dest_acc_en, G=G)
    return tx, tg, tb, kwargs, cfg, out


R, L = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
RING, LINE = ttnn.Topology.Ring, ttnn.Topology.Linear
HS, WS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED

# (id, kwargs) — cluster_axis 1 on the 1x4 ring.
RING_CASES = [
    ("single_tile_rms_gammaTILE", dict(shape=(1, 1, 32, 32), gamma_layout=L)),
    ("multi_tile_rms_gammaRM_line_2links", dict(shape=(1, 1, 64, 128), gamma_layout=R, topology=LINE, num_links=2)),
    ("focus_640x384_rms_gammaRM_2links", dict(shape=(1, 1, 640, 384), gamma_layout=R, num_links=2)),
    (
        "batched_fp32_rms_gammaFP32",
        dict(shape=(2, 1, 64, 64), dtype=ttnn.float32, gamma_layout=L, gamma_dtype=ttnn.float32),
    ),
    (
        "indexer_640x32_ln_gamma_beta_RM_2links",
        dict(shape=(1, 1, 640, 32), norm_type="layer_norm", gamma_layout=R, with_beta=True, num_links=2),
    ),
    ("rank2_ln_gamma_only_line", dict(shape=(128, 256), norm_type="layer_norm", gamma_layout=L, topology=LINE)),
    ("rank3_rms_no_gamma_2links", dict(shape=(1, 64, 96), num_links=2)),
    ("w_non_aligned_rms_gammaRM", dict(shape=(1, 1, 32, 50), gamma_layout=R)),
    ("w_non_aligned_ln_gamma_beta", dict(shape=(1, 1, 64, 17), norm_type="layer_norm", gamma_layout=L, with_beta=True)),
    ("h_non_aligned_ln_no_affine", dict(shape=(1, 1, 17, 64), norm_type="layer_norm")),
    (
        "big_offset_ln_bf16",
        dict(shape=(1, 1, 64, 256), norm_type="layer_norm", gamma_layout=L, with_beta=True, offset=50.0, num_links=2),
    ),
    (
        "big_offset_ln_fp32",
        dict(
            shape=(1, 1, 32, 128),
            dtype=ttnn.float32,
            norm_type="layer_norm",
            gamma_layout=R,
            gamma_dtype=ttnn.float32,
            with_beta=True,
            offset=50.0,
        ),
    ),
    ("bf16_fp32acc_off_rms", dict(shape=(1, 1, 64, 128), gamma_layout=L, fp32_dest_acc_en=False, num_links=2)),
    ("bf8b_rms_gammaBF16", dict(shape=(1, 1, 64, 128), dtype=ttnn.bfloat8_b, gamma_layout=L, num_links=2)),
    ("height_sharded_rms", dict(shape=(1, 1, 256, 128), gamma_layout=R, memory_layout=HS, num_links=2)),
    ("width_sharded_rms", dict(shape=(1, 1, 64, 256), gamma_layout=R, memory_layout=WS, num_links=2)),
]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
@pytest.mark.parametrize("case", [c[1] for c in RING_CASES], ids=[c[0] for c in RING_CASES])
def test_all_gather_rms_norm_ring(mesh_device, case):
    run_case(mesh_device, cluster_axis=1, **case)


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], ids=["mesh2x2"], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_all_gather_rms_norm_mesh_axes(mesh_device, cluster_axis):
    run_case(mesh_device, cluster_axis=cluster_axis, shape=(1, 1, 64, 128), gamma_layout=R, topology=LINE, eps=1e-5)


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
def test_all_gather_rms_norm_output_tensor_and_repeat(mesh_device):
    """Preallocated output is written and returned; repeated calls are bit-identical (semaphores re-armed)."""
    tx, tg, tb, kwargs, cfg, first = run_case(
        mesh_device, cluster_axis=1, shape=(1, 1, 640, 384), gamma_layout=R, num_links=2
    )
    ref = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(first)]
    buf = first
    for _ in range(3):
        out = all_gather_rms_norm(
            tx,
            cluster_axis=1,
            gamma=tg,
            epsilon=EPS,
            topology=RING,
            num_links=2,
            compute_kernel_config=cfg,
            output_tensor=buf,
            **kwargs,
        )
        for o, b in zip(ttnn.get_device_tensors(out), ttnn.get_device_tensors(buf)):
            assert o.buffer_address() == b.buffer_address(), "output_tensor was not written in place"
        got = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        for a, b in zip(got, ref):
            assert torch.equal(a, b), "repeat call not bit-identical"
    # every member of the group returns the same full row
    for t in ref[1:]:
        assert torch.equal(t, ref[0]), "group members disagree"
