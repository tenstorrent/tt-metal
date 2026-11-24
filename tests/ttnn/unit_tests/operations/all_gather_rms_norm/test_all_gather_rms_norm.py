# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for all_gather_rms_norm (immutable spec — the implementer must not modify this file).

Every device (r, c) holds its own (..., M, W_local) slice; the G devices of a cluster_axis group together hold the
full hidden dim W = G * W_local. Every member must return the same normalized (..., M, W) row:
torch rms_norm (or layer_norm with norm_type="layer_norm") over the concatenated slices, in group-position order.

The mesh is the root `mesh_device` fixture (indirect): a 4-device group opened as 1x4 (cluster_axis 1) or 4x1
(cluster_axis 0) under FABRIC_2D_TORUS_XY with the production router payload, the group the models run (a Galaxy TP
axis; on a QuietBox 2 the box's snake ring). Accuracy is gated with the golden suite's own limits
(eval/golden_tests/all_gather_rms_norm/helpers.py TOLERANCES via check_output: PCC and relative RMS, every device
and every group-position slice on its own).

Regime pins (op_design.md, Regime selection):
  R_A interleaved, k > 1 ......... (1, 1, 640, 384)  rms_norm, the PERF FOCUS shape
  R_A, one tile per slice ........ (1, 1, 640, 32)   layer_norm + gamma + beta, the LN PERF FOCUS shape
  R_A single tile ................ (1, 1, 32, 32)
  R_A wide decode ................ (1, 1, 32, 1792)
  R_A multi-batch rank 4 ......... (2, 1, 128, 256), (2, 4, 64, 128)
  R_A rank 3 / rank 2 ............ (1, 32, 384), (640, 384)
  R_U W_local non-aligned ........ (1, 1, 17, 50), (1, 1, 32, 50)
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.all_gather_rms_norm.helpers import (
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gathered_shape,
    pytorch_all_gather_rms_norm,
    replicate,
)
from eval.golden_tests.all_gather_rms_norm.topology import ring_mock_meshes, system_mesh_shape, wrap_axes
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

EPS = 1e-6

_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def _router_config():
    """Production router payload: 14 KiB + 64 B on Blackhole, 7 KiB + 64 B on Wormhole (as the golden conftest)."""
    try:
        cfg = ttnn._ttnn.fabric.FabricRouterConfig()
        cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
        return cfg
    except Exception:  # noqa: BLE001 — fall back to the fabric default payload (correctness is unaffected)
        return None


def _device_params():
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}
    cfg = _router_config()
    if cfg is not None:
        params["fabric_router_config"] = cfg
    return params


DEVICE_PARAMS = [pytest.param(_device_params(), id="torus_xy")]


def _ring_ok(mesh_shape, cluster_axis):
    """Ring needs a closed 4-cycle along the group: the box's snake ring (no wrap links) or a real wrap axis."""
    if (tuple(mesh_shape), cluster_axis) in [(tuple(s), a) for s, a in ring_mock_meshes()]:
        return True
    return system_mesh_shape()[cluster_axis] == mesh_shape[cluster_axis] and cluster_axis in wrap_axes()


def _skip_if_unrunnable(mesh_device, cluster_axis, topology):
    shape = tuple(mesh_device.shape)
    if shape[cluster_axis] < 2:
        pytest.skip(f"cluster_axis {cluster_axis} has size {shape[cluster_axis]}")
    if topology == ttnn.Topology.Ring and not _ring_ok(shape, cluster_axis):
        pytest.skip(f"no ring along cluster_axis {cluster_axis} of mesh {shape} on this cluster")


def _make_inputs(mesh_device, shape, dtype, *, layer_norm, offset=0.0):
    """Stacked (R, C, *shape) x: torch.randn with seed 42; LayerNorm rows get a per-row mean offset so a mean taken
    per slice, or no mean subtraction, cannot pass."""
    torch.manual_seed(42)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32) + offset
    if layer_norm:
        x = x + torch.randn((rows, cols, *shape[:-1], 1), dtype=torch.float32) * 2.0
    return x.to(_TORCH_DTYPE[dtype])


def _make_weight(W, dtype, seed):
    torch.manual_seed(seed)
    return torch.randn((W,), dtype=torch.float32).to(_TORCH_DTYPE[dtype])


def _run_case(
    mesh_device,
    shape,
    *,
    cluster_axis,
    dtype,
    norm_type,
    gamma_layout,
    with_beta,
    topology,
    num_links,
    offset=0.0,
    output_tensor=None,
    memory_config=None,
    fp32_dest_acc_en=True,
):
    G = tuple(mesh_device.shape)[cluster_axis]
    out_shape = gathered_shape(shape, G)
    layer_norm = norm_type == "layer_norm"
    x = _make_inputs(mesh_device, shape, dtype, layer_norm=layer_norm, offset=offset)
    gamma = beta = None
    kwargs = {}
    if gamma_layout is not None:
        gamma = _make_weight(out_shape[-1], dtype, seed=43)
        kwargs["gamma"] = create_ttnn_gamma(gamma, mesh_device, dtype=dtype, layout=gamma_layout)
        if with_beta:
            beta = _make_weight(out_shape[-1], dtype, seed=44)
            kwargs["beta"] = create_ttnn_gamma(beta, mesh_device, dtype=dtype, layout=gamma_layout)
    if layer_norm:
        kwargs["norm_type"] = "layer_norm"
    if output_tensor is not None:
        kwargs["output_tensor"] = output_tensor
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    if not fp32_dest_acc_en:
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = False
        cfg.math_approx_mode = False
        kwargs["compute_kernel_config"] = cfg
    expected = pytorch_all_gather_rms_norm(x, gamma, cluster_axis, EPS, norm_type=norm_type, beta=beta)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    out = all_gather_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        epsilon=EPS,
        topology=topology,
        num_links=num_links,
        **kwargs,
    )
    check_output(
        out,
        expected,
        shape=out_shape,
        dtype=dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        expected_layout=ttnn.TILE_LAYOUT,
        group=G,
    )
    return out


R = ttnn.Topology.Ring
L = ttnn.Topology.Linear
RM = ttnn.ROW_MAJOR_LAYOUT
TL = ttnn.TILE_LAYOUT

# (shape, norm_type, gamma_layout or None, with_beta, topology, num_links, dtype)
CASES_AXIS1 = [
    # PERF FOCUS (q_a latent): 640 x 384 per chip, gamma ROW_MAJOR sticks, ring, 2 links
    ((1, 1, 640, 384), "rms_norm", RM, False, R, 2, ttnn.bfloat16),
    # PERF FOCUS (indexer k, LayerNorm): one tile per slice, gamma + beta sticks, ring, 2 links
    ((1, 1, 640, 32), "layer_norm", RM, True, R, 2, ttnn.bfloat16),
    # single tile, no gamma, line, 1 link
    ((1, 1, 32, 32), "rms_norm", None, False, L, 1, ttnn.bfloat16),
    # wide decode (dense-FFN norm), TILE gamma, ring, 1 link
    ((1, 1, 32, 1792), "rms_norm", TL, False, R, 1, ttnn.bfloat16),
    # multi-batch rank 4, fp32 end to end, TILE gamma, line, 2 links
    ((2, 1, 128, 256), "rms_norm", TL, False, L, 2, ttnn.float32),
    ((2, 4, 64, 128), "layer_norm", TL, True, R, 2, ttnn.bfloat16),
    # rank 3 and rank 2
    ((1, 32, 384), "rms_norm", RM, False, R, 2, ttnn.bfloat16),
    ((640, 384), "layer_norm", None, False, L, 1, ttnn.bfloat16),
    # W_local non-aligned (R_U), H non-aligned too
    ((1, 1, 17, 50), "rms_norm", RM, False, R, 1, ttnn.bfloat16),
    ((1, 1, 32, 50), "layer_norm", TL, False, L, 2, ttnn.float32),
    # H non-aligned, aligned W
    ((1, 1, 100, 512), "rms_norm", TL, False, R, 2, ttnn.bfloat16),
]


def _case_id(c):
    shape, norm, gl, beta, topo, links, dtype = c
    g = "nogamma" if gl is None else ("gRM" if gl == RM else "gTILE")
    return (
        f"{'x'.join(map(str, shape))}-{norm}-{g}{'-beta' if beta else ''}-{topo.name}-L{links}-"
        f"{'fp32' if dtype == ttnn.float32 else 'bf16'}"
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("case", CASES_AXIS1, ids=[_case_id(c) for c in CASES_AXIS1])
def test_all_gather_rms_norm(mesh_device, case):
    shape, norm_type, gamma_layout, with_beta, topology, num_links, dtype = case
    _skip_if_unrunnable(mesh_device, 1, topology)
    _run_case(
        mesh_device,
        shape,
        cluster_axis=1,
        dtype=dtype,
        norm_type=norm_type,
        gamma_layout=gamma_layout,
        with_beta=with_beta,
        topology=topology,
        num_links=num_links,
    )


CASES_AXIS0 = [
    ((1, 1, 640, 384), "rms_norm", RM, False, R, 2, ttnn.bfloat16),
    ((1, 1, 64, 128), "layer_norm", TL, True, L, 1, ttnn.bfloat16),
]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 1), id="4x1")], indirect=True)
@pytest.mark.parametrize("case", CASES_AXIS0, ids=[_case_id(c) for c in CASES_AXIS0])
def test_all_gather_rms_norm_cluster_axis0(mesh_device, case):
    shape, norm_type, gamma_layout, with_beta, topology, num_links, dtype = case
    _skip_if_unrunnable(mesh_device, 0, topology)
    _run_case(
        mesh_device,
        shape,
        cluster_axis=0,
        dtype=dtype,
        norm_type=norm_type,
        gamma_layout=gamma_layout,
        with_beta=with_beta,
        topology=topology,
        num_links=num_links,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
def test_layer_norm_big_offset_fp32(mesh_device):
    """LayerNorm, fp32 end to end, row mean ~50 with spread ~1: the variance must not cancel (E[x^2] - mean^2) and
    the centering must keep fp32 precision (op_design.md, Key Risks: LN precision)."""
    _skip_if_unrunnable(mesh_device, 1, ttnn.Topology.Linear)
    _run_case(
        mesh_device,
        (1, 1, 128, 384),
        cluster_axis=1,
        dtype=ttnn.float32,
        norm_type="layer_norm",
        gamma_layout=TL,
        with_beta=True,
        topology=ttnn.Topology.Linear,
        num_links=1,
        offset=50.0,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
def test_output_tensor_reuse_and_determinism(mesh_device):
    """Preallocated output (as MLA passes it): written in place, returned, and bit-identical over repeated calls
    (counters re-armed, no stale slices or partials between calls)."""
    topology = ttnn.Topology.Ring if _ring_ok(tuple(mesh_device.shape), 1) else ttnn.Topology.Linear
    _skip_if_unrunnable(mesh_device, 1, topology)
    shape = (1, 1, 640, 384)
    G = tuple(mesh_device.shape)[1]
    out_shape = gathered_shape(shape, G)
    buf = create_ttnn_input_tensor(
        replicate(torch.full(out_shape, 7.0), mesh_device), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    first = None
    for _ in range(3):
        out = _run_case(
            mesh_device,
            shape,
            cluster_axis=1,
            dtype=ttnn.bfloat16,
            norm_type="rms_norm",
            gamma_layout=RM,
            with_beta=False,
            topology=topology,
            num_links=2,
            output_tensor=buf,
        )
        for dev_out, dev_buf in zip(ttnn.get_device_tensors(out), ttnn.get_device_tensors(buf)):
            assert dev_out.buffer_address() == dev_buf.buffer_address(), "op must return the provided output_tensor"
        now = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(buf)]
        if first is None:
            first = now
        else:
            for a, b in zip(first, now):
                assert torch.equal(a, b), "output differs between identical calls"


# ---------------------------------------------------------------------------------------------------------------------
# Refinement 5: L1 INTERLEAVED output (ttnn.L1_MEMORY_CONFIG) — the MLA q_a norm's real call. Input stays DRAM.
# ---------------------------------------------------------------------------------------------------------------------

# (shape, norm_type, gamma_layout or None, with_beta, topology, num_links, dtype, fp32_dest_acc_en)
CASES_L1_OUT = [
    ((1, 1, 640, 384), "rms_norm", RM, False, R, 2, ttnn.bfloat16, True),  # the real call
    ((1, 1, 640, 384), "rms_norm", RM, False, R, 2, ttnn.bfloat16, False),
    ((1, 1, 640, 32), "layer_norm", RM, True, R, 2, ttnn.bfloat16, True),
    ((1, 1, 640, 32), "layer_norm", RM, True, R, 2, ttnn.bfloat16, False),
    ((1, 1, 640, 512), "rms_norm", TL, False, L, 1, ttnn.bfloat16, True),
    ((1, 1, 128, 384), "layer_norm", TL, True, R, 2, ttnn.float32, True),
    # non-aligned (R_U: slot-contiguous DRAM staging, then the assembly writes the L1 output pages)
    ((1, 1, 17, 50), "rms_norm", RM, False, R, 1, ttnn.bfloat16, True),
    ((1, 1, 100, 512), "layer_norm", TL, True, L, 2, ttnn.bfloat16, True),
]


def _l1_case_id(c):
    return _case_id(c[:7]) + ("-fp32acc" if c[7] else "-bf16acc")


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("case", CASES_L1_OUT, ids=[_l1_case_id(c) for c in CASES_L1_OUT])
def test_l1_interleaved_output(mesh_device, case):
    shape, norm_type, gamma_layout, with_beta, topology, num_links, dtype, fp32_acc = case
    _skip_if_unrunnable(mesh_device, 1, topology)
    out = _run_case(
        mesh_device,
        shape,
        cluster_axis=1,
        dtype=dtype,
        norm_type=norm_type,
        gamma_layout=gamma_layout,
        with_beta=with_beta,
        topology=topology,
        num_links=num_links,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        fp32_dest_acc_en=fp32_acc,
    )
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
def test_l1_interleaved_preallocated_output(mesh_device):
    """A preallocated L1 interleaved output_tensor: written in place, returned, repeatable."""
    topology = ttnn.Topology.Ring if _ring_ok(tuple(mesh_device.shape), 1) else ttnn.Topology.Linear
    _skip_if_unrunnable(mesh_device, 1, topology)
    shape = (1, 1, 640, 384)
    out_shape = gathered_shape(shape, tuple(mesh_device.shape)[1])
    buf = create_ttnn_input_tensor(
        replicate(torch.full(out_shape, 7.0), mesh_device),
        mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    for _ in range(2):
        out = _run_case(
            mesh_device,
            shape,
            cluster_axis=1,
            dtype=ttnn.bfloat16,
            norm_type="rms_norm",
            gamma_layout=RM,
            with_beta=False,
            topology=topology,
            num_links=2,
            output_tensor=buf,
        )
        for dev_out, dev_buf in zip(ttnn.get_device_tensors(out), ttnn.get_device_tensors(buf)):
            assert dev_out.buffer_address() == dev_buf.buffer_address(), "op must return the provided output_tensor"
        assert out.memory_config().buffer_type == ttnn.BufferType.L1


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
def test_l1_interleaved_input_refused(mesh_device, expect_error):
    """The input stays DRAM interleaved: an L1 interleaved INPUT is refused at validate()."""
    from ttnn.operations._op_contract import UnsupportedAxisValue

    x = torch.randn((1, 4, 1, 1, 64, 128), dtype=torch.bfloat16)
    tx_l1 = create_ttnn_input_tensor(
        x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    with expect_error(UnsupportedAxisValue, "interleaved input must be in DRAM"):
        all_gather_rms_norm(tx_l1, cluster_axis=1, epsilon=EPS, memory_config=ttnn.L1_MEMORY_CONFIG)
