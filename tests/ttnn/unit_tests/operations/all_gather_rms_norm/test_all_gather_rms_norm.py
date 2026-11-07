# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for all_gather_rms_norm (the spec; do not modify).

Every device of a cluster_axis group holds an (..., M, W_local) slice of the hidden dim; every member must return
torch.nn.functional.rms_norm(concat_p x[p], (W,), gamma, eps) over the full W = G * W_local.

Meshes: a 4-device group opened as 1x4 (cluster_axis=1) or 4x1 (cluster_axis=0) under FABRIC_2D_TORUS_XY with the
production router payload (14 KiB + 64 B on Blackhole). Ring cases run only where that 1x4 / 4x1 mesh closes a ring
(the QuietBox snake mock, eval/golden_tests/all_gather_rms_norm/topology.ring_mock_meshes); real Galaxy rings are
covered by the golden suite.

Regime pins (op_design.md -> Regime / split selection): A k=1 (1,1,32,64); A symmetric (1,1,640,384);
A owner (1,1,640,1792); N (1,1,17,50), (1,1,64,1000), (4,128,47), (32,17).

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm.py
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

torch.manual_seed(42)

EPS = 1e-6
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT
RING = ttnn.Topology.Ring
LINE = ttnn.Topology.Linear


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


def _ring_ok(mesh_shape, cluster_axis):
    try:
        from eval.golden_tests.all_gather_rms_norm.topology import ring_mock_meshes
    except ImportError:
        return False
    return (tuple(mesh_shape), cluster_axis) in ring_mock_meshes()


# --- reference and tensor helpers ----------------------------------------------------------------------------


def torch_reference(x_stacked, gamma, cluster_axis, eps=EPS):
    """x_stacked: (R, C, ..., M, W_local) — what each device holds. Returns (R, C, ..., M, W) fp32."""
    x = x_stacked.to(torch.float32)
    rows, cols = x.shape[0], x.shape[1]
    out_shape = (*x.shape[2:-1], x.shape[-1] * (rows, cols)[cluster_axis])
    out = torch.empty((rows, cols, *out_shape), dtype=torch.float32)
    w = None if gamma is None else gamma.to(torch.float32).reshape(-1)
    for other in range(cols if cluster_axis == 0 else rows):
        members = [(r, other) for r in range(rows)] if cluster_axis == 0 else [(other, c) for c in range(cols)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=w, eps=eps)
        for r, c in members:
            out[r, c] = normed
    return out


def to_mesh(stacked, mesh_device, *, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """stacked (R, C, *shape) -> mesh tensor where device (r, c) holds stacked[r, c]."""
    rows, cols = tuple(mesh_device.shape)
    assert tuple(stacked.shape[:2]) == (rows, cols)
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def replicate(t, mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return t.reshape(1, 1, *t.shape).expand(rows, cols, *t.shape).contiguous()


def gamma_to_mesh(gamma, mesh_device, *, dtype, layout):
    """Full-W gamma (W,) -> TILE (1,1,1,W) or ROW_MAJOR sticks (1,1,ceil(W/32),32), same on every device."""
    W = gamma.shape[-1]
    if layout == RM:
        padded = -(-W // 32) * 32
        per_dev = torch.nn.functional.pad(gamma, (0, padded - W)).reshape(1, 1, padded // 32, 32)
    else:
        per_dev = gamma.reshape(1, 1, 1, W)
    return to_mesh(replicate(per_dev, mesh_device), mesh_device, dtype=dtype, layout=layout)


def quantize_like_device(stacked, dtype):
    if dtype != ttnn.bfloat8_b:
        return stacked
    flat = stacked.reshape(-1, stacked.shape[-1])
    return ttnn.to_torch(ttnn.from_torch(flat, dtype=dtype, layout=TILE)).reshape(stacked.shape).to(torch.bfloat16)


def pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return 1.0 if den == 0 else float((a * b).sum() / den)


def check(out, expected, *, dtype, group, w_local):
    """Every device: the whole row and every group-position slice of it (a slice from the wrong peer, at the wrong
    offset, or normalized with a wrong denominator must not hide behind the rest of the row)."""
    rows, cols = expected.shape[0], expected.shape[1]
    assert list(out.shape) == list(expected.shape[2:]), f"shape {list(out.shape)} != {list(expected.shape[2:])}"
    assert out.dtype == dtype, f"dtype {out.dtype} != {dtype}"
    assert out.layout == TILE, f"layout {out.layout}"
    device_tensors = ttnn.get_device_tensors(out)
    assert len(device_tensors) == rows * cols
    target = PCC[dtype]
    for idx, t in enumerate(device_tensors):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(t).to(torch.float32)
        want = expected[r, c]
        assert torch.isfinite(got).all(), f"device ({r},{c}): non-finite output"
        assert pcc(got, want) >= target, f"device ({r},{c}): pcc {pcc(got, want):.6f} < {target}"
        for p in range(group):
            s = slice(p * w_local, (p + 1) * w_local)
            v = pcc(got[..., s], want[..., s])
            assert v >= target, f"device ({r},{c}) slice {p}: pcc {v:.6f} < {target}"


def run_case(
    mesh_device,
    *,
    cluster_axis,
    shape,
    dtype,
    topology,
    num_links,
    gamma_layout,
    gamma_dtype,
    fp32_dest_acc_en=True,
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    seed=42,
):
    if topology == RING and not _ring_ok(tuple(mesh_device.shape), cluster_axis):
        pytest.skip("this cluster cannot close a 4-device ring on the opened mesh")
    G = tuple(mesh_device.shape)[cluster_axis]
    w_local = shape[-1]
    torch.manual_seed(seed)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    x = quantize_like_device(x, dtype)
    gamma = None
    tg = None
    if gamma_layout is not None:
        gamma = torch.randn((G * w_local,), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tg = gamma_to_mesh(gamma, mesh_device, dtype=gamma_dtype, layout=gamma_layout)
    expected = torch_reference(x, gamma, cluster_axis)

    kwargs = {}
    in_mc = ttnn.DRAM_MEMORY_CONFIG
    if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        from eval.sharding import auto_shard_config

        in_mc = auto_shard_config(list(shape), memory_layout, layout=TILE, dtype=dtype, device=mesh_device)
        out_shape = [*shape[:-1], G * w_local]
        kwargs["memory_config"] = auto_shard_config(
            out_shape, memory_layout, layout=TILE, dtype=dtype, device=mesh_device
        )
    tx = to_mesh(x, mesh_device, dtype=dtype, layout=TILE, memory_config=in_mc)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False

    out = all_gather_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        gamma=tg,
        epsilon=EPS,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=cfg,
        **kwargs,
    )
    check(out, expected, dtype=dtype, group=G, w_local=w_local)


# --- the 4-device group along cluster_axis 1 (mesh 1x4) ------------------------------------------------------

# (shape, dtype, topology, num_links, gamma_layout, gamma_dtype)
AXIS1_CASES = [
    # perf cases (MLA q_a latent and the dense-FFN norm), production config: ring, 2 links, bf16, RM-stick gamma
    pytest.param((1, 1, 640, 384), ttnn.bfloat16, RING, 2, RM, ttnn.bfloat16, id="focus_640x384-ring-L2-gRM"),
    pytest.param((1, 1, 640, 512), ttnn.bfloat16, RING, 2, RM, ttnn.bfloat16, id="640x512-ring-L2-gRM"),
    pytest.param((1, 1, 640, 1792), ttnn.bfloat16, RING, 2, RM, ttnn.bfloat16, id="640x1792-ring-L2-gRM"),
    # single tile-row, smallest member split (regime A, k = 1)
    pytest.param((1, 1, 32, 64), ttnn.bfloat16, RING, 1, None, None, id="32x64-ring-L1-nogamma"),
    # decode-wide fp32 with fp32 TILE gamma
    pytest.param((1, 1, 32, 1792), ttnn.float32, RING, 1, TILE, ttnn.float32, id="32x1792-fp32-ring-L1-gTILE"),
    # multi-batch rank 4, non-square
    pytest.param((2, 4, 128, 512), ttnn.bfloat16, RING, 2, TILE, ttnn.bfloat16, id="2x4x128x512-ring-L2-gTILE"),
    # rank 3 and rank 2
    pytest.param((1, 32, 384), ttnn.bfloat16, RING, 1, RM, ttnn.bfloat16, id="r3_1x32x384-ring-L1-gRM"),
    pytest.param((640, 384), ttnn.bfloat8_b, RING, 2, TILE, ttnn.bfloat16, id="r2_640x384-bf8b-ring-L2-gTILE"),
    # mixed precision: bf16 activations, fp32 weights
    pytest.param((1, 1, 128, 384), ttnn.bfloat16, RING, 2, RM, ttnn.float32, id="128x384-ring-L2-gRM_fp32"),
    # H not tile-aligned
    pytest.param((1, 1, 100, 512), ttnn.bfloat16, RING, 2, RM, ttnn.bfloat16, id="100x512-ring-L2-gRM"),
    # W_local not tile-aligned: slices land at non-tile offsets of the dense output row (regime N)
    pytest.param((1, 1, 17, 50), ttnn.bfloat16, RING, 1, RM, ttnn.bfloat16, id="17x50-ring-L1-gRM"),
    pytest.param((1, 1, 64, 1000), ttnn.float32, RING, 2, TILE, ttnn.float32, id="64x1000-fp32-ring-L2-gTILE"),
    pytest.param((4, 128, 47), ttnn.bfloat16, RING, 1, None, None, id="r3_4x128x47-ring-L1-nogamma"),
    # bfloat8_b x W_local non-aligned (Refinement 2): regime N composes from a bf16 staging row
    pytest.param((1, 1, 17, 50), ttnn.bfloat8_b, RING, 1, RM, ttnn.bfloat16, id="17x50-bf8b-ring-L1-gRM"),
    pytest.param((1, 1, 64, 1000), ttnn.bfloat8_b, RING, 2, TILE, ttnn.float32, id="64x1000-bf8b-ring-L2-gTILE"),
    pytest.param((4, 128, 47), ttnn.bfloat8_b, LINE, 1, None, None, id="r3_4x128x47-bf8b-line-L1-nogamma"),
    # far slice split between directions by row parity (Refinement 3): odd tile-row counts, so a 1-row last item
    # leaves one direction's far half empty (R_t 5 / 3 with 4 / 2 rows per item), and 1-row items (xfer_rows 1, wide
    # rows) where every other far item is empty
    pytest.param((1, 1, 160, 384), ttnn.bfloat16, RING, 2, RM, ttnn.bfloat16, id="160x384-ring-L2-rowsplit_tail"),
    pytest.param((1, 1, 96, 384), ttnn.bfloat16, RING, 1, None, None, id="96x384-ring-L1-rowsplit_tail"),
    pytest.param((1, 1, 96, 1792), ttnn.bfloat16, RING, 2, RM, ttnn.bfloat16, id="96x1792-ring-L2-rowsplit_xr1"),
    # Linear over the same four devices
    pytest.param((1, 1, 640, 384), ttnn.bfloat16, LINE, 2, RM, ttnn.bfloat16, id="focus_640x384-line-L2-gRM"),
    pytest.param((1, 1, 64, 128), ttnn.float32, LINE, 1, TILE, ttnn.float32, id="64x128-fp32-line-L1-gTILE"),
    pytest.param((32, 17), ttnn.bfloat16, LINE, 1, None, None, id="r2_32x17-line-L1-nogamma"),
    pytest.param((1, 1, 17, 1280), ttnn.bfloat8_b, LINE, 2, RM, ttnn.bfloat16, id="17x1280-bf8b-line-L2-gRM"),
]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
@pytest.mark.parametrize("shape,dtype,topology,num_links,gamma_layout,gamma_dtype", AXIS1_CASES)
def test_all_gather_rms_norm_axis1(mesh_device, shape, dtype, topology, num_links, gamma_layout, gamma_dtype):
    run_case(
        mesh_device,
        cluster_axis=1,
        shape=shape,
        dtype=dtype,
        topology=topology,
        num_links=num_links,
        gamma_layout=gamma_layout,
        gamma_dtype=gamma_dtype,
    )


# --- the same group along cluster_axis 0 (mesh 4x1) ----------------------------------------------------------

AXIS0_CASES = [
    pytest.param((1, 1, 640, 384), ttnn.bfloat16, RING, 2, RM, ttnn.bfloat16, id="focus_640x384-ring-L2-gRM"),
    pytest.param((1, 1, 128, 256), ttnn.bfloat16, LINE, 1, TILE, ttnn.bfloat16, id="128x256-line-L1-gTILE"),
    pytest.param((1, 1, 64, 100), ttnn.float32, LINE, 1, None, None, id="64x100-fp32-line-L1-nogamma"),
]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True, ids=["mesh4x1"])
@pytest.mark.parametrize("shape,dtype,topology,num_links,gamma_layout,gamma_dtype", AXIS0_CASES)
def test_all_gather_rms_norm_axis0(mesh_device, shape, dtype, topology, num_links, gamma_layout, gamma_dtype):
    run_case(
        mesh_device,
        cluster_axis=0,
        shape=shape,
        dtype=dtype,
        topology=topology,
        num_links=num_links,
        gamma_layout=gamma_layout,
        gamma_dtype=gamma_dtype,
    )


# --- precision mode, placement, defaults, preallocated output ------------------------------------------------


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
@pytest.mark.parametrize("shape", [(1, 1, 640, 384), (1, 1, 32, 1792)], ids=["640x384", "32x1792"])
def test_all_gather_rms_norm_fp16_dest(mesh_device, shape):
    """bfloat16 with fp32_dest_acc_en=False (16-bit DEST accumulation)."""
    run_case(
        mesh_device,
        cluster_axis=1,
        shape=shape,
        dtype=ttnn.bfloat16,
        topology=LINE,
        num_links=2,
        gamma_layout=RM,
        gamma_dtype=ttnn.bfloat16,
        fp32_dest_acc_en=False,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
@pytest.mark.parametrize(
    "memory_layout,shape",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 1, 640, 384)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1, 64, 512)),
    ],
    ids=["height_640x384", "width_64x512"],
)
def test_all_gather_rms_norm_sharded(mesh_device, memory_layout, shape):
    """L1-sharded input with an output sharded the same way for the gathered shape."""
    run_case(
        mesh_device,
        cluster_axis=1,
        shape=shape,
        dtype=ttnn.bfloat16,
        topology=LINE,
        num_links=1,
        gamma_layout=TILE,
        gamma_dtype=ttnn.bfloat16,
        memory_layout=memory_layout,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
def test_all_gather_rms_norm_defaults(mesh_device):
    """all_gather_rms_norm(x, cluster_axis=1): Linear, every usable link, no gamma, default compute config."""
    shape = (1, 1, 64, 256)
    G = 4
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch.bfloat16)
    expected = torch_reference(x, None, 1)
    out = all_gather_rms_norm(to_mesh(x, mesh_device, dtype=ttnn.bfloat16, layout=TILE), cluster_axis=1)
    check(out, expected, dtype=ttnn.bfloat16, group=G, w_local=shape[-1])


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
def test_all_gather_rms_norm_output_tensor_and_repeat(mesh_device):
    """MLA preallocates the gathered buffer: the op writes into it and returns it. Back-to-back calls (the ready
    fence and every per-call counter re-armed) give the new result, and identical calls are bit-identical."""
    topology = RING if _ring_ok(tuple(mesh_device.shape), 1) else LINE
    shape = (1, 1, 640, 384)
    G = 4
    rows, cols = tuple(mesh_device.shape)
    out_shape = (*shape[:-1], G * shape[-1])
    buf = to_mesh(replicate(torch.full(out_shape, 7.0), mesh_device), mesh_device, dtype=ttnn.bfloat16, layout=TILE)
    gamma = torch.randn((G * shape[-1],), dtype=torch.float32).to(torch.bfloat16)
    tg = gamma_to_mesh(gamma, mesh_device, dtype=ttnn.bfloat16, layout=RM)
    first = None
    for seed in (1, 2, 2):
        torch.manual_seed(seed)
        x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch.bfloat16)
        expected = torch_reference(x, gamma, 1)
        out = all_gather_rms_norm(
            to_mesh(x, mesh_device, dtype=ttnn.bfloat16, layout=TILE),
            cluster_axis=1,
            gamma=tg,
            topology=topology,
            num_links=2,
            output_tensor=buf,
        )
        for d_out, d_buf in zip(ttnn.get_device_tensors(out), ttnn.get_device_tensors(buf)):
            assert d_out.buffer_address() == d_buf.buffer_address(), "the op must return the provided output_tensor"
        check(buf, expected, dtype=ttnn.bfloat16, group=G, w_local=shape[-1])
        now = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(buf)]
        if seed == 2:
            if first is None:
                first = now
            else:
                for a, b in zip(first, now):
                    assert torch.equal(a, b), "identical calls must give bit-identical output"
