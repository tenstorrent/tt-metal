# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""all_gather_rms_norm debug tests (implementer). DO NOT DELETE — documents the debugging process."""

import pytest
import torch
import ttnn


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
def test_mesh_open_only(mesh_device):
    """Baseline: open the fabric mesh, move a tensor, no op. Tells whether teardown Ethernet timeouts are ours."""
    t = ttnn.from_torch(
        torch.ones(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    assert ttnn.to_torch(ttnn.get_device_tensors(t)[0]).sum() == 32 * 32


# ---- owner combine (k > K_SYM_MAX) and split pins ---------------------------------------------------------------
def _run(mesh_device, shape, *, topology, num_links, gamma_layout=None, dtype=ttnn.bfloat16, seed=7):
    import importlib

    from tests.ttnn.unit_tests.operations.all_gather_rms_norm import test_all_gather_rms_norm as acc

    acc.run_case(
        mesh_device,
        cluster_axis=1,
        shape=shape,
        dtype=dtype,
        topology=topology,
        num_links=num_links,
        gamma_layout=gamma_layout,
        gamma_dtype=dtype if gamma_layout is not None else None,
        seed=seed,
    )


def _split_of(mesh_device, shape, *, ring, L, dtype=ttnn.bfloat16):
    from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm_program_descriptor as pd

    chips, G = pd.plan(mesh_device, 1, ring, L)
    allowed = pd.allowed_cores(mesh_device)
    pu = {c for ch in chips.values() for c in ch.ports.values()}
    geo = pd.derive_geometry(shape, G)
    T = ttnn.tile_size(dtype)
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    rp = max(1, payload // T)
    sz = pd.Sizes(T, 4096, 192, 10, rp, max(1, min(8, pd.PORT_CB_BYTES // (2 * rp * T))), True)
    return pd.select_split(geo, sz, len(allowed - pu), pd.cb_l1_budget(mesh_device), ring, L)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
def test_split_pins(mesh_device):
    """The split rule on the fused grid picks the design's k for the perf shapes (op_design.md -> Regime / split
    selection): 640x384 / 640x512 / 640x1792 symmetric k = 8 on 2 links (Refinement 4: the packet-aware objective
    moved 640x1792 off k = 12 owner, 86.1 -> 81.5 us measured), and 640x1792 owner k = 16 on 1 link (145.0 us at the
    old k = 28 -> 108.0 us). The x * gamma prescale is on for every one of them (regime A, k > 1, gamma)."""
    from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm_program_descriptor as pd

    if pd.FORCE_K:
        pytest.skip("AGRMS_FORCE_K set")
    for shape, L, k, owner in (
        ((1, 1, 640, 384), 2, 8, False),
        ((1, 1, 640, 512), 2, 8, False),
        ((1, 1, 640, 1792), 2, 8, False),
        ((1, 1, 640, 1792), 1, 16, True),
    ):
        sp = _split_of(mesh_device, shape, ring=True, L=L)
        assert (sp.k, sp.combine_owner) == (k, owner), f"{shape} L{L}: k={sp.k} owner={sp.combine_owner}"
        geo = pd.derive_geometry(shape, 4)
        sz = pd.Sizes(ttnn.tile_size(ttnn.bfloat16), 4096, 192, 10, 7, 4, True)
        assert pd.prescale_on(geo, sz, sp) == pd.PRESCALE_GAMMA


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
@pytest.mark.parametrize(
    "shape,force_k",
    [((1, 1, 2048, 1792), 0), ((1, 1, 1024, 1792), 40), ((1, 1, 256, 1792), 16)],
    ids=["owner_multiblock_auto", "owner_k40_rowgather2", "owner_k16"],
)
def test_owner_combine(mesh_device, shape, force_k, monkeypatch):
    """Owner combine with several blocks per core (owner rotation) and k > 32 (two gather tiles per owned row)."""
    from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "FORCE_K", force_k)
    _run(mesh_device, shape, topology=ttnn.Topology.Linear, num_links=1, gamma_layout=ttnn.ROW_MAJOR_LAYOUT)


# ---- Refinement 2: bfloat8_b x W_local non-aligned (regime N bf16 staging row) ---------------------------------
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 50), (1, 1, 17, 50)], ids=["32x50", "17x50"])
def test_bf8b_regime_n_columns(mesh_device, shape):
    """Device d holds value (d + 1) * (1 + col/64): every output column's error is printed, so a wrong compose
    source / offset or a zeroing that hits real datums shows up as a column pattern."""
    from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm
    from tests.ttnn.unit_tests.operations.all_gather_rms_norm import test_all_gather_rms_norm as acc

    G, w = 4, shape[-1]
    base = torch.zeros((1, G, *shape))
    for d in range(G):
        base[0, d] = (d + 1) * (1 + torch.arange(w).float() / 64)
    x = acc.quantize_like_device(base.to(torch.bfloat16), ttnn.bfloat8_b)
    exp = acc.torch_reference(x, None, 1)[0, 0]
    tx = acc.to_mesh(x, mesh_device, dtype=ttnn.bfloat8_b, layout=acc.TILE)
    out = all_gather_rms_norm(tx, cluster_axis=1, epsilon=acc.EPS, topology=ttnn.Topology.Linear, num_links=1)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    err = (got - exp).abs().amax(dim=tuple(range(got.dim() - 1)))
    bad = (err > 0.05).nonzero().flatten().tolist()
    print("bad columns:", bad)
    print("row0 got:", got.reshape(-1, G * w)[0].tolist())
    print("row0 exp:", exp.reshape(-1, G * w)[0].tolist())
    assert not bad
