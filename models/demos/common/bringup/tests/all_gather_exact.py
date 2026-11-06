# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact model-case check of ttnn.experimental.high_bw_all_gather (stock), shared by the models' own tests
(models/demos/glm53_flash_d_p/tests/test_all_gather_stock.py, models/demos/deepseek_v41_flash_d_p/tests/
test_all_gather_stock.py). Not a test file itself.

The cases are the retired fork ttnn.bringup.fabric_all_gather's model cases (fabric_all_gather_ttnn/tests/cases.py,
O.1; the fork was retired 2026-11-06: upstream #61387 made the stock op the same program), unchanged, now run on the
stock op the models call. Pure data movement: every element of every chip's output is compared exactly (bits),
including, for a partial gather (gathered_dim_size), the bytes outside each rank's prefix, which must keep the
preallocated output's random fill. Every chip's input differs, so a chip gathered into the wrong group or slot fails.
Then determinism (models/demos/common/bringup/testing/determinism.py). Don't-care: nothing.

A case: id, sig (the fork-call capture's signature, for the record), mesh, device_params (fabric_config name,
l1_small_size, optional fabric_payload_bytes), input {shape, dtype, layout} (per chip), output {shape}, dim,
cluster_axis (0 / 1 / None = the whole mesh), num_links, optional gathered_dim_size and semaphores (the caller-owned
L1_SMALL pair the model passes), seed, exact."""

from __future__ import annotations

import torch

import ttnn
from models.demos.common.bringup.testing import determinism


# ---- torch semantics (high_bw_all_gather's contract)
def group(mesh: tuple[int, int], d: int, cluster_axis: int | None) -> list[int]:
    """Row-major device indices of device d's group, in rank order: cluster_axis 0 -> the chips of one mesh column,
    1 -> one mesh row, None -> every chip."""
    rows, cols = mesh
    r, c = divmod(d, cols)
    if cluster_axis is None:
        return list(range(rows * cols))
    if cluster_axis == 0:
        return [i * cols + c for i in range(rows)]
    return [r * cols + j for j in range(cols)]


def reference(xs, mesh, d, *, dim, cluster_axis, fill, gathered_dim_size=None):
    """xs: every device's input (row-major list); fill: the output's prior contents. -> device d's expected output: its
    group's inputs concatenated along `dim` in row-major chip order. gathered_dim_size (a partial gather): rank g writes
    the first gathered_dim_size / G entries of its input along `dim` into its fixed slot [g L, g L + gathered_dim_size /
    G) (L = the input's length along `dim`); every other output byte keeps what the preallocated output held."""
    members = group(mesh, d, cluster_axis)
    if gathered_dim_size is None:
        return torch.cat([xs[m] for m in members], dim=dim)
    out = fill.clone()
    L = xs[0].shape[dim]
    active = gathered_dim_size // len(members)
    assert gathered_dim_size % len(members) == 0 and active <= L
    for g, m in enumerate(members):
        out.narrow(dim, g * L, active).copy_(xs[m].narrow(dim, 0, active))
    return out


# ---- the device side
def device_params(c: dict) -> dict:
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    if "fabric_payload_bytes" in p:  # the model's fabric packet payload (harness.device_params)
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = int(p.pop("fabric_payload_bytes"))
        p["fabric_router_config"] = rc
    # A case runs on a box of its own mesh size only: a smaller mesh opened on a bigger box fails the FABRIC_2D router
    # handshake, and the groups depend on the mesh.
    p["require_exact_physical_num_devices"] = True
    return p


def parametrize(cases: list[dict]) -> dict:
    """pytest.mark.parametrize arguments for ("mesh_device, device_params, case", ...)."""
    return dict(
        argnames="mesh_device, device_params, case",
        argvalues=[(tuple(c["mesh"]), device_params(c), c) for c in cases],
        ids=[c["id"] for c in cases],
        indirect=["mesh_device", "device_params"],
    )


def _put(mesh_device, host, spec, mapper):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=getattr(ttnn.DataType, spec["dtype"]),
        layout=getattr(ttnn.Layout, spec["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


_BITS = {"BFLOAT16": torch.int16, "FLOAT32": torch.int32, "UINT32": torch.int32}


def _random(shape, dtype, g):
    """Random values in the captured dtype: bf16 / fp32 normal, uint32 random 31-bit words (as torch int32)."""
    if dtype == "UINT32":
        return torch.randint(0, 2**31 - 1, shape, generator=g, dtype=torch.int32)
    return torch.randn(shape, generator=g).to(torch.float32 if dtype == "FLOAT32" else torch.bfloat16)


def _semaphores(mesh_device, c):
    """The caller-owned pair the model passes (ready, data_valid): zero, the whole worker grid, L1_SMALL."""
    if not c.get("semaphores"):
        return {}
    g = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    s0, s1 = [ttnn.create_global_semaphore(mesh_device, crs, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    return {"ready_semaphore": s0, "data_valid_semaphore": s1}


def _inputs(mesh_device, c, seed):
    """Host per-device inputs (row-major list) and the output's prior fill; the device input, chip (r, c) holding
    xs[r * cols + c]. Distribution as the model declares it: along an axis, 2D (chip (r, c) = host[r, c]); over the
    whole mesh, `dim` sharded over every chip in row-major order (what the op validates)."""
    rows, cols = c["mesh"]
    g = torch.Generator().manual_seed(seed)
    _, _, n, w = c["input"]["shape"]
    host = _random((rows, cols, n, w), c["input"]["dtype"], g)
    fill = _random(c["output"]["shape"], c["input"]["dtype"], g)
    xs = [host[r, cc].reshape(c["input"]["shape"]) for r in range(rows) for cc in range(cols)]
    if c["cluster_axis"] is None:
        tt_in = _put(
            mesh_device, torch.cat(xs, dim=c["dim"]), c["input"], ttnn.ShardTensorToMesh(mesh_device, dim=c["dim"])
        )
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1))
        tt_in = _put(mesh_device, host, c["input"], mapper)
    assert list(tt_in.shape) == c["input"]["shape"]
    return xs, fill, tt_in


def _call(mesh_device, c, tt_in, fill, sems=None):
    """One call into a fresh output holding `fill` on every chip."""
    out = _put(mesh_device, fill, c["input"], ttnn.ReplicateTensorToMesh(mesh_device))
    kw = {"gathered_dim_size": c["gathered_dim_size"]} if c.get("gathered_dim_size") is not None else {}
    kw.update(sems or {})
    res = ttnn.experimental.high_bw_all_gather(
        tt_in, dim=c["dim"], output_tensor=out, cluster_axis=c["cluster_axis"], num_links=c["num_links"], **kw
    )
    return res if res is not None else out


def run_case(mesh_device, c: dict) -> None:
    """The case's call on random inputs: every chip's output bit for bit vs ``reference``, then determinism."""
    assert c["exact"]
    rows, cols = c["mesh"]
    sems = _semaphores(mesh_device, c)
    xs, fill, tt_in = _inputs(mesh_device, c, c["seed"])
    out = _call(mesh_device, c, tt_in, fill, sems)
    outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == rows * cols
    for d in range(rows * cols):
        want = reference(
            xs,
            (rows, cols),
            d,
            dim=c["dim"],
            cluster_axis=c["cluster_axis"],
            fill=fill,
            gathered_dim_size=c.get("gathered_dim_size"),
        )
        got = outs[d].reshape(want.shape).to(want.dtype)
        bits = _BITS[c["input"]["dtype"]]
        bad = got.view(bits) != want.view(bits)
        n_bad = int(bad.sum())
        assert n_bad == 0, f"dev {d}: {n_bad}/{want.numel()} elements differ, first at {bad.nonzero()[0].tolist()}"
    _, fill_b, tt_in_b = _inputs(mesh_device, c, c["seed"] + 1)
    determinism.assert_deterministic(
        lambda: _call(mesh_device, c, tt_in, fill, sems),
        lambda: _call(mesh_device, c, tt_in_b, fill_b, sems),
        first=out,
        label=c["id"],
    )
