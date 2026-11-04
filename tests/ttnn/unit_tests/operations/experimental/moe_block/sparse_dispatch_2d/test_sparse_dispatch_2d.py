# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.moe_block.sparse_dispatch_2d on device.

- test_dispatch: random rows and uniform routing on the system mesh (LoudBox 2x4 on FABRIC_2D, Galaxy 8x4 on
  FABRIC_2D_TORUS_XY), bit-exact against the torch model of the result (sparse_dispatch_2d_inputs.assert_dispatched: the
  destinations' rows; pass-through chips may keep the home's row);
  a second call on new buffers must hit the program cache and patch the addresses.
- test_reference_case: bit-exact against saved reference outputs; runs only with $MOE_BLOCK_REFS set.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sparse_dispatch_2d_inputs as inputs  # noqa: E402
from ref_harness import assert_bit_exact, case_dir, from_mesh, list_cases, load_case, to_mesh  # noqa: E402

mb = ttnn.experimental.moe_block
OP = "sparse_dispatch_2d"
CHUNK = 5120  # tokens of the mesh a call (S = CHUNK / N a chip)
TOP_K = 8


def _system_mesh():
    n = ttnn.get_num_devices()
    return {8: (2, 4), 32: (8, 4)}.get(n, (2, 4))


def _device_params(mesh_shape, payload):
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload
    fabric = ttnn.FabricConfig.FABRIC_2D_TORUS_XY if tuple(mesh_shape) == (8, 4) else ttnn.FabricConfig.FABRIC_2D
    return {
        "fabric_config": fabric,
        "fabric_router_config": router,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }


def _resources(mesh_device, width):
    R, C = tuple(mesh_device.shape)
    # (only the experts a chip matters to the dispatch)
    dims = mb.MoEModelDims(num_experts=8 * R * C, top_k=TOP_K, emb_dim=width, routed_expert_hidden_dim=2048)
    return mb.MoEBlockResources(mesh_device, dims, mb.MoEBlockConfig())


def _inputs(mesh_device, resources, width, seed):
    R, C = tuple(mesh_device.shape)
    S = CHUNK // (R * C)
    topo = resources.topology
    gen = torch.Generator().manual_seed(seed)
    route = inputs.route_words(
        R, C, S, 8 * R * C, TOP_K, resources.config.num_links, topo["wrap_x"], topo["wrap_y"], mb.span_entry, gen
    )
    xbuf_in = inputs.xbuf_inputs(R, C, S, width, gen)
    per_chip = {ch: {"xbuf_in": xbuf_in[ch], "route": route[ch]} for ch in route}
    return S, per_chip, (xbuf_in, route)


def _to_device(per_chip, mesh_device):
    xbuf = to_mesh(per_chip, "xbuf_in", mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    route = to_mesh(per_chip, "route", mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return xbuf, route


def _check(mesh_device, xbuf, model):
    R, C = tuple(mesh_device.shape)
    xbuf_in, route = model
    inputs.assert_dispatched(from_mesh(xbuf, (R, C)), xbuf_in, route, R, C, CHUNK // (R * C))


_MESH = _system_mesh()
WIDTHS = [3584, 6144]  # K3's latent, GLM-5.3's emb_dim


@pytest.mark.parametrize(
    "mesh_device, device_params, width",
    [
        pytest.param(_MESH, _device_params(_MESH, inputs.fabric_payload(w)), w, id=f"{_MESH[0]}x{_MESH[1]}-W{w}")
        for w in WIDTHS
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dispatch(mesh_device, width):
    resources = _resources(mesh_device, width)
    S, per_chip, model = _inputs(mesh_device, resources, width, seed=1234)
    xbuf, route = _to_device(per_chip, mesh_device)
    out = mb.sparse_dispatch_2d(xbuf, route, resources)
    ttnn.synchronize_device(mesh_device)
    _check(mesh_device, out, model)
    entries = mesh_device.num_program_cache_entries()

    # (new buffers at other addresses (the first ones stay alive), other contents: a cache hit patching every address)
    S, per_chip, model = _inputs(mesh_device, resources, width, seed=99)
    xbuf2, route2 = _to_device(per_chip, mesh_device)
    out = mb.sparse_dispatch_2d(xbuf2, route2, resources)
    ttnn.synchronize_device(mesh_device)
    _check(mesh_device, out, model)
    assert mesh_device.num_program_cache_entries() == entries, "the second call missed the program cache"
    resources.release()


def _ref_params():
    if not os.environ.get("MOE_BLOCK_REFS"):
        return [pytest.param(None, None, None, id="no-refs", marks=pytest.mark.skip(reason="MOE_BLOCK_REFS not set"))]
    cases = list_cases(OP)
    if not cases:
        return [pytest.param(None, None, None, id="no-cases", marks=pytest.mark.skip(reason=f"no {OP} cases"))]
    out = []
    for case in cases:
        meta = json.loads((case_dir(OP, case) / "meta.json").read_text())
        shape = tuple(meta["mesh_shape"])
        out.append(pytest.param(shape, _device_params(shape, meta["config"]["fabric_payload"]), case, id=case))
    return out


@pytest.mark.parametrize("mesh_device, device_params, case", _ref_params(), indirect=["mesh_device", "device_params"])
def test_reference_case(mesh_device, case):
    meta, ref = load_case(OP, case)
    cfg = meta["config"]
    R, C = tuple(mesh_device.shape)
    topo_wrap = (bool(cfg["torus"][0]) and C > 2, bool(cfg["torus"][1]) and R > 2)
    resources = _resources(mesh_device, cfg["width"])
    assert (
        resources.topology["wrap_x"],
        resources.topology["wrap_y"],
    ) == topo_wrap, "fabric wrap differs from the case"
    xbuf, route = _to_device(ref, mesh_device)
    out = mb.sparse_dispatch_2d(xbuf, route, resources, group=cfg["group"], manifest_batch=cfg["manifest_batch"])
    ttnn.synchronize_device(mesh_device)
    got = {ch: {"xbuf": t} for ch, t in from_mesh(out, (R, C)).items()}
    assert_bit_exact(got, ref, ["xbuf"])
    resources.release()
