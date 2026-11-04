# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""flat_gather (models/demos/moe_block/tt/flat_gather.py) against the C++ flat op's own weight prep
(flat_routed_expert_weights.flat_weight_regions + bank_sharded), GLM-5.3's flat layout (8 experts a chip):

* test_flat_gather_layout: host only once the plan is made - the face map applied to float matrices equals the weight
  prep's packed pages, value for value.
* test_flat_gather_device: the device gather from bfp4 sources equals bank_sharded of the same weights, bit for bit.

The plan needs the 8x4 Galaxy (or the mock Galaxy for the layout test); both skip elsewhere."""

import os

import pytest
import torch

import ttnn
from models.demos.moe_block.tt.flat_gather import (
    _pages,
    flat_face_map,
    flat_weights_module,
    gather,
    gather_batches,
)

R, C = 8, 4
H, I, E, NG, CHUNK = 6144, 2048, 8, 256, 5120
MOCK = "blackhole_galaxy" in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")


def emulate_gather(srcs: list, face_map: torch.Tensor) -> torch.Tensor:
    """gather on the host: float source matrices (s order) -> the dest pages [pages, 32, 32]."""

    def faces_of(s):  # (each source's own [rows, cols]: Wd is [I, H]; every source has NT tiles)
        R, C = s.shape
        return s.reshape(R // 32, 2, 16, C // 32, 2, 16).permute(0, 3, 1, 4, 2, 5).reshape(-1, 256)

    faces = torch.cat([faces_of(s) for s in srcs])
    fm = face_map.to(torch.int64)
    got = torch.where((fm >= 0).unsqueeze(-1), faces[fm.clamp(min=0)], torch.zeros(()))  # [pages, 4, 256]
    P = fm.shape[0]
    return got.reshape(P, 2, 2, 16, 16).permute(0, 1, 3, 2, 4).reshape(P, 32, 32)


@pytest.fixture(scope="module")
def mesh():
    if not MOCK and ttnn.get_num_devices() < R * C:
        pytest.skip(f"needs a {R}x{C} Galaxy or the mock Galaxy")
    m = ttnn.open_mesh_device(ttnn.MeshShape(R, C))
    yield m
    ttnn.close_mesh_device(m)


def _lay(mesh):
    mb = ttnn.experimental.moe_block
    layout = mb.FlatLayoutOptions()
    return mb.flat_routed_expert_plan(mesh, H, I, E, NG, CHUNK, weights_bf8=False, pin=1, layout=layout).as_dict()


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return [(r(H, I), r(H, I), r(I, H)) for _ in range(E)]


def test_flat_gather_layout(mesh):
    lay = _lay(mesh)
    fm = flat_face_map(H, I, E, lay)
    W = _weights()
    regions, dregs = flat_weights_module().flat_weight_regions(W, lay)
    srcs = [m_ for w in W for m_ in w]
    for name, regs in (("w_gu", regions), ("w_d", dregs)):
        want = _pages(regs, lay["banks"])
        got = emulate_gather(srcs, fm[name])
        assert torch.equal(got, want), f"{name}: {int((got != want).sum())} values differ"
    # (in batches of 3 experts - the K3 loader's way, sources renumbered from the batch's first: the same pages)
    NT = (H // 32) * (I // 32)
    for name, regs in (("w_gu", regions), ("w_d", dregs)):
        want = _pages(regs, lay["banks"])
        got = torch.full_like(want, float("nan"))
        for e0, e1, sub in gather_batches(fm, NT, E, 3):
            rows, m = sub[name]
            got[rows.long()] = emulate_gather(srcs[3 * e0 : 3 * e1], m)
        assert torch.equal(got, want), f"{name} in batches: {int((got != want).sum())} values differ"


@pytest.mark.parametrize("per", [None, 3], ids=["whole", "batches-of-3"])
@pytest.mark.parametrize("rounds", [2])
def test_flat_gather_device(mesh, rounds, per):
    """Two rounds on fresh tensors (a model gathers every layer: the second hits the program cache); per: gather in
    batches of that many experts (gather_batches, the sources of a batch only), else all at once."""
    fw = flat_weights_module()
    lay = _lay(mesh)
    fm = flat_face_map(H, I, E, lay)
    kept = []  # (the rounds' sources stay allocated: each round's at other addresses)
    for seed in range(1, rounds + 1):
        kept += _device_round(mesh, fw, lay, fm, seed, per)
    for t in kept:
        ttnn.deallocate(t)


def _device_round(mesh, fw, lay, fm, seed, per=None):
    W = [tuple(m_.to(torch.bfloat16) for m_ in w) for w in _weights(seed)]
    dt = ttnn.bfloat4_b
    rep = ttnn.ReplicateTensorToMesh(mesh)
    srcs = [
        ttnn.from_torch(m_.reshape(1, 1, *m_.shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        for w in W
        for m_ in w
    ]
    regions, dregs = fw.flat_weight_regions(W, lay)
    banks = lay["banks"]
    for name, regs in (("w_gu", regions), ("w_d", dregs)):
        want = fw.bank_sharded([regs], banks, dt, mesh, replicate=True)
        dst = fw.uninitialized_bank_sharded(len(regs), regs[0].shape[0], banks, dt, mesh)
        if per is None:
            gather(mesh, srcs, fm[name], dst)
        else:
            for e0, e1, sub in gather_batches({name: fm[name]}, (H // 32) * (I // 32), E, per):
                rows, m = sub[name]
                gather(mesh, srcs[3 * e0 : 3 * e1], m, dst, pages=rows)
        a = ttnn.to_torch(ttnn.get_device_tensors(dst)[0])
        b = ttnn.to_torch(ttnn.get_device_tensors(want)[0])
        assert torch.equal(a, b), f"{name}: {int((a != b).sum())} values differ from the weight prep"
        ttnn.deallocate(dst)
        ttnn.deallocate(want)
    return srcs
