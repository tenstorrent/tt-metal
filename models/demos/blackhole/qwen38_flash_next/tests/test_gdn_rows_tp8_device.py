# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One GDN verify layer's rows body at 5 rows on the 1x8 mesh (TP=8), checkpoint layer 0: the fused body against the
composed chain on the same weights, inputs and states -- the output rows, the final state, the committed recurrent
state and the committed FIR history for accepted = 0, 2, 4 on every device.

  test_gdn_rows_wrap_is_bitwise_the_chain_at_tp8   gdn_rows_wrap (BITWISE class): every result torch.equal.
  test_gdn_rows_scan_is_within_component_tolerance gdn_rows_scan (COMPONENT class, QWEN38_MTP_ROWS_TP8=1): the output
      rows and the states within a relative L2 error of SCAN_REL_L2 of the chain's (the chain and the scan are two
      roundings of the same recurrence, 0.0048 state / 0.0098 output each against the oracle), the history bitwise.
      Prints max abs / max rel (to max |chain|) / rel L2 per result.

Run:  QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 pytest -s models/demos/blackhole/qwen38_flash_next/tests/test_gdn_rows_tp8_device.py
Env:  HF_MODEL, TT_CACHE_PATH (the component cache with the TP=8 GDN weights), the 1x8 ring mesh graph descriptor.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE, Qwen38TPHarness
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, Qwen38TTNNGDNWeights

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1" or MESH_SHAPE[1] != 8,
    reason="requires the held 1x8 mesh (QWEN38_FUSED_DEVICE_TEST=1, MESH_DEVICE=P150x8)",
)

ROWS = 5
TP = MESH_SHAPE[1]
ACCEPTED = (0, 2, 4)
SEEDS = (4101, 4102)
SCAN_REL_L2 = 0.03
NAMES = ("output rows", "final state", "committed state", "committed history")


@pytest.fixture
def tp_harness(mesh_device, tmp_path):
    return Qwen38TPHarness(mesh_device, tmp_path)


def _up(mesh, host, dtype, dim):
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, dim)),
    )


def _down(tensor):
    return [ttnn.to_torch(shard).to(torch.float32) for shard in ttnn.get_device_tensors(tensor)]


class _Variant:
    def __init__(self, harness, weights):
        self.harness, self.mesh = harness, harness.mesh_device
        self.module = Qwen38TTNNGDN(self.mesh, harness.contract, weights, collective_topology=harness.topology)
        self.constants = self.module.allocate_rows_constants(ROWS)
        self.rows = self.module.allocate_rows_state(self.constants)
        self.state = self.module.allocate_state()

    @property
    def wrap(self):
        return getattr(self.rows, "wrap_buffers", None) is not None

    @property
    def scan(self):
        return getattr(self.rows, "scan_buffers", None) is not None or hasattr(self.rows, "scan")

    def run(self, hidden, recurrent, history, accepted):
        for host, target, dtype, dim in (
            (recurrent, self.state.recurrent, ttnn.float32, 1),
            (history, self.rows.history, ttnn.bfloat16, 3),
        ):
            source = _up(self.mesh, host, dtype, dim)
            gdn_module._copy_inplace(source, target, label="test state")
            ttnn.deallocate(source)
        tile = self.harness.upload_sharded(hidden)
        result = self.module.forward_rows(tile, self.state, self.rows, full_tile=True)
        final = None if result.final_state is None else _down(result.final_state)
        out = (_down(result.hidden_rows), final)
        ttnn.deallocate(tile)
        ttnn.deallocate(result.hidden_rows)
        if result.final_state is not None:
            ttnn.deallocate(result.final_state)
        count = ttnn.from_torch(
            torch.full((1, 1, 1, 1), float(accepted)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        selectors = gdn_module.build_rows_selectors(count, self.constants)
        self.module.commit_rows(self.state, self.rows, selectors)
        selectors.deallocate()
        ttnn.deallocate(count)
        return out + (_down(self.state.recurrent), _down(self.rows.history))


def _inputs(shapes, seed):
    generator = torch.Generator().manual_seed(seed)
    rshape, hshape = shapes
    hidden = torch.zeros(1, 1, 32, HIDDEN_SIZE)
    hidden[:, :, :ROWS] = torch.randn(1, 1, ROWS, HIDDEN_SIZE, generator=generator)
    recurrent = torch.randn(rshape[0], rshape[1] * TP, rshape[2], rshape[3], generator=generator) * 0.05
    history = torch.zeros(hshape[0], hshape[1], hshape[2], hshape[3] * TP)
    history[:, :, :3] = torch.randn(hshape[0], hshape[1], 3, hshape[3] * TP, generator=generator) * 0.5
    return hidden.to(torch.bfloat16), recurrent, history.to(torch.bfloat16)


def _setup(harness, monkeypatch, fused_env):
    weights = Qwen38TTNNGDNWeights.from_checkpoint(
        harness.checkpoint,
        harness.mesh_device,
        harness.contract,
        harness.component_cache_root,
        layer_index=0,
        tt_metal_sha=harness.tt_metal_sha,
    )
    for name in ("QWEN38_FUSED_OFF", "QWEN38_MTP_ROWS_TP8"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QWEN38_FUSED_OFF", "gdn_rows_wrap")  # the composed chain
    chain = _Variant(harness, weights)
    monkeypatch.delenv("QWEN38_FUSED_OFF", raising=False)
    for name, value in fused_env.items():
        monkeypatch.setenv(name, value)
    fused = _Variant(harness, weights)
    shapes = (
        tuple(int(v) for v in chain.state.recurrent.shape),
        tuple(int(v) for v in chain.rows.history.shape),
    )
    return chain, fused, shapes


def _pairs(a, b):
    for name, xa, xb in zip(NAMES, a, b):
        if xa is None or xb is None:
            continue
        for device, (p, q) in enumerate(zip(xa, xb)):
            yield name, device, p, q


@pytest.mark.timeout(3000)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_gdn_rows_wrap_is_bitwise_the_chain_at_tp8(tp_harness, monkeypatch):
    chain, wrap, shapes = _setup(tp_harness, monkeypatch, {})
    assert wrap.wrap and not chain.wrap and not wrap.scan
    bad = []
    for seed in SEEDS:
        hidden, recurrent, history = _inputs(shapes, seed)
        for accepted in ACCEPTED:
            a = chain.run(hidden, recurrent, history, accepted)
            b = wrap.run(hidden, recurrent, history, accepted)
            assert float(max(t[:, :, :ROWS].abs().max() for t in a[0])) > 0.1  # not a vacuous zero comparison
            for name, device, p, q in _pairs(a, b):
                if not torch.equal(p, q):
                    d = (p - q).abs()
                    bad.append(f"seed {seed} accepted {accepted} {name} device {device}: max|d| {d.max():.3e}")
    assert not bad, "\n".join(bad[:20])


@pytest.mark.timeout(3000)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_gdn_rows_scan_is_within_component_tolerance_at_tp8(tp_harness, monkeypatch):
    chain, scan, shapes = _setup(tp_harness, monkeypatch, {"QWEN38_MTP_ROWS_TP8": "1"})
    assert scan.scan and not chain.scan and not scan.wrap and not chain.wrap
    worst = {}
    bad = []
    for seed in SEEDS:
        hidden, recurrent, history = _inputs(shapes, seed)
        for accepted in ACCEPTED:
            a = chain.run(hidden, recurrent, history, accepted)
            b = scan.run(hidden, recurrent, history, accepted)
            for name, device, p, q in _pairs(a, b):
                d = (p - q).abs()
                scale = float(p.abs().max()) or 1.0
                rel_l2 = float(torch.linalg.vector_norm(p - q) / (torch.linalg.vector_norm(p) or 1.0))
                row = worst.setdefault(name, [0.0, 0.0, 0.0])
                row[0], row[1], row[2] = (
                    max(row[0], float(d.max())),
                    max(row[1], float(d.max()) / scale),
                    max(row[2], rel_l2),
                )
                if name == "committed history":
                    if not torch.equal(p, q):
                        bad.append(f"history differs: seed {seed} accepted {accepted} device {device}")
                elif rel_l2 > SCAN_REL_L2 or not torch.isfinite(q).all():
                    bad.append(f"{name}: seed {seed} accepted {accepted} device {device}: rel L2 {rel_l2:.4f}")
    for name, (max_abs, max_rel, rel_l2) in worst.items():
        print(f"SCAN_VS_CHAIN {name}: max abs {max_abs:.4e}, max rel {max_rel:.4e}, rel L2 {rel_l2:.4e}", flush=True)
    assert not bad, "\n".join(bad[:20])
