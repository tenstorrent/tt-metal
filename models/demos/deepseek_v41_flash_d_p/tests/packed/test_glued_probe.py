# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7 diagnostic probe (not gated): (1) tt/packed_device.GluedMover's row moves vs a host roll on random rows;
(2) a glued segment at model row 0 but segment index 1 ([Q, O]: Q computes nothing) vs O alone, layers
DS41_PACKED_LAYERS (the segment-index machinery without a move).

    BRINGUP_CHUNK=8192 DS41_PACKED=on DS41_PACKED_GLUED=on DS41_PACKED_LAYERS=0 scripts/run_safe_pytest.sh --run-all \\
        --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_probe.py
"""

from __future__ import annotations

import importlib

import pytest
import torch

import ttnn
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import test_glued_oracle as O

O = importlib.import_module(O.__name__)
S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("DS41_PACKED_LAYERS",)


def _mover(mesh):
    from models.demos.deepseek_v41_flash_d_p.tt.common import DRAM
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import GluedMover

    sp, tp = tuple(mesh.shape)
    n = sp * tp
    chunk, w = 8192, 256
    q = chunk // n
    host = torch.randn(chunk, w).bfloat16()
    per = host.reshape(sp, tp, q, w)
    t = ttnn.from_torch(
        per,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(sp, tp)),
    )
    mv = GluedMover(mesh, chunk)
    g = mv.gather(t)
    worst = {}
    for off in (32, 256, 1600, 2368, 4096):
        x = mv.to_row0(g, off)
        dev = ttnn.get_device_tensors(x)
        got = torch.cat([ttnn.to_torch(dev[k]).reshape(q, w) for k in range(n)])
        want = torch.roll(host, -off, dims=0)
        worst[off] = float((got.float() - want.float()).abs().max())
        ttnn.deallocate(x)
    # part + assemble: rows [0, 600) of a rolled copy back at row 32 + 0..
    ttnn.deallocate(g)
    ttnn.deallocate(t)
    print(f"mover: max abs diff per offset {worst}")
    return worst


@mesh_parametrize
def test_glued_probe(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    worst = _mover(mesh_device)
    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    if model.skip_mode != "off":
        model.use_decoder_skip("off")
    layers = O._layers(model)
    run = O.Run(Golden.for_rung(S, "s56320").tokens())
    full = {}
    orig_add = O.Capture.add

    def add(self, key, rows):  # (the probe keeps O's whole rows of every layer)
        full.setdefault(id(self), {})[key] = rows.clone()
        orig_add(self, key, rows)

    O.Capture.add = add
    a = O._run(model, layers, run, ["O"], "on")
    b = O._run(model, layers, run, ["Q", "O"], "on")
    c = O._run(model, layers, run, ["I", "O"], "on")
    keys = [("O", i) for i in layers]
    d_idx = O.Capture.diff(b[0], a[0], keys)
    d_mv = O.Capture.diff(c[0], a[0], keys)
    O.Capture.add = orig_add
    from models.demos.deepseek_v41_flash_d_p.tt.layout import HC

    for i in layers:
        x, y = full[id(a[0])][("O", i)], full[id(c[0])][("O", i)]
        d = (x - y).abs()
        rows = torch.nonzero(d.amax(-1) > 0).reshape(-1)
        cols = torch.nonzero(d.amax(0) > 0).reshape(-1)
        w = x.shape[1] - HC
        print(
            f"L{i}: [I, O] vs alone: {rows.numel()} rows differ (first {rows[:12].tolist()}, last {rows[-4:].tolist()}), "
            f"{cols.numel()} cols (streams {int((cols < w).sum())}, pre {int((cols >= w).sum())}; first {cols[:8].tolist()}), max {float(d.max())}"
        )
    print(f"[Q, O] (index 1, row 0) vs O alone: {d_idx}, digests {O._dig_mismatch(b[1]['O'], a[1]['O'])}")
    print(f"[I, O] (index 1, row 128) vs O alone: {d_mv}, digests {O._dig_mismatch(c[1]['O'], a[1]['O'])}")
    assert max(worst.values()) == 0
