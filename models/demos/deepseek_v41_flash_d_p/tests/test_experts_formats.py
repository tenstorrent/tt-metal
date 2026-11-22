# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.27: every EXPERTS_IMPL value (bfp4, bfp8, bfp4_mse, bfp8_down4) on the shipped MoE path (MOE_IMPL=block on the
Galaxy torus) - the all-device model with layer 0 only, chunk 0 of the s10240 golden (start 0), its layer-0 output vs
the golden's (rel = ||d|| / ||golden||, testing/ladder.rel; PCC). Not a gate: evidence that each format builds and runs
through the block (bfp8 and bfp8_down4 were refused there before P.27) and how far each lands, one model per value (a
fresh mesh each, so the shared MoE block is rebuilt in the value's weight format). The weight caches of layer 0 are
made on a miss (~2 min a value).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_experts_formats.py -s
"""

from __future__ import annotations

import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import begin_chunk, spec
from models.demos.common.bringup.testing.ladder import rel

S = spec()
IMPLS = ["bfp4", "bfp8", "bfp4_mse", "bfp8_down4"]
_OUT: dict = {}


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
def test_layer0(mesh_device, impl, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_adapter

    monkeypatch.setenv("DS41_EXPERTS_IMPL", impl)
    moe_block_adapter._BLOCKS.clear()  # (a fresh mesh: the block is rebuilt with this value's op)
    moe_block_adapter._DONE.clear()
    g = Golden.for_rung(S, "s10240")
    chunk = g.chunk
    t0 = time.time()
    model = S.hooks().device_model(mesh_device, S, [0], lm_head=False)
    load = time.time() - t0
    state = model.new_state(g.seq)
    tokens = g.tokens()[:chunk]
    begin_chunk(model, tokens, 0)
    h = model.embed(tokens)
    out = model.layer(0, h, 0, state)
    model.free(h)
    model.sync()
    y = model.to_host(out).float()
    model.free(out)
    want = g.layer(0, 0)["out"].float().reshape(y.shape)
    r, p = float(rel(y, want)), metrics.pcc(y, want)
    _OUT[impl] = y
    stats = model.blocks[0].moe_block.bfp4_stats if model.blocks[0].moe_block is not None else None
    print(
        f"EXPERTS_FORMAT {impl}: layer 0 out vs golden rel {r:.5f} pcc {p:.6f} (load {load:.0f} s) bfp4_stats {stats}"
    )
    for k, v in _OUT.items():
        if k != impl:
            print(f"EXPERTS_FORMAT {impl} vs {k}: rel {float(rel(y, v)):.5f}")
    assert torch.isfinite(y).all(), impl
