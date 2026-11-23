# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.9 / OV-58 (diagnostic, not a gate unit): does MM_TUNED160 reach the gate ladder's path, and does it change the
ladder rung's numbers? The gate's test_ladder (BRINGUP_RUNG=last, DS41_MM_TUNED160=on) gave every err_* metric equal to
MB.8's (the code before the switch).

The ladder's own driver (testing/ladder.run_ladder) on the rung MM160_RUNG (default last) over the held accuracy model
(batch.device_model, the gate's shared load), twice: MM_TUNED160=off, then on. Per side: the tuned-config hits at call
time (tests/test_mm160_offon.Hits: a dense call whose key is a GALAXY160 key while the switch reads on, a gmm_knobs call
answered from GMM160) and the digest of every layer output the ladder reads back (model.to_host at the last chunk).

Metrics: mm160_ladder_hits_<off|on> (total) and _<site>; mm160_ladder_layers_differ (layers whose output bytes differ
off vs on), mm160_ladder_max_abs_diff (over those layers' outputs). Asserts only off hits == 0 and on hits > 0.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_RUNG=last
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_mm160_ladder_hits.py"
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("MM160_RUNG",)


@mesh_parametrize
def test_mm160_ladder_hits(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.common.bringup.testing import batch
    from models.demos.common.bringup.testing import ladder as L

    O = importlib.import_module("models.demos.deepseek_v41_flash_d_p.tests.test_mm160_offon")
    rung_name = os.environ.get("MM160_RUNG", "last")  # diagnostic
    rung = S.rung(rung_name)
    from models.demos.common.bringup.reference.golden import Golden

    g = Golden.for_rung(S, rung_name)
    layers = L.rung_layers(S, rung, g)
    model = batch.device_model(S, mesh_device, layers, lm_head=layers[-1] == S.num_layers - 1)
    res = {}
    for side in ("off", "on"):
        outs = []
        to_host = model.to_host

        def keep(h, _f=to_host, _o=outs):
            t = _f(h)
            _o.append(t.clone())
            return t

        model.to_host = keep
        try:
            with O.switch("MM_TUNED160", side), O.Hits() as hc:
                out = L.run_ladder(S, rung_name, mesh_device)
        finally:
            if "to_host" in vars(model):
                del model.to_host
        res[side] = (hc, outs, out)
        metrics.record(f"mm160_ladder_hits_{side}", hc.total)
        for site, n in hc.n.items():
            metrics.record(f"mm160_ladder_hits_{side}_{site}", n)
        print(f"MM160 ladder {rung_name} {side}: tuned-config hits {hc.n}, {len(outs)} outputs read", flush=True)
    a, b = res["off"][1], res["on"][1]
    assert len(a) == len(b), (len(a), len(b))
    differ, worst = [], 0.0
    for k, (p, q) in enumerate(zip(a, b)):
        if O.digest(p) != O.digest(q):
            differ.append(k)
            worst = max(worst, float(torch.nan_to_num((p.double() - q.double()).abs(), nan=float("inf")).max()))
    metrics.record("mm160_ladder_layers_differ", len(differ))
    metrics.record("mm160_ladder_max_abs_diff", worst)
    print(
        f"MM160 ladder {rung_name}: off vs on outputs differ at {len(differ)} of {len(a)} reads {differ[:12]} "
        f"(max abs {worst})",
        flush=True,
    )
    assert res["off"][0].total == 0 and res["on"][0].total > 0, (res["off"][0].n, res["on"][0].n)
