# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host front-end vs the reference: routing exact, bias chain across forwards (flip halves, rollout steps,
unrestored calls), x_final / masks / s_features / loc-scale. CPU only (python_env torch)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))
import dataflow as DF  # noqa: E402
from models.experimental.kairos_forecast.tests.pcc.test_dataflow import CASES, GOLD, fronts  # noqa: E402
from models.experimental.kairos_forecast.tt.host_frontend import front_end  # noqa: E402
from models.experimental.kairos_forecast.tt.router_state import RouterState, routing_record  # noqa: E402

BENCH = Path("/home/andy/bringup/goldens/kairos/bench")
BENCH_CASES = ["paper", "long_horizon", "ragged", "single", "single_plain", "single_short", "single_long_horizon"]


@pytest.fixture(scope="module")
def W():
    return DF.load_weights()


@pytest.mark.parametrize("case", BENCH_CASES)
def test_bench_routing(W, case):
    rec = torch.load(BENCH / f"{case}.pt", weights_only=False)
    router = RouterState(rec["bias0"])
    worst = {"route_mismatch_rows": 0, "bias_max_abs_diff": 0.0}
    for k, ctx in enumerate(rec["step_context"]):
        front, r = front_end(ctx, router, W)
        ref_w, ref_i = rec["gate"][k]
        rows = ctx.shape[0]
        rr = routing_record(ref_i.view(rows, -1, 3), r["indices"].view(rows, -1, 3), rec["bias_after"][k],
                            r["bias_after"], rec["scores"][k])
        assert torch.allclose(r["weights"], ref_w, rtol=1e-6, atol=0), (case, k)
        assert torch.equal(front["loc_scale"][0], rec["loc_scale"][k][0]) or torch.allclose(
            front["loc_scale"][0], rec["loc_scale"][k][0], rtol=1e-6), (case, k)
        worst["route_mismatch_rows"] += rr["route_mismatch_rows"]
        worst["bias_max_abs_diff"] = max(worst["bias_max_abs_diff"], rr["bias_max_abs_diff"])
    print(case, len(rec["step_context"]), "forwards", worst, f"min_top3_margin {rr['min_top3_margin']:.2e}")
    assert worst["route_mismatch_rows"] == 0 and worst["bias_max_abs_diff"] <= 1e-6, worst


@pytest.mark.parametrize("case", CASES)
def test_test_goldens(W, case):
    """Front dicts from the host front-end vs the captured ones, on every forward (contexts: the golden inputs,
    the flip's -x, the rollout's appended medians are rebuilt from the captured outputs)."""
    rec = torch.load(GOLD / f"{case}.pt", weights_only=False)
    ref = fronts(rec)
    router = RouterState(rec["bias0"])
    x = rec["x"]
    kw = rec["kwargs"]
    steps = -(-rec["prediction_length"] // 128)
    ctxs = []
    for call in range(rec["calls"]):
        for sign in ((1, -1) if kw.get("force_flip_invariance") else (1,)):
            past = sign * x
            for s in range(steps):
                ctxs.append(past)
                if s + 1 < steps:  # append this step's reference median (the golden's own, from its head output)
                    f = ref[len(ctxs) - 1]
                    med = DF.head_to_quantiles(f["head"], *f["loc_scale"])[:, 4]
                    past = torch.cat([past, med], -1)
    assert len(ctxs) == len(ref)
    for k, (ctx, f) in enumerate(zip(ctxs, ref)):
        got, r = front_end(ctx, router, W)
        g = rec["gate"][k]
        assert torch.equal(r["indices"], g["indices"]), (case, k)
        assert (r["bias_after"] - g["bias_after"]).abs().max() <= 1e-6, (case, k)
        assert torch.equal(got["attn_mask"], f["attn_mask"]) and torch.equal(got["indices"], f["indices"])
        for key in ("x_final", "weights", "s_features"):
            a, b = got[key].nan_to_num(0.0), f[key].nan_to_num(0.0)
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-6), (case, k, key, float((a - b).abs().max()))
