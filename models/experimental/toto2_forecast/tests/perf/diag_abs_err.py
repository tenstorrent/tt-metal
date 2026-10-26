# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Where a case's abs_err_max sits (D4 condition 3): row, quantile, step, and that row's golden scale.

    python models/experimental/toto2_forecast/tests/perf/diag_abs_err.py <ckpt> <case> <precision> [out.json]
"""

from __future__ import annotations

import json
import sys

import ttnn

from models.experimental.toto2_forecast.tests.golden.goldens import _rows, active_rows, degenerate_rows, load_golden
from models.experimental.toto2_forecast.tests.pcc.test_model import run_forecast
from models.experimental.toto2_forecast.tt.common import Toto2Precision
from models.experimental.toto2_forecast.tt.model import TtToto2


def main(ckpt: str, case: str, precision: str, out: str | None = None) -> None:
    device = ttnn.open_device(device_id=0)
    try:
        golden = load_golden(ckpt, case)
        model = TtToto2.from_checkpoint(device, ckpt, Toto2Precision.named(precision))
        _, head = run_forecast(model, golden)
        del model
    finally:
        ttnn.close_device(device)
    ref = golden["captures"]["head_out"]
    err = (head.double() - ref.double()).abs()
    q, b, v, h, t = (int(i) for i in (err == err.max()).nonzero()[0])
    rows_ref, rows_err = _rows(ref), _rows(err)
    row = b * ref.shape[2] + v
    mask = golden["inputs"]["target_mask"]
    rec = {
        "ckpt": ckpt,
        "case": case,
        "precision": precision,
        "abs_err_max": err.max().item(),
        "quantile_index": q,
        "series": b,
        "variate": v,
        "patch": h,
        "step": t,
        "ref_value": ref[q, b, v, h, t].item(),
        "dev_value": head[q, b, v, h, t].item(),
        "row_ref_std": rows_ref[row].std().item(),
        "row_ref_absmean": rows_ref[row].abs().mean().item(),
        "row_ref_band": (ref - ref.mean(0))[:, b, v].abs().mean().item(),
        "row_active": bool(active_rows(ref)[row]),
        "row_degenerate": bool(degenerate_rows(ref)[row]),
        "case_ref_std": ref.double().std().item(),
        "row_err_p99": rows_err[row].quantile(0.99).item(),
        "row_mask_observed_frac": mask.reshape(-1, mask.shape[-1])[row].float().mean().item()
        if mask is not None
        else None,
        "other_rows_err_max": rows_err[[i for i in range(rows_err.shape[0]) if i != row]].max().item()
        if rows_err.shape[0] > 1
        else None,
    }
    got_rows = _rows(head)
    a, e = got_rows[row] - got_rows[row].mean(), rows_ref[row] - rows_ref[row].mean()
    rec["row_pcc"] = (a @ e / (a.norm() * e.norm())).item()
    band = lambda t: (t - t.mean(0))[:, b, v].double().norm().item()
    rec["row_spread_ratio"] = band(head) / band(ref)
    rec["row_err_by_quantile"] = [err[i, b, v].max().item() for i in range(err.shape[0])]
    print(json.dumps(rec, indent=2))
    if out:
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:])
