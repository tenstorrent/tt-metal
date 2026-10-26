# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Per-row diagnostics of the device head on the paper bench golden, by series kind (D1 item 1). One chip:

    ttq run --model timesfm3 --part 0 --kind pcc -- python \
        models/experimental/timesfm3_forecast/tests/perf/row_diagnostics.py <out.json> [--precision accuracy]

For every series kind of the realistic mix (``make_goldens.REALISTIC_MIX``): the per-row PCC and spread ratio
distribution and the error std; the 12 worst active rows by spread ratio and by PCC with their golden band
size; and the gate metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.experimental.timesfm3_forecast.tests.golden import make_goldens as mg
from models.experimental.timesfm3_forecast.tests.golden_io import _pcc_rows, active_rows, head_metrics, load_golden


def main():
    import ttnn

    from models.experimental.timesfm3_forecast.tt.common import TtPrecision
    from models.experimental.timesfm3_forecast.tt.model import TtTimesFM3

    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--precision", default="accuracy")
    args = ap.parse_args()
    g = load_golden("bench_paper")
    device = ttnn.open_device(device_id=0)
    try:
        tt = TtTimesFM3.from_checkpoint(device, TtPrecision.by_name(args.precision))
        hi = tt.prepare_inputs(list(g["contexts"].numpy()), g["horizon"])
        inputs = tt.upload_inputs(hi)
        out = tt.forward_device(inputs)
        head = tt.head_to_torch(out, inputs)
        ttnn.deallocate(out)
        tt.deallocate_inputs(inputs)
    finally:
        ttnn.close_device(device)
    ref = g["head_readout"]
    e = ref.double().reshape(ref.shape[0], -1, 9)
    d = head.double().reshape(head.shape[0], -1, 9)
    es, ds = (e - e.mean(-1, keepdim=True)).flatten(1), (d - d.mean(-1, keepdim=True)).flatten(1)
    ratio = ds.norm(dim=-1) / es.norm(dim=-1)
    pccs = _pcc_rows(e.flatten(1), d.flatten(1))
    err = (d - e).flatten(1).std(-1)
    band = es.norm(dim=-1)
    active = active_rows(ref)
    kinds = np.random.default_rng(20).choice(
        list(mg.REALISTIC_MIX), size=ref.shape[0], p=list(mg.REALISTIC_MIX.values())
    )
    report = {
        "precision": args.precision,
        "policy": tt.precision.describe(),
        "metrics": head_metrics(ref, head),
        "kinds": {},
    }
    for k in mg.REALISTIC_MIX:
        sel = torch.from_numpy(kinds == k) & active
        q = lambda t, p: float(torch.quantile(t[sel], p))  # noqa: E731
        report["kinds"][k] = {
            "active_rows": int(sel.sum()),
            "pcc_p01": q(pccs, 0.01),
            "pcc_median": q(pccs, 0.5),
            "spread_ratio_p01": q(ratio, 0.01),
            "spread_ratio_median": q(ratio, 0.5),
            "spread_ratio_p99": q(ratio, 0.99),
            "err_std_median": q(err, 0.5),
            "band_norm_median": q(band, 0.5),
        }
    idx = torch.nonzero(active).flatten()

    def rows(order):
        return [
            {
                "row": int(r),
                "kind": str(kinds[r]),
                "spread_ratio": float(ratio[r]),
                "pcc": float(pccs[r]),
                "err_std": float(err[r]),
                "band_norm": float(band[r]),
            }
            for r in order[:12].tolist()
        ]

    report["worst_by_spread_ratio"] = rows(idx[torch.argsort((ratio[idx] - 1).abs(), descending=True)])
    report["worst_by_pcc"] = rows(idx[torch.argsort(pccs[idx])])
    report["band_norm_median_active"] = float(band[active].median())
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("precision", "metrics")}, indent=1))
    for k, v in report["kinds"].items():
        print(f"[ROWS] {k}: " + ", ".join(f"{a} {b:.4f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))


if __name__ == "__main__":
    main()
