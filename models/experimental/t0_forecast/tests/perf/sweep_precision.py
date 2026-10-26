# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Precision sweep: per-role HiFi4 and an fp32 residual stream, scored on the gate metrics, timed at the paper shape.

    ttq run --model t0-gamma --part 3 --kind bench -- python models/experimental/t0_forecast/tests/perf/sweep_precision.py \
        --out models/experimental/t0_forecast/doc/optimize/iter_00/precision_sweep.json [--variants base,ff,...]

For every variant (the accuracy policy plus changes) it records the quantile metrics (``tests/helpers.quantile_metrics``)
on ``single`` (1 x 2048 -> 64), ``single_long`` (1 x 8192 -> 1024) and the three bench goldens,
and the paper device time (median of 3 ``forward_device`` calls, each after a 0.3 s idle). ``--trace`` adds the
per-layer read-out drift of ``single_long`` (PCC of each layer's read-out patches against the reference).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROLES_ALL = ("embed", "qkv", "wv", "wo", "ff1", "ff2", "decoder")
VARIANTS = {
    "base": {},
    "qkv": {"roles": ("qkv",)},
    "wo": {"roles": ("wo", "wv")},
    "ff1": {"roles": ("ff1",)},
    "ff2": {"roles": ("ff2",)},
    "ff": {"roles": ("ff1", "ff2")},
    "attn": {"roles": ("qkv", "wo", "wv")},
    "all_hifi4": {"roles": ROLES_ALL},
    "fp32res": {"fp32_residual": True},
    "fp32res_ff": {"fp32_residual": True, "roles": ("ff1", "ff2")},
    "fp32res_all": {"fp32_residual": True, "roles": ROLES_ALL},
    # The performance precision (bfp8 weights, LoFi) and per-role exceptions to it (PLAN 9 v1.2 F4).
    "perf": {"base": "performance"},
    "perf_ff_bf16": {"base": "performance", "bf16_roles": ("ff1", "ff2")},
    "perf_attn_bf16": {"base": "performance", "bf16_roles": ("qkv", "wv", "wo")},
    # D4 item 3: bf16 wo + w_down (the residual epilogue folds), bf16 w1/w2, HiFi2 per role.
    "perf_wo_down_bf16": {"base": "performance", "bf16_roles": ("wo", "wv", "ff2")},
    "perf_hifi2_ff": {"base": "performance", "hifi2_roles": ("ff1", "ff2")},
    "perf_hifi2_attn": {"base": "performance", "hifi2_roles": ("qkv", "wv", "wo")},
    "perf_hifi2_all": {"base": "performance", "hifi2_roles": ROLES_ALL},
    "perf_wo_down_bf16_hifi2": {"base": "performance", "bf16_roles": ("wo", "wv", "ff2"), "hifi2_roles": ROLES_ALL},
}


def precision_for(name: str):
    from models.experimental.t0_forecast.tt.common import TtT0Precision

    spec = VARIANTS[name]
    base = TtT0Precision.named(spec.get("base", "accuracy"))
    roles = frozenset(base.hifi4_roles) | frozenset(spec.get("roles", ()))
    return base.replaced(
        name=f"{base.name}+{name}",
        hifi4_roles=roles,
        fp32_residual=spec.get("fp32_residual", False),
        bf16_roles=frozenset(spec.get("bf16_roles", ())),
        hifi2_roles=frozenset(spec.get("hifi2_roles", ())),
    )


def main():
    import torch
    import ttnn

    from models.experimental.t0_forecast.tests.helpers import download_rows, golden, quantile_metrics, weights
    from models.experimental.t0_forecast.tt.model import TtT0

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--trace", default="base,fp32res", help="variants whose single_long per-layer drift is traced")
    args = ap.parse_args()

    cases = {}
    for name, file, lo, n in (("single", "single.pt", 63, 2), ("single_long", "single_long.pt", 255, 32)):
        g = golden(file)
        cases[name] = (g["context"], g["horizon"], g["head"][:, lo : lo + n])
    for name, file in (
        ("long_context", "bench_long_context.pt"),
        ("paper", "bench_paper.pt"),
        ("multivariate", "bench_multivariate.pt"),
    ):
        g = golden(file)
        cases[name] = (g["context"], g["horizon"], g["head_readout"])
    timing_context = cases["paper"][0]

    device = ttnn.open_device(device_id=0)
    results = []
    try:
        for name in args.variants.split(","):
            tt = TtT0(device, weights(), precision_for(name))
            record = {"variant": name, "precision": tt.precision.name, "hifi4_roles": sorted(tt.precision.hifi4_roles)}
            record["bf16_roles"] = sorted(tt.precision.bf16_roles)
            record["fp32_residual"] = tt.precision.fp32_residual
            for case, (context, horizon, ref) in cases.items():
                got = tt.network(tt.prepare_inputs(context, horizon))
                record[case] = quantile_metrics(ref, got)
                kinds = (
                    golden(f"bench_{case}.pt").get("kinds")
                    if case in ("paper", "long_context", "multivariate")
                    else None
                )
                if kinds:
                    record[case]["pcc_row_min_kind"] = kinds[
                        record[case]["pcc_row_min_row"] // (8 if case == "multivariate" else 1)
                    ]
            prepared = tt.prepare_inputs(timing_context, 64)
            inputs = tt.upload_inputs(prepared)
            ttnn.deallocate(tt.forward_device(inputs))
            times = []
            for _ in range(3):
                time.sleep(0.3)
                start = time.perf_counter()
                out = tt.forward_device(inputs)
                ttnn.synchronize_device(device)
                times.append(time.perf_counter() - start)
                ttnn.deallocate(out)
            tt.deallocate_inputs(inputs)
            record["paper_device_s"] = statistics.median(times)
            if name in args.trace.split(","):
                g = golden("single_long.pt")
                prepared = tt.prepare_inputs(g["context"], 1024)
                inputs = tt.upload_inputs(prepared)
                x = tt.embed(inputs)
                drift = {}
                for i in range(len(tt.layers)):
                    x = tt.run_layer(i, x, inputs)
                    if i < tt.cut_index:
                        h = download_rows(x, 1, prepared.num_patches)[:, 255:287]
                    else:  # already cut to the 32 read-out patches
                        h = ttnn.to_torch(x).float().reshape(1, 32, -1)
                    ref = g[f"layer_{i}"][:, 255:287]
                    drift[i] = float(torch.corrcoef(torch.stack([h.flatten(), ref.flatten()]))[0, 1])
                ttnn.deallocate(x)
                tt.deallocate_inputs(inputs)
                record["single_long_layer_readout_pcc"] = drift
            results.append(record)
            m = lambda c, k: record[c][k]
            print(
                f"[sweep] {name:12s} paper {record['paper_device_s'] * 1e3:7.1f} ms | single norm {m('single', 'pcc_normalized'):.6f}"
                f" spread {m('single', 'pcc_spread'):.6f} | single_long norm {m('single_long', 'pcc_normalized'):.6f}"
                f" spread {m('single_long', 'pcc_spread'):.6f} | long32 norm {m('long_context', 'pcc_normalized'):.6f}"
                f" spread {m('long_context', 'pcc_spread'):.6f} row_min {m('long_context', 'pcc_row_min'):.6f}"
                f" | paper spread {m('paper', 'pcc_spread'):.6f} row_min {m('paper', 'pcc_row_min'):.6f}",
                flush=True,
            )
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            del tt
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
