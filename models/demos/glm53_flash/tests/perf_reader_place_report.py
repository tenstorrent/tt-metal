# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (torch) report of a GLM53_STACK_PROBE_DUMP file (test_mk_backend_probe.py): per layer and chip, the layer
span, the owners' qkv-landed mark and every reader's weight-stream marks, all in us after the chip's own earliest mark
of the layer (each chip's clock is its own: compare durations, not absolute times).

    python models/demos/glm53_flash/tests/perf_reader_place_report.py dump.pt [dump_b.pt]

Rows (mk_layer.decode_probe's layout): owner BRISC k = row k (marks: mhc_pre stream x 3, qkv landed, ...), readers'
weight streamer = row 160 + 2 n (start, then one mark per phase it streams), activation mover = 161 + 2 n. The east /
west split of a bank: banks 0-3 / 4-7 (Blackhole: DRAM columns physical x 0 / x 9)."""

from __future__ import annotations

import statistics
import sys

import torch

MHZ = 1350.0


def _marks(row) -> list:
    return [int(v) for v in row.tolist() if int(v) > 0]


def chip_rows(p, assign, k_cores: int) -> dict:
    """one chip's one layer [256, 16] -> spans in us after its earliest mark"""
    nz = p[p > 0]
    if not nz.numel():
        return {}
    t0 = int(nz.min())
    us = lambda v: (int(v) - t0) / MHZ  # noqa: E731
    own = [_marks(p[k]) for k in range(k_cores)]
    own = [m for m in own if len(m) >= 4]
    end = max(int(nz.max()), 0)
    out = dict(span=us(end), qkv_landed=statistics.median(us(m[3]) for m in own) if own else None)
    for side, banks in (("west", range(0, 4)), ("east", range(4, 8))):
        w1 = [us(_marks(p[160 + 2 * n])[1]) for n, (_, b, i) in enumerate(assign)
              if b in banks and len(_marks(p[160 + 2 * n])) > 1]  # fmt: skip
        out[f"{side}_w1_max"] = max(w1) if w1 else None
        out[f"{side}_w1_med"] = statistics.median(w1) if w1 else None
    return out


def report(path: str) -> list:
    rows = []
    for st in torch.load(path):
        P = st["probe"]  # [chips, layers, 256, 16]
        for j, layer in enumerate(st["layers"]):
            per = [chip_rows(P[d, j], st["assign"], st["k_cores"]) for d in range(P.shape[0])]
            per = [r for r in per if r]
            if not per:
                continue
            spans = [r["span"] for r in per]
            q = [r["qkv_landed"] for r in per if r["qkv_landed"] is not None]
            line = dict(layer=layer, sparse=st["sparse"][j], chips=len(per), span_med=statistics.median(spans),
                        span_max=max(spans), span_argmax=spans.index(max(spans)),
                        qkv_med=statistics.median(q) if q else None, qkv_max=max(q) if q else None)  # fmt: skip
            for side in ("west", "east"):
                v = [r[f"{side}_w1_max"] for r in per if r[f"{side}_w1_max"] is not None]
                line[f"{side}_w1_max_med"] = statistics.median(v) if v else None
                line[f"{side}_w1_max_max"] = max(v) if v else None
            rows.append(line)
    return rows


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"== {path}")
        for r in report(path):
            print(" ".join(f"{k}={round(v, 1) if isinstance(v, float) else v}" for k, v in r.items()))
