# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the qkv prefetch section and the other users of its kernels (tt/proj_stream.py), Galaxy
row 0 (qkv_prefetch_common.py).

STEPS launches of the section, each with x poisoned (NaN) until the stub writes it, and STEPS calls of the standalone
qkv stream, the o stream and the router scores. Every output keeps its format (the section's [1, 1, 32, 1920] bf16
TILE interleaved in L1), is finite on the user rows 0-3, agrees across the row's chips, and matches the base
implementation's saved outputs (``--write-base``) per row with PCC >= 0.99999 (same fidelities and fp32
accumulation; only the accumulation order may differ). The section's rows must also match x @ w on the host
(PCC >= 0.9999). Exact matches against the base are reported.

Prints QKV_PREFETCH_CORRECTNESS_PASS after every check passed.
"""

import argparse
import os

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import qkv_prefetch_common as qp

BASE = os.environ.get("MIMO_QKV_PREFETCH_BASE", "/home/zni/models/mimo-golden/qkv_prefetch_base.pt")
STEPS = 4


def stream_config():
    return qp.compute_config(ttnn.MathFidelity.HiFi2)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-base", action="store_true")
    args = parser.parse_args()
    mesh, row = qp.open_row()
    got = {"section": [], "qkv": [], "o": [], "scores": [], "biased": []}
    plain = []  # the model's standalone qkv stream on the same x and weight
    ref = []
    ok_format = True
    try:
        plan, stream, w, w_stream = qp.build(row)
        w_host = ttnn.to_torch(ttnn.get_device_tensors(w)[0]).float().reshape(qp.HIDDEN, qp.QKV_WIDTH)
        qkv_stream = qp.BankStreamMatmul(plan, qp.HIDDEN, qp.QKV_WIDTH, stream_config(), skip_bank_core=True)
        x = qp.x_tensor(row, qp.x_host(0))
        for step in range(STEPS):
            xh = qp.x_host(step)
            x_src = qp.x_tensor(row, xh)
            qp.poison(row, x)
            out = qp.launch(row, stream, w_stream, x, x_src)
            mc = out.memory_config()
            fmt = (
                tuple(out.shape) == (1, 1, 32, qp.QKV_WIDTH)
                and out.dtype == ttnn.bfloat16
                and out.layout == ttnn.TILE_LAYOUT
                and mc.buffer_type == ttnn.BufferType.L1
                and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            )
            if not fmt:
                print(f"section: wrong output format {tuple(out.shape)} {out.dtype} {out.layout} {mc}", flush=True)
            ok_format &= fmt
            got["section"].append(qp.rows(out))
            ref.append(xh.float()[0, 0, : qp.USERS] @ w_host)
            out.deallocate(True)
            out = qkv_stream(x_src, w_stream)
            plain.append(qp.rows(out))
            out.deallocate(True)
            x_src.deallocate(True)
        run = qp.standalone(row)
        for step in range(STEPS):
            outs, ins = run(step)
            for name, t in outs.items():
                got[name].append(qp.rows(t))
            for name in ("qkv", "o"):
                outs[name].deallocate(True)
            for t in ins:
                t.deallocate(True)
    finally:
        qp.close(mesh, row)
    got = {k: torch.cat(v) for k, v in got.items()}
    ref = torch.cat(ref)
    if args.write_base:
        torch.save(got, BASE)
        print(f"wrote base outputs {BASE}", flush=True)
    base = got if args.write_base else torch.load(BASE)
    ok = ok_format
    for name, rows_ in got.items():
        finite = bool(torch.isfinite(rows_).all())
        worst = min(pcc(a, b) for a, b in zip(rows_, base[name])) if finite else float("nan")
        exact = int((rows_ == base[name]).all(dim=-1).sum()) if finite else 0
        print(f"{name}: finite {finite}, worst row PCC vs base {worst:.7f}, rows equal to base {exact}/{len(rows_)}")
        ok &= finite and worst >= 0.99999
    worst_ref = min(pcc(a, b) for a, b in zip(got["section"], ref))
    print(f"section: worst row PCC vs host x @ w {worst_ref:.7f}", flush=True)
    ok &= worst_ref >= 0.9999
    plain = torch.cat(plain)
    worst_plain = min(pcc(a, b) for a, b in zip(got["section"], plain))
    exact = int((got["section"] == plain).all(dim=-1).sum())
    print(f"section: worst row PCC vs the standalone qkv stream {worst_plain:.7f}, rows equal {exact}/{len(plain)}")
    ok &= worst_plain >= 0.99999
    print(f"formats ok {ok_format}", flush=True)
    if ok:
        print("QKV_PREFETCH_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("qkv prefetch correctness check failed")


if __name__ == "__main__":
    main()
