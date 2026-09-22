#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: ttnn.experimental.kda.qkv_causal_conv1d_silu vs the current fused ttnn.conv1d
prefill path (models/demos/blackhole/qwen36/tt/gdn/tp.py:367-471, _conv1d_prefill), for the
Qwen3.5-2B tp=1 shape: T=1024, Q=K=V=2048 (C=6144), kernel=4, bf16.

Sweeps channel_chunk_size (sweep A, op-default compute config), then compute_kernel_config
fidelity/fp32acc/packer_l1_acc for the best 3 chunk sizes (sweep B), then die-0 realism
(zero history, larger-magnitude activations) for the overall best config (sweep C), plus one
reference timing of the fused ttnn.conv1d path it would replace.

No correctness/perf gate: this is a data-collection probe. Results: CSV + MD in the results dir.
"""
import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn

RESULTS_DIR = Path(
    "/tmp/claude-1000/-home-ttuser-atupe-tt-metal/3e58daa7-01b3-474e-bdd9-1ad3dc7dcd5d/scratchpad/kda_probe"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "kda_conv_probe_results.csv"
MD_PATH = RESULTS_DIR / "kda_conv_probe_results.md"

T = 1024
QW, KW, VW = 2048, 2048, 2048
C = QW + KW + VW  # 6144
K = 4  # conv kernel size / history rows

N_TRACE_CALLS = 10
N_TIMED_EXEC = 5

CHUNKS_A = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 6144]
FIDELITIES_B = [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2]

FIELDS = [
    "sweep",
    "chunk",
    "fidelity",
    "fp32acc",
    "packer_l1",
    "us",
    "pcc_q",
    "pcc_k",
    "pcc_v",
    "maxerr",
    "frac_gt_0.05",
    "status",
    "err",
]


# ---------------- reference / data setup ----------------


def pearson_pcc(dev_out: torch.Tensor, ref_out: torch.Tensor) -> float:
    a = dev_out.double().flatten()
    b = ref_out.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def make_host_fixture(*, history_zero: bool = False, x_scale: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    x = (torch.randn(1, T, C) * x_scale).to(torch.bfloat16)
    if history_zero:
        history = torch.zeros(1, K - 1, C, dtype=torch.bfloat16)
    else:
        history = torch.randn(1, K - 1, C, dtype=torch.bfloat16)
    w = (torch.randn(C, 1, K) * 0.3).to(torch.bfloat16)
    return x, history, w


def golden(x: torch.Tensor, history: torch.Tensor, w: torch.Tensor):
    """fp32: out[t] = sum_j w[:,0,j] * xfull[t+j], xfull = cat(history, x); q,k,v = split(silu(out))."""
    xfull = torch.cat([history, x], dim=1).float()  # [1, (K-1)+T, C]
    wf = w.float()
    out = torch.zeros(1, T, C, dtype=torch.float32)
    for j in range(K):
        out = out + wf[:, 0, j].view(1, 1, C) * xfull[:, j : j + T, :]
    out = F.silu(out)
    return out.split([QW, KW, VW], dim=-1)


def to_device_fixture(mesh, x: torch.Tensor, history: torch.Tensor, w: torch.Tensor):
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    hist_tt = ttnn.from_torch(
        history,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    taps_tt = tuple(
        ttnn.from_torch(
            w[:, 0, j].reshape(1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for j in range(K)
    )
    return x_tt, hist_tt, taps_tt


# ---------------- row bookkeeping ----------------


def mk_row(sweep, chunk, fidelity, fp32acc, packer_l1, us, pcc_q, pcc_k, pcc_v, maxerr, frac_gt, status, err):
    def fmt_pcc(p):
        return f"{p:.5f}" if (p is not None and p == p) else ""

    return {
        "sweep": sweep,
        "chunk": chunk,
        "fidelity": fidelity,
        "fp32acc": fp32acc,
        "packer_l1": packer_l1,
        "us": f"{us:.1f}" if us is not None else "",
        "pcc_q": fmt_pcc(pcc_q),
        "pcc_k": fmt_pcc(pcc_k),
        "pcc_v": fmt_pcc(pcc_v),
        "maxerr": f"{maxerr:.5f}" if maxerr is not None else "",
        "frac_gt_0.05": f"{frac_gt:.6f}" if frac_gt is not None else "",
        "status": status,
        "err": err or "",
    }


def print_row(r):
    print(
        f"  {r['sweep']:<16} chunk={str(r['chunk']):<8} fid={r['fidelity']:<8} fp32={r['fp32acc']:<6} "
        f"pl1={r['packer_l1']:<6} us={r['us']:<9} pcc_q={r['pcc_q']:<9} pcc_k={r['pcc_k']:<9} "
        f"pcc_v={r['pcc_v']:<9} maxerr={r['maxerr']:<9} frac>0.05={r['frac_gt_0.05']:<9} "
        f"{r['status']} {r['err']}",
        flush=True,
    )


def write_csv_md(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    def sort_key(r):
        try:
            return (0, float(r["us"]))
        except (ValueError, TypeError):
            return (1, 0.0)

    sorted_rows = sorted(rows, key=sort_key)
    lines = [
        "# KDA qkv_causal_conv1d_silu probe (Qwen3.5-2B tp=1: T=1024, C=6144, K=4, bf16)",
        "",
        "Sorted by us ascending; ERR/blank-us rows sink to the bottom.",
        "",
    ]
    lines.append("| " + " | ".join(FIELDS) + " |")
    lines.append("|" + "---|" * len(FIELDS))
    for r in sorted_rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in FIELDS) + " |")
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {CSV_PATH} ({len(rows)} rows)")
    print(f"Wrote {MD_PATH}")


# ---------------- KDA op runner ----------------


def run_kda_config(mesh, sweep, chunk, x_tt, hist_tt, taps_tt, golden_qkv, *, ckc=None, ckc_label=("default", "", "")):
    fidelity_label, fp32acc_label, packer_label = ckc_label
    golden_q, golden_k, golden_v = golden_qkv

    def call():
        return ttnn.experimental.kda.qkv_causal_conv1d_silu(
            x_tt,
            hist_tt,
            *taps_tt,
            QW,
            KW,
            VW,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=chunk),
            compute_kernel_config=ckc,
        )

    try:
        q, k, v = call()
        ttnn.synchronize_device(mesh)
        qf, kf, vf = ttnn.to_torch(q).float(), ttnn.to_torch(k).float(), ttnn.to_torch(v).float()
        pcc_q = pearson_pcc(qf, golden_q)
        pcc_k = pearson_pcc(kf, golden_k)
        pcc_v = pearson_pcc(vf, golden_v)
        err_q, err_k, err_v = (qf - golden_q).abs(), (kf - golden_k).abs(), (vf - golden_v).abs()
        maxerr = max(err_q.max().item(), err_k.max().item(), err_v.max().item())
        total = err_q.numel() + err_k.numel() + err_v.numel()
        n_bad = (err_q > 0.05).sum().item() + (err_k > 0.05).sum().item() + (err_v > 0.05).sum().item()
        frac_gt = n_bad / total
        q.deallocate(force=True)
        k.deallocate(force=True)
        v.deallocate(force=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}".splitlines()[0][:200]
        row = mk_row(
            sweep, chunk, fidelity_label, fp32acc_label, packer_label, None, None, None, None, None, None, "ERR", err
        )
        print_row(row)
        return row

    try:
        outs = []
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(N_TRACE_CALLS):
            outs.append(call())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)

        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)

        samples = []
        for _ in range(N_TIMED_EXEC):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e6)
        us = statistics.median(samples) / N_TRACE_CALLS

        ttnn.release_trace(mesh, tid)
        for triple in outs:
            for t in triple:
                t.deallocate(force=True)
        row = mk_row(
            sweep,
            chunk,
            fidelity_label,
            fp32acc_label,
            packer_label,
            us,
            pcc_q,
            pcc_k,
            pcc_v,
            maxerr,
            frac_gt,
            "OK",
            "",
        )
        print_row(row)
        return row
    except Exception as e:
        err = f"TRACE {type(e).__name__}: {e}".splitlines()[0][:200]
        row = mk_row(
            sweep,
            chunk,
            fidelity_label,
            fp32acc_label,
            packer_label,
            None,
            pcc_q,
            pcc_k,
            pcc_v,
            maxerr,
            frac_gt,
            "ERR",
            err,
        )
        print_row(row)
        return row


def ckc_from_row(row):
    if row["fidelity"] == "default":
        return None, ("default", "", "")
    fid_map = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2}
    fid = fid_map[row["fidelity"]]
    fp32acc = row["fp32acc"] == "True"
    packer = row["packer_l1"] == "True"
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32acc, packer_l1_acc=packer
    )
    return ckc, (row["fidelity"], row["fp32acc"], row["packer_l1"])


# ---------------- reference (fused ttnn.conv1d) path ----------------


def run_reference(mesh, x_tt, hist_tt, w: torch.Tensor, golden_qkv):
    """Time the current fused ttnn.conv1d prefill path this KDA op would replace
    (mirrors gdn/tp.py:367-471 _conv1d_prefill, n_cc=2 chunks of 3072)."""
    cw = 3072
    n_cc = 2
    Lin = (K - 1) + T
    golden_q, golden_k, golden_v = golden_qkv

    try:
        cc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        conv_cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

        wprep_chunks = []
        for i in range(n_cc):
            w_chunk = w[i * cw : (i + 1) * cw].contiguous()
            w_tt = ttnn.from_torch(w_chunk, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            wprep = ttnn.prepare_conv_weights(
                weight_tensor=w_tt,
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=cw,
                out_channels=cw,
                batch_size=1,
                input_height=1,
                input_width=Lin,
                kernel_size=(1, K),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=False,
                groups=cw,
                device=mesh,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
            )
            wprep_chunks.append(wprep)

        def call():
            xin = ttnn.concat([hist_tt, x_tt], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            xin = ttnn.reshape(xin, (1, Lin, 1, C))
            outs = []
            for i, wprep in enumerate(wprep_chunks):
                xin_i = xin if n_cc == 1 else ttnn.slice(xin, (0, 0, 0, i * cw), (1, Lin, 1, (i + 1) * cw))
                out_i = ttnn.conv1d(
                    input_tensor=xin_i,
                    weight_tensor=wprep,
                    device=mesh,
                    in_channels=cw,
                    out_channels=cw,
                    batch_size=1,
                    input_length=Lin,
                    kernel_size=K,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=cw,
                    dtype=ttnn.bfloat16,
                    conv_config=conv_cfg,
                    compute_config=cc,
                    slice_config=ttnn.Conv2dL1FullSliceConfig,
                    return_output_dim=False,
                    return_weights_and_bias=False,
                )
                if n_cc > 1:
                    ttnn.deallocate(xin_i)
                out_i = ttnn.reshape(ttnn.sharded_to_interleaved(out_i, ttnn.DRAM_MEMORY_CONFIG), (1, T, cw))
                outs.append(ttnn.silu(out_i, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(xin)
            chunk0, chunk1 = outs
            q = ttnn.slice(chunk0, (0, 0, 0), (1, T, QW))
            k = ttnn.concat(
                [
                    ttnn.slice(chunk0, (0, 0, QW), (1, T, cw)),
                    ttnn.slice(chunk1, (0, 0, 0), (1, T, cw - QW)),
                ],
                dim=2,
            )
            v = ttnn.slice(chunk1, (0, 0, cw - QW), (1, T, cw))
            return q, k, v

        q, k, v = call()
        ttnn.synchronize_device(mesh)
        qf, kf, vf = ttnn.to_torch(q).float(), ttnn.to_torch(k).float(), ttnn.to_torch(v).float()
        pcc_q, pcc_k, pcc_v = pearson_pcc(qf, golden_q), pearson_pcc(kf, golden_k), pearson_pcc(vf, golden_v)
        err_q, err_k, err_v = (qf - golden_q).abs(), (kf - golden_k).abs(), (vf - golden_v).abs()
        maxerr = max(err_q.max().item(), err_k.max().item(), err_v.max().item())
        total = err_q.numel() + err_k.numel() + err_v.numel()
        n_bad = (err_q > 0.05).sum().item() + (err_k > 0.05).sum().item() + (err_v > 0.05).sum().item()
        frac_gt = n_bad / total
        q.deallocate(force=True)
        k.deallocate(force=True)
        v.deallocate(force=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}".splitlines()[0][:200]
        row = mk_row("reference", "3072x2", "HiFi4", "True", "True", None, None, None, None, None, None, "ERR", err)
        print_row(row)
        return row

    try:
        outs = []
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(N_TRACE_CALLS):
            outs.append(call())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)

        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)

        samples = []
        for _ in range(N_TIMED_EXEC):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e6)
        us = statistics.median(samples) / N_TRACE_CALLS

        ttnn.release_trace(mesh, tid)
        for triple in outs:
            for t in triple:
                t.deallocate(force=True)
        row = mk_row("reference", "3072x2", "HiFi4", "True", "True", us, pcc_q, pcc_k, pcc_v, maxerr, frac_gt, "OK", "")
        print_row(row)
        return row
    except Exception as e:
        err = f"TRACE {type(e).__name__}: {e}".splitlines()[0][:200]
        row = mk_row(
            "reference", "3072x2", "HiFi4", "True", "True", None, pcc_q, pcc_k, pcc_v, maxerr, frac_gt, "ERR", err
        )
        print_row(row)
        return row


# ---------------- main ----------------


def main():
    print("Opening mesh device (1x1) ...", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    rows = []
    try:
        x, history, w = make_host_fixture()
        golden_qkv = golden(x, history, w)
        x_tt, hist_tt, taps_tt = to_device_fixture(mesh, x, history, w)

        print("\n-- Sweep A: channel_chunk_size, op-default compute config --", flush=True)
        for chunk in CHUNKS_A:
            row = run_kda_config(mesh, "A", chunk, x_tt, hist_tt, taps_tt, golden_qkv)
            rows.append(row)
            write_csv_md(rows)

        ok_a = [r for r in rows if r["sweep"] == "A" and r["status"] == "OK" and r["us"]]
        best3 = sorted(ok_a, key=lambda r: float(r["us"]))[:3]
        best3_chunks = [int(r["chunk"]) for r in best3]
        print(f"\n-- Sweep A best 3 chunks by us: {best3_chunks} --", flush=True)

        print("\n-- Sweep B: compute_kernel_config sweep for best 3 chunks --", flush=True)
        for chunk in best3_chunks:
            for fidelity in FIDELITIES_B:
                fid_name = "HiFi4" if fidelity == ttnn.MathFidelity.HiFi4 else "HiFi2"
                for fp32acc in (True, False):
                    for packer in (True, False):
                        ckc = ttnn.WormholeComputeKernelConfig(
                            math_fidelity=fidelity,
                            math_approx_mode=False,
                            fp32_dest_acc_en=fp32acc,
                            packer_l1_acc=packer,
                        )
                        row = run_kda_config(
                            mesh,
                            "B",
                            chunk,
                            x_tt,
                            hist_tt,
                            taps_tt,
                            golden_qkv,
                            ckc=ckc,
                            ckc_label=(fid_name, str(fp32acc), str(packer)),
                        )
                        rows.append(row)
                        write_csv_md(rows)

        ok_ab = [r for r in rows if r["sweep"] in ("A", "B") and r["status"] == "OK" and r["us"]]
        if not ok_ab:
            print("\n!! No OK rows in sweep A/B; skipping sweep C (no best config to reuse).", flush=True)
        else:
            best_row = min(ok_ab, key=lambda r: float(r["us"]))
            best_chunk = int(best_row["chunk"])
            best_ckc, best_label = ckc_from_row(best_row)
            print(f"\n-- Overall best config: chunk={best_chunk} label={best_label} us={best_row['us']} --", flush=True)

            print("\n-- Sweep C: die-0 realism (zero history / x*3.0) on best config --", flush=True)
            x0, hist0, w0 = make_host_fixture(history_zero=True)
            golden0 = golden(x0, hist0, w0)
            x0_tt, hist0_tt, taps0_tt = to_device_fixture(mesh, x0, hist0, w0)
            row = run_kda_config(
                mesh,
                "C_history_zero",
                best_chunk,
                x0_tt,
                hist0_tt,
                taps0_tt,
                golden0,
                ckc=best_ckc,
                ckc_label=best_label,
            )
            rows.append(row)
            write_csv_md(rows)

            x3, hist3, w3 = make_host_fixture(x_scale=3.0)
            golden3 = golden(x3, hist3, w3)
            x3_tt, hist3_tt, taps3_tt = to_device_fixture(mesh, x3, hist3, w3)
            row = run_kda_config(
                mesh, "C_x_scale3", best_chunk, x3_tt, hist3_tt, taps3_tt, golden3, ckc=best_ckc, ckc_label=best_label
            )
            rows.append(row)
            write_csv_md(rows)

        print("\n-- Reference: fused ttnn.conv1d prefill path (n_cc=2, chunks of 3072) --", flush=True)
        row = run_reference(mesh, x_tt, hist_tt, w, golden_qkv)
        rows.append(row)
        write_csv_md(rows)

        ok_rows = [r for r in rows if r["status"] == "OK" and r["us"]]
        top10 = sorted(ok_rows, key=lambda r: float(r["us"]))[:10]
        print("\n-- Top 10 by us --", flush=True)
        for r in top10:
            print_row(r)

        err_rows = [r for r in rows if r["status"] == "ERR"]
        print(f"\n-- ERR summary ({len(err_rows)} rows) --", flush=True)
        for r in err_rows:
            print_row(r)

    finally:
        write_csv_md(rows)
        ttnn.close_mesh_device(mesh)
        print("\nDevice closed. Done.", flush=True)


if __name__ == "__main__":
    main()
