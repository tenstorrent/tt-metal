#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated benchmark of ttnn.transformer.chunked_scaled_dot_product_attention exactly as
Qwen3.5-2B SP prefill calls it on die 3 (see attention/tp.py:765-823, sp_prefill.py:136-144,
350-358).

Fixed "die-3" shape: batch 1, 8 query heads, 2 KV heads, head_dim 256, S=1024 queries,
chunk_start_idx=3072, keys 0..4095 in a paged cache of 64 blocks x 64 tokens.

Run one phase per invocation (env per the task doc), in order:
  python sp_sdpa_sweep.py --phase baseline --out-dir DIR
  python sp_sdpa_sweep.py --phase sweep    --out-dir DIR
  python sp_sdpa_sweep.py --phase fidelity --out-dir DIR
  python sp_sdpa_sweep.py --phase bf8      --out-dir DIR
  python sp_sdpa_sweep.py --phase die0     --out-dir DIR

Each run APPENDS its rows to <out-dir>/sdpa_sweep_results.csv and regenerates
sdpa_sweep_results.md from the full accumulated CSV (all phases run so far), sorted by us.
fidelity/bf8/die0 phases read the "best 3" (qc, kc, grid, exp_approx) configs from the
name=="sweep" rows already in that CSV, so --phase sweep must run before them.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

import ttnn

# ---- fixed problem (die-3 SP-prefill SDPA shape) ----
NQH, NKV, HD = 8, 2, 256
S_Q = 1024
S_KV = 4096
BLOCK_SIZE = 64
NUM_BLOCKS = S_KV // BLOCK_SIZE  # 64
DIE3_CHUNK_START = 3072
DIE0_CHUNK_START = 0

N_TRACE_CALLS = 10
N_TIMED_EXEC = 5

FIELDS = [
    "name",
    "qc",
    "kc",
    "grid",
    "max_cores",
    "exp_approx",
    "fidelity",
    "fp32acc",
    "packer_l1_acc",
    "kv_dtype",
    "us",
    "pcc",
    "status",
    "err",
]

FIDELITY_NAMES = {
    ttnn.MathFidelity.LoFi: "LoFi",
    ttnn.MathFidelity.HiFi2: "HiFi2",
    ttnn.MathFidelity.HiFi4: "HiFi4",
}
FIDELITY_LIST = [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi4]


# ---------------- reference / data setup ----------------


def make_qkv_torch():
    torch.manual_seed(0)
    Q = torch.randn(1, NQH, S_Q, HD, dtype=torch.bfloat16)
    K = torch.randn(1, NKV, S_KV, HD, dtype=torch.bfloat16)
    V = torch.randn(1, NKV, S_KV, HD, dtype=torch.bfloat16)
    return Q, K, V


def torch_ref(Q, K, V, chunk_start_idx):
    """float32 softmax(Q@K^T*scale + causal_mask)@V; query p (absolute chunk_start_idx+p)
    attends keys 0..chunk_start_idx+p inclusive. GQA: each KV head repeats contiguously."""
    rep = NQH // NKV
    Qf = Q.float()
    Kf = K.float().repeat_interleave(rep, dim=1)
    Vf = V.float().repeat_interleave(rep, dim=1)
    scale = HD**-0.5
    scores = torch.matmul(Qf, Kf.transpose(-1, -2)) * scale
    s_kv = Kf.shape[2]
    q_pos = torch.arange(S_Q).view(1, 1, S_Q, 1) + chunk_start_idx
    k_pos = torch.arange(s_kv).view(1, 1, 1, s_kv)
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf)


def page_kv_torch(K, V):
    def to_paged(t):
        return t.reshape(1, NKV, NUM_BLOCKS, BLOCK_SIZE, HD).transpose(1, 2).reshape(NUM_BLOCKS, NKV, BLOCK_SIZE, HD)

    return to_paged(K), to_paged(V)


def pearson_pcc(dev_out, ref_out):
    a = dev_out.double().flatten()
    b = ref_out.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def upload_fixture(mesh, Q, K_paged, V_paged, dtype):
    q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.from_torch(
        K_paged, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v = ttnn.from_torch(
        V_paged, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return q, k, v


def upload_page_table(mesh, num_blocks=NUM_BLOCKS):
    rows = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    return ttnn.from_torch(
        rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


# ---------------- program / compute config builders ----------------


def build_progcfg(gx, gy, qc, kc, exp_approx, max_cores):
    """Returns (config, max_cores_method) where max_cores_method in {"ctor", "attr", "skip"}."""
    base_kwargs = dict(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        exp_approx_mode=exp_approx,
        q_chunk_size=qc,
        k_chunk_size=kc,
    )
    try:
        cfg = ttnn.SDPAProgramConfig(max_cores_per_head_batch=max_cores, **base_kwargs)
        return cfg, "ctor"
    except TypeError:
        pass
    cfg = ttnn.SDPAProgramConfig(**base_kwargs)
    try:
        cfg.max_cores_per_head_batch = max_cores
        return cfg, "attr"
    except Exception:
        return cfg, "skip"


def build_ckc(fidelity, fp32acc, packer_l1_acc, math_approx_mode=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32acc,
        packer_l1_acc=packer_l1_acc,
    )


# ---------------- timing ----------------


def time_and_pcc(mesh, q, k, v, pt, chunk_start_idx, progcfg, ckc, ref_out):
    """1 untraced warm call (correctness + PCC) -> 10-call trace capture -> 1 warm replay ->
    5 timed replays. Returns (us, pcc, err); err is None on success. us = median/10."""

    def call():
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q,
            input_tensor_k=k,
            input_tensor_v=v,
            page_table_tensor=pt,
            chunk_start_idx=chunk_start_idx,
            compute_kernel_config=ckc,
            program_config=progcfg,
        )

    try:
        out = call()
        ttnn.synchronize_device(mesh)
        pcc = pearson_pcc(ttnn.to_torch(out), ref_out)
        out.deallocate(force=True)
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}".splitlines()[0][:200]

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
        for o in outs:
            o.deallocate(force=True)
        return us, pcc, None
    except Exception as e:
        return None, pcc, f"TRACE {type(e).__name__}: {e}".splitlines()[0][:200]


# ---------------- rows ----------------


def mk_row(
    name, qc, kc, gx, gy, max_cores, exp_approx, fidelity_name, fp32acc, packer_l1_acc, kv_dtype, us, pcc, status, err
):
    return {
        "name": name,
        "qc": qc,
        "kc": kc,
        "grid": f"{gx}x{gy}",
        "max_cores": max_cores,
        "exp_approx": str(exp_approx),
        "fidelity": fidelity_name,
        "fp32acc": str(fp32acc),
        "packer_l1_acc": str(packer_l1_acc),
        "kv_dtype": kv_dtype,
        "us": f"{us:.1f}" if us is not None else "",
        "pcc": f"{pcc:.5f}" if (pcc is not None and pcc == pcc) else "",
        "status": status,
        "err": err or "",
    }


def print_row(r):
    print(
        f"  {r['name']:<14} qc={r['qc']:<4} kc={r['kc']:<4} grid={r['grid']:<8} mc={r['max_cores']:<4} "
        f"exp={r['exp_approx']:<5} fid={r['fidelity']:<6} fp32={r['fp32acc']:<5} pl1={r['packer_l1_acc']:<5} "
        f"kv={r['kv_dtype']:<5} us={r['us']:<9} pcc={r['pcc']:<9} {r['status']} {r['err']}",
        flush=True,
    )


def run_config(
    mesh,
    name,
    q,
    k,
    v,
    pt,
    chunk_start_idx,
    ref_out,
    *,
    qc,
    kc,
    gx,
    gy,
    exp_approx,
    fidelity,
    fp32acc,
    packer_l1_acc,
    kv_dtype,
    max_cores=16,
    math_approx_mode=True,
    rows=None,
):
    fidelity_name = FIDELITY_NAMES.get(fidelity, str(fidelity))
    note = ""
    try:
        progcfg, mc_method = build_progcfg(gx, gy, qc, kc, exp_approx, max_cores)
        if mc_method != "ctor":
            note = f"max_cores_per_head_batch:{mc_method}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}".splitlines()[0][:200]
        row = mk_row(
            name,
            qc,
            kc,
            gx,
            gy,
            max_cores,
            exp_approx,
            fidelity_name,
            fp32acc,
            packer_l1_acc,
            kv_dtype,
            None,
            None,
            "ERR",
            err,
        )
        print_row(row)
        if rows is not None:
            rows.append(row)
        return row

    ckc = build_ckc(fidelity, fp32acc, packer_l1_acc, math_approx_mode=math_approx_mode)
    us, pcc, err = time_and_pcc(mesh, q, k, v, pt, chunk_start_idx, progcfg, ckc, ref_out)
    status = "OK" if err is None else "ERR"
    err_field = err or note
    row = mk_row(
        name,
        qc,
        kc,
        gx,
        gy,
        max_cores,
        exp_approx,
        fidelity_name,
        fp32acc,
        packer_l1_acc,
        kv_dtype,
        us,
        pcc,
        status,
        err_field,
    )
    print_row(row)
    if rows is not None:
        rows.append(row)
    return row


def append_and_write(out_dir, new_rows):
    csv_path = out_dir / "sdpa_sweep_results.csv"
    md_path = out_dir / "sdpa_sweep_results.md"

    existing = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing = list(csv.DictReader(f))

    all_rows = existing + new_rows

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    def sort_key(r):
        try:
            return (0, float(r["us"]))
        except (ValueError, TypeError):
            return (1, 0.0)

    sorted_rows = sorted(all_rows, key=sort_key)

    lines = [
        "# SDPA sweep results (die-3 / die-0 SP-prefill chunked SDPA shape)",
        "",
        "Sorted by us ascending; ERR/blank-us rows sink to the bottom.",
        "",
    ]
    lines.append("| " + " | ".join(FIELDS) + " |")
    lines.append("|" + "---|" * len(FIELDS))
    for r in sorted_rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in FIELDS) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {csv_path} ({len(all_rows)} total rows)")
    print(f"Wrote {md_path}")
    return csv_path, md_path


def read_best_sweep_configs(out_dir, n=3):
    csv_path = out_dir / "sdpa_sweep_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run --phase sweep first")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    sweep_rows = [r for r in rows if r["name"] == "sweep" and r["status"] == "OK" and r["us"]]
    if not sweep_rows:
        raise SystemExit("no OK rows with name=='sweep' found in the CSV -- run --phase sweep first")
    sweep_rows.sort(key=lambda r: float(r["us"]))
    best = sweep_rows[:n]
    configs = []
    for r in best:
        gx, gy = map(int, r["grid"].split("x"))
        configs.append(
            dict(
                qc=int(r["qc"]),
                kc=int(r["kc"]),
                gx=gx,
                gy=gy,
                exp_approx=(r["exp_approx"] == "True"),
                max_cores=int(r["max_cores"]) if str(r["max_cores"]).isdigit() else 16,
            )
        )
    return configs


# ---------------- phases ----------------


def phase_baseline(mesh, out_dir):
    grid = mesh.compute_with_storage_grid_size()
    print(f"compute_with_storage_grid_size = {grid} ({grid.x}x{grid.y} = {grid.x * grid.y} cores)")

    Q, K, V = make_qkv_torch()
    ref = torch_ref(Q, K, V, DIE3_CHUNK_START)
    Kp, Vp = page_kv_torch(K, V)
    q, k, v = upload_fixture(mesh, Q, Kp, Vp, dtype=ttnn.bfloat16)
    pt = upload_page_table(mesh)

    rows = []
    us_values = []
    for _ in range(3):
        row = run_config(
            mesh,
            "baseline",
            q,
            k,
            v,
            pt,
            DIE3_CHUNK_START,
            ref,
            qc=64,
            kc=64,
            gx=grid.x,
            gy=grid.y,
            exp_approx=False,
            fidelity=ttnn.MathFidelity.HiFi2,
            fp32acc=True,
            packer_l1_acc=True,
            kv_dtype="bf16",
            max_cores=16,
            rows=rows,
        )
        if row["status"] == "OK":
            us_values.append(float(row["us"]))

    append_and_write(out_dir, rows)

    in_range = len(us_values) == 3 and all(1050 <= u <= 1350 for u in us_values)
    print(f"\nbaseline us values: {us_values}")
    print(f"baseline PCC values: {[r['pcc'] for r in rows]}")
    if not in_range:
        print("!! baseline OUT OF EXPECTED RANGE (1050-1350 us) or errored -- per instructions, STOP here.")
    return in_range


def phase_sweep(mesh, out_dir):
    grid = mesh.compute_with_storage_grid_size()
    full_grid = (grid.x, grid.y)
    grids = [full_grid, (8, 8)]
    print(f"compute_with_storage_grid_size = {grid} ({grid.x}x{grid.y}); grids swept: {grids}")

    Q, K, V = make_qkv_torch()
    ref = torch_ref(Q, K, V, DIE3_CHUNK_START)
    Kp, Vp = page_kv_torch(K, V)
    q, k, v = upload_fixture(mesh, Q, Kp, Vp, dtype=ttnn.bfloat16)
    pt = upload_page_table(mesh)

    rows = []
    for qc in (32, 64, 128, 256, 512):
        for kc in (32, 64, 128, 256, 512):
            for gx, gy in grids:
                for exp_approx in (False, True):
                    run_config(
                        mesh,
                        "sweep",
                        q,
                        k,
                        v,
                        pt,
                        DIE3_CHUNK_START,
                        ref,
                        qc=qc,
                        kc=kc,
                        gx=gx,
                        gy=gy,
                        exp_approx=exp_approx,
                        fidelity=ttnn.MathFidelity.HiFi2,
                        fp32acc=True,
                        packer_l1_acc=True,
                        kv_dtype="bf16",
                        max_cores=16,
                        rows=rows,
                    )

    ok_rows = [r for r in rows if r["status"] == "OK" and r["us"]]
    best5 = sorted(ok_rows, key=lambda r: float(r["us"]))[:5]
    print(f"\n-- best 5 (qc,kc,grid,exp) from the 100-config sweep; retrying max_cores in {{8,32,64}} --")
    for r in best5:
        gx, gy = map(int, r["grid"].split("x"))
        for mc in (8, 32, 64):
            run_config(
                mesh,
                "sweep_mc",
                q,
                k,
                v,
                pt,
                DIE3_CHUNK_START,
                ref,
                qc=int(r["qc"]),
                kc=int(r["kc"]),
                gx=gx,
                gy=gy,
                exp_approx=(r["exp_approx"] == "True"),
                fidelity=ttnn.MathFidelity.HiFi2,
                fp32acc=True,
                packer_l1_acc=True,
                kv_dtype="bf16",
                max_cores=mc,
                rows=rows,
            )

    append_and_write(out_dir, rows)


def phase_fidelity(mesh, out_dir):
    grid = mesh.compute_with_storage_grid_size()
    Q, K, V = make_qkv_torch()
    ref = torch_ref(Q, K, V, DIE3_CHUNK_START)
    Kp, Vp = page_kv_torch(K, V)
    q, k, v = upload_fixture(mesh, Q, Kp, Vp, dtype=ttnn.bfloat16)
    pt = upload_page_table(mesh)

    configs = [dict(qc=64, kc=64, gx=grid.x, gy=grid.y, exp_approx=False, max_cores=16, label="baseline")]
    for i, c in enumerate(read_best_sweep_configs(out_dir, n=3)):
        c = dict(c)
        c["label"] = f"sweep_best{i + 1}"
        configs.append(c)

    rows = []
    for cfg in configs:
        for fidelity in FIDELITY_LIST:
            for fp32acc in (True, False):
                for packer_l1_acc in (True, False):
                    run_config(
                        mesh,
                        f"fidelity_{cfg['label']}",
                        q,
                        k,
                        v,
                        pt,
                        DIE3_CHUNK_START,
                        ref,
                        qc=cfg["qc"],
                        kc=cfg["kc"],
                        gx=cfg["gx"],
                        gy=cfg["gy"],
                        exp_approx=cfg["exp_approx"],
                        fidelity=fidelity,
                        fp32acc=fp32acc,
                        packer_l1_acc=packer_l1_acc,
                        kv_dtype="bf16",
                        max_cores=cfg["max_cores"],
                        rows=rows,
                    )

    append_and_write(out_dir, rows)


def phase_bf8(mesh, out_dir):
    grid = mesh.compute_with_storage_grid_size()
    Q, K, V = make_qkv_torch()
    ref = torch_ref(Q, K, V, DIE3_CHUNK_START)
    Kp, Vp = page_kv_torch(K, V)
    q, k, v = upload_fixture(mesh, Q, Kp, Vp, dtype=ttnn.bfloat8_b)
    pt = upload_page_table(mesh)

    configs = [dict(qc=64, kc=64, gx=grid.x, gy=grid.y, exp_approx=False, max_cores=16, label="baseline")]
    for i, c in enumerate(read_best_sweep_configs(out_dir, n=3)):
        c = dict(c)
        c["label"] = f"sweep_best{i + 1}"
        configs.append(c)

    rows = []
    for cfg in configs:
        for fp32acc in (True, False):
            run_config(
                mesh,
                f"bf8_{cfg['label']}",
                q,
                k,
                v,
                pt,
                DIE3_CHUNK_START,
                ref,
                qc=cfg["qc"],
                kc=cfg["kc"],
                gx=cfg["gx"],
                gy=cfg["gy"],
                exp_approx=cfg["exp_approx"],
                fidelity=ttnn.MathFidelity.HiFi2,
                fp32acc=fp32acc,
                packer_l1_acc=True,
                kv_dtype="bf8",
                max_cores=cfg["max_cores"],
                rows=rows,
            )

    append_and_write(out_dir, rows)


def phase_die0(mesh, out_dir):
    grid = mesh.compute_with_storage_grid_size()
    Q, K, V = make_qkv_torch()
    ref0 = torch_ref(Q, K, V, DIE0_CHUNK_START)
    Kp, Vp = page_kv_torch(K, V)
    q, k, v = upload_fixture(mesh, Q, Kp, Vp, dtype=ttnn.bfloat16)
    pt = upload_page_table(mesh)

    configs = [dict(qc=64, kc=64, gx=grid.x, gy=grid.y, exp_approx=False, max_cores=16, label="baseline")]
    for i, c in enumerate(read_best_sweep_configs(out_dir, n=3)):
        c = dict(c)
        c["label"] = f"sweep_best{i + 1}"
        configs.append(c)

    rows = []
    for cfg in configs:
        run_config(
            mesh,
            f"die0_{cfg['label']}",
            q,
            k,
            v,
            pt,
            DIE0_CHUNK_START,
            ref0,
            qc=cfg["qc"],
            kc=cfg["kc"],
            gx=cfg["gx"],
            gy=cfg["gy"],
            exp_approx=cfg["exp_approx"],
            fidelity=ttnn.MathFidelity.HiFi2,
            fp32acc=True,
            packer_l1_acc=True,
            kv_dtype="bf16",
            max_cores=cfg["max_cores"],
            rows=rows,
        )

    append_and_write(out_dir, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["baseline", "sweep", "fidelity", "bf8", "die0"])
    ap.add_argument("--out-dir", type=Path, default=Path.cwd())
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening mesh device for phase={args.phase} ...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    t0 = time.time()
    try:
        if args.phase == "baseline":
            ok = phase_baseline(mesh, args.out_dir)
            if not ok:
                print("STOPPING after baseline phase; not proceeding to sweep/fidelity/bf8/die0.")
        elif args.phase == "sweep":
            phase_sweep(mesh, args.out_dir)
        elif args.phase == "fidelity":
            phase_fidelity(mesh, args.out_dir)
        elif args.phase == "bf8":
            phase_bf8(mesh, args.out_dir)
        elif args.phase == "die0":
            phase_die0(mesh, args.out_dir)
    finally:
        ttnn.close_mesh_device(mesh)
        print(f"\nPhase {args.phase} done in {time.time() - t0:.1f}s. Device closed.")


if __name__ == "__main__":
    main()
