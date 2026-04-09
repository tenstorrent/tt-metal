#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Analyze optimal mesh layout for DeepSeek V3 MLA on a single galaxy (32 BH chips).

For each (RP x UP) factorization of 32 and target sequence lengths,
models per-device matmul compute time and CCL communication time for
each operation in the MLA forward pass.

Follows the same compute model as parse_ring_joint_perf.py:
  tile_ops = (M/32) * (K/32) * (N/32)  [* batch for batched matmuls]
  cycles   = tile_ops * fidelity_cycles
  time_ns  = cycles / num_cores / clock_ghz

Matmul roofline: max(compute-bound, DRAM-bound). Weights are bf8, activations bf16.
Communication modeled as ring collectives at 100 GB/s (2 links x 50 GB/s bidir).
Ring SDPA overlap: comm pipelined behind compute across ring steps.

Usage:
    python models/demos/deepseek_v3_d_p/utils/chunked_prefill/analyze_mla_layouts.py
"""

import math
from dataclasses import dataclass

# ── DeepSeek V3 MLA dimensions ──────────────────────────────────────────────
HIDDEN_SIZE = 7168
NUM_HEADS = 128
KV_LORA_RANK = 512
Q_LORA_RANK = 1536
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192

# ── Blackhole hardware parameters ───────────────────────────────────────────
CLOCK_GHZ = 1.35
COMPUTE_GRID = (11, 10)  # di/dt limited from 12x10
NUM_CORES = COMPUTE_GRID[0] * COMPUTE_GRID[1]  # 110
HIFI2_CYCLES = 32  # cycles per tile matmul at HiFi2
TILE = 32
MATMUL_UTIL = 0.7  # practical compute utilization factor for matmuls

# Ethernet: BH has 25 GB/s unidirectional / 50 GB/s bidirectional per link.
# Always uses bidirectional (even for linear topology).
# Measured: ~24.16 GB/s uni-dir at 8192B packets (test_all_ethernet_links_bandwidth.py).
# 2 links per hop for BH CCL.
ETH_BW_PER_LINK_GBps = 50.0  # bidirectional
NUM_ETH_LINKS = 2
ETH_BW_GBps = ETH_BW_PER_LINK_GBps * NUM_ETH_LINKS  # 100 GB/s
ETH_UTIL = 0.8  # practical ethernet utilization factor

# DRAM bandwidth per device
DRAM_BW_GBps = 512
DRAM_UTIL = 0.6  # practical DRAM bandwidth utilization factor

# Data type sizes (bytes per element)
BF16 = 2
BF8 = 1


# ── Compute helpers ─────────────────────────────────────────────────────────


def tiles(x):
    """Number of tiles (ceil) for dimension x."""
    return math.ceil(x / TILE)


def matmul_compute_ms(M, K, N, batch=1):
    """
    Compute-bound matmul time in ms for [batch, M, K] @ [batch, K, N].

    tile_ops = batch * ceil(M/32) * ceil(K/32) * ceil(N/32)
    Derated by MATMUL_UTIL (practical utilization factor).
    """
    tile_ops = batch * tiles(M) * tiles(K) * tiles(N)
    return tile_ops * HIFI2_CYCLES / NUM_CORES / CLOCK_GHZ / MATMUL_UTIL / 1e6


def matmul_dram_ms(M, K, N, batch=1, act_bytes=BF16, wt_bytes=BF8, out_bytes=BF16):
    """
    Memory-bound matmul time in ms for [batch, M, K] @ [batch, K, N].

    Reads activation + weight, writes output, all through DRAM.
    """
    read_act = batch * M * K * act_bytes
    read_wt = batch * K * N * wt_bytes
    write_out = batch * M * N * out_bytes
    total_bytes = read_act + read_wt + write_out
    return total_bytes / (DRAM_BW_GBps * 1e9) * 1e3


def matmul_detail(M, K, N, batch=1, act_bytes=BF16, wt_bytes=BF8, out_bytes=BF16):
    """
    Return matmul roofline breakdown: read, compute, write times and bottleneck.
    """
    comp = matmul_compute_ms(M, K, N, batch)
    read_bytes = batch * M * K * act_bytes + batch * K * N * wt_bytes
    write_bytes = batch * M * N * out_bytes
    effective_dram_bw = DRAM_BW_GBps * DRAM_UTIL
    read_ms = read_bytes / (effective_dram_bw * 1e9) * 1e3
    write_ms = write_bytes / (effective_dram_bw * 1e9) * 1e3
    bottleneck = max((read_ms, "read"), (comp, "compute"), (write_ms, "write"), key=lambda x: x[0])
    return {
        "read_ms": read_ms,
        "compute_ms": comp,
        "write_ms": write_ms,
        "bound": bottleneck[1],
        "time_ms": bottleneck[0],
    }


def ring_collective_ms(data_bytes, num_devices, ring=False):
    """
    Time for one collective (allgather or reduce-scatter) in ms.

    Linear: (N-1)/N * data / BW
    Ring:   (N-1)/N * data / BW / 2  (halves hops vs linear)

    Returns 0 if num_devices <= 1.
    """
    if num_devices <= 1:
        return 0.0
    effective_bw = ETH_BW_GBps * ETH_UTIL
    t = (num_devices - 1) * data_bytes / num_devices / (effective_bw * 1e9) * 1e3
    if ring:
        t /= 2
    return t


def sdpa_global_compute_ms(seq_len, nh, dk, dv, num_devices):
    """
    Global causal SDPA compute time in ms.

    Total tile ops across all devices, divided by total cores across all devices.
    QK^T: NH * ceil(S/32) * ceil(Dk/32) * ceil(S/32) / 2  (causal)
    AV:   NH * ceil(S/32) * ceil(S/32) * ceil(Dv/32) / 2  (causal)
    """
    s_tiles = tiles(seq_len)
    qkt = nh * s_tiles * tiles(dk) * s_tiles / 2
    av = nh * s_tiles * s_tiles * tiles(dv) / 2
    total_cores = num_devices * NUM_CORES
    return (qkt + av) * HIFI2_CYCLES / total_cores / CLOCK_GHZ / MATMUL_UTIL / 1e6


def sdpa_global_kv_transfer_ms(seq_len, nh_local, dk, dv, rp):
    """
    Global KV transfer time for ring attention in ms.

    Each link sees (RP-1) chunks pass through. Ring halves this vs linear.
    K chunk: [1, 1, sl_local, dk] in bf8
    V chunk: [1, nh_local, sl_local, dv] in bf8
    """
    if rp <= 1:
        return 0.0
    sl_local = seq_len // rp
    k_bytes = sl_local * dk * BF8
    v_bytes = nh_local * sl_local * dv * BF8
    chunk_bytes = k_bytes + v_bytes
    effective_bw = ETH_BW_GBps * ETH_UTIL
    # Ring: (RP-1) chunks through each link, /2 for bidirectional ring
    return (rp - 1) * chunk_bytes / 2 / (effective_bw * 1e9) * 1e3


# ── Analysis ────────────────────────────────────────────────────────────────


GROUPS = ("matmul", "ccl", "ring_attn")


@dataclass
class Op:
    name: str
    compute_ms: float
    comm_ms: float = 0.0
    group: str = "matmul"  # "matmul", "ccl", or "ring_attn"
    detail: dict = None  # matmul breakdown: read_ms, compute_ms, write_ms, bound

    @property
    def total_ms(self):
        return self.compute_ms + self.comm_ms


def analyze_layout(seq_len, rp, up):
    """
    Model MLA forward pass timing for one (RP, UP) layout.

    Follows the forward() method in mla.py step by step:
      Q path:  q_a_proj → TP AR → RMSNorm → q_b_proj → heads → wkv_b1 + RoPE
      KV path: kv_a_proj → TP AR → RMSNorm + RoPE → wkv_b2 (V expand)
      Attn:    ring SDPA
      Output:  concat_heads → o_proj → TP RS

    RMSNorm, RoPE, slice/concat are ignored (small relative to matmuls).
    """
    sl = seq_len // rp  # local sequence length
    nh = NUM_HEADS // up  # local head count
    ops = []

    def mm_op(name, M, K, N, batch=1, act_bytes=BF16, wt_bytes=BF8, out_bytes=BF16):
        d = matmul_detail(M, K, N, batch, act_bytes, wt_bytes, out_bytes)
        return Op(name, d["time_ms"], detail=d)

    # ── Q path ──────────────────────────────────────────────────────────
    # q_a_proj: [1,1,sl,H/UP] @ [H/UP, q_lr]  act=bf16, wt=bf8, out=bf16
    ops.append(mm_op("q_a_proj", sl, HIDDEN_SIZE // up, Q_LORA_RANK))

    # TP all-reduce for q_a (reduce-scatter + all-gather on dim 3)
    d_qa = sl * Q_LORA_RANK * BF16
    ops.append(Op("q_a_tp_ar", 0.0, 2 * ring_collective_ms(d_qa, up, ring=True), group="ccl"))

    # q_b_proj: [1,1,sl,q_lr] @ [q_lr, NH*QK_HD/UP]  act=bf16, wt=bf8, out=bf16
    ops.append(mm_op("q_b_proj", sl, Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM // up))

    # wkv_b1: batched [nh,sl,qk_nope_hd] @ [nh,qk_nope_hd,kv_lr]  act=bf16, wt=bf8, out=bf16
    ops.append(mm_op("wkv_b1", sl, QK_NOPE_HEAD_DIM, KV_LORA_RANK, batch=nh))

    # ── KV path ─────────────────────────────────────────────────────────
    # kv_a_proj: [1,1,sl,H/UP] @ [H/UP, kv_lr+rope_dim]  act=bf16, wt=bf8, out=bf16
    kv_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
    ops.append(mm_op("kv_a_proj", sl, HIDDEN_SIZE // up, kv_dim))

    # TP all-gather (dim 1) + reduce_nc → effectively all-reduce
    # AG gathers UP shards of [1,1,sl,kv_dim] → [1,UP,sl,kv_dim]
    d_kv = up * sl * kv_dim * BF16
    ops.append(Op("kv_a_tp_ar", 0.0, ring_collective_ms(d_kv, up, ring=True), group="ccl"))

    # wkv_b2: batched [nh,sl,kv_lr] @ [nh,kv_lr,v_hd]  act=bf16, wt=bf8, out=bf8
    ops.append(mm_op("wkv_b2", sl, KV_LORA_RANK, V_HEAD_DIM, batch=nh, out_bytes=BF8))

    # ── Attention ───────────────────────────────────────────────────────
    # Ring SDPA: global compute vs global KV transfer
    sdpa_dk = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
    num_devices = rp * up
    comp = sdpa_global_compute_ms(seq_len, NUM_HEADS, sdpa_dk, V_HEAD_DIM, num_devices)
    comm = sdpa_global_kv_transfer_ms(seq_len, nh, sdpa_dk, V_HEAD_DIM, rp)
    ops.append(Op("ring_sdpa", comp, comm, group="ring_attn"))

    # ── Output ──────────────────────────────────────────────────────────
    # o_proj: [1,1,sl,NH*V_HD/UP] @ [NH*V_HD/UP, H]  act=bf16, wt=bf8, out=bf16
    ops.append(mm_op("o_proj", sl, NUM_HEADS * V_HEAD_DIM // up, HIDDEN_SIZE))

    # TP reduce-scatter (dim 3)
    d_o = sl * HIDDEN_SIZE * BF16
    ops.append(Op("o_proj_tp_rs", 0.0, ring_collective_ms(d_o, up, ring=True), group="ccl"))

    return ops


def summarize_layout(seq_len, rp, up, ops, group_filter=None):
    """Print per-op breakdown, return summary metrics.

    group_filter: if set, only show ops matching this group ("matmul", "ccl", "ring_attn").
    """
    sl = seq_len // rp
    nh = NUM_HEADS // up

    show_ops = [o for o in ops if o.group == group_filter] if group_filter else ops

    total_comp = sum(o.compute_ms for o in show_ops)
    total_comm = sum(o.comm_ms for o in show_ops)
    no_overlap = total_comp + total_comm

    # With ring SDPA overlap: replace comp+comm with max(comp, comm)
    sdpa = next((o for o in show_ops if o.name == "ring_sdpa"), None)
    if sdpa:
        with_overlap = no_overlap - (sdpa.compute_ms + sdpa.comm_ms) + max(sdpa.compute_ms, sdpa.comm_ms)
    else:
        with_overlap = no_overlap

    group_label = f"  [{group_filter}]" if group_filter else ""
    print(f"\n  RP={rp} x UP={up}  |  seq_local={sl}  heads_local={nh}{group_label}")

    if group_filter == "matmul":
        # Detailed matmul view: read / compute / write / bound
        print(f"  {'Op':<16} {'Read':>10} {'Compute':>10} {'Write':>10} {'Time':>10}  {'Bound'}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*8}")
        sum_read = sum_comp = sum_write = 0.0
        for o in show_ops:
            d = o.detail
            sum_read += d["read_ms"]
            sum_comp += d["compute_ms"]
            sum_write += d["write_ms"]
            print(
                f"  {o.name:<16} {d['read_ms']:>10.4f} {d['compute_ms']:>10.4f} "
                f"{d['write_ms']:>10.4f} {d['time_ms']:>10.4f}  {d['bound']}"
            )
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'SUM':<16} {sum_read:>10.4f} {sum_comp:>10.4f} {sum_write:>10.4f} {total_comp:>10.4f}")
    elif group_filter == "ring_attn":
        # Ring attention view: compute / comm / time / bound (per step overlap)
        print(f"  {'Op':<16} {'Compute':>10} {'Comm':>10} {'Time':>10}  {'Bound'}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}  {'-'*8}")
        for o in show_ops:
            if o.comm_ms > 0 and rp > 1:
                comp_step = o.compute_ms / rp
                comm_step = o.comm_ms / (rp - 1)
                bound = "compute" if comp_step >= comm_step else "comm"
                time = max(o.compute_ms, o.comm_ms)
            else:
                bound = "compute"
                time = o.compute_ms
            print(f"  {o.name:<16} {o.compute_ms:>10.4f} {o.comm_ms:>10.4f} {time:>10.4f}  {bound}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'SUM':<16} {total_comp:>10.4f} {total_comm:>10.4f} {with_overlap:>10.4f}")
        if sdpa and rp > 1:
            comp_per_step = sdpa.compute_ms / rp
            comm_per_step = sdpa.comm_ms / (rp - 1)
            ratio = comp_per_step / comm_per_step if comm_per_step > 0 else float("inf")
            print(f"  per step: comp={comp_per_step:.4f} comm={comm_per_step:.4f} ({ratio:.2f}x)")
    else:
        # Combined view: ops sorted by group, with bound info
        group_labels = {"matmul": "Matmuls", "ccl": "CCLs (TP)", "ring_attn": "Ring Attention"}
        group_totals = {}

        for g in GROUPS:
            g_ops = [o for o in show_ops if o.group == g]
            if not g_ops:
                continue
            print(f"  ── {group_labels[g]}")
            g_total = 0.0
            for o in g_ops:
                if g == "matmul" and o.detail:
                    bound = o.detail["bound"]
                    time = o.detail["time_ms"]
                elif g == "ring_attn" and o.comm_ms > 0 and rp > 1:
                    bound = "compute" if o.compute_ms >= o.comm_ms else "comm"
                    time = max(o.compute_ms, o.comm_ms)
                elif g == "ccl":
                    bound = "eth"
                    time = o.comm_ms
                else:
                    bound = ""
                    time = o.compute_ms + o.comm_ms
                print(f"    {o.name:<14} {time:>10.4f}  {bound}")
                g_total += time
            print(f"    {'subtotal':<14} {g_total:>10.4f}")
            group_totals[g] = g_total

        print(f"  {'-'*38}")
        print(f"  {'TOTAL':<16} {with_overlap:>10.4f}")

    return {
        "compute": total_comp,
        "comm": total_comm,
        "no_overlap": no_overlap,
        "overlap": with_overlap,
    }


def power_of_2_range(lo, hi):
    """Generate powers of 2 from lo to hi inclusive."""
    vals = []
    v = 1
    while v <= hi:
        if v >= lo:
            vals.append(v)
        v *= 2
    return vals


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek V3 MLA layout analysis for single galaxy")
    parser.add_argument(
        "--group",
        choices=GROUPS,
        default=None,
        help="Show only one group of operations: matmul, ccl, or ring_attn",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=2,
        metavar=("RP", "UP"),
        help="Single mesh layout to analyze (e.g. --mesh 8 4). Must satisfy RP*UP=32.",
    )
    parser.add_argument(
        "--seq",
        type=int,
        nargs="+",
        metavar="LEN",
        help="Sequence lengths to analyze. Two values gives a power-of-2 range (e.g. --seq 1024 8192 → 1024,2048,4096,8192).",
    )
    args = parser.parse_args()

    if args.mesh:
        rp, up = args.mesh
        assert rp * up == 32, f"RP * UP must be 32, got {rp} * {up} = {rp * up}"
        layouts = [(rp, up)]
    else:
        layouts = [(32, 1), (16, 2), (8, 4), (4, 8), (2, 16), (1, 32)]

    if args.seq:
        if len(args.seq) == 2:
            seq_lens = power_of_2_range(args.seq[0], args.seq[1])
        else:
            seq_lens = args.seq
    else:
        seq_lens = [4096, 8192, 16384]

    print("=" * 72)
    title = "DeepSeek V3 MLA — Single Galaxy (32 BH) Layout Analysis"
    if args.group:
        title += f"  [{args.group}]"
    print(title)
    print("=" * 72)
    print(f"HW:  BH @ {CLOCK_GHZ} GHz, {NUM_CORES} cores/dev, HiFi2 ({HIFI2_CYCLES} cyc/tile), {MATMUL_UTIL:.0%} util")
    print(f"DRAM: {DRAM_BW_GBps} GB/s per device, {DRAM_UTIL:.0%} util ({DRAM_BW_GBps * DRAM_UTIL:.0f} GB/s effective)")
    print(
        f"ETH: {ETH_BW_GBps} GB/s  ({NUM_ETH_LINKS} links x {ETH_BW_PER_LINK_GBps} GB/s bidir), {ETH_UTIL:.0%} util ({ETH_BW_GBps * ETH_UTIL:.0f} GB/s effective)"
    )
    print(f"MLA: H={HIDDEN_SIZE} NH={NUM_HEADS} kv_lr={KV_LORA_RANK} q_lr={Q_LORA_RANK}")
    print(f"     Dk_nope={QK_NOPE_HEAD_DIM} Dk_rope={QK_ROPE_HEAD_DIM} Dv={V_HEAD_DIM}")

    results = {}
    for seq_len in seq_lens:
        print(f"\n{'='*72}")
        print(f"  Sequence length: {seq_len}")
        print(f"{'='*72}")
        for rp, up in layouts:
            sl = seq_len // rp
            if sl < TILE:
                print(f"\n  RP={rp} x UP={up}  — SKIPPED (seq_local={sl} < tile={TILE})")
                continue
            ops = analyze_layout(seq_len, rp, up)
            results[(seq_len, rp, up)] = summarize_layout(seq_len, rp, up, ops, group_filter=args.group)

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    label = f" [{args.group}]" if args.group else ""
    print(f"SUMMARY: Estimated MLA forward time per device [ms]{label}")
    print(f"{'='*72}")

    col_w = 20
    header = f"{'Layout':<10}"
    for s in seq_lens:
        header += f"  {'seq=' + str(s):>{col_w}}"
    print(header)
    print("-" * len(header))

    for rp, up in layouts:
        row = f"{rp}x{up:<8}"
        for s in seq_lens:
            key = (s, rp, up)
            if key in results:
                r = results[key]
                cell = f"{r['overlap']:.3f} ({r['compute']:.3f}+{r['comm']:.3f})"
                row += f"  {cell:>{col_w}}"
            else:
                row += f"  {'n/a':>{col_w}}"
        print(row)

    print()
    print("Format: total_w_overlap (compute + comm)")
    print()

    for s in seq_lens:
        valid = {k: v for k, v in results.items() if k[0] == s}
        if valid:
            best = min(valid, key=lambda k: valid[k]["overlap"])
            r = valid[best]
            print(f"  Best for seq={s}: RP={best[1]}xUP={best[2]} → {r['overlap']:.3f} ms")


if __name__ == "__main__":
    main()
