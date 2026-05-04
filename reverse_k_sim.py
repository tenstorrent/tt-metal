#!/usr/bin/env python3
"""
Reverse-K simulation + visualization for ring-joint-SDPA mla-100k, ring_iter=0.

Models per-core K-chain and V-chain roles plus per-timestamp actions across
110 cores x 63 timestamps. Emits a single self-contained HTML report.

Mirrors host distribution (program_factory.cpp 861-889, 937-976, 978-990,
1184-1281) and zigzag remap (q_chunk_remapping.hpp 30-44).
"""
from __future__ import annotations

import json
from pathlib import Path

# -------- configs --------
# mla-100k Galaxy config (the user's target):
MLA_100K = dict(
    name="mla-100k",
    grid_x=12,
    grid_y=10,
    sdpa_cols=11,
    num_cores=110,
    B=1,
    NHK=1,
    NH=32,
    num_q_chunks=20,
    distribution="head_grouped",
    layout="col_major",
)
# CSV oracle config (matches existing idea-ring_iter0_timestamps.csv):
CSV_ORACLE = dict(
    name="csv-oracle",
    grid_x=10,
    grid_y=10,
    sdpa_cols=10,
    num_cores=100,
    B=1,
    NHK=1,
    NH=29,
    num_q_chunks=20,
    distribution="flat",
    layout="row_major",
)

ENABLE_ZIGZAG = True
T_PER_PAIR = 21

# action codes
K_NONE, K_DRAM, K_MCAST_SEND, K_AWAIT, K_RECV_DONE, K_IDLE = 0, 1, 2, 3, 4, 5
V_NONE, V_DRAM, V_FORWARD, V_AWAIT, V_SNK_WAIT, V_IDLE, V_DRAM_LOCAL = 0, 1, 2, 3, 4, 5, 6
COMP_NONE, COMP_COMPUTE, COMP_IDLE = 0, 1, 2


# -------- layer 1: distribution --------
def compute_split(cfg: dict) -> tuple[int, int, int]:
    """(base_chunks_per_core, extra_chunks_per_core, cores_doing_extra)."""
    total_q = cfg["B"] * cfg["NH"] * cfg["num_q_chunks"]
    total_pairs = total_q // 2
    nc = cfg["num_cores"]
    if ENABLE_ZIGZAG:
        cores_extra = total_pairs % nc
        base = (total_pairs // nc) * 2
        extra = 2
    else:
        cores_extra = total_q % nc
        base = total_q // nc
        extra = 1
    return base, extra, cores_extra


def physical_core(cfg: dict, core_idx: int) -> tuple[int, int]:
    if cfg.get("layout") == "col_major":
        rows = cfg["num_cores"] // cfg["sdpa_cols"]
        return core_idx // rows, core_idx % rows
    return core_idx % cfg["sdpa_cols"], core_idx // cfg["sdpa_cols"]


def _positions_form_line(positions: list[tuple[int, int]]) -> bool:
    """True iff `positions` are colinear and contiguous on either axis (1xN or Nx1)."""
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    same_x = all(x == xs[0] for x in xs)
    same_y = all(y == ys[0] for y in ys)
    contig_x = (max(xs) - min(xs) + 1) == len(xs)
    contig_y = (max(ys) - min(ys) + 1) == len(ys)
    return (same_y and contig_x) or (same_x and contig_y)


def zigzag_q(cfg: dict, pos_in_head: int) -> tuple[int, bool]:
    """pos_in_head -> (q_chunk, is_light)."""
    n = cfg["num_q_chunks"]
    if pos_in_head % 2 == 0:
        return pos_in_head // 2, True
    return n - 1 - pos_in_head // 2, False


def build_core_work(cfg: dict) -> list[dict]:
    """Dispatch to flat (program_factory.cpp) or head_grouped distribution."""
    if cfg.get("distribution") == "head_grouped":
        return _build_core_work_head_grouped(cfg)
    return _build_core_work_flat(cfg)


def _build_core_work_flat(cfg: dict) -> list[dict]:
    """Per-core work assignment mirroring program_factory.cpp 861-929."""
    base, extra, cores_extra = compute_split(cfg)
    n = cfg["num_q_chunks"]
    nc = cfg["num_cores"]
    total_q = cfg["B"] * cfg["NH"] * n
    cores = []
    flat = 0
    for ci in range(nc):
        cnt = base + (extra if ci < cores_extra else 0)
        gqs = flat
        head_work: list[dict] = []
        for j in range(cnt):
            pos = flat + j
            h = pos // n
            if head_work and head_work[-1]["head"] == h:
                head_work[-1]["q_chunk_count"] += 1
            else:
                head_work.append({"batch": 0, "head": h, "q_chunk_start": pos % n, "q_chunk_count": 1})
        pairs = []
        n_pairs = cnt // 2
        for p in range(n_pairs):
            pos1 = flat + 2 * p
            pos2 = pos1 + 1
            h1 = pos1 // n
            h2 = pos2 // n
            assert h1 == h2, f"pair straddles heads at core {ci} pair {p}"
            pih1 = pos1 % n
            assert pih1 % 2 == 0, f"unexpected odd pos_in_head: {pih1}"
            ql, _ = zigzag_q(cfg, pih1)
            qh, _ = zigzag_q(cfg, pih1 + 1)
            assert ql + qh == n - 1
            pairs.append(
                {
                    "pair_idx": p,
                    "head": h1,
                    "l": ql,
                    "h": qh,
                    "ql": ql,
                    "qh": qh,
                    "q_iter_light": 2 * p,
                    "q_iter_heavy": 2 * p + 1,
                }
            )
        x, y = physical_core(cfg, ci)
        cores.append(
            {
                "core_idx": ci,
                "x": x,
                "y": y,
                "global_q_start": gqs,
                "global_q_count": cnt,
                "head_work": head_work,
                "pairs": pairs,
            }
        )
        flat += cnt
    assert flat == total_q, (flat, total_q)
    return cores


def _build_core_work_head_grouped(cfg: dict) -> list[dict]:
    """Head-grouped distribution.

    Group cores into blocks of `cores_per_group = pairs_per_head = num_q_chunks // 2`.
    Each block serves a contiguous range of heads; within a block, core i (0..cps-1)
    owns pair index i = (light=i, heavy=N-1-i) of every head in the block.

    For mla-100k (110 cores, 32 heads, 20 q-chunks): 11 groups of 10 cores; the
    first 10 groups serve 3 heads each, the last group serves 2. Cores 0-99 own
    3 pairs (6 chunks); cores 100-109 own 2 pairs (4 chunks).
    """
    n = cfg["num_q_chunks"]
    nc = cfg["num_cores"]
    NH = cfg["NH"]
    pairs_per_head = n // 2
    cores_per_group = pairs_per_head
    assert nc % cores_per_group == 0, (nc, cores_per_group)
    num_groups = nc // cores_per_group
    heads_per_group = NH // num_groups
    extra_head_groups = NH % num_groups

    group_heads: list[list[int]] = []
    head_cursor = 0
    for g in range(num_groups):
        cnt = heads_per_group + (1 if g < extra_head_groups else 0)
        assert cnt >= 1, f"group {g} would get 0 heads (NH={NH}, num_groups={num_groups})"
        group_heads.append(list(range(head_cursor, head_cursor + cnt)))
        head_cursor += cnt
    assert head_cursor == NH, (head_cursor, NH)

    cores = []
    flat_q_cursor = 0
    for ci in range(nc):
        g = ci // cores_per_group
        i = ci % cores_per_group
        heads = group_heads[g]
        chunk_count = len(heads) * 2
        head_work = [{"batch": 0, "head": h, "q_chunk_start": i, "q_chunk_count": 2} for h in heads]
        pairs = []
        for p, h in enumerate(heads):
            ql = i
            qh = n - 1 - i
            assert ql + qh == n - 1
            pairs.append(
                {
                    "pair_idx": p,
                    "head": h,
                    "l": ql,
                    "h": qh,
                    "ql": ql,
                    "qh": qh,
                    "q_iter_light": 2 * p,
                    "q_iter_heavy": 2 * p + 1,
                }
            )
        x, y = physical_core(cfg, ci)
        cores.append(
            {
                "core_idx": ci,
                "x": x,
                "y": y,
                "global_q_start": flat_q_cursor,
                "global_q_count": chunk_count,
                "head_work": head_work,
                "pairs": pairs,
            }
        )
        flat_q_cursor += chunk_count
    total_q = cfg["B"] * NH * n
    assert flat_q_cursor == total_q, (flat_q_cursor, total_q)
    return cores


# -------- layer 2: chain roles --------
def select_k_injector(core_work: list[dict]) -> int:
    best, best_idx = -1, -1
    for c in core_work:
        if c["global_q_count"] > best:
            best = c["global_q_count"]
            best_idx = c["core_idx"]
    return best_idx


def build_v_chains(cfg: dict, core_work: list[dict]) -> dict[int, dict]:
    """Dispatch to flat or head_grouped V-chain construction."""
    if cfg.get("distribution") == "head_grouped":
        return _build_v_chains_head_grouped(cfg, core_work)
    return _build_v_chains_flat(cfg, core_work)


def _build_v_chains_flat(cfg: dict, core_work: list[dict]) -> dict[int, dict]:
    chains: dict[int, dict] = {}
    by_head: dict[int, list[int]] = {}
    for c in core_work:
        for hw in c["head_work"]:
            by_head.setdefault(hw["head"], []).append(c["core_idx"])
    for head, cores in by_head.items():
        cores = sorted(cores)
        inj_pos = None
        for i, ci in enumerate(cores):
            if len(core_work[ci]["head_work"]) == 1:
                inj_pos = i
                break
        if inj_pos is None or inj_pos == len(cores) - 1:
            chains[head] = {"chain": [], "inj": None, "snk": None, "mcast": False, "excluded": cores[:]}
            continue
        excluded = cores[:inj_pos]
        chain = cores[inj_pos:]
        positions = [physical_core(cfg, ci) for ci in chain]
        uniform = all(core_work[ci]["global_q_count"] == core_work[chain[0]]["global_q_count"] for ci in chain)
        is_mcast = _positions_form_line(positions) and uniform
        chains[head] = {"chain": chain, "inj": chain[0], "snk": chain[-1], "mcast": is_mcast, "excluded": excluded}
    return chains


def _build_v_chains_head_grouped(cfg: dict, core_work: list[dict]) -> dict[int, dict]:
    """Head-grouped V chains: each head's chain is exactly its core-group, in order.

    No exclusion is needed — every core that serves a head participates in that
    head's chain. `inj = chain[0]`, `snk = chain[-1]`.
    """
    chains: dict[int, dict] = {}
    by_head: dict[int, list[int]] = {}
    for c in core_work:
        for hw in c["head_work"]:
            by_head.setdefault(hw["head"], []).append(c["core_idx"])
    for head, cores in by_head.items():
        chain = sorted(cores)
        positions = [physical_core(cfg, ci) for ci in chain]
        uniform = all(core_work[ci]["global_q_count"] == core_work[chain[0]]["global_q_count"] for ci in chain)
        is_mcast = _positions_form_line(positions) and uniform
        chains[head] = {
            "chain": chain,
            "inj": chain[0],
            "snk": chain[-1],
            "mcast": is_mcast,
            "excluded": [],
        }
    return chains


def compute_static_roles(cfg: dict, core_work, v_chains, k_inj) -> dict:
    nc = cfg["num_cores"]
    k_role = [0] * nc
    k_role[k_inj] = 1
    v_role_per_head = [{} for _ in range(nc)]
    for head, info in v_chains.items():
        for ci in info["excluded"]:
            v_role_per_head[ci][head] = 0
        chain = info["chain"]
        for i, ci in enumerate(chain):
            if i == 0:
                v_role_per_head[ci][head] = 1
            elif i == len(chain) - 1:
                v_role_per_head[ci][head] = 3
            else:
                v_role_per_head[ci][head] = 2
    return {"k_role": k_role, "v_role_per_head": v_role_per_head}


# -------- layer 3: per (core, t) actions --------
def core_compute_at(cfg: dict, core: dict, t: int) -> dict | None:
    pair_idx = t // T_PER_PAIR
    if pair_idx >= len(core["pairs"]):
        return None
    pair = core["pairs"][pair_idx]
    l, h = pair["l"], pair["h"]
    head = pair["head"]
    ll = t % T_PER_PAIR
    if ll <= l:
        q = pair["ql"]
        k = ll
        in_light = True
    else:
        heavy_step = ll - (l + 1)
        q = pair["qh"]
        k = h - heavy_step
        in_light = False
    if ll == 0:
        phase = 1
    elif 1 <= ll <= 10:
        phase = 2
    else:
        phase = 3
    return {
        "q": q,
        "k": k,
        "v": k,
        "head": head,
        "l": l,
        "h": h,
        "pair_idx": pair_idx,
        "pair_local_t": ll,
        "phase": phase,
        "in_light_phase": in_light,
    }


def k_set_for_pair_local_t(cfg: dict, ll: int) -> list[int]:
    n = cfg["num_q_chunks"]
    half = n // 2
    if ll == 0:
        return [0]
    if 1 <= ll <= half - 1:
        return sorted({ll, n - ll})
    if ll == half:
        return [half]
    return [n - ll]


def compute_per_t_actions(cfg, core_work, v_chains, static_roles, k_inj) -> list[dict]:
    nc = cfg["num_cores"]
    # T_MAX = max number of pairs across all cores * T_PER_PAIR
    max_pairs = max(len(c["pairs"]) for c in core_work)
    t_max = max_pairs * T_PER_PAIR
    timestamps = []
    k_role = static_roles["k_role"]
    v_rph = static_roles["v_role_per_head"]
    for t in range(t_max):
        ll = t % T_PER_PAIR
        kset = k_set_for_pair_local_t(cfg, ll)
        cells_q = [None] * nc
        cells_k = [None] * nc
        cells_v = [None] * nc
        cells_head = [None] * nc
        cells_compute = [COMP_NONE] * nc
        cells_k_action = [K_NONE] * nc
        cells_v_action = [V_NONE] * nc
        cells_pair_idx = [-1] * nc
        cells_is_light = [0] * nc
        active = 0
        seen_v_chains = set()
        for ci in range(nc):
            info = core_compute_at(cfg, core_work[ci], t)
            if info is None:
                cells_compute[ci] = COMP_IDLE
                cells_k_action[ci] = K_IDLE
                cells_v_action[ci] = V_IDLE
                cells_pair_idx[ci] = -1
                continue
            active += 1
            cells_q[ci] = info["q"]
            cells_k[ci] = info["k"]
            cells_v[ci] = info["v"]
            cells_head[ci] = info["head"]
            cells_compute[ci] = COMP_COMPUTE
            cells_pair_idx[ci] = info["pair_idx"]
            cells_is_light[ci] = 1 if info["in_light_phase"] else 0
            # K action
            if k_role[ci] == 1:
                cells_k_action[ci] = K_DRAM  # injector reads + mcasts
            else:
                cells_k_action[ci] = K_AWAIT
            # V action: depends on this core's role for this head
            head = info["head"]
            v_role = v_rph[ci].get(head, 0)
            if v_role == 1:
                cells_v_action[ci] = V_DRAM
            elif v_role == 2:
                cells_v_action[ci] = V_FORWARD
            elif v_role == 3:
                cells_v_action[ci] = V_SNK_WAIT
            else:
                cells_v_action[ci] = V_DRAM_LOCAL
            seen_v_chains.add((head, info["v"]))
        # v_reads_total: count distinct (head, v_idx) pairs that need a V this t
        v_reads_total = len(seen_v_chains)
        if ll == 0:
            phase = 1
        elif 1 <= ll <= 10:
            phase = 2
        else:
            phase = 3
        timestamps.append(
            {
                "t": t,
                "phase": phase,
                "pair_local_t": ll,
                "k_set": kset,
                "v_reads_total": v_reads_total,
                "active_cores": active,
                "cells_q": cells_q,
                "cells_k": cells_k,
                "cells_v": cells_v,
                "cells_head": cells_head,
                "cells_compute": cells_compute,
                "cells_k_action": cells_k_action,
                "cells_v_action": cells_v_action,
                "cells_pair_idx": cells_pair_idx,
                "cells_is_light": cells_is_light,
            }
        )
    return timestamps


# -------- verification --------
def verify_mla_100k(cfg, core_work, v_chains, static_roles, timestamps, k_inj):
    assert cfg.get("distribution") == "head_grouped", cfg.get("distribution")
    nc = cfg["num_cores"]
    NH = cfg["NH"]
    n = cfg["num_q_chunks"]
    total_q = cfg["B"] * NH * n
    assert sum(c["global_q_count"] for c in core_work) == total_q

    # head-group structure (mla-100k: 11 groups × 10 cores; first 10 groups own 3
    # heads, last group owns 2)
    pairs_per_head = n // 2
    cores_per_group = pairs_per_head
    assert nc % cores_per_group == 0
    num_groups = nc // cores_per_group
    heads_per_group = NH // num_groups
    extra_head_groups = NH % num_groups
    group_heads = []
    head_cursor = 0
    for g in range(num_groups):
        cnt = heads_per_group + (1 if g < extra_head_groups else 0)
        group_heads.append(list(range(head_cursor, head_cursor + cnt)))
        head_cursor += cnt
    assert head_cursor == NH
    assert num_groups == 11 and group_heads[0] == [0, 1, 2] and group_heads[10] == [30, 31]

    # per-core: chunk count and pair contents
    for ci in range(nc):
        g = ci // cores_per_group
        i = ci % cores_per_group
        heads = group_heads[g]
        expected_chunks = len(heads) * 2
        assert core_work[ci]["global_q_count"] == expected_chunks, (ci, expected_chunks)
        pairs = core_work[ci]["pairs"]
        assert len(pairs) == len(heads), (ci, len(pairs), len(heads))
        for p, h in enumerate(heads):
            pr = pairs[p]
            assert pr["head"] == h, (ci, p, pr["head"], h)
            assert pr["ql"] == i and pr["qh"] == n - 1 - i, (ci, p, pr["ql"], pr["qh"])

    # K injector: still the first max-work core (cores 0..99 all tie at 6 chunks)
    assert sum(static_roles["k_role"]) == 1
    assert k_inj == 0, k_inj

    # V chains: one per head, each exactly cores_per_group long, chain = group cores
    assert len(v_chains) == NH, len(v_chains)
    for g in range(num_groups):
        expected_chain = list(range(g * cores_per_group, (g + 1) * cores_per_group))
        for h in group_heads[g]:
            info = v_chains[h]
            assert info["chain"] == expected_chain, (h, info["chain"], expected_chain)
            assert info["inj"] == expected_chain[0]
            assert info["snk"] == expected_chain[-1]
            assert info["excluded"] == []

    # col_major layout: each group's cores form a single full column.
    if cfg.get("layout") == "col_major":
        rows = nc // cfg["sdpa_cols"]
        assert physical_core(cfg, 0) == (0, 0)
        assert physical_core(cfg, 9) == (0, 9)
        assert physical_core(cfg, 10) == (1, 0)
        assert physical_core(cfg, 109) == (10, 9)
        for g in range(num_groups):
            chain = list(range(g * cores_per_group, (g + 1) * cores_per_group))
            positions = [physical_core(cfg, ci) for ci in chain]
            xs = {p[0] for p in positions}
            ys = sorted(p[1] for p in positions)
            assert len(xs) == 1 and xs == {g}, (g, positions)
            assert ys == list(range(rows)), (g, ys)
        # every chain is mcast-eligible: column-aligned + uniform work
        for h, info in v_chains.items():
            assert info["mcast"] is True, (h, info)

    # K-set schedule (driven only by pair_local_t, unchanged by distribution)
    half = n // 2
    for t in range(len(timestamps)):
        ll = t % T_PER_PAIR
        kset = timestamps[t]["k_set"]
        if ll == 0:
            assert kset == [0]
        elif 1 <= ll <= half - 1:
            assert kset == sorted({ll, n - ll}), (t, ll, kset)
        elif ll == half:
            assert kset == [half]
        else:
            assert kset == [n - ll], (t, ll, kset)

    # active cores per t: max_pairs is 3 (from groups 0..9). After t=41 (pair 2
    # finishes at t=62), the last group (2 pairs) goes idle from t=42 onward.
    for t in range(len(timestamps)):
        expected = 110 if t < 42 else 100
        assert timestamps[t]["active_cores"] == expected, (t, timestamps[t]["active_cores"])
    print(f"[verify {cfg['name']}] all asserts pass")


def verify_against_csv(cfg, timestamps, csv_path: Path):
    """Cross-check (q,k,v) against the existing 100-core CSV oracle."""
    if not csv_path.exists():
        print(f"[verify-csv] missing: {csv_path}")
        return
    lines = csv_path.read_text().strip().split("\n")
    header = lines[0].split(",")
    n_t_csv = len(header) - 3
    assert n_t_csv == len(timestamps), (n_t_csv, len(timestamps))
    nc = cfg["num_cores"]
    for line in lines[1:]:
        parts = line.split(",")
        if not parts[0].isdigit():
            continue  # skip DISTINCT_KV / IDEAL_KV summary rows
        ci = int(parts[0])
        if ci >= nc:
            continue
        for t in range(len(timestamps)):
            token = parts[3 + t]
            if not token or token in ("D", "-"):
                continue
            qkv = token.split("_")
            q, k, v = int(qkv[0][1:]), int(qkv[1][1:]), int(qkv[2][1:])
            got = (timestamps[t]["cells_q"][ci], timestamps[t]["cells_k"][ci], timestamps[t]["cells_v"][ci])
            assert got == (q, k, v), f"core {ci} t={t}: csv={token}, sim=Q{got[0]}_K{got[1]}_V{got[2]}"
    print(f"[verify-csv {cfg['name']}] all rows match")


# -------- payload + HTML --------
def build_payload(cfg, core_work, v_chains, static_roles, timestamps, k_inj) -> dict:
    nc = cfg["num_cores"]
    return {
        "config": {
            "name": cfg["name"],
            "num_cores": nc,
            "grid_x": cfg["grid_x"],
            "grid_y": cfg["grid_y"],
            "sdpa_cols": cfg["sdpa_cols"],
            "NH": cfg["NH"],
            "num_q_chunks": cfg["num_q_chunks"],
            "T_MAX": len(timestamps),
            "T_PER_PAIR": T_PER_PAIR,
            "k_inj_core": k_inj,
            "v_chains_by_head": {
                str(h): {
                    "chain": info["chain"],
                    "inj": info["inj"],
                    "snk": info["snk"],
                    "mcast": info["mcast"],
                    "excluded": info["excluded"],
                }
                for h, info in v_chains.items()
            },
        },
        "static": {
            "core_positions": [[c["x"], c["y"]] for c in core_work],
            "k_role": static_roles["k_role"],
            "v_role_per_head": [
                {str(h): r for h, r in static_roles["v_role_per_head"][ci].items()} for ci in range(nc)
            ],
            "global_q_count": [c["global_q_count"] for c in core_work],
            "head_work": [
                [{"head": hw["head"], "q_chunk_count": hw["q_chunk_count"]} for hw in c["head_work"]] for c in core_work
            ],
            "pairs_per_core": [
                [{"pair_idx": p["pair_idx"], "head": p["head"], "l": p["l"], "h": p["h"]} for p in c["pairs"]]
                for c in core_work
            ],
        },
        "timestamps": timestamps,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Reverse-K — Per-Core Chain Roles & Per-Timestamp Actions</title>
<style>
  :root {
    --bg: #0b0d12;
    --bg-card: #161922;
    --bg-card-hi: #1d2230;
    --fg: #e6e8ee;
    --fg-dim: #9aa3b2;
    --accent: #6ea8fe;
    --accent-2: #f59e0b;
    --good: #4ade80;
    --bad: #ef4444;
    --border: #262b38;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; }
  header { padding: 24px 32px 14px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; }
  header .sub { color: var(--fg-dim); font-size: 13px; }
  main { padding: 22px 32px 64px; max-width: 1700px; }
  section { margin-bottom: 22px; }
  section h2 { font-size: 15px; font-weight: 600; margin: 0 0 8px; color: var(--fg); }

  .controls { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; padding: 12px 16px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; }
  .controls label { display: flex; gap: 8px; align-items: center; color: var(--fg-dim); font-size: 13px; }
  .controls select, .controls input[type=range] { background: var(--bg-card-hi); color: var(--fg); border: 1px solid var(--border); border-radius: 5px; padding: 4px 8px; font: inherit; font-size: 13px; }
  .controls input[type=range] { padding: 0; width: 220px; }
  .controls .size-display { color: var(--fg); font-variant-numeric: tabular-nums; min-width: 36px; }
  .controls button { background: var(--bg-card-hi); color: var(--fg); border: 1px solid var(--border); border-radius: 5px; padding: 4px 12px; font: inherit; font-size: 13px; cursor: pointer; }
  .controls button:hover { background: var(--accent); color: #0b0d12; border-color: var(--accent); }
  .mode-controls { margin-bottom: 12px; }
  .mode-controls label { gap: 6px; }
  .modebtn { margin-right: 0; }
  .modebtn + .modebtn { margin-left: -1px; }
  .modebtn.active { background: var(--accent); color: #0b0d12; border-color: var(--accent); font-weight: 600; }
  .modebtn.active:hover { background: var(--accent); color: #0b0d12; }

  .stats-row { display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; margin-bottom: 16px; }
  .stat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; min-width: 120px; flex: 1; }
  .stat .label { color: var(--fg-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }
  .stat .num { font-size: 22px; font-weight: 600; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .stat .sub { color: var(--fg-dim); font-size: 11px; margin-top: 2px; }
  .stat.phase1 .num { color: var(--accent); }
  .stat.phase2 .num { color: var(--accent-2); }
  .stat.phase3 .num { color: var(--good); }

  .grid-block { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; }
  .grid-block .title-row { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 8px; flex-wrap: wrap; gap: 12px; }
  .grid-block .meta { color: var(--fg-dim); font-size: 12px; }
  .canvas-wrap { position: relative; overflow-x: auto; }
  .canvas-stack { position: relative; display: inline-block; }
  canvas.grid { display: block; background: #0a0a14; }
  canvas.overlay { display: block; position: absolute; left: 0; top: 0; pointer-events: none; }
  canvas.strip { display: block; image-rendering: pixelated; background: #0a0a14; cursor: pointer; }

  .legend { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
  .legend .swatch { display: inline-block; width: 13px; height: 13px; border-radius: 3px; vertical-align: middle; margin-right: 5px; }
  .legend > div { display: flex; align-items: center; font-size: 12px; color: var(--fg-dim); }

  #tooltip { position: fixed; pointer-events: none; background: #1d2230; border: 1px solid var(--border); border-radius: 6px; padding: 8px 11px; font-size: 12px; line-height: 1.5; color: var(--fg); box-shadow: 0 6px 20px rgba(0,0,0,0.5); z-index: 100; opacity: 0; transition: opacity 0.08s; max-width: 320px; }
  #tooltip.show { opacity: 1; }
  #tooltip .ttkey { color: var(--fg-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  #tooltip .ttbig { font-size: 14px; font-weight: 600; color: var(--accent); margin-bottom: 4px; }
  #tooltip .row { display: flex; justify-content: space-between; gap: 12px; margin-top: 2px; }
  #tooltip .row .k { color: var(--fg-dim); }
  #tooltip .row .v { font-weight: 500; }
</style>
</head>
<body>
  <header>
    <h1>Reverse-K — Per-Core Chain Roles &amp; Per-Timestamp Actions</h1>
    <div class="sub" id="header-sub">loading…</div>
  </header>

  <main>
    <section class="controls mode-controls">
      <label>Distribution
        <button class="modebtn" data-dist="flat">flat</button>
        <button class="modebtn" data-dist="head_grouped">head-grouped</button>
      </label>
      <label>Layout
        <button class="modebtn" data-layout="row_major">row-major</button>
        <button class="modebtn" data-layout="col_major">col-major</button>
      </label>
    </section>

    <section class="controls">
      <label>Cell content
        <select id="cell-mode">
          <option value="role">Chain role (static)</option>
          <option value="k_action" selected>K action</option>
          <option value="v_action">V action</option>
          <option value="compute">Compute / idle</option>
          <option value="q">Q chunk (color)</option>
          <option value="k">K chunk (color)</option>
          <option value="phase">Phase 1/2/3</option>
          <option value="head">Head (color)</option>
        </select>
      </label>
      <label>t=
        <input type="range" id="t-slider" min="0" max="62" value="0">
        <span class="size-display" id="t-display">0</span>
      </label>
      <button id="play-pause">▶ play</button>
      <label>Highlight V chain
        <select id="head-select">
          <option value="-1">— none —</option>
        </select>
      </label>
      <label><input type="checkbox" id="show-edges" checked> V chain edges</label>
    </section>

    <section class="stats-row">
      <div class="stat"><div class="label">timestamp</div><div class="num" id="stat-t">0</div><div class="sub" id="stat-t-sub">pair 0/3 · light Q</div></div>
      <div class="stat" id="stat-phase-card"><div class="label">phase</div><div class="num" id="stat-phase">1</div><div class="sub" id="stat-phase-sub">K0 only</div></div>
      <div class="stat"><div class="label">pair_local_t</div><div class="num" id="stat-plt">0</div><div class="sub">step within pair</div></div>
      <div class="stat"><div class="label">active cores</div><div class="num" id="stat-active">110</div><div class="sub" id="stat-active-sub">/ 110</div></div>
      <div class="stat"><div class="label">K reads</div><div class="num" id="stat-kr">1</div><div class="sub" id="stat-kr-sub">K0</div></div>
      <div class="stat"><div class="label">V reads</div><div class="num" id="stat-vr">0</div><div class="sub">distinct (head, V_idx)</div></div>
    </section>

    <section class="grid-block">
      <div class="title-row">
        <h2>Physical 11×10 grid — one cell per core</h2>
        <div class="meta">Hover any cell for details · ←/→ to step · spacebar to play</div>
      </div>
      <div class="canvas-wrap">
        <div class="canvas-stack">
          <canvas class="grid" id="grid-canvas"></canvas>
          <canvas class="overlay" id="grid-overlay"></canvas>
        </div>
      </div>
    </section>

    <section class="grid-block">
      <div class="title-row">
        <h2>Mini matrix — rows=cores, cols=timestamps</h2>
        <div class="meta">Click any column to jump there · vertical line marks current t</div>
      </div>
      <div class="canvas-wrap"><canvas class="strip" id="strip-canvas"></canvas></div>
    </section>

    <section class="legend" id="legend"></section>
  </main>

  <div id="tooltip"></div>

  <script type="application/json" id="payload">__PAYLOAD_JSON__</script>
  <script>
    const ENVELOPE = JSON.parse(document.getElementById('payload').textContent);
    let PAYLOAD, CFG, STATIC, TS;
    let NUM_CORES, SDPA_COLS, GRID_Y, T_MAX, T_PER_PAIR, NH;
    let CURRENT_MODE = ENVELOPE.default;

    // action codes
    const K_NONE=0, K_DRAM=1, K_MCAST_SEND=2, K_AWAIT=3, K_RECV_DONE=4, K_IDLE=5;
    const V_NONE=0, V_DRAM=1, V_FORWARD=2, V_AWAIT=3, V_SNK_WAIT=4, V_IDLE=5, V_DRAM_LOCAL=6;
    const COMP_NONE=0, COMP_COMPUTE=1, COMP_IDLE=2;

    // colors
    const ROLE_COLORS = {
      kInj: '#ef4444',  // red
      kRcv: '#475569',  // slate
      vInj: '#10b981',  // emerald
      vMid: '#3b82f6',  // blue
      vSnk: '#a855f7',  // purple
      vNone: '#374151', // gray
      idle: '#1a1a25',
    };
    // K_DRAM and K_MCAST_SEND share the receiver color so the K injector cell
    // visually blends with the rest of the chain; the K-INJ badge identifies it.
    // Action codes stay distinct — only the colors collapse.
    const KACT_COLORS = {[K_NONE]: '#1a1a25', [K_DRAM]: '#475569', [K_MCAST_SEND]: '#475569', [K_AWAIT]: '#475569', [K_RECV_DONE]: '#94a3b8', [K_IDLE]: '#1a1a25'};
    const VACT_COLORS = {[V_NONE]: '#1a1a25', [V_DRAM]: '#10b981', [V_FORWARD]: '#3b82f6', [V_AWAIT]: '#475569', [V_SNK_WAIT]: '#a855f7', [V_IDLE]: '#1a1a25', [V_DRAM_LOCAL]: '#f59e0b'};
    const COMP_COLORS = {[COMP_NONE]: '#1a1a25', [COMP_COMPUTE]: '#4ade80', [COMP_IDLE]: '#1a1a25'};
    const PHASE_COLORS = {1: '#6ea8fe', 2: '#f59e0b', 3: '#4ade80'};

    // Golden-angle shuffle so consecutive heads get maximally-distant hues.
    function headColor(h, alpha=1) {
      const hue = (h * 137.508) % 360;
      // alternate lightness between two bands for extra contrast
      const light = (h % 2 === 0) ? 60 : 50;
      return alpha < 1 ? `hsla(${hue}, 65%, ${light}%, ${alpha})` : `hsl(${hue}, 65%, ${light}%)`;
    }
    function hueColor(idx, n, alpha=1) {
      const hue = (idx * 360 / n) | 0;
      return alpha < 1 ? `hsla(${hue}, 70%, 56%, ${alpha})` : `hsl(${hue}, 70%, 56%)`;
    }

    const state = {
      t: 0, cellMode: 'k_action', highlightHead: -1,
      showEdges: true,
      playing: false, playInterval: null,
    };

    // ---------- canvas geometry ----------
    const GRID_CELL = 92;
    const GRID_PAD  = 2;
    const STRIP_W = 13;
    const STRIP_H = 5;

    const elGrid    = document.getElementById('grid-canvas');
    const elOverlay = document.getElementById('grid-overlay');
    const elStrip   = document.getElementById('strip-canvas');
    const elTip     = document.getElementById('tooltip');

    function loadMode(mode) {
      CURRENT_MODE = mode;
      PAYLOAD = ENVELOPE.payloads[mode];
      CFG = PAYLOAD.config;
      STATIC = PAYLOAD.static;
      TS = PAYLOAD.timestamps;
      NUM_CORES = CFG.num_cores;
      SDPA_COLS = CFG.sdpa_cols;
      GRID_Y = CFG.grid_y;
      T_MAX = CFG.T_MAX;
      T_PER_PAIR = CFG.T_PER_PAIR;
      NH = CFG.NH;

      elGrid.width     = SDPA_COLS * GRID_CELL;
      elGrid.height    = GRID_Y * GRID_CELL;
      elOverlay.width  = elGrid.width;
      elOverlay.height = elGrid.height;
      elStrip.width    = T_MAX * STRIP_W;
      elStrip.height   = NUM_CORES * STRIP_H + 4;

      const slider = document.getElementById('t-slider');
      slider.max = T_MAX - 1;
      if (state.t > T_MAX - 1) state.t = T_MAX - 1;
      slider.value = state.t;

      // populate head-select (clear + repopulate; preserves "— none —" at index 0)
      const hsel = document.getElementById('head-select');
      while (hsel.options.length > 1) hsel.remove(1);
      for (let h = 0; h < NH; h++) {
        const o = document.createElement('option');
        o.value = h;
        o.textContent = `Head ${h}`;
        hsel.appendChild(o);
      }
      if (state.highlightHead >= NH) state.highlightHead = -1;
      hsel.value = String(state.highlightHead);

      const dist = mode.startsWith('grouped') ? 'head_grouped' : 'flat';
      const layout = mode.endsWith('col') ? 'col_major' : 'row_major';
      document.getElementById('header-sub').textContent =
        `${CFG.name} · ring_iter=0 · ${NUM_CORES} cores × ${T_MAX} timestamps · ${NH} V chains · distribution=${dist} · layout=${layout}`;

      // sync mode-button .active state
      document.querySelectorAll('.modebtn[data-dist]').forEach(b => {
        b.classList.toggle('active', b.dataset.dist === dist);
      });
      document.querySelectorAll('.modebtn[data-layout]').forEach(b => {
        b.classList.toggle('active', b.dataset.layout === layout);
      });
    }

    loadMode(CURRENT_MODE);

    function cellColor(ci, t, mode) {
      const ts = TS[t];
      if (mode === 'role') {
        // K-INJ is shown via the bottom-left label badge, not by cell color.
        const head = ts.cells_head[ci];
        if (head !== null) {
          const r = STATIC.v_role_per_head[ci][String(head)];
          if (r === 1) return ROLE_COLORS.vInj;
          if (r === 2) return ROLE_COLORS.vMid;
          if (r === 3) return ROLE_COLORS.vSnk;
          if (r === 0) return ROLE_COLORS.vNone;
        }
        return ROLE_COLORS.idle;
      }
      if (mode === 'k_action') return KACT_COLORS[ts.cells_k_action[ci]];
      if (mode === 'v_action') return VACT_COLORS[ts.cells_v_action[ci]];
      if (mode === 'compute')  return COMP_COLORS[ts.cells_compute[ci]];
      if (mode === 'phase')    return ts.cells_compute[ci] === COMP_COMPUTE ? PHASE_COLORS[ts.phase] : ROLE_COLORS.idle;
      if (mode === 'q') {
        const q = ts.cells_q[ci];
        return q === null ? ROLE_COLORS.idle : hueColor(q, 20);
      }
      if (mode === 'k') {
        const k = ts.cells_k[ci];
        return k === null ? ROLE_COLORS.idle : hueColor(k, 20);
      }
      if (mode === 'head') {
        const h = ts.cells_head[ci];
        return h === null ? ROLE_COLORS.idle : headColor(h);
      }
      return ROLE_COLORS.idle;
    }

    function roleTagsFor(ci, t) {
      const tags = [];
      if (STATIC.k_role[ci] === 1) tags.push('K-INJ');
      else tags.push('K-RCV');
      const head = TS[t].cells_head[ci];
      if (head !== null) {
        const r = STATIC.v_role_per_head[ci][String(head)];
        if (r === 1) tags.push(`V-INJ:H${head}`);
        else if (r === 2) tags.push(`V-MID:H${head}`);
        else if (r === 3) tags.push(`V-SNK:H${head}`);
        else if (r === 0) tags.push(`V-EX:H${head}`);
      }
      return tags;
    }

    // ---------- main grid render ----------
    function drawGrid() {
      const ctx = elGrid.getContext('2d');
      const t = state.t;
      const ts = TS[t];
      ctx.clearRect(0, 0, elGrid.width, elGrid.height);

      const hl = state.highlightHead;
      const hlChain = hl >= 0 ? new Set(CFG.v_chains_by_head[String(hl)].chain) : null;
      const hlExcluded = hl >= 0 ? new Set(CFG.v_chains_by_head[String(hl)].excluded) : null;

      const GAP = 2;        // gap between adjacent cells (1px on each side from each cell)
      const FRAME = 5;      // head-color outer frame thickness

      for (let ci = 0; ci < NUM_CORES; ci++) {
        const [px, py] = STATIC.core_positions[ci];
        // outer cell (with gap)
        const ox = px * GRID_CELL + GAP / 2;
        const oy = py * GRID_CELL + GAP / 2;
        const ow = GRID_CELL - GAP;
        const oh = GRID_CELL - GAP;
        // inner content area (after head-color frame)
        const ix = ox + FRAME;
        const iy = oy + FRAME;
        const iw = ow - 2 * FRAME;
        const ih = oh - 2 * FRAME;

        let dim = false;
        if (hl >= 0 && !hlChain.has(ci) && !hlExcluded.has(ci)) dim = true;

        // outer frame colored by current head (cores in the same head share color)
        const headForBorder = ts.cells_head[ci];
        if (dim) ctx.globalAlpha = 0.20;
        ctx.fillStyle = headForBorder !== null ? headColor(headForBorder) : '#1a1a25';
        ctx.fillRect(ox, oy, ow, oh);
        // inner action-colored area
        ctx.fillStyle = cellColor(ci, t, state.cellMode);
        ctx.fillRect(ix, iy, iw, ih);
        ctx.globalAlpha = 1.0;

        // highlighted head's chain accent (drawn just inside the frame)
        if (hl >= 0 && hlChain && hlChain.has(ci)) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.strokeRect(ix + 0.5, iy + 0.5, iw - 1, ih - 1);
        }

        if (dim) continue;

        // ALL labels inside the inner area so they don't sit on the head frame.
        const PAD = 3;
        const tlX = ix + PAD, tlY = iy + PAD;
        const trX = ix + iw - PAD, trY = iy + PAD;
        const blX = ix + PAD, blY = iy + ih - PAD;
        const brX = ix + iw - PAD, brY = iy + ih - PAD;
        const cX  = ix + iw / 2, cY = iy + ih / 2;

        // top-left: core index
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`c${ci}`, tlX, tlY);

        if (ts.cells_compute[ci] === COMP_COMPUTE) {
          // top-right: head·pair
          ctx.font = '9px ui-monospace, monospace';
          ctx.fillStyle = 'rgba(255,255,255,0.6)';
          ctx.textAlign = 'right'; ctx.textBaseline = 'top';
          ctx.fillText(`H${ts.cells_head[ci]}·p${ts.cells_pair_idx[ci]}`, trX, trY);

          // direction arrow above Q·K
          const isLight = ts.cells_is_light[ci] === 1;
          ctx.font = 'bold 13px ui-monospace, monospace';
          ctx.fillStyle = isLight ? '#3b82f6' : '#f97316';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(isLight ? 'fwd →' : '← rev', cX, cY - 16);

          // big Q·K
          ctx.fillStyle = 'rgba(255,255,255,0.95)';
          ctx.font = 'bold 15px ui-monospace, monospace';
          ctx.fillText(`Q${ts.cells_q[ci]}·K${ts.cells_k[ci]}`, cX, cY + 4);
        } else {
          ctx.fillStyle = 'rgba(255,255,255,0.4)';
          ctx.font = 'bold 11px ui-monospace, monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText('IDLE', cX, cY);
        }

        // bottom-left, stacked: V role (top) over K role (bottom)
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';

        const head = ts.cells_head[ci];
        if (head !== null) {
          const r = STATIC.v_role_per_head[ci][String(head)];
          let label = '', color = 'rgba(255,255,255,0.7)', font = '9px ui-monospace, monospace';
          if (r === 1)      { label = `★ V-INJ:H${head}`; color = '#4ade80'; font = 'bold 10px ui-monospace, monospace'; }
          else if (r === 2) { label = `V-MID:H${head}`;   color = '#93c5fd'; }
          else if (r === 3) { label = `V-SNK:H${head}`;   color = '#d8b4fe'; }
          else              { label = `V-EX:H${head}`;    color = '#fcd34d'; }
          ctx.font = font;
          ctx.fillStyle = color;
          ctx.fillText(label, blX, blY - 11);
        }

        const isKInj = STATIC.k_role[ci] === 1;
        if (isKInj) {
          // red filled badge with white text — visible regardless of cell color
          const label = '★ K-INJ';
          ctx.font = 'bold 11px ui-monospace, monospace';
          const tw = ctx.measureText(label).width;
          const padX = 4, padY = 2, badgeH = 14;
          const bx = blX, by = blY - badgeH;
          ctx.fillStyle = '#ef4444';
          ctx.fillRect(bx - 1, by, tw + 2 * padX, badgeH);
          ctx.fillStyle = '#ffffff';
          ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
          ctx.fillText(label, bx + padX - 1, by + badgeH / 2);
        } else {
          ctx.font = '9px ui-monospace, monospace';
          ctx.fillStyle = 'rgba(255,255,255,0.55)';
          ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
          ctx.fillText('K-RCV', blX, blY);
        }
      }
    }

    // ---------- overlay (chain edges + mcast glow) ----------
    function drawOverlay() {
      const ctx = elOverlay.getContext('2d');
      ctx.clearRect(0, 0, elOverlay.width, elOverlay.height);

      // chain edges for highlighted head
      if (state.showEdges && state.highlightHead >= 0) {
        const info = CFG.v_chains_by_head[String(state.highlightHead)];
        const chain = info.chain;
        if (chain.length >= 2) {
          ctx.strokeStyle = headColor(state.highlightHead, 0.9);
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          for (let i = 0; i < chain.length; i++) {
            const ci = chain[i];
            const [px, py] = STATIC.core_positions[ci];
            const cx = px * GRID_CELL + GRID_CELL / 2;
            const cy = py * GRID_CELL + GRID_CELL / 2;
            if (i === 0) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
          }
          ctx.stroke();
          // arrowheads at each step
          for (let i = 1; i < chain.length; i++) {
            const a = STATIC.core_positions[chain[i-1]];
            const b = STATIC.core_positions[chain[i]];
            const ax = a[0] * GRID_CELL + GRID_CELL/2, ay = a[1] * GRID_CELL + GRID_CELL/2;
            const bx = b[0] * GRID_CELL + GRID_CELL/2, by = b[1] * GRID_CELL + GRID_CELL/2;
            const dx = bx - ax, dy = by - ay;
            const len = Math.hypot(dx, dy);
            const ux = dx/len, uy = dy/len;
            const tx = bx - ux * 14, ty = by - uy * 14;
            ctx.beginPath();
            ctx.moveTo(bx - ux*4, by - uy*4);
            ctx.lineTo(tx + uy*5, ty - ux*5);
            ctx.lineTo(tx - uy*5, ty + ux*5);
            ctx.closePath();
            ctx.fillStyle = headColor(state.highlightHead, 0.9);
            ctx.fill();
          }
        }
      }
    }

    // ---------- mini strip ----------
    function drawStrip() {
      const ctx = elStrip.getContext('2d');
      ctx.clearRect(0, 0, elStrip.width, elStrip.height);
      for (let t = 0; t < T_MAX; t++) {
        for (let ci = 0; ci < NUM_CORES; ci++) {
          ctx.fillStyle = cellColor(ci, t, state.cellMode);
          ctx.fillRect(t * STRIP_W, ci * STRIP_H, STRIP_W, STRIP_H);
        }
      }
      // current t marker
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(state.t * STRIP_W + 0.5, 0.5, STRIP_W - 1, NUM_CORES * STRIP_H - 1);
      // pair boundaries
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      for (let p = 1; p < 3; p++) {
        const x = p * T_PER_PAIR * STRIP_W;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, NUM_CORES * STRIP_H); ctx.stroke();
      }
    }

    // ---------- stats ----------
    function updateStats() {
      const t = state.t;
      const ts = TS[t];
      const pair = Math.floor(t / T_PER_PAIR);
      const lightOrHeavy = ts.pair_local_t === 0 ? 'light Q (start)' :
        ts.pair_local_t <= 10 ? 'fwd+rev' : 'rev only';
      document.getElementById('stat-t').textContent = t;
      document.getElementById('stat-t-sub').textContent = `pair ${pair}/3 · ${lightOrHeavy}`;
      document.getElementById('stat-phase').textContent = ts.phase;
      const phaseSub = ts.phase === 1 ? 'K0 only' : ts.phase === 2 ? 'fwd+rev K' : 'rev K only';
      document.getElementById('stat-phase-sub').textContent = phaseSub;
      const phaseCard = document.getElementById('stat-phase-card');
      phaseCard.className = 'stat phase' + ts.phase;
      document.getElementById('stat-plt').textContent = ts.pair_local_t;
      document.getElementById('stat-active').textContent = ts.active_cores;
      document.getElementById('stat-active-sub').textContent = `/ ${NUM_CORES}` + (ts.active_cores < NUM_CORES ? ' · padded' : '');
      document.getElementById('stat-kr').textContent = ts.k_set.length;
      document.getElementById('stat-kr-sub').textContent = ts.k_set.map(k => 'K' + k).join(', ');
      document.getElementById('stat-vr').textContent = ts.v_reads_total;
    }

    // ---------- legend ----------
    function updateLegend() {
      const el = document.getElementById('legend');
      const m = state.cellMode;
      let items = [];
      if (m === 'k_action') {
        items = [
          ['K_DRAM_READ (injector)', KACT_COLORS[K_DRAM]],
          ['K_AWAIT_MCAST (receivers)', KACT_COLORS[K_AWAIT]],
          ['IDLE', KACT_COLORS[K_IDLE]],
        ];
      } else if (m === 'v_action') {
        items = [
          ['V_DRAM_READ (chain inj)', VACT_COLORS[V_DRAM]],
          ['V_FORWARD (chain mid)', VACT_COLORS[V_FORWARD]],
          ['V_SNK_WAIT (chain sink)', VACT_COLORS[V_SNK_WAIT]],
          ['V_DRAM_LOCAL (excluded)', VACT_COLORS[V_DRAM_LOCAL]],
          ['IDLE', VACT_COLORS[V_IDLE]],
        ];
      } else if (m === 'role') {
        items = [
          ['K-INJ', ROLE_COLORS.kInj],
          ['V-INJ', ROLE_COLORS.vInj],
          ['V-MID', ROLE_COLORS.vMid],
          ['V-SNK', ROLE_COLORS.vSnk],
          ['V excluded', ROLE_COLORS.vNone],
          ['(K-RCV default)', ROLE_COLORS.kRcv],
        ];
      } else if (m === 'compute') {
        items = [['COMPUTE', COMP_COLORS[COMP_COMPUTE]], ['IDLE', COMP_COLORS[COMP_IDLE]]];
      } else if (m === 'phase') {
        items = [['phase 1', PHASE_COLORS[1]], ['phase 2', PHASE_COLORS[2]], ['phase 3', PHASE_COLORS[3]]];
      } else if (m === 'q' || m === 'k') {
        el.innerHTML = `<div>${m.toUpperCase()} chunk index: <span style="display:inline-block;width:240px;height:13px;border-radius:3px;margin:0 6px;vertical-align:middle;background:linear-gradient(to right,hsl(0,70%,56%),hsl(60,70%,56%),hsl(120,70%,56%),hsl(180,70%,56%),hsl(240,70%,56%),hsl(300,70%,56%),hsl(360,70%,56%))"></span> 0 → 19</div>`;
        return;
      } else if (m === 'head') {
        el.innerHTML = `<div>Head index 0..31 mapped to distinct hues</div>`;
        return;
      }
      el.innerHTML = items.map(([l, c]) => `<div><span class="swatch" style="background:${c}"></span>${l}</div>`).join('');
    }

    // ---------- tooltip ----------
    function showTip(e, html) {
      elTip.innerHTML = html;
      elTip.classList.add('show');
      elTip.style.left = Math.min(e.clientX + 14, window.innerWidth - 340) + 'px';
      elTip.style.top  = Math.min(e.clientY + 14, window.innerHeight - 200) + 'px';
    }
    function hideTip() { elTip.classList.remove('show'); }

    elGrid.addEventListener('mousemove', (e) => {
      const rect = elGrid.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const px = Math.floor(x / GRID_CELL);
      const py = Math.floor(y / GRID_CELL);
      // find core at (px, py)
      let ci = -1;
      for (let i = 0; i < NUM_CORES; i++) {
        if (STATIC.core_positions[i][0] === px && STATIC.core_positions[i][1] === py) { ci = i; break; }
      }
      if (ci < 0) { hideTip(); return; }
      const t = state.t;
      const ts = TS[t];
      const cellInfo = ts.cells_compute[ci];
      const head = ts.cells_head[ci];
      const tags = roleTagsFor(ci, t);
      const vRoles = STATIC.v_role_per_head[ci];
      const allVRoles = Object.keys(vRoles).map(h => {
        const r = vRoles[h];
        const lbl = r === 1 ? 'INJ' : r === 2 ? 'MID' : r === 3 ? 'SNK' : 'EX';
        return `H${h}:${lbl}`;
      }).join(', ');
      const kActLabel = {[K_DRAM]: 'K_DRAM_READ', [K_AWAIT]: 'K_AWAIT_MCAST',
                        [K_MCAST_SEND]: 'K_MCAST_SEND', [K_RECV_DONE]: 'K_RECV_DONE',
                        [K_IDLE]: 'IDLE', [K_NONE]: '—'}[ts.cells_k_action[ci]];
      const vActLabel = {[V_DRAM]: 'V_DRAM_READ', [V_FORWARD]: 'V_FORWARD',
                        [V_AWAIT]: 'V_AWAIT_RECV', [V_SNK_WAIT]: 'V_SNK_WAIT',
                        [V_IDLE]: 'IDLE', [V_DRAM_LOCAL]: 'V_DRAM_LOCAL',
                        [V_NONE]: '—'}[ts.cells_v_action[ci]];
      let body = `<div class="ttbig">core ${ci} · phys (${px},${py})</div>`;
      body += `<div class="row"><span class="k">K role</span><span class="v">${tags[0]}</span></div>`;
      body += `<div class="row"><span class="k">V roles (per head)</span><span class="v">${allVRoles || '—'}</span></div>`;
      const pairs = STATIC.pairs_per_core[ci];
      const pairIdx = ts.cells_pair_idx[ci];
      body += `<div class="row"><span class="k">global Q count</span><span class="v">${STATIC.global_q_count[ci]}</span></div>`;
      body += `<div class="row"><span class="k">pairs</span><span class="v">${pairs.length}</span></div>`;
      const pairList = pairs.map((p, i) => {
        const tag = `H${p.head}(l=${p.l},h=${p.h})`;
        return i === pairIdx ? `<b style="color:var(--accent)">${tag}</b>` : tag;
      }).join(', ');
      body += `<div class="row"><span class="k">all pairs</span><span class="v">${pairList}</span></div>`;
      if (cellInfo === COMP_COMPUTE) {
        const p = pairs[pairIdx];
        body += `<div class="row"><span class="k">current pair</span><span class="v">${pairIdx+1}/${pairs.length} · H${p.head} (l=${p.l},h=${p.h})</span></div>`;
        body += `<div class="row"><span class="k">phase · pair_local_t</span><span class="v">${ts.phase} · ${ts.pair_local_t}</span></div>`;
        body += `<div class="row"><span class="k">compute</span><span class="v">Q${ts.cells_q[ci]}·K${ts.cells_k[ci]}·V${ts.cells_v[ci]} (${ts.cells_is_light[ci] ? 'light' : 'heavy'})</span></div>`;
        body += `<div class="row"><span class="k">K action</span><span class="v">${kActLabel}</span></div>`;
        body += `<div class="row"><span class="k">V action (H${head})</span><span class="v">${vActLabel}</span></div>`;
      } else {
        body += `<div class="row"><span class="k">status</span><span class="v">IDLE (padded)</span></div>`;
      }
      showTip(e, body);
    });
    elGrid.addEventListener('mouseleave', hideTip);

    // strip click jumps to t
    elStrip.addEventListener('click', (e) => {
      const rect = elStrip.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const t = Math.max(0, Math.min(T_MAX - 1, Math.floor(x / STRIP_W)));
      state.t = t;
      document.getElementById('t-slider').value = t;
      render();
    });

    // controls
    function applyMode(dist, layout) {
      const d = dist === 'head_grouped' ? 'grouped' : 'flat';
      const l = layout === 'col_major' ? 'col' : 'row';
      const key = `${d}_${l}`;
      if (!ENVELOPE.payloads[key]) return;
      loadMode(key);
      render();
    }
    document.querySelectorAll('.modebtn[data-dist]').forEach(b => {
      b.addEventListener('click', () => {
        const layout = CURRENT_MODE.endsWith('col') ? 'col_major' : 'row_major';
        applyMode(b.dataset.dist, layout);
      });
    });
    document.querySelectorAll('.modebtn[data-layout]').forEach(b => {
      b.addEventListener('click', () => {
        const dist = CURRENT_MODE.startsWith('grouped') ? 'head_grouped' : 'flat';
        applyMode(dist, b.dataset.layout);
      });
    });

    document.getElementById('cell-mode').addEventListener('change', (e) => {
      state.cellMode = e.target.value; render();
    });
    document.getElementById('t-slider').addEventListener('input', (e) => {
      state.t = parseInt(e.target.value, 10); render();
    });
    document.getElementById('head-select').addEventListener('change', (e) => {
      state.highlightHead = parseInt(e.target.value, 10); render();
    });
    document.getElementById('show-edges').addEventListener('change', (e) => {
      state.showEdges = e.target.checked; render();
    });
    const playBtn = document.getElementById('play-pause');
    playBtn.addEventListener('click', () => {
      state.playing = !state.playing;
      playBtn.textContent = state.playing ? '⏸ pause' : '▶ play';
      if (state.playing) {
        state.playInterval = setInterval(() => {
          state.t = (state.t + 1) % T_MAX;
          document.getElementById('t-slider').value = state.t;
          render();
        }, 350);
      } else {
        clearInterval(state.playInterval);
      }
    });
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowRight') {
        state.t = Math.min(T_MAX - 1, state.t + 1);
        document.getElementById('t-slider').value = state.t;
        render();
      } else if (e.key === 'ArrowLeft') {
        state.t = Math.max(0, state.t - 1);
        document.getElementById('t-slider').value = state.t;
        render();
      } else if (e.key === ' ') {
        e.preventDefault();
        playBtn.click();
      }
    });

    function render() {
      document.getElementById('t-display').textContent = state.t;
      drawGrid();
      drawOverlay();
      drawStrip();
      updateStats();
      updateLegend();
    }
    render();
  </script>
</body>
</html>
"""


def run_sim(cfg: dict):
    core_work = build_core_work(cfg)
    v_chains = build_v_chains(cfg, core_work)
    k_inj = select_k_injector(core_work)
    static_roles = compute_static_roles(cfg, core_work, v_chains, k_inj)
    timestamps = compute_per_t_actions(cfg, core_work, v_chains, static_roles, k_inj)
    return core_work, v_chains, static_roles, timestamps, k_inj


def main():
    here = Path(__file__).parent
    csv_path = here / "idea-ring_iter0_timestamps.csv"

    # 1) Verify against the existing 100-core CSV oracle
    print(f"[run] {CSV_ORACLE['name']}")
    cw, vc, sr, ts, ki = run_sim(CSV_ORACLE)
    verify_against_csv(CSV_ORACLE, ts, csv_path)
    print(f"  cores={CSV_ORACLE['num_cores']} timestamps={len(ts)} k_inj={ki}")

    # 2) Build all 4 mla-100k variants for interactive toggling in the HTML
    canonical = "grouped_col"
    payloads: dict[str, dict] = {}
    for distribution in ("flat", "head_grouped"):
        for layout in ("row_major", "col_major"):
            cfg = dict(MLA_100K, distribution=distribution, layout=layout)
            d_short = "grouped" if distribution == "head_grouped" else "flat"
            l_short = "col" if layout == "col_major" else "row"
            key = f"{d_short}_{l_short}"
            print(f"[run] {cfg['name']} {distribution}/{layout}")
            cw, vc, sr, ts, ki = run_sim(cfg)
            if key == canonical:
                verify_mla_100k(cfg, cw, vc, sr, ts, ki)
            payloads[key] = build_payload(cfg, cw, vc, sr, ts, ki)
            print(f"  cores={cfg['num_cores']} timestamps={len(ts)} k_inj={ki}")

    envelope = {"default": canonical, "payloads": payloads}
    payload_json = json.dumps(envelope, separators=(",", ":"))
    print(f"[payload] {len(payload_json):,} bytes ({len(payloads)} variants)")
    out = here / "reverse_k_visualization.html"
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)
    out.write_text(html)
    print(f"[write] {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
