#!/usr/bin/env python3
"""Tier-1 chunk-skip win forecast for ttnn.experimental.topk_large_indices.

Host-side exact simulation of the column-parallel leaf loop with a
data-dependent chunk skip:
  skip chunk  <=>  chunk_max <= running K-th survivor min
Two threshold variants:
  cons : K = llk_k   (the full LLK survivor window -- what the DST holds)
  aggr : K = user k  (sound: elements below the leaf's running user-k-th
                      can never enter the global top-k after tree merges)

Factory model reproduced exactly from
ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/
topk_large_indices_program_factory.cpp:
  snap_to_llk_target_k (line 38): k<=512 -> 512, <=1024 -> 1024, else 2048
  compute_model_column_split_config (line 358): a x b rectangle search,
    cost(P) = 2*ceil(chunks/P) + ceil(log2 P), P in [2, min(chunks, 128)],
    ties prefer fewer cores; grid = 13 x 10 on p150a.
  compute_slice_runtime (line 500): base = chunks//P, extra = chunks%P,
    slice s gets base + (s < extra); contiguous chunk ranges.
"""

import math
import numpy as np

GRID_X, GRID_Y = 13, 10          # p150a compute_with_storage_grid_size (factory comment line 329)
MAX_SLICES = 128                  # factory line 330 (post P-cap-raise)
AICLK_GHZ = 1.35                  # Blackhole p150a AI clock assumption

# ---- measured constants from the campaign (mission brief) ----
# leaf chunk ~ 2 merge units (process_chunk ~1 + merge+rebuild ~1); tree level ~1 unit
# skip decision: tensix_sync + cross-lane fold + TRISC Dst readback ~81 cyc best
# SFPU max class ~2-4 cyc/vec; unpack-to-dest ~3.9 cyc/vec + ~91-123 cyc/tile handshake
CYC_READBACK = 81
CYC_MAXFOLD = 16 * 3 + 5 * 4      # 16 vecs/512elem @3cyc + ~5 fold steps -> ~68
CYC_UNPACK_TILE = 130             # 1-tile unpack-to-dest incl handshake (SORTING.md 0a-bis)


def bf16_round(x):
    """Round float array to bfloat16 (round-to-nearest-even), return float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u2 = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u2.astype(np.uint32).view(np.float32)


def tree_levels(p):
    l = 0
    while (1 << l) < p:
        l += 1
    return l


def snap_llk_k(k):
    return 512 if k <= 512 else (1024 if k <= 1024 else 2048)


def pick_P(chunks):
    """Exact replica of compute_model_column_split_config's rectangle search."""
    ceiling = min(chunks, MAX_SLICES)
    best_cost, best = None, None
    for a in range(1, GRID_X + 1):
        for b in range(1, GRID_Y + 1):
            p = a * b
            if p < 2 or p > ceiling:
                continue
            cost = 2 * math.ceil(chunks / p) + tree_levels(p)
            if best_cost is None or cost < best_cost or (cost == best_cost and p < best[0]):
                best_cost, best = cost, (p, a, b)
    if best is None or best_cost >= 2 * chunks:
        return None, None, None
    return best[0], best[1:], best_cost


def slice_chunk_counts(chunks, P):
    base, extra = chunks // P, chunks % P
    return [base + (1 if s < extra else 0) for s in range(P)]


def gen_row(dist, n, rng):
    if dist == "normal":
        x = rng.standard_normal(n)
    elif dist == "uniform":
        x = rng.uniform(0.0, 1.0, n)
    elif dist == "softmax":
        z = rng.standard_normal(n)
        z -= z.max()
        e = np.exp(z)
        x = e / e.sum()
    elif dist == "ascending":
        x = np.sort(rng.standard_normal(n))     # adversarial, tie-free ascending
    elif dist == "allequal":
        x = np.full(n, 1.0)
    else:
        raise ValueError(dist)
    return bf16_round(x)


def simulate_row(row, llk_k, user_k, P):
    """Stream chunks per slice exactly as the leaf loop does.

    Returns per-slice (chunks, later, skip_cons, skip_aggr).
    Threshold evolution is unaffected by skips (skipped elements are <= the
    threshold that gated them), so one exact survivor stream serves both rules.
    """
    chunks = len(row) // llk_k
    assert chunks * llk_k == len(row)
    counts = slice_chunk_counts(chunks, P)
    out = []
    start = 0
    for c in counts:
        sl = row[start * llk_k:(start + c) * llk_k]
        start += c
        surv = np.sort(sl[:llk_k])[::-1]          # chunk 0 seeds the window
        sc = sa = 0
        for j in range(1, c):
            chunk = sl[j * llk_k:(j + 1) * llk_k]
            cmax = chunk.max()
            if cmax <= surv[llk_k - 1]:
                sc += 1
            if cmax <= surv[user_k - 1]:
                sa += 1
            merged = np.concatenate([surv, chunk])
            surv = np.sort(merged)[::-1][:llk_k]
        out.append((c, c - 1, sc, sa))
    return out


def leaf_units(c):
    """Baseline leaf cost in merge units: chunk0 sort ~1, later chunks ~2."""
    return 0.0 if c == 0 else 1.0 + 2.0 * (c - 1)


def tree_time(leaf_us, unit_us):
    """Pairwise merge tree completion time (root = slice 0)."""
    P = len(leaf_us)
    done = list(leaf_us)
    L = tree_levels(P)
    for level in range(L):
        step = 1 << level
        for i in range(0, P, step * 2):
            j = i + step
            if j < P:
                done[i] = max(done[i], done[j]) + unit_us
    return done[0]


def analytic_pskip(c, M, K):
    """P(chunk c+1 skippable | iid continuous): top-K of union all from prev.
    prev = c*M elements, chunk = M elements -> C(cM,K)/C(cM+M,K)."""
    lp = 0.0
    for i in range(K):
        lp += math.log(c * M - i) - math.log(c * M + M - i)
    return math.exp(lp)


CELLS = [
    # name, user_k, n, measured_us, routed(bool)
    ("k32@65536 (routed)", 32, 65536, 36.6, True),
    ("k64@65536 (routed)", 64, 65536, 42.4, True),
    ("k512@262144", 512, 262144, 23.3, False),
    ("k512@65536", 512, 65536, 13.1, False),
    ("k2048@262144", 2048, 262144, 49.6, False),
]
INNER_K512_65536_US = 13.1        # routed cells wrap this same inner op shape
DISTS = ["normal", "uniform", "softmax", "ascending", "allequal"]
SEEDS = 5


def main():
    print(f"grid {GRID_X}x{GRID_Y}, max_slices {MAX_SLICES}, aiclk {AICLK_GHZ} GHz")
    results = {}
    for name, k, n, meas_us, routed in CELLS:
        llk_k = snap_llk_k(k)
        chunks = n // llk_k
        P, rect, model_cost = pick_P(chunks)
        counts = slice_chunk_counts(chunks, P)
        cmax = max(counts)
        levels = tree_levels(P)
        leaf_max = leaf_units(cmax)
        total_units = leaf_max + levels

        # per-unit us: routed cells decompose as fixed routing overhead + inner op
        if routed:
            inner_us = INNER_K512_65536_US
            overhead_us = meas_us - inner_us
        else:
            inner_us = meas_us
            overhead_us = 0.0
        unit_us = inner_us / total_units
        unit_cyc = unit_us * AICLK_GHZ * 1e3

        tiles = 1 if llk_k <= 1024 else 2
        cyc_test_full = CYC_READBACK + CYC_MAXFOLD + tiles * CYC_UNPACK_TILE  # skipped chunk pays this
        cyc_test_extra = CYC_READBACK + CYC_MAXFOLD                            # unpack reused if not skipped
        us_test_full = cyc_test_full / (AICLK_GHZ * 1e3)
        us_test_extra = cyc_test_extra / (AICLK_GHZ * 1e3)

        # reader floor per slice (Tier-1 leaves fetch unchanged)
        bytes_slice = cmax * llk_k * 2
        us_noc = bytes_slice / 64.0 / (AICLK_GHZ * 1e3)
        us_dram_total = n * 2 / (400e9) * 1e6   # aggregate DRAM stream @ ~400 GB/s

        print(f"\n=== {name}: llk_k={llk_k} chunks={chunks} P={P} rect={rect[0]}x{rect[1]} "
              f"counts={cmax}/{min(counts)} levels={levels}")
        print(f"    factory cost={model_cost}u; finer decomp leaf_max={leaf_max}u + {levels}u = {total_units}u; "
              f"unit={unit_us:.3f}us ({unit_cyc:.0f} cyc); overhead={overhead_us:.1f}us")
        print(f"    skip-test: full {cyc_test_full} cyc = {us_test_full*1e3:.0f}ns, "
              f"extra(no-skip) {cyc_test_extra} cyc = {us_test_extra*1e3:.0f}ns")
        print(f"    reader floor: {bytes_slice} B/slice = {us_noc:.3f}us NoC-class; "
              f"aggregate DRAM ~{us_dram_total:.2f}us")
        # analytic iid skip probabilities per chunk position
        for K, tag in [(llk_k, "cons"), (k, "aggr")]:
            probs = [analytic_pskip(c, llk_k, K) for c in range(1, cmax)]
            print(f"    analytic iid P(skip) [{tag} K={K}]: " +
                  ", ".join(f"c{c+1}={p:.2e}" for c, p in enumerate(probs)))

        for dist in DISTS:
            fr_c, fr_a, pred_c, pred_a, base_chk = [], [], [], [], []
            for seed in range(SEEDS):
                rng = np.random.default_rng(1000 + seed)
                row = gen_row(dist, n, rng)
                slices = simulate_row(row, llk_k, k, P)
                later = sum(s[1] for s in slices)
                fr_c.append(sum(s[2] for s in slices) / max(later, 1))
                fr_a.append(sum(s[3] for s in slices) / max(later, 1))
                for skip_idx, acc in [(2, pred_c), (3, pred_a)]:
                    leaf_t = []
                    for (c, lat, sc, sa) in slices:
                        sk = (c, lat, sc, sa)[skip_idx]
                        t = (1.0 * unit_us if c else 0.0)
                        t += sk * us_test_full
                        t += (lat - sk) * (2.0 * unit_us + us_test_extra)
                        leaf_t.append(t)
                    acc.append(tree_time(leaf_t, unit_us) + overhead_us)
                base_chk.append(tree_time([leaf_units(c) * unit_us for (c, _, _, _) in slices],
                                          unit_us) + overhead_us)
                if dist in ("ascending", "allequal"):
                    break  # deterministic
            results[(name, dist)] = (np.mean(fr_c), np.mean(fr_a),
                                     np.mean(pred_c), np.mean(pred_a), np.mean(base_chk))
            print(f"    {dist:10s} skip cons={np.mean(fr_c)*100:6.2f}%  aggr={np.mean(fr_a)*100:6.2f}%  "
                  f"pred cons={np.mean(pred_c):6.2f}us  aggr={np.mean(pred_a):6.2f}us  "
                  f"(model-baseline {np.mean(base_chk):6.2f}us, measured {meas_us}us)")

    # single-core row-parallel contrast (where the streams are long)
    print("\n=== contrast: row-parallel single-core stream (chunks all on one core) ===")
    for k, n in [(32, 65536), (512, 262144)]:
        llk_k = snap_llk_k(k)
        chunks = n // llk_k
        fr_c = fr_a = 0.0
        rng = np.random.default_rng(7)
        row = gen_row("normal", n, rng)
        surv = np.sort(row[:llk_k])[::-1]
        sc = sa = 0
        for j in range(1, chunks):
            chunk = row[j * llk_k:(j + 1) * llk_k]
            m = chunk.max()
            if m <= surv[llk_k - 1]:
                sc += 1
            if m <= surv[k - 1]:
                sa += 1
            surv = np.sort(np.concatenate([surv, chunk]))[::-1][:llk_k]
        print(f"    k={k} n={n} chunks={chunks}: cons {sc}/{chunks-1} = {sc/(chunks-1)*100:.1f}%,"
              f" aggr {sa}/{chunks-1} = {sa/(chunks-1)*100:.1f}%")


if __name__ == "__main__":
    main()
