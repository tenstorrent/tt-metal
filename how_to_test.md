# Fabric per-hop latency — 1D vs 2D (T3K / Galaxy / Blackhole)

## Method

Round-trip latency measured on a **single clock** to avoid cross-chip skew: the source sends a
tiny atomic-inc to the destination over fabric, the destination bounces an atomic-inc straight
back, and the source times `t0→t1` with its own wall clock (`get_timestamp`). Per-hop latency is
the **slope** of `RTT vs hops`, divided by 2 (the round trip traverses `2×hops`). Tiny payload
(256 B) so latency dominates; `--trace-iters 1` so each sample is one clean round trip; p50 over
`--iters` samples. Tool: `bench_unicast` (`--fabric 2d|1d`). Fit: `analyze_hop_latency.py`.

## Headline — measured per-hop one-way latency

| Machine | Arch | 2D per-hop | 1D per-hop | 1D vs 2D (same machine) |
|---|---|---|---|---|
| **T3K** (2×4) | Wormhole | **874 ns** | **711 ns** | 1D ~19% lower |
| **Galaxy** (4×8 / 1×32) | Wormhole | **907 ns** (h 1–10) | **734 ns** near → ~915 far | 1D ~19% lower (near) |
| **p150_x4** (2×2 / 1×4) | Blackhole | **619 ns** | **515 ns** | 1D ~17% lower |

**Takeaways**
- **1D is ~17–19% lower per hop than 2D**, consistently on all three machines (same-machine
  comparisons). Smaller 16 B LowLatency header + simpler routing vs the 96 B Hybrid 2D path; lower
  endpoint cost too (smaller fit intercepts).
- **Blackhole hops are ~28–30% faster than Wormhole** (1D 515 vs ~711; 2D 619 vs 874–907) — BH's
  400 Gb/s eth + 2 ERISCs vs WH's 100 Gb/s + 1.
- **2D is linear** in hops; **1D is mildly super-linear** over long lines (see §Shape). The two WH
  machines agree on the near-hop 1D value (T3K 711, Galaxy 734).

## Roofline model values (what to put in the CCL perf model)

A roofline is an **optimistic floor** — real perf must never come in *faster* than it. So the model
uses the **lowest** measured per-hop for each (arch, topology), applied **linearly** (β = 0):

| arch | 2D (Mesh/Torus) | 1D (Line/Ring) |
|---|---|---|
| **Wormhole** | **874 ns** | **711 ns** |
| **Blackhole** | **619 ns** | **515 ns** |

- WH-2D uses **874 (T3K)** over **907 (Galaxy)**: ~4% machine-to-machine spread, lower = safer floor
  (endpoints matched: T3K 1627 vs Galaxy 1660 ns, so the slope diff is the only variable). WH-1D uses
  the near-hop **711**, not Galaxy's 734/838.
- The round-trip fit **intercept is excluded** — it's far-end worker turnaround, which has no analog
  in a one-way CCL pipeline-fill term.

### Why we exclude the super-linear (quadratic) term

1D over a long line (Galaxy 1×32) is mildly super-linear. A quadratic fits the 14-point 1D data well:

```
RTT(h)        ≈ 890 + 1390·h + 9.3·h²     (round trip, ns; +890 endpoint is physical/positive)
one-way fill  ≈ 695·h + 4.7·h²            (drop endpoint, halve)  → ~700 ns/hop near, ~840 far
```

It even reconciles both regimes (h=1 → ~700, matching T3K 711; h=31 → ~839, matching the Galaxy
1–31 average). **But it must not go in the roofline:** the `β·h²` term *raises* predicted time at
long distances, pulling the floor *up* so a real run hitting better than ~840 ns/hop at 31 hops would
dip **below** the floor. The super-linearity is real **degradation that lives below the roofline** —
using the low near-hop value linearly keeps real perf safely *above* the floor (slower) at long
distances, which is exactly what a roofline should do. So β = 0. (2D is linear anyway, so this only
ever applied to 1D.)

## Detailed measurements

| Machine | Fabric | hops swept | per-hop one-way | RTT slope (ns/hop) | intercept (ns) | R² | shape |
|---|---|---|---|---|---|---|---|
| T3K (2×4) | 2D | 1–4 | 874.3 | 1748.5 | +1627 | 0.996 | linear |
| T3K (2×4) | 1D | 1–3¹ | 711.5 | 1423.0 | +837 | 0.992 | linear |
| Galaxy (4×8) | 2D | 1–10 (16 pts) | 906.9 | 1813.9 | **+1660** | 0.9995 | **linear** |
| Galaxy (1×32) | 1D | 1–4 (near) | 734.1 | 1468.2 | +704 | 0.9996 | linear |
| Galaxy (1×32) | 1D | 1–31 (14 pts) | 844.1 | 1688.2 | **−584²** | 0.997 | **super-linear** |
| p150_x4 (2×2) | 2D | 1–2³ | 619.1 | 1238.2 | +982 | (2 pts) | linear |
| p150_x4 (1×4) | 1D | 1–3 | 515.4 | 1030.7 | +870 | 0.999 | linear |

AICLK: Wormhole 1000 MHz (cyc==ns), Blackhole 1350 MHz.

## Shape findings & caveats

- **2D is linear** (Galaxy 4×8, hops 1–10): positive intercept (+1660), R²=0.9995, RTT/hop decreases
  monotonically toward the slope (no upward curl). Same-hop chips agree to ~1% (it's hop-count, not
  path/direction). We have **no 2D data beyond 10 hops** (4×8 diameter), so a much larger 2D fabric
  *could* eventually super-linearize — unmeasured.
- **1D is super-linear** over long lines: incremental per-hop grows ~720 ns (near, h≤4) → ~915 ns
  (far, h 24–31). That's why the full 1–31 linear fit has a (²) negative intercept. Quadratic form
  in §Roofline. The onset is ~h=8–10; below that it's effectively linear at the near-hop rate.
- (¹) T3K 1D 1-hop varies by direction (row EW ≈ 2110 cyc vs column NS ≈ 2415 cyc), lowering R²;
  sweep `dst 1,2,3` (row 0) for a clean fit.
- (³) BH 2D is a 2-point fit (2×2 diameter = 2 hops); backed by the dst1≈dst2 (both 1-hop) agreement.

## Reproduction

On **each machine's host**, from the repo root. Build once:

```bash
./build_metal.sh --build-metal-tests
BIN=build/test/tt_metal/tt_fabric/bench_unicast
ANALYZE=tests/tt_metal/tt_fabric/benchmark/collectives/unicast/analyze_hop_latency.py
# common flags: tiny payload + 1 round trip per sample (latency regime)
COMMON="--size 256 --page 256 --send-core 0,0 --recv-core 0,0 --iters 50 --warmup 5 --trace-iters 1"
```

### T3K (Wormhole, native 2×4 = 8 chips) — both modes on native mesh
```bash
unset TT_MESH_GRAPH_DESC_PATH        # ensure native topology

# 2D: dst 1..7 -> hops 1..4
rm -f t3k_2d.csv
for d in 1 2 3 4 5 6 7; do $BIN --fabric 2d --src-dev 0:0 --dst-dev 0:$d $COMMON --csv t3k_2d.csv; done
python $ANALYZE t3k_2d.csv

# 1D: dst 1,2,3 along row 0 -> hops 1,2,3 (multi-hop 1D works on the native mesh)
rm -f t3k_1d.csv
for d in 1 2 3; do $BIN --fabric 1d --src-dev 0:0 --dst-dev 0:$d $COMMON --csv t3k_1d.csv; done
python $ANALYZE t3k_1d.csv
```

### Galaxy (Wormhole, 32 chips) — 1D over a 1×32 line, 2D over the native 4×8
```bash
# 1D: 1x32 line descriptor (full curve shows super-linearity)
export TT_MESH_GRAPH_DESC_PATH=$PWD/tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto
rm -f galaxy_1d.csv
for d in 1 2 4 8 10 12 14 18 20 22 24 28 30 31; do $BIN --fabric 1d --src-dev 0:0 --dst-dev 0:$d $COMMON --csv galaxy_1d.csv; done
python $ANALYZE galaxy_1d.csv
# (restrict to `for d in 1 2 3 4` for the near-neighbor per-hop comparable to T3K)

# 2D: NATIVE 4x8 mesh -> Manhattan hops up to 10 (corner chip 31 = (3,7))
unset TT_MESH_GRAPH_DESC_PATH
rm -f galaxy_2d.csv
for d in 1 2 3 4 5 6 7 15 23 31; do $BIN --fabric 2d --src-dev 0:0 --dst-dev 0:$d $COMMON --csv galaxy_2d.csv; done
python $ANALYZE galaxy_2d.csv
# (check the printed `hops` column is 1..10 monotonic; if it folds, the default came up as a torus)
```

### Blackhole p150_x4 (native 2×2 = 4 chips)
```bash
unset TT_MESH_GRAPH_DESC_PATH

# 2D: native 2x2, dst 1,2,3 -> hops 1,1,2
rm -f bh_2d.csv
for d in 1 2 3; do $BIN --fabric 2d --src-dev 0:0 --dst-dev 0:$d $COMMON --csv bh_2d.csv; done
python $ANALYZE bh_2d.csv

# 1D: bring the 4 chips up as a 1x4 line, dst 1,2,3 -> hops 1,2,3
rm -f bh_1d.csv
for d in 1 2 3; do $BIN --fabric 1d --mesh-shape 1x4 --src-dev 0:0 --dst-dev 0:$d $COMMON --csv bh_1d.csv; done
python $ANALYZE bh_1d.csv
```

### Notes for reproduction
- **Latency runs use `--trace-iters 1`** (one round trip per sample). Throughput/host-time uses a
  larger `--trace-iters` + big `--size`; that path is unaffected by the latency columns.
- **`--fabric 1d|2d`** picks `FABRIC_1D` vs `FABRIC_2D` (`1d_ring` = FABRIC_1D_RING also exists).
- **Multi-hop 1D needs a line topology.** T3K's native 2×4 gives a 1D line along a row (hops 1–3);
  Galaxy uses the `galaxy_1x32` descriptor (hops 1–31); BH uses `--mesh-shape 1x4`.
- **`hops` is derived from live mesh coordinates** and is authoritative; on a torus it folds past N/2.
- CSV: `...,hops,rtt_cyc_p50,rtt_cyc_p95,rtt_ns_p50,fabric`. `analyze_hop_latency.py` fits
  `rtt_ns_p50` vs `hops` → `per-hop one-way = slope/2`, intercept, R².
