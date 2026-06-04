# Fabric per-hop latency — 1D vs 2D (T3K / Galaxy / Blackhole)

## Method

Round-trip latency measured on a **single clock** to avoid cross-chip skew: the source sends a
tiny atomic-inc to the destination over fabric, the destination bounces an atomic-inc straight
back, and the source times `t0→t1` with its own wall clock (`get_timestamp`). Per-hop latency is
the **slope** of `RTT vs hops`, divided by 2 (the round trip traverses `2×hops`). Tiny payload
(256 B) so latency dominates; `--trace-iters 1` so each sample is one clean round trip; p50 over
`--iters` samples. Tool: `bench_unicast` (`--fabric 2d|1d`). Fit: `analyze_hop_latency.py`.

## Headline results — per-hop one-way latency

| Machine | Arch | 2D per-hop | 1D per-hop | 1D vs 2D (same machine) |
|---|---|---|---|---|
| **T3K** | Wormhole | **874 ns** | **711 ns** | **1D ~19% lower** |
| **p150_x4** | Blackhole | **619 ns** | **515 ns** | **1D ~17% lower** |
| **Galaxy** | Wormhole | (not measured) | **734 ns** (near) / 838 ns (1–31) | — |

**Takeaways**
- **1D fabric is ~17–19% lower latency per hop than 2D**, consistently on *both* architectures
  (controlled, same-machine comparisons). The 16 B LowLatency 1D header + simpler routing beats the
  96 B Hybrid 2D path — at the endpoints too (lower fit intercepts).
- **Blackhole hops are ~28–30% faster than Wormhole** (1D: 515 vs ~711–734; 2D: 619 vs 874),
  consistent with BH's 400 Gb/s eth + 2 ERISCs vs WH's 100 Gb/s + 1.
- **Wormhole 1D near-neighbor hop ≈ 711–734 ns** (T3K 711, Galaxy-near 734 — consistent across two
  WH machines).

## Detailed measurements

| Machine | Fabric | hops swept | per-hop one-way | RTT slope (ns/hop) | intercept (ns) | R² |
|---|---|---|---|---|---|---|
| T3K (2×4) | 2D | 1–4 | 874.3 | 1748.5 | +1627 | 0.996 |
| T3K (2×4) | 1D | 1–3¹ | 711.5 | 1423.0 | +837 | 0.992 |
| Galaxy (1×32) | 1D | 1,2,4 | 734.1 | 1468.2 | +704 | 0.9996 |
| Galaxy (1×32) | 1D | 1…31 | 838.4 | 1676.9 | −244² | 0.998 |
| p150_x4 (2×2) | 2D | 1–2³ | 619.1 | 1238.2 | +982 | (2 pts) |
| p150_x4 (1×4) | 1D | 1–3 | 515.4 | 1030.7 | +870 | 0.999 |

AICLK: Wormhole 1000 MHz (cyc==ns), Blackhole 1350 MHz.

**Caveats**
1. T3K 1D 1-hop varies by direction (row EW ≈ 2110 cyc vs column NS ≈ 2415 cyc), which lowers R²
   slightly; sweep `dst 1,2,3` (row 0) for a clean fit.
2. Galaxy 1D is slightly **super-linear**: incremental per-hop grows from ~720 ns (near, hops 1–4)
   to ~915 ns (far, hops 24–31), so the full 1–31 fit has a negative intercept. The 838 ns is the
   1–31 average; the near-neighbor hop is ~734 ns. Likely accumulating buffering/credit latency over
   many hops.
3. BH 2D is a 2-point fit (2×2 mesh diameter is 2 hops); backed by the dst1≈dst2 (both 1-hop)
   agreement rather than a multi-point R².

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

### Galaxy (Wormhole) — 1D over the 1×32 line
```bash
export TT_MESH_GRAPH_DESC_PATH=$PWD/tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto

# 1D: full curve (shows super-linearity)
rm -f galaxy_1d.csv
for d in 1 2 4 8 16 24 31; do $BIN --fabric 1d --src-dev 0:0 --dst-dev 0:$d $COMMON --csv galaxy_1d.csv; done
python $ANALYZE galaxy_1d.csv
# (restrict to `for d in 1 2 3 4` for a near-neighbor per-hop comparable to T3K)

unset TT_MESH_GRAPH_DESC_PATH        # for any native/2D Galaxy runs
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
- **Latency runs use `--trace-iters 1`** (one round trip per sample). For throughput/host-time, use a
  larger `--trace-iters` and a big `--size`; that path is unaffected by these latency columns.
- **`--fabric 1d|2d`** picks `FABRIC_1D` vs `FABRIC_2D`. `--fabric 1d_ring` (FABRIC_1D_RING) also exists
  but is not needed here.
- **Multi-hop 1D needs a line topology.** T3K's native 2×4 gives a usable 1D line along a row (hops
  1–3). Galaxy uses the `galaxy_1x32` descriptor (hops 1–31). BH uses `--mesh-shape 1x4`. A 2×2/2×4
  mesh under 1D only reaches colinear chips.
- **Hops are derived from the live mesh coordinates**, so the `hops` CSV column is authoritative; on a
  ring topology (not used here) hops would fold past N/2.
- CSV columns: `...,hops,rtt_cyc_p50,rtt_cyc_p95,rtt_ns_p50,fabric`. `analyze_hop_latency.py` fits
  `rtt_ns_p50` vs `hops` and prints `per-hop one-way = slope/2`, intercept, R².
