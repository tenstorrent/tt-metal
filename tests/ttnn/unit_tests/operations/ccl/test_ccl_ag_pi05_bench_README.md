# pi0.5 all_gather bench: `all_gather_async` vs PR #48301 `ttnn.all_gather`

Benchmarks the pi0.5 prefill all_gather against the new multicast/unicast `ttnn.all_gather`
from PR #48301 (branch `scardoza/new-ag`), on the pi0.5 AG shape (gathered `[1,1,768,2048]`).

## What it runs — 4 combinations × {bf16, bf8}

| test | impl | fabric | mesh | topology | per-device in |
|------|------|--------|------|----------|---------------|
| `test_ag_8dev_async`  | `ttnn.experimental.all_gather_async` | `FABRIC_1D`      | 1×8 (TP=8) | Ring (op arg)   | `[1,1,768,256]` |
| `test_ag_8dev_ttnn`   | `ttnn.all_gather`                    | `FABRIC_1D`      | 1×8 (TP=8) | Linear (derived)| `[1,1,768,256]` |
| `test_ag_ring4_async` | `ttnn.experimental.all_gather_async` | `FABRIC_1D_RING` | 1×4 (TP=4) | Ring (op arg)   | `[1,1,768,512]` |
| `test_ag_ring4_ttnn`  | `ttnn.all_gather`                    | `FABRIC_1D_RING` | 1×4 (TP=4) | Ring (derived)  | `[1,1,768,512]` |

## How to run (Blackhole galaxy, chips 8–15)

```bash
TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
python -m tracy -p -r -n ag -o /tmp/tracy_ag $(which pytest) \
  tests/ttnn/unit_tests/operations/ccl/test_ccl_ag_pi05_bench.py -s
```

Then read `DEVICE KERNEL DURATION [ns]` for the `all_gather` rows from
`ops_perf_results_*.csv`. **Pick a clean `DEVICE ID`** — a profiler marker bug corrupts
some chips with ~`4.5e12` ns values; filter those out (`< 1e8`) and take the median.

Optional env:
- `PI0_CCL_PACKET_BYTES=8192` — set the fabric max packet payload (Blackhole max).

## Key facts about the two implementations

- **`all_gather_async`** takes `topology=Ring` as an **op argument** and honors it, so it does
  a true ring over `FABRIC_1D` (the 1×8 tray has a physical wrap link). This is what pi0.5 uses.
- **PR #48301 `ttnn.all_gather`** derives ring-vs-linear from the **fabric config**
  (`FABRIC_1D_RING` → ring, `FABRIC_1D` → linear) and **ignores** the (now-deprecated) op-level
  `topology` arg. So it can only ring under `FABRIC_1D_RING`.
- On these chips, `FABRIC_1D` enumerates **8 devices** (1×8) but `FABRIC_1D_RING` enumerates only
  **4 devices** (2×2). Hence the new op's ring path is only reachable at **TP=4** here — an 8-device
  ring with the new op is not currently possible on this hardware.

## Measured (device kernel duration, median, no packet override)

| combination | bf16 | bf8 |
|---|---|---|
| 8-dev `all_gather_async` (ring, FABRIC_1D) | **77.5 µs** | **58.3 µs** |
| 8-dev `ttnn.all_gather` (linear, FABRIC_1D) | 137.6 µs | 94.5 µs |
| 4-dev `all_gather_async` (ring, FABRIC_1D_RING) | **65.7 µs** | **55.2 µs** |
| 4-dev `ttnn.all_gather` (ring, FABRIC_1D_RING) | 103.2 µs | 73.1 µs |

## Conclusion

On this Blackhole config, the PR #48301 `ttnn.all_gather` is **slower than `all_gather_async`**
for the pi0.5 AG shape in every runnable combination — ~1.8× at 8-device `FABRIC_1D` (where it's
forced to linear) and ~1.3–1.6× at 4-device `FABRIC_1D_RING` (ring vs ring). It is **not** a win
for pi0.5's all_gather, and it cannot reproduce pi0.5's 8-device ring (that needs `FABRIC_1D_RING`,
which only spans 4 chips here).

Open question: why `FABRIC_1D_RING` enumerates only 4 devices when the 8-chip tray is physically a
ring — likely a mesh-discovery/config quirk rather than a hard limit.
