# pi0.5 all_gather bench: `all_gather_async` vs PR #48301 `ttnn.all_gather` (TP=8)

Benchmarks the pi0.5 prefill all_gather against the new multicast/unicast `ttnn.all_gather`
from PR #48301 (branch `scardoza/new-ag`), on the pi0.5 AG shape (gathered `[1,1,768,2048]`).

Scope: **the 8-device (TP=8) use case only** — that's what pi0.5 prefill runs. (The new op's
`FABRIC_1D_RING` path only spans 4 devices on this hardware, which is out of scope for TP=8;
see below.)

## What it runs — 2 impls × {bf16, bf8}

| test | impl | fabric | mesh | topology | per-device in |
|------|------|--------|------|----------|---------------|
| `test_ag_8dev_async` | `ttnn.experimental.all_gather_async` | `FABRIC_1D` | 1×8 (TP=8) | Ring (op arg)    | `[1,1,768,256]` |
| `test_ag_8dev_ttnn`  | `ttnn.all_gather`                    | `FABRIC_1D` | 1×8 (TP=8) | Linear (derived) | `[1,1,768,256]` |

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

## Measured (device kernel duration, median across clean device IDs, no packet override)

| combination | bf16 | bf8 |
|---|---|---|
| 8-dev `all_gather_async` (ring, `FABRIC_1D`) — pi0.5's path | **82.5 µs** | **66.7 µs** |
| 8-dev `ttnn.all_gather` (linear, `FABRIC_1D`)              | 137.8 µs | 94.8 µs |
| ratio (new / async) | 1.67× | 1.42× |

(Fastest clean chip for the async path: ~74 µs bf16 / ~58 µs bf8 — per-chip ring variance
pulls the median up; the fastest-chip figures match our historical numbers.)

## Why the new op is forced to Linear at TP=8

- **`all_gather_async`** takes `topology=Ring` as an **op argument** and honors it directly, so
  it does a true ring over `FABRIC_1D` across all 8 chips. This is what pi0.5 uses.
- **PR #48301 `ttnn.all_gather`** derives ring-vs-linear from the **fabric config**
  (`ccl_common.cpp::get_axis_topology`: `axis_can_wrap = (fabric_config == FABRIC_1D_RING)`),
  and **ignores** the deprecated op-level `topology` arg. Under `FABRIC_1D` that is `false` →
  **Linear for any device count**, including 8.
- To ring, it needs `FABRIC_1D_RING` **and** the device set to span the full wrapping axis. But
  the 8 visible chips (8–15) are physically a **2×4 subtorus** (`dim_types: [RING, RING]`), and
  only the **size-4** dimension closes into a ring. So `FABRIC_1D_RING` tops out at **4 devices**
  here — it cannot form an 8-device ring. Hence at TP=8 the new op has no ring option.

## Conclusion

For the pi0.5 TP=8 prefill all_gather, PR #48301's `ttnn.all_gather` is **1.4–1.7× slower** than
`all_gather_async` and cannot ring all 8 chips on this Blackhole config (it's stuck on Linear,
since `FABRIC_1D_RING` only spans the 4-chip torus dimension). It is **not** a win for pi0.5;
`all_gather_async` remains the right choice.
