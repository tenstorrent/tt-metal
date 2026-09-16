# SUPERSEDED (2026-09-16 pm) — see [MULTIHOP_SWA_HALO.md](MULTIHOP_SWA_HALO.md) §11

> Hops now **time-share** the 2 links (sequential EDM channel hand-off), so 2k at CP=8 runs:
> 131.3 ms TTFT, 4-hop accuracy test green. The link count is no longer a limit. The halo
> payload was invariant at ~2 MiB/device/layer all along, so links were never the bottleneck;
> the multi-hop adder is a flat 21.5 ms regardless of hop count. Kept for the probe results,
> which still stand: this box physically has 2 chip-to-chip links.

# Why 2k prefill chunks need 4 fabric links, and we have 2

**One-line:** small prefill chunks need the sliding-window halo fetched from several CP neighbours at
once, each concurrent fetch needs its own fabric link, and a BH Galaxy gives us 2 — so 4k works and
2k does not, at CP=8.

## The mechanism

Gemma4's sliding window is **1024 tokens**, and under context parallelism each rank holds a
`chunk / CP` slab of the sequence. A rank whose slab is narrower than the window has to pull the
missing history from its predecessors — one **hop** per predecessor:

| CP | chunk | per-rank slab | hops needed | status |
|---:|---:|---:|---:|---|
| 8 | 8192 | 1024 | 1 | shipped today |
| 8 | **4096** | 512 | **2** | **works — 1.40x better TTFT, PCC 0.9997 vs torch** |
| 8 | 2048 | 256 | **4** | ~~blocked~~ **works via link sharing — 131.3 ms TTFT** |
| 4 | **2048** | 512 | **2** | **works — best TTFT measured** |

**Why each hop needs its own link:** one worker core per (fabric link, direction) owns an EDM
channel. Two workers sending on the same channel is not a supported configuration — the first sender
stalls mid-transfer and the op deadlocks (we hit this, it is not theoretical). Same convention is
visible in `all_gather_async` and stated outright in `reduce_scatter_minimal_direct`.

## We have 2 links, and none of the easy knobs change that

Probe (20 lines, no pytest, ~40 s/config — `scratchpad/probe_links.py`):
`len(ttnn.get_forwarding_link_indices(src_node, dst_node))`

| thing tried | result |
|---|---|
| baseline, auto-discovery | **2** links, both mesh axes, every hop distance |
| `FABRIC_1D` / `FABRIC_1D_RING` / `FABRIC_2D` | **2** — fabric type is irrelevant here |
| `tt-smi -glx_reset`, then re-probe | **2** — links are not degraded, nothing to recover |
| mesh-graph descriptor patched to `channels { count: 4 }` | **rejected**: `TT_FATAL: Expected 4 eth links from physical chip 0 to physical chip 1` |

The stock single-galaxy descriptors all declare `channels { count: 2 }`, and the hardware agrees:
of 128 chip/direction pairs only **4** report 4 eth channels (the edge/wrap links on D13/D14/D17/D18).
So the "4 eth channels but only 2 routing planes" warning in the logs is about those few pairs, not
a config we are leaving on the table. **2 is physical for chip-to-chip links on this box.**

## Measured TTFT (chunk-0 device time = what a prompt ≤ chunk pays), ctx 32k

| mesh | chunk 8192 | chunk 4096 | chunk 2048 |
|---|---:|---:|---:|
| **8x4** (CP8/TP4, production) | 242.7 ms | **174.2 ms (1.39x)** | **131.3 ms (1.85x)** |
| **4x8** (CP4/TP8) | 335.3 ms | 185.2 ms | **118.3 ms (2.05x vs 8x4/8192)** |

## Three alternatives

1. **Ship 4k at 8x4 now.** 1.40x better TTFT for prompts ≤4096, numerically gated (PCC 0.99968 vs
   torch on the SP8+linear path the model runs), one-hop path unchanged. Nothing further needed.
2. **Run short-prompt traffic at 4x8 and use 2k.** 118 ms, the best number measured, **no code change
   beyond what is already in** — at CP=4 a 2048 chunk only needs 2 hops. Cost: 4x8 is ~25% slower at
   long context, and mesh shape is fixed at model load, so this is a separate instance, not a
   per-request switch.
3. **Make hops share links** (~1–1.5 h): one worker per link carries `ceil(hops/links)` hops
   *sequentially* instead of one hop per worker. Same bytes, same channels, roughly doubles halo
   latency (a few ms/chunk against a ~50 ms/chunk saving). Needs `hops_per_worker` as a compile-time
   arg, per-hop runtime arg blocks, and the packet route moved from compile-time to per-hop. Unlocks
   2k at CP=8 — and any future window/chunk ratio — without touching fabric or hardware.

Nothing here needs a fabric change, a firmware change, or more links.
