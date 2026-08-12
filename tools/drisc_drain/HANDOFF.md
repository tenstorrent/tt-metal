# Handoff: `yusuf/drisc` host DRISC profiler (2026-08-12)

Branch tip when written: `31842011d1f` on `yusuf/drisc` (from `mo/drisc_drain_fast`).
Prior box: Ryzen 5 7600X (6c/12t). Moving to a beefier host to remeasure / push further.

## Goal

Push honest **marker-wire** throughput on:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PERF_DEBUG_PROFILER=1 \
TT_METAL_PERF_DEBUG_ROLE_SPLIT=1 TT_METAL_PERF_DEBUG_RING_RECS=16777216 \
TT_METAL_PERF_DEBUG_NO_TRACY=1 TT_METAL_PERF_DEBUG_NO_CONSUMER=1 \
./build_Release/programming_examples/test_perf_debug_zones --gx 12 --gy 10 --iters 500 --delay 15
```

Rebuild: `ninja -C build_Release tt_metal` (not a full `build_metal.sh` unless needed).

**Honest metric:** `HOST busy = max(copy+ack, decode, publish)`; copy = **max across sockets** (not sum).
Numerator ≈ **45.8 MB** marker-wire (6M zone-markers × 8 B wire). Ring must stay fed under `NO_CONSUMER`.

## Current winning path (do not regress)

| Piece | Detail |
|-------|--------|
| Threads | **2 drain + 3 decode + 1 publisher = 6** (default) |
| Drain | `peek` + `memcpy` + `pop` + early ACK (not `D2HSocket::read`) |
| Decode | scratch `PerfDebugRec` batches → dedicated publisher |
| Publish | AVX-32 NT `publish_batch` → **single** `BroadcastRing` |
| Records | **12 B** `{uint64_t ts, uint32_t meta}` — **full timestamps** (user rejected 8 B truncated “win”) |
| Pin | prod phys 0–1, 3 decode phys, **publisher on a free phys SMT** |
| Staging | `kStageSlots=8` (4 stalls decode badly under role-split) |
| Wire | live-pack device path + `SPSC_SPAN_PACKED_FLAG` |

Prior box: **~20–21 GB/s** marker-wire; publish ~2.2–2.4 ms on ~72 MB of 12 B records is the ceiling.

## Ablation floor (same box)

| Config | Threads | Marker-wire |
|--------|---------|-------------|
| w=4 | 7 | ~same as w=3, not better |
| **w=3 (default)** | **6** | **~20–21 GB/s** |
| w=2 | 5 | ~14–17 (decode-bound ~3.3 ms) |

## Pin lesson (easy to re-break)

Publisher **must not** share an SMT sibling with a live decode worker — that alone dropped ~20 → ~15 GB/s.
Old accidental win: pin plan reserved 4 decode phys while only 3 workers ran, so publisher sat alone on the 4th core’s SMT. Current code intentionally gives publisher the next free phys.

`TT_METAL_PERF_DEBUG_PIN_BASE=-1` disables pinning.

## Dead ends — do not revive as defaults

- Skip ring under `NO_CONSUMER`
- 8 B truncated timestamps
- AVX-512 NT publish, `PUBLISH_MEMCPY`
- D2H `STREAM_LOAD` / `vmovntdqa` in `read()` (profiler doesn’t use `read()`)
- Dual ring / Claim / `claim_contiguous` / `DIRECT_RING` / `INLINE_PUBLISH` / `ONE_PRODUCER` RR
- `TIME_PHASE_PUB` / `DRAIN_FIRST` (copy better, publish colder → net loss)
- Coalesce unfenced batches
- Dedicated whole phys core for publisher while starving decode (with 4 workers)
- Gutting record fidelity for GB/s; dual rings (consumer nightmare)

Deleted knobs (gone from code): `DIRECT_RING`, `DUAL_RING`, `INLINE_PUBLISH`, `ONE_PRODUCER`, `TIME_PHASE_PUB`, `DRAIN_FIRST`, `PUBLISH_AVX512`, `PUBLISH_MEMCPY`, `D2H_STREAM_LOAD`.

Bring-up ablation envs (`ABLATE`, `NOC`, `STALL_ONLY`, `D2H_DISCARD`, …) still exist for `tools/drisc_drain` harness — leave them.

## Key files

- `tt_metal/tools/profiler/perf_debug_profiler.{cpp,hpp}` — host pipeline
- `tt_metal/common/broadcast_ring.hpp` — AVX-32 NT `publish_batch` + `warm_pages`
- `tt_metal/distributed/d2h_socket.cpp` + API hpp — `peek`/`pop`
- `tt_metal/tools/profiler/spsc_marker_decode.hpp`
- `tt_metal/tools/profiler/kernels/drisc_profiler_drain.cpp`
- `tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h`
- `tools/drisc_drain/HOST_FRONTIER.md` — short frontier card
- `tools/drisc_drain/FINDINGS.md` / harness scripts — silicon bring-up (orthogonal)

## What’s next (open)

1. **Remeasure on beefier box** (more phys cores / better DRAM BW) — same env as above; expect pin plan to scale if ≥6 phys cores after producers.
2. **Publish is the GB/s ceiling** with full 12 B records (~2.3 ms). Need ≤~1.83 ms busy for 25 GB/s.
3. **Fewer than 6 threads** needs faster decode so w=2 can hold the band (~25%+ decode speedup).
4. Do **not** cut fidelity or reintroduce dual rings without an explicit product decision.

## Paste into a new Cursor chat

```
Continue perf-debug host DRISC profiler work on branch yusuf/drisc.
Read tools/drisc_drain/HANDOFF.md and tools/drisc_drain/HOST_FRONTIER.md first.
Remeasure zones --gx 12 --gy 10 --iters 500 --delay 15 with ROLE_SPLIT + NO_TRACY + NO_CONSUMER.
Keep 12 B full-timestamp records and the single BroadcastRing. Goal: more marker-wire GB/s and/or fewer than 6 host threads without fake wins.
```
