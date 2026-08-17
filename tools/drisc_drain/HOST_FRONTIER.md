# Host pipeline frontier (pruned 2026-08-12)

Target stress: `zones --gx 12 --gy 10 --iters 500 --delay 15` with
`ROLE_SPLIT=1 NO_TRACY=1 NO_CONSUMER=1`.

## Keep (winning path)

- **2 drain threads** (one per D2H socket) → peek + memcpy staging + early ACK
- **3 decode workers** → START/END **pairing** (per-lane stacks) → scratch `PerfDebugRec` batches
- **1 publisher** → AVX-32 NT `publish_batch` into a **single** BroadcastRing
- **16 B records**: `{uint64_t ts, uint32_t dur, uint32_t meta}` — one COMPLETE zone per record
  (full timestamps; `dur == UINT32_MAX` + trailing EXT record is the >3 s escape)
- Pairing fast path: an **adjacent [START, END]** pair is a leaf zone at any depth — pairs positionally,
  no stack traffic (the depth-gated version never fired: lanes live inside FW/KERNEL wrappers)
- Consumer → **one lock-free Tracy item per zone** (`GpuZone`), cached contexts + interned srclocs —
  no serial-queue mutex, no per-zone malloc, no orphan bookkeeping (pairing drops orphan ENDs in decode)
- Ring `warm_pages` + phys-core pin (`PIN_BASE=-1` disables)
- Pin: producers phys 0–1, 3 decode phys, **publisher on a free phys SMT** (not sharing a decode worker — that alone costs ~5 GB/s)
- HOST busy = `max(copy, decode, publish)` with copy = **max across sockets** (not sum)

Honest marker-wire (64-core box, 2026-08-17): **~21–24 GB/s** at **6 host threads**
(decode ~2.0–2.25 ms is the ceiling; publish dropped to ~1.5 ms on 48 MB of zone records).
Same box, same-commit baseline (separate START/END records): ~20–21 GB/s, publish-bound at ~2.3 ms.
`DECODE_WORKERS=4` (7 threads) now reaches **~24–25 GB/s** and is copy-bound (~2.0 ms).
Keep `kStageSlots=8` (4 stalls decode under role-split).

## Live knobs

- `TT_METAL_PERF_DEBUG_DECODE_WORKERS` (default **3**, max 4)
- `TT_METAL_PERF_DEBUG_PIN_BASE` (`-1` disables)
- `TT_METAL_PERF_DEBUG_NO_CONSUMER` / `NO_TRACY`
- `TT_METAL_PERF_DEBUG_RING_RECS`

## Cut (measured dead ends / unused)

- D2H `STREAM_LOAD` / `vmovntdqa` in `read()` — profiler uses peek+memcpy only
- BroadcastRing AVX-512 NT, `PUBLISH_MEMCPY`, Claim/`claim_contiguous`, unfenced coalesce
- Profiler: `DIRECT_RING`, `DUAL_RING`, `INLINE_PUBLISH`, `ONE_PRODUCER`, `TIME_PHASE_PUB`, `DRAIN_FIRST`
- Truncated 8 B timestamps (not a real win)
