# r04-b02-a02: streamed forwarder multicast: peers' fused incs carry a per-source-device bit field in out_ready, so the forwarder multicasts each peer's gathered page to the worker rectangle the moment that page lands, leaving only the last page's multicast + the go flag after the final arrival

## Motivation
The parent r04-b02-a01 (1.2988) moved the gathered stats on-chip by forwarder multicast. The worker side worked:
go -> C_COMB start fell from 0.62 to 0.065 µs. But the forwarder issued all 3 peer-page multicasts (3 x 3328 B) +
the go flag as one serial burst **after** out_ready completed. F_MCAST was 1.21 µs on every shape, with a near
constant spread (1.20-1.23, `r04-b02-a01/ag_out.txt`). Go landed 1.03 / 1.15 µs (first / last worker) after
F_FABRIC end, versus r03-b02-a02's 0.17 / 0.52 µs go plus a 0.60 µs stick read.

The out_ready wait itself is mostly cross-chip skew: F_FABRIC median 2.4-2.7 µs, min 0.8-1.3 µs. So on a typical
chip 2 of the 3 peer pages have been in the forwarder's L1 for a long time before the last one arrives, and the
parent did nothing with them. r04-b02-a01's reflection #1 proposes exactly this overlap. It has not been tried.

## Mechanism
Forwarder kernel only (`device/kernels/dataflow/dit_rmsnorm_forwarder.cpp`), `gather_mcast` path only (RMS, one
forwarder, one round, ring <= 8: all four campaign shapes):
1. **Per-source arrival field.** Each forwarder's fused fabric write+inc increments the peers' out_ready by
   `1 << (field_bits * my_device_index)` instead of 1 (`field_bits` = 8 for ring <= 4, else 4; one round, so a field
   only reaches 1). The fused inc carries a 32-bit value (`NocUnicastAtomicIncFusedCommandHeader::val`), and
   flush=true + same NoC/VC still orders the payload before the inc. The final out_ready value is the OR of the peers'
   fields, so the existing end-of-kernel reset to 0 is unchanged.
2. **Streamed multicast.** After the fabric send, the forwarder multicasts its own page (as in the parent), then
   polls out_ready. Each time a peer's field appears, it multicasts that page from its L1 shard to
   `gathered_base + d * page_stride` on the worker rectangle immediately. Once all peer fields are set, it multicasts
   the page(s) not yet sent, then the go flag on the same NoC/VC (ordered behind the data), then one write barrier.
3. Zones: `F_FABRIC` = send -> last arrival seen (same meaning as before). `F_MCAST` = what remains after the last
   arrival (normally 1 page + go + barrier); `F_MCASTN` is used instead when more than one page was still unsent,
   so the per-multicast cost is measurable.
Host, compute and worker writer are untouched (kernel JIT only, no rebuild). The non-gather path is unchanged.

## Why this is not a repeat
- r04-b02-a01 (parent): same data path, but all multicasts after the last arrival. This changes **when** the
  multicasts are issued, which is that reflection's #1.
- r02-b02-a04: multicast of the go flag only, workers still read. Different.
- r03-b04-a02 used 16-bit per-wave fields in out_ready for a two-wave AG. Here the fields are per **source chip**
  and serve to identify which page has landed. The waves idea itself (flawed) is not revisited.

## Expected effect and risk
- Go after the last arrival: ~1.03 -> ~0.35-0.45 µs, if one multicast costs ~1/3 of the parent's burst. Combine start
  ~0.4-0.5 µs after the last arrival, versus ~0.8-1.1 µs in r03-b02-a02 and 1.1 µs in the parent. So -0.6 µs vs the
  parent per shape (~12.3 / 13.9 / 17.8 / 18.8 µs), roughly +1..3% over r03-b02-a02.
- On the chip that sends last, every peer page is already present at send time, so it multicasts all 4 pages in a
  row. Its own packet still needs ~1 µs to reach the peers, which hides most of that burst.
- Risks: (a) a wrong field shift makes the forwarder wait forever -> hang (eval `hang`); (b) the inc arriving before
  the payload would multicast stale data -> accuracy_fail. Same-VC ordering is what the parent already relied on.
  (c) If one multicast costs as much as three (a fixed per-burst cost, not per page), F_MCAST stays ~1.2 µs and
  the result is neutral vs the parent. F_MCAST vs F_MCASTN durations will tell.
