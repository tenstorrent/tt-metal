# Router code size and stack use

Started 2026-10-30 (user: main is at the edge of its code and stack space; not a constraint, but track it). Every new
lead or candidate variant gets a line in the log below, against main and the current lead.

## Method (`size_report.py`)

From the offline coverage builds (`coverage.py build`, production cases: study instrumentation, stream credits and
synthesized configs are left out), per router ELF:
- **code**: loadable bytes (PT_LOAD file sizes), what the kernel config buffer holds. For two-ERISC routers, E0 + E1 as
  the largest of each over a case's configs (an upper bound for one router) against the 26,624 B buffer
  (`MEM_ERISC_KERNEL_CONFIG_SIZE`, which also holds runtime args).
- **stack**: deepest static use from the kernel's `_start`, frames from each function's sp adjustments, the call graph
  from `jal` (tail jumps add no frame). E0's indirect calls enter the base firmware: +68 B below the caller's frame
  (RESULTS §18). **free** = stack pointer at kernel entry (E0 0xFFB01EF0, E1 0xFFB01FC0, RESULTS §18; E1's matches the
  firmware's `__stack_top` minus its `_start` and `main` frames, 16 + 48 B) minus the end of the kernel's data and bss,
  minus the deepest use.

Check against the stack paint (RESULTS §18, older main): E1's static deepest use on main's census ELFs is 752 B, the
painted value exactly. E0's is 1,460 B against a painted 1,364 B: main's `kernel_main` frame grew 1,296 → 1,328 B, and
the static bound lets the base-firmware call happen one function deeper (`wait_for_static_connection_to_ready`, 64 B).
An upper bound, as intended.

Main's router is built from `variants/mainv2cap` over the post-merge base (`/home/aagarwal/.tt-metal-p1-base-m`, main
6d8a103 of 2026-10-28). Main builds VC1 at -Os and everything else at -O3 (`get_kernel_opt_level`). Main does not build
`bh_2d_vc2_1erisc` (static assertion: VC2 needs two ERISCs).

To check a variant (CPU only, a few minutes per level):

    cd fabric_exp/consolidation
    export TT_COV_DIR=/home/aagarwal/.tt-metal-v3r-builds/cov_m OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m
    python3 coverage.py build --variant ../variants/<V> --opt O3 --name <V> --cases <production cases>
    python3 size_report.py --opt O3 main v4pl6 <V>        # --detail: the deepest call chain per router
    # the same at -Os

Production cases (35): `bh_1d bh_1d_1erisc bh_1d_mux bh_1d_telemetry bh_1d_trim bh_2d bh_2d_1erisc bh_2d_bwtelemetry
bh_2d_mux bh_2d_telemetry bh_2d_trim bh_2d_udm_gtest bh_2d_vc2 bh_2d_vc2_1erisc bh_neighbor bh_neighbor_1erisc bh_ring
bh_ring_1erisc bh_ring_mux bh_torus bh_torus_1erisc bh_trim_gtest bh_trim_gtest_1erisc bh_vc1 glx_1d glx_2d glx_express
glx_express_1erisc glx_express_line glx_express_line_1erisc glx_express_mesh glx_ring glx_torus` (comma-separated for
`--cases`).

**Caution (2026-11-03):** size_report.py rewrites `<NAME>_<opt>/size_report.json` for every NAME it is given, with only
the cases it read. A run with `--cases` therefore cuts each named build's json down to those cases (v4rxg3v_O3: 2,450 ->
131 entries; restored from a backup), and build_size.py and the size tables read those files. Back up the json files
first, or give only the build being measured plus a reference whose json can be regenerated in full.

## Baseline (2026-10-30; 32 cases both build)

| | main -O3 | main -Os | v4pl6 -O3 | v4pl6 -Os |
| --- | ---: | ---: | ---: | ---: |
| VC1 E0 + E1 (limit 26,624 B) | **29,940 (over by 3,316)** | 25,252 (1,372 left) | 23,264 (3,360 left) | 22,596 (4,028 left) |
| largest E0 | 17,760 (VC1) | 13,192 (VC1) | 12,444 (VC1) | 11,536 (VC1) |
| largest one-ERISC binary | 12,604 (Galaxy express) | 11,732 (Galaxy express) | 14,020 (Galaxy express) | 13,728 (Galaxy express) |
| least free stack E0 | **424 (VC1)**, ~900 on 2D/torus/Galaxy | **320 (VC1)**, ~800 on 2D | 1,604 (VC1) | 1,556 (VC1) |
| least free stack E1 | 3,024 (2D mux) | 2,852 (2D mux) | 3,864 (Galaxy express, one ERISC) | 3,608 (same) |
| deepest stack E0 / E1 | 1,940 / 1,744 | 2,036 / 1,856 | 996 / 1,056 | 1,044 / 1,312 |

- Main's E0 keeps its sender state in `kernel_main`'s frame (1,792 B on VC1 at -O3); v4's loop functions have 300-400 B
  frames (`kernel_main` 592 B).
- v4 is larger than main only for one-ERISC routers (one binary holds both programs): Galaxy express +1,416 B at -O3
  (+2,004 at -Os), 2D one-ERISC +708 at -Os. Far from the limit on Blackhole; Wormhole keeps router `.text` in IRAM
  (V4_SINGLE_ERISC §1.5): measure there before Wormhole bring-up.
- Earlier leads at -O3 (14 cases all of them have): VC1 E0 + E1 v3qm 25,368, v4trm 23,920; least free E0 v3qm 1,592,
  v4trm 1,604; least free E1 v3qm 3,092, v4trm 3,688 (2D VC2 one-ERISC).

## Log

| Date | Variant | Level | VC1 E0+E1 | largest one-ERISC | least free E0 / E1 | vs lead |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-10-30 | v4pl6 (lead) | -O3 | 23,264 | 14,020 | 1,604 / 3,864 | baseline |
| 2026-10-30 | v4pl6 (lead) | -Os | 22,596 | 13,728 | 1,556 / 3,608 | baseline |
| 2026-10-31 | v4rx1k (acks first, on v4rx1) | -O3 | 23,264 | 14,160 | 1,604 / 3,864 | equal; only ring/torus receivers change (-76 to +60 B), stack unchanged |
| 2026-10-31 | v4rx1j (one ack per pass, header group, end-of-pass proof+status group) | -O3 | 23,264 | 14,160 | 1,604 / 3,864 | equal |
| 2026-10-31 | v4rx1je (v4rx1j + early-if-ready completion and frame) | -O3 | 23,264 | 14,308 | 1,604 / 3,848 | +148 B one-ERISC torus, E1 stack -16 B |
| 2026-10-31 | v4rx1js (v4rx1j, end-of-pass block as scalars) | -O3 | 23,264 | 14,160 | 1,604 / 3,864 | equal |
| 2026-10-31 | v4rx1rg (v4rx1js + one-insert header ring, ack pointer) | -O3 | 23,268 | 14,268 | 1,604 / 3,712 | +4 B VC1, E1 stack -152 B (the ring) |
| 2026-10-31 | v4rx1r1 (v4rx1rg + 1D routing as one word) | -O3 | 23,268 | 14,268 | 1,604 / 3,712 | as v4rx1rg |
| 2026-10-31 | v4rxdas2 (temporary lead: receiver redesign A stage 2) | -O3 | 23,248 | 14,108 | 1,604 / 3,864 | vs v4rx1r1: -20 B VC1, E1 free stack +152 B (no header ring) |
| 2026-10-31 | v4rxc1 (design C: header group before the acks) | -O3 | 23,280 | 14,104 | 1,604 / 3,864 | +32 B VC1 |
| 2026-10-31 | v4rxc2b (design C: every packets-received register in the pass group) | -O3 | 23,376 | 14,188 | 1,604 / 3,864 | +128 B VC1, +80 B one-ERISC |
| 2026-10-31 | v4rxc3 (design C: one completion skeleton) | -O3 | 23,248 | 14,108 | 1,604 / 3,864 | byte-identical on all 2,086 images |
| 2026-10-31 | v4rxc4 (design C: keep_in_memory on VC1/VC2 cursors) | -O3 | 23,284 | 14,104 | 1,604 / 3,864 | +36 B VC1 |
| 2026-10-31 | v4rxc5 (design C: c3 + one-slot catch-up) | -O3 | 23,356 | 14,140 | 1,604 / 3,864 | +108 B VC1, +32 B one-ERISC |
| 2026-10-31 | v4rxc6 (design C: c1 + c2b) | -O3 | 23,384 | 14,176 | 1,604 / 3,864 | +136 B VC1, +68 B one-ERISC |
| 2026-10-31 | v4rxc7 (lead candidate: c3 + c2b + ChannelState comment) | -O3 | 23,376 | 14,188 | 1,604 / 3,864 | byte-identical to v4rxc2b on all 2,086 images |
| 2026-11-01 | v4rxc7v (v4rxc7 + VC1 with VC2 admitted) | -O3 | 23,376 (VC1); VC1+VC2 26,076 | 21,128 (VC1+VC2 one ERISC) | 1,476 / 3,408 | other 2,086 images byte-identical to v4rxc7; VC1+VC2 E0 13,792 + E1 12,284, 548 B under the 26,624 B buffer |
| 2026-11-01 | v4rxc7c (worker connection semaphore at a compile-time address) | -O3 | 23,356 | | | largest E0 2D+VC2 11,448 (lead 11,508) |
| 2026-11-01 | v4rxv2i1 (v4rxc7c + both worker semaphores in the pass group, connections checked together) | -O3 | 23,332 | | | largest E0 2D+VC2 11,388 |
| 2026-11-01 | v4rxv2i3 (v4rxv2i1 + VC2's credit off the write command buffer, VC0 credit preset kept) | -O3 | | | | largest E0 2D+VC2 11,584 |
| 2026-11-01 | v4tx1 (sender restructure stage 1: Vc<V>/Channel<CH> features, Each<>, visit_vc; today's placement) | -O3 | 23,336 | | 1,604 / 3,864 | E0 -4 B; sender loops +0/+0 on every config; 878 images byte-identical (every E1) |
| 2026-11-01 | v4tx1b (v4tx1 + every worker's credit from its own connection-info read counter) | -O3 | 23,356 | | 1,604 / 3,864 | +8 to +20 B (init) |
| 2026-11-01 | v4rxv2i3 (lead; VC1 opt-level A/B, job 367; bh_vc1 only) | -Os | 22,564 | | 1,556 / 4,120 | VC1+VC2 size probe (lead + v4rxc7v's gate line, scratch): -Os 25,232, -O3 26,156 (468 B left before runtime args), one ERISC 20,556 / 21,104 |
| 2026-11-01 | v4rxv2i3ni (lead + `tt_buf_rw::note<>` always_inline, v4txr5's buf_rw_note.h; probe) | -Os | 21,532 | | 1,588 / 4,140 | -1,032 B VC1 at -Os (VC1+VC2 probe 24,120, one ERISC 19,768); -O3 .text identical to the lead on all 44 VC1 images |
| 2026-11-01 | v4rxo7 (receiver ownership pass: stage 0 inline-write fix (ii′) + renames, host-owned facts, per-VC downstreams with the buffer-2 fix (c), one cursor rule (B2), slots due for credit; V4_RECEIVER_OWNERSHIP.md) | -O3 | 22,840 | 13,848 (glx express) | 1,604 / 3,944 | vs v4rxv2i3: -500 B VC1, E1 least free stack +80 B; E0 unchanged |
| 2026-11-01 | v4rxo7 | -Os | 22,180 | 13,512 (glx express) | 1,556 / 3,832 | vs v4rxbt0base (lead): -384 B VC1, E1 least free stack +224 B |
| 2026-11-02 | v4rxs2x (S2-explicit: S2's placement stated, `ChannelMemory` + keep_in_memory) | -O3 | 22,956 | 16,856 (bh_vc1_trim_w0, one ERISC) | 1,604 / 3,700 | worst over the 50 cases all builds have |
| 2026-11-02 | v4rxs2x | -Os | 22,140 | 16,280 (bh_vc1_trim_w0, one ERISC) | 1,556 / 3,556 | |
| 2026-11-02 | v4rxe1 (S2-explicit readability polish: uniform `ChannelState`, `ChannelStates`, `step_channel`, first channel's cursor copied into run()'s local) | -O3 | 22,984 | 16,856 | 1,604 / 3,700 | vs v4rxs2x: two-ERISC E1 code +28 B; **E1 frame +16 B (1D, ring, 2D, Galaxy 2D) to +32 B (express, VC1)**: the first channel's ChannelState cursor field (dead in run()) and its initial stores; per-config E1 free stack -16 to -32 B (e.g. glx_express 4,400 → 4,368); least free E1 unchanged (3,700, a one-ERISC config) |
| 2026-11-02 | v4rxe1 | -Os | 22,152 | 16,280 | 1,556 / 3,556 | vs v4rxs2x: two-ERISC E1 code +12 B (a memset call for the larger zero-init); E1 free stack -12 to -32 B per config; least free E1 unchanged (3,556) |
| 2026-11-02 | v4rxf1 (naming and structure polish on v4rxe1: receiver keyed by VC, `Each<>`, flat `RingProgress`, read-group primitives) | -O3 | 22,984 | 16,856 | 1,604 / 3,700 | vs v4rxe1: .text identical on 2,697 of 2,713 images; the 16 UDM images +4 B (one-ERISC `receiver_pass`, accepted) |
| 2026-11-02 | v4rxf1 | -Os | 22,152 | 16,280 | 1,556 / 3,556 | .text identical to v4rxe1 on all 2,713 images |
| 2026-11-02 | v4rxf4 (v4rxf1 + S4 slots due for credit + the codec decode + C1 comment trim; receiver lead candidate) | -O3 | 22,996 | 16,860 | 1,604 / 3,700 | vs v4rxe1: VC1 E0+E1 +12 B. vs lead v4rxv2i3 (32 cases shared with main): VC1 -344 B, least free E1 3,864 -> 3,944 |
| 2026-11-02 | v4rxf4 | -Os | 22,156 | 16,268 | 1,556 / 3,556 | vs v4rxe1: VC1 +4 B; one-ERISC E1 -8 to -12 B |
| 2026-11-02 | v4rxg1 / v4rxg2 / v4rxg3 (channel-kind names, asserts out; `ReceiverChannelStates` owns the placement rule and the downstream progress field order; derived `loop_local_vc`) | -O3 | 22,996 | 16,860 | 1,604 / 3,700 | equal to v4rxf4 at every stage (g1 byte-identical; g2 layout-only; g3 changes only VC0-receiver-trimmed shapes, which are not in the 50-case set) |
| 2026-11-02 | v4rxg1 / v4rxg2 / v4rxg3 | -Os | 22,156 | 16,268 | 1,556 / 3,556 | equal to v4rxf4 |
| 2026-11-02 | v4rxnsa / v4rxnsb (drafts, N/S fix 1: partial presets on forwards, (a) multi-downstream routers only, (b) every forward; V4_NS_PRESETS.md) | -O3 | 22,608 | 16,860 | 1,604 / 3,700 | vs v4rxg3: VC1 E0+E1 -388 B (both); largest E1 bh_2d 6,392 -> 6,352 (a) / 6,328 (b), glx_2d 7,004 -> 6,828, glx_express 7,440 -> 7,224; least free unchanged |
| 2026-11-02 | v4rxnsa / v4rxnsb (drafts) | -Os | 21,452 | 16,268 | 1,556 / 3,556 | vs v4rxg3: VC1 E0+E1 -704 B (both) |
| 2026-11-02 | v4rxnsa / v4rxnsb as run (user decisions: two trims, relay unchanged, UDM without presets; V4_NS_PRESETS §7) | -O3 | 22,536 | 16,860 | 1,604 / 3,700 | vs v4rxg3: VC1 E0+E1 -460 B (both) |
| 2026-11-02 | v4rxnsa / v4rxnsb as run | -Os | 21,380 | 16,268 | 1,556 / 3,556 | vs v4rxg3: VC1 E0+E1 -776 B (both) |
| 2026-11-03 | v4rxnsf (N/S fix 1 as ForwardCommands: full / partial / stateless forms chosen at compile time; V4_NS_PRESETS §9.1) | -O3 / -Os | 22,536 / 21,380 | 16,860 / 16,268 | 1,604 / 3,700; 1,556 / 3,556 | byte-identical to v4rxnsa on all 2,863 images at both levels |
| 2026-11-03 | v4txo8 (sender ownership pass on v4rxg3, V4_SENDER_OWNERSHIP.md §12: names and structure, packed P-B placement, worker interfaces at a fixed location, worker credits on the host's command buffer, one cold connection path, telemetry's count from `sent`, round helpers, comments) | -O3 | 23,056 | 16,840 | 1,684 / 3,728 | vs v4rxg3: VC1 E0+E1 +60 B (X4 -124, X5b +72); E0 least free +80 B, deepest E0 996 -> 580 B (from X3a); least free E1 over the 54 shared cases +28 B. **Cleanup item:** bh_vc1's E1 free 4,236 -> 3,896 B: `worker_interface_storage` (336 B .bss) and its construction (+188 B) on an E1 that serves no workers (stage A builds the interfaces unconditionally; v4txo8w, pending the user's call, builds them only where the ERISC serves sender channels) |
| 2026-11-03 | v4txo8 | -Os | 21,844 | 16,244 | 1,636 / 3,584 | vs v4rxg3: VC1 E0+E1 -312 B; E0 least free +80 B; bh_vc1 E1 free 4,092 -> 3,752 B (the same storage; at -Os every receiver-only E1 carries 48-336 B of it) |
| 2026-11-03 | v4txc0 (sender consolidation C0: v4txo8 + v4rxnsf2; V4_SENDER_CONSOLIDATION.md §5) | -O3 | 22,596 | 16,840 | 1,684 / 3,728 | E0 = v4txo8's (byte-identical), E1 = v4rxnsf2's forwards + v4txo8's interface storage: VC1 E0+E1 -460 B vs v4txo8; least free unchanged |
| 2026-11-03 | v4txc0 | -Os | 21,068 | 16,244 | 1,636 / 3,584 | VC1 E0+E1 -776 B vs v4txo8 |
| 2026-11-03 | v4txc1 (C1: sender facts into their owners) | -O3 / -Os | 22,596 / 21,068 | 16,840 / 16,244 | as v4txc0 | byte-identical to v4txc0 on all 2,863 images at both levels |
| 2026-11-03 | v4txc1b (C1b: a VC's registers in the pass's first read group only where its worker channel is served) | -O3 | 22,596 | 16,840 | 1,684 / 3,728 | changes only the 112 T1/T7 images (VC0 worker trimmed, VC2 worker served): two-ERISC E0 -8 B, one-ERISC +12 B |
| 2026-11-03 | v4txc1b | -Os | 21,068 | 16,244 | 1,636 / 3,584 | T1/T7: E0 +64 B (least free E0 there 1,828 -> 1,812), one-ERISC +24 B |
| 2026-11-03 | v4txc2 (C2: one CommandBuffers table for both ERISCs) | -O3 / -Os | as v4txc1b | | | byte-identical to v4txc1b on all 2,863 images at both levels |
| 2026-11-03 | v4txc3 (C3: WorkerChannel / ForwardingChannel / SenderChannelInterfaces; interfaces built only where the ERISC serves sender channels, v4txo8w) | -O3 | 22,248 | 16,684 | 1,684 / 3,728 | vs v4txc2: VC1 E0+E1 -348 B (E0 -160, E1 -188); bh_vc1 E1 free 3,896 -> 4,236 B (v4rxg3's again: no interface storage on a sender-less E1) |
| 2026-11-03 | v4txc3 | -Os | 20,720 | 16,088 | 1,636 / 3,584 | vs v4txc2: VC1 E0+E1 -348 B; bh_vc1 E1 free 3,768 -> 4,108 B |
| 2026-11-03 | v4txc4 (C4: comments and names) | -O3 / -Os | as v4txc3 | | | byte-identical to v4txc3 on all 2,863 images at both levels |
| 2026-11-03 | v4txa1 (Part A1: router_common facts only, host-arg arrays in the ct args, router_utils.hpp) | -O3 / -Os | as v4txc4 | | | byte-identical to v4txc4 on all 2,863 images at both levels |
| 2026-11-03 | v4txa9 (Part A: A1-A7, A9 on v4txc4; V4_SENDER_CONSOLIDATION.md §6) | -O3 | 22,320 | 16,676 | 1,684 / 3,728 | vs v4txc4: VC1 E0+E1 +72 B (A2's per-owner stream init, A4's round state); stack unchanged |
| 2026-11-03 | v4txa9 | -Os | 20,752 | 16,160 | 1,636 / 3,584 | vs v4txc4: VC1 E0+E1 +32 B |
| 2026-11-03 | v4txa10 (A10: the kernel file as the phase skeleton; V4_SENDER_CONSOLIDATION.md §6.8) | -O3 / -Os | as v4txa9 | | | every loop function identical to v4txa9 on all 2,863 images; only _start / kernel_main differ |
| 2026-11-03 | v4txa11 (A11: A10 + the this_chip() revert; §6.9) | -O3 | 22,232 | 16,680 | 1,684 / 3,728 | vs v4txc4: VC1 E0+E1 -16 B; vs v4txa10 -88 B |
| 2026-11-03 | v4txa11 | -Os | 20,760 | 16,164 | 1,636 / 3,584 | vs v4txc4: VC1 E0+E1 +40 B; vs v4txa10 +8 B |
