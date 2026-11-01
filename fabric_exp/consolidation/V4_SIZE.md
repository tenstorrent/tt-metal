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
