<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Streaming profiler: detailed reference

The streaming profiler drains device zones off Blackhole continuously: worker RISCs write markers into
per-RISC L1 rings, resident DRISC relays on the DRAM cores pump them to the host over D2H sockets, and a host
receiver hands whole zones to any consumer that registers a callback.

Start with the short illustrated intro, [`STREAMING_PROFILER.md`](STREAMING_PROFILER.md). This document is the
full reference: usage (§1), the current design (§2), the designs it replaced (§3), the zone primitives and the
wire format (§4), the offline Tracy tools (§5), and — in the companion
[`STREAMING_PROFILER_FINDINGS.md`](STREAMING_PROFILER_FINDINGS.md), §6 — the complete dated record of the findings
and benchmarks that produced all of it. The findings record is a separate file only because of the 500 KB
per-file pre-commit limit.

It consolidates, unchanged in substance, the former `STREAMING_PROFILER_ZONES.md` and
`tools/drisc_drain/{ARCHITECTURE,HANGS,README,DIRECT_PUSH_PLAN,FINDINGS}.md`.

## Contents

- [1. Overview and usage](#1-overview-and-usage)
  - [1.1 Three device-profiler modes (mutually exclusive)](#11-three-device-profiler-modes-mutually-exclusive)
  - [1.2 Environment variables](#12-environment-variables)
  - [1.3 Enable (Blackhole)](#13-enable-blackhole)
  - [1.4 Register a callback](#14-register-a-callback)
  - [1.5 What you get](#15-what-you-get)
  - [1.6 The call pattern](#16-the-call-pattern)
  - [1.7 Try it end to end](#17-try-it-end-to-end)
  - [1.8 Two rules](#18-two-rules)
- [2. Architecture (current design)](#2-architecture-current-design)
  - [2.1 The pipeline](#21-the-pipeline)
  - [2.2 The relay](#22-the-relay)
  - [2.3 The host](#23-the-host)
  - [2.4 Sizing and where the numbers come from](#24-sizing-and-where-the-numbers-come-from)
- [3. Design history (superseded designs)](#3-design-history-superseded-designs)
  - [3.1 Fillers + movers and the DRAM frame ring (2026-08-11; superseded 2026-08-25)](#31-fillers--movers-and-the-dram-frame-ring-2026-08-11-superseded-2026-08-25)
  - [3.2 Blackhole device/host hang runbook (2026-08-07 … 08-08)](#32-blackhole-devicehost-hang-runbook-2026-08-07--08-08)
  - [3.3 The DRISC hang-investigation harness (2026-08)](#33-the-drisc-hang-investigation-harness-2026-08)
- [4. Zone primitives and the wire format](#4-zone-primitives-and-the-wire-format)
  - [4.1 Zones and point markers in the Tracy GUI](#41-zones-and-point-markers-in-the-tracy-gui)
  - [4.2 The zone wire format, and what it measures like](#42-the-zone-wire-format-and-what-it-measures-like)
- [5. Dev tools: reading a `.tracy` without the GUI](#5-dev-tools-reading-a-tracy-without-the-gui)
- [6. Findings and benchmarks](#6-findings-and-benchmarks)
  - [6.1 How to read this, and the name mapping](STREAMING_PROFILER_FINDINGS.md#61-how-to-read-this-and-the-name-mapping)
  - [6.2 Direct host push: fillers own D2H sockets, movers deleted (plan, 2026-08-25)](STREAMING_PROFILER_FINDINGS.md#62-direct-host-push-fillers-own-d2h-sockets-movers-deleted-plan-2026-08-25)
  - [6.3 The record (`FINDINGS.md`, 2026-07 → 2026-08-26)](STREAMING_PROFILER_FINDINGS.md#63-the-record-findingsmd-2026-07--2026-08-26)
  - [Running it](STREAMING_PROFILER_FINDINGS.md#running-it)
  - [The cost model](STREAMING_PROFILER_FINDINGS.md#the-cost-model)
  - [Ingest — reading markers out of worker L1](STREAMING_PROFILER_FINDINGS.md#ingest--reading-markers-out-of-worker-l1)
  - [The poll, and why it dominates](STREAMING_PROFILER_FINDINGS.md#the-poll-and-why-it-dominates)
  - [Egress — D2H socket](STREAMING_PROFILER_FINDINGS.md#egress--d2h-socket)
  - [Egress alternative: DMA to a GDDR buffer (leg A)](STREAMING_PROFILER_FINDINGS.md#egress-alternative-dma-to-a-gddr-buffer-leg-a)
  - [The reserved DRAM profiler region](STREAMING_PROFILER_FINDINGS.md#the-reserved-dram-profiler-region)
  - [Full circle: A -> DRAM ping/pong -> B -> host](STREAMING_PROFILER_FINDINGS.md#full-circle-a---dram-pingpong---b---host)
  - [Direct: ingest -> host on one DRISC](STREAMING_PROFILER_FINDINGS.md#direct-ingest---host-on-one-drisc)
  - [Scaling the direct drainer: ingest fans out, egress does not](STREAMING_PROFILER_FINDINGS.md#scaling-the-direct-drainer-ingest-fans-out-egress-does-not)
  - [Two-tier adaptive drainer: monitor, then drain at the right granularity](STREAMING_PROFILER_FINDINGS.md#two-tier-adaptive-drainer-monitor-then-drain-at-the-right-granularity)
  - [End to end: a DRISC services REAL producers](STREAMING_PROFILER_FINDINGS.md#end-to-end-a-drisc-services-real-producers)
  - [End to end to the HOST: framed egress over the D2H socket](STREAMING_PROFILER_FINDINGS.md#end-to-end-to-the-host-framed-egress-over-the-d2h-socket)
  - [Zone decode: the stream is real](STREAMING_PROFILER_FINDINGS.md#zone-decode-the-stream-is-real)
  - [Verdict: the DRAM round-trip does not pay](STREAMING_PROFILER_FINDINGS.md#verdict-the-dram-round-trip-does-not-pay)
  - [What it means: the zone budget](STREAMING_PROFILER_FINDINGS.md#what-it-means-the-zone-budget)
  - [Design implications](STREAMING_PROFILER_FINDINGS.md#design-implications)
  - [Gotchas](STREAMING_PROFILER_FINDINGS.md#gotchas)
  - [Open questions](STREAMING_PROFILER_FINDINGS.md#open-questions)
  - [A caution on the instrumentation](STREAMING_PROFILER_FINDINGS.md#a-caution-on-the-instrumentation)
  - [§N — The conduit drainer, and the knee at 125 (bh-05, 2026-08-04)](STREAMING_PROFILER_FINDINGS.md#n--the-conduit-drainer-and-the-knee-at-125-bh-05-2026-08-04)
  - [§N+1 — Read/write NoC split, and the two-DRISC probe (bh-05, 2026-08-04)](STREAMING_PROFILER_FINDINGS.md#n1--readwrite-noc-split-and-the-two-drisc-probe-bh-05-2026-08-04)
  - [§N+2 — The static TLB lever, and what the degraded state actually costs (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n2--the-static-tlb-lever-and-what-the-degraded-state-actually-costs-bh-05-2026-08-06)
  - [§N+3 — Matched test: why the Tensix drainer cannot hang the card (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n3--matched-test-why-the-tensix-drainer-cannot-hang-the-card-bh-05-2026-08-06)
  - [§N+4 — The egress amplifier: PCIe bandwidth is NOT the hang trigger (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n4--the-egress-amplifier-pcie-bandwidth-is-not-the-hang-trigger-bh-05-2026-08-06)
  - [§N+5 — The slow-dispatch DRISC wedge: an UNSATISFIABLE write barrier (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n5--the-slow-dispatch-drisc-wedge-an-unsatisfiable-write-barrier-bh-05-2026-08-06)
  - [§N+6 — Egress saturates at ~17 GB/s, and the hang is NOT on that axis (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n6--egress-saturates-at-17-gbs-and-the-hang-is-not-on-that-axis-bh-05-2026-08-06)
  - [§N+7 — Tensix vs DRISC: every difference in the path, host and device](STREAMING_PROFILER_FINDINGS.md#n7--tensix-vs-drisc-every-difference-in-the-path-host-and-device)
  - [§N+8 — The hang is CUMULATIVE, not a property of (delay, repeat). §N+6's "ingest axis" is confounded.](STREAMING_PROFILER_FINDINGS.md#n8--the-hang-is-cumulative-not-a-property-of-delay-repeat-n6s-ingest-axis-is-confounded)
  - [§N+9 — Fixed config has never hung; every hang came from a VARYING sequence. Rare and stochastic. (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n9--fixed-config-has-never-hung-every-hang-came-from-a-varying-sequence-rare-and-stochastic-bh-05-2026-08-06)
  - [§N+10 — The Tensix-vs-DRISC comparison was confounded with DISPATCH MODE all along](STREAMING_PROFILER_FINDINGS.md#n10--the-tensix-vs-drisc-comparison-was-confounded-with-dispatch-mode-all-along)
  - [§N+11 — THE 2x2 IS COMPLETE: only DRISC *AND* fast dispatch hangs (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n11--the-2x2-is-complete-only-drisc-and-fast-dispatch-hangs-bh-05-2026-08-06)
  - [§N+12 — Bisection: egress-only is clean, NoC swap changes nothing (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n12--bisection-egress-only-is-clean-noc-swap-changes-nothing-bh-05-2026-08-06)
  - [§N+13 — The degradation is CARD-WIDE, not the DRAM endpoint (bh-05, 2026-08-06)](STREAMING_PROFILER_FINDINGS.md#n13--the-degradation-is-card-wide-not-the-dram-endpoint-bh-05-2026-08-06)
  - [§N+14 — CAUGHT LIVE: the "PCIe hang" is an ENDPOINT wedge, not a link failure (bh-05, 2026-08-07)](STREAMING_PROFILER_FINDINGS.md#n14--caught-live-the-pcie-hang-is-an-endpoint-wedge-not-a-link-failure-bh-05-2026-08-07)
  - [§N+15 — WHY the CQ gets stuck in `finish`: the host spins on a pointer only the dead device can move](STREAMING_PROFILER_FINDINGS.md#n15--why-the-cq-gets-stuck-in-finish-the-host-spins-on-a-pointer-only-the-dead-device-can-move)
  - [§N+16 — Arming the operation timeout SUPPRESSES the hang; and degradation needs a FREEZE, not a hang (bh-05, 2026-08-07)](STREAMING_PROFILER_FINDINGS.md#n16--arming-the-operation-timeout-suppresses-the-hang-and-degradation-needs-a-freeze-not-a-hang-bh-05-2026-08-07)
  - [§N+17 — POLL PRESSURE IS REFUTED, and §N+16's "the timeout suppresses the hang" is in serious doubt](STREAMING_PROFILER_FINDINGS.md#n17--poll-pressure-is-refuted-and-n16s-the-timeout-suppresses-the-hang-is-in-serious-doubt)
  - [§N+18 — FOUND IT: a periodic DEVICE READ prevents the endpoint wedge (bh-05, 2026-08-07)](STREAMING_PROFILER_FINDINGS.md#n18--found-it-a-periodic-device-read-prevents-the-endpoint-wedge-bh-05-2026-08-07)
  - [§N+19 — RETRACT §N+18. The timeout escapes a TEARDOWN wait; it does not prevent the wedge.](STREAMING_PROFILER_FINDINGS.md#n19--retract-n18-the-timeout-escapes-a-teardown-wait-it-does-not-prevent-the-wedge)
  - [§N+20 — §N+11's 2x2 IS SUSPECT: it was scored before WEDGE and TEARDOWN could be told apart](STREAMING_PROFILER_FINDINGS.md#n20--n11s-2x2-is-suspect-it-was-scored-before-wedge-and-teardown-could-be-told-apart)
  - [§N+21 — CONSOLIDATED STATE (2026-08-07). Read this section first.](STREAMING_PROFILER_FINDINGS.md#n21--consolidated-state-2026-08-07-read-this-section-first)
  - [§N+22 — Tensix does not wedge: 0/200 against DRISC's 4/200, egress-matched](STREAMING_PROFILER_FINDINGS.md#n22--tensix-does-not-wedge-0200-against-driscs-4200-egress-matched)
  - [§N+23 — Tensix does not wedge UNDER STRESS either: 0/400 vs DRISC 4/200 (p ~ 0.02)](STREAMING_PROFILER_FINDINGS.md#n23--tensix-does-not-wedge-under-stress-either-0400-vs-drisc-4200-p--002)
  - [§N+24 — SOLVED: the slow-dispatch TEARDOWN is a HARNESS GRID BUG, not a device fault (bh-05, 2026-08-07)](STREAMING_PROFILER_FINDINGS.md#n24--solved-the-slow-dispatch-teardown-is-a-harness-grid-bug-not-a-device-fault-bh-05-2026-08-07)
  - [§N+25 — DRISC now polls the FULL 120-core grid by default (bh-05, 2026-08-07)](STREAMING_PROFILER_FINDINGS.md#n25--drisc-now-polls-the-full-120-core-grid-by-default-bh-05-2026-08-07)
  - [§N+26 — The IOMMU lead is DEAD, and three failure names are one state (bh-05, 2026-08-07/08)](STREAMING_PROFILER_FINDINGS.md#n26--the-iommu-lead-is-dead-and-three-failure-names-are-one-state-bh-05-2026-08-0708)
  - [§N+27 — REFUTED: the drainer/`dram_membar` subchannel collision is not the wedge (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n27--refuted-the-drainerdram_membar-subchannel-collision-is-not-the-wedge-bh-26-2026-08-08)
  - [§N+28 — Where the 120-core knee actually goes: the SCAN, not read and not egress (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n28--where-the-120-core-knee-actually-goes-the-scan-not-read-and-not-egress-bh-26-2026-08-08)
  - [§N+29 — A DRISC drainer is only safe on a DRAM core in NoC row y == 0 (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n29--a-drisc-drainer-is-only-safe-on-a-dram-core-in-noc-row-y--0-bh-26-2026-08-08)
  - [§N+30 — Two DRISCs lower the knee 1.67x AND FREEZE THE HOST. Not worth it. (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n30--two-driscs-lower-the-knee-167x-and-freeze-the-host-not-worth-it-bh-26-2026-08-08)
  - [§N+31 — The 220 ms MMIO stall IS the root-port completion timeout, and the endpoint logs UnsupReq (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n31--the-220-ms-mmio-stall-is-the-root-port-completion-timeout-and-the-endpoint-logs-unsupreq-bh-26-2026-08-08)
  - [§N+32 — UnsupReq REFUTED; the hang strikes at DRAINER BRING-UP, inside set_drisc_niu_mode (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n32--unsupreq-refuted-the-hang-strikes-at-drainer-bring-up-inside-set_drisc_niu_mode-bh-26-2026-08-08)
  - [§N+33 — A Gen1 downtrain is software-recoverable: force re-equalization, no cold power cycle](STREAMING_PROFILER_FINDINGS.md#n33--a-gen1-downtrain-is-software-recoverable-force-re-equalization-no-cold-power-cycle)
  - [§N+34 — FIXED: the bring-up hang was one-launch-per-NIU-flip. Two drainers now default, knee 100 -> 20 (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n34--fixed-the-bring-up-hang-was-one-launch-per-niu-flip-two-drainers-now-default-knee-100---20-bh-26-2026-08-08)
  - [§N+35 — The single-drainer wedge does NOT reproduce on bh-26: it is BOX-DEPENDENT (2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n35--the-single-drainer-wedge-does-not-reproduce-on-bh-26-it-is-box-dependent-2026-08-08)
  - [§N+36 — Decode+publish on a healthy card: the HOST IS NOT THE CONSTRAINT, and several earlier claims were degraded-card artifacts (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n36--decodepublish-on-a-healthy-card-the-host-is-not-the-constraint-and-several-earlier-claims-were-degraded-card-artifacts-bh-26-2026-08-08)
  - [§N+37 — The REAL knee on a healthy card: 1 drainer ~250, 2 drainers ~150 (~1.7x). "Knee 20" was a desynchronized-launch artifact (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n37--the-real-knee-on-a-healthy-card-1-drainer-250-2-drainers-150-17x-knee-20-was-a-desynchronized-launch-artifact-bh-26-2026-08-08)
  - [§N+38 — The knee is set by the WORST sweep's CREDIT-WAIT, and the per-read page cap must clear a whole FRAME (bh-26, 2026-08-08)](STREAMING_PROFILER_FINDINGS.md#n38--the-knee-is-set-by-the-worst-sweeps-credit-wait-and-the-per-read-page-cap-must-clear-a-whole-frame-bh-26-2026-08-08)
  - [§N+39 — The DRAM frame ring is RUNWAY, not headroom: the volume knee is 20k zones/RISC, and 12 MiB is enough for 5k (bh-26, 2026-08-11)](STREAMING_PROFILER_FINDINGS.md#n39--the-dram-frame-ring-is-runway-not-headroom-the-volume-knee-is-20k-zonesrisc-and-12-mib-is-enough-for-5k-bh-26-2026-08-11)
  - [§N+40 — 4 FILLERS + 2 DUAL-RING MOVERS: the knee moves 60 -> 15 (4x), and `max batch` does NOT halve (bh-26, 2026-08-11)](STREAMING_PROFILER_FINDINGS.md#n40--4-fillers--2-dual-ring-movers-the-knee-moves-60---15-4x-and-max-batch-does-not-halve-bh-26-2026-08-11)
  - [§N+41 — DRISC SELF-PROFILING: the drainer frames its own zones as a worker span, and the sampler has to trigger on WORK, not on sweep number (bh-26, 2026-08-12)](STREAMING_PROFILER_FINDINGS.md#n41--drisc-self-profiling-the-drainer-frames-its-own-zones-as-a-worker-span-and-the-sampler-has-to-trigger-on-work-not-on-sweep-number-bh-26-2026-08-12)
  - [§N+42 — The profiler's NoC footprint, measured by the hardware instead of derived (bh-26, 2026-08-13)](STREAMING_PROFILER_FINDINGS.md#n42--the-profilers-noc-footprint-measured-by-the-hardware-instead-of-derived-bh-26-2026-08-13)
  - [§N+43 — The NoC-footprint PLOTS land on the device timebase, and every code-size intuition was wrong (bh-26, 2026-08-13)](STREAMING_PROFILER_FINDINGS.md#n43--the-noc-footprint-plots-land-on-the-device-timebase-and-every-code-size-intuition-was-wrong-bh-26-2026-08-13)
  - [§N+44 — The profiler on a REAL model: ResNet-50 trace+2cq (bh-26, 2026-08-13)](STREAMING_PROFILER_FINDINGS.md#n44--the-profiler-on-a-real-model-resnet-50-trace2cq-bh-26-2026-08-13)
  - [§N+45 — e2e overhead across FOUR real models, and the LLM decode regime (bh-05 / p100a, 2026-08-14)](STREAMING_PROFILER_FINDINGS.md#n45--e2e-overhead-across-four-real-models-and-the-llm-decode-regime-bh-05--p100a-2026-08-14)
  - [§N+46 — The DRAM-core clock has its own ORIGIN, and it is board-dependent (bh-05 / p100a, 2026-08-14)](STREAMING_PROFILER_FINDINGS.md#n46--the-dram-core-clock-has-its-own-origin-and-it-is-board-dependent-bh-05--p100a-2026-08-14)
  - [§N+47 — ONE SHARED FREQUENCY, measured on a long baseline: the per-core fits were BIASED, not noisy (bh-05 / p100a, 2026-08-14)](STREAMING_PROFILER_FINDINGS.md#n47--one-shared-frequency-measured-on-a-long-baseline-the-per-core-fits-were-biased-not-noisy-bh-05--p100a-2026-08-14)
  - [§N+48 — COMMON-TRIGGER SYNC EVENT: the anchor residual is 0.18 us, not 25 us (bh-05 / p100a, 2026-08-17)](STREAMING_PROFILER_FINDINGS.md#n48--common-trigger-sync-event-the-anchor-residual-is-018-us-not-25-us-bh-05--p100a-2026-08-17)
  - [§N+49 — THE CROSS-DOMAIN NUMBER, and why in-capture alignment holds while between-capture offsets reach minutes (bh-05 / p100a, 2026-08-17)](STREAMING_PROFILER_FINDINGS.md#n49--the-cross-domain-number-and-why-in-capture-alignment-holds-while-between-capture-offsets-reach-minutes-bh-05--p100a-2026-08-17)
  - [§N+50 — e2e overhead on ResNet-50 BATCH 32: +2.64% (bh-05 / p100a, 2026-08-17)](STREAMING_PROFILER_FINDINGS.md#n50--e2e-overhead-on-resnet-50-batch-32-264-bh-05--p100a-2026-08-17)
  - [§N+51 — Receiver v2 (zero-copy decode into per-stream rings) is lossless and count-exact; the delay-15 stall floor is DEVICE-side (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n51--receiver-v2-zero-copy-decode-into-per-stream-rings-is-lossless-and-count-exact-the-delay-15-stall-floor-is-device-side-bh-18-2026-08-19)
  - [§N+52 — --delay calibration: 1 unit = 10.00 cycles = 7.41 ns; 2 GiB rings retain a whole capture; ring hugepages are a wash (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n52----delay-calibration-1-unit--1000-cycles--741-ns-2-gib-rings-retain-a-whole-capture-ring-hugepages-are-a-wash-bh-18-2026-08-19)
  - [§N+53 — The D2H ladder: PCIe writes ~37 GB/s, pinned-FIFO reads 27 GB/s, DECODE 5.6-6.4 GB/s -- the wall is interpretation, not the pipe (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n53--the-d2h-ladder-pcie-writes-37-gbs-pinned-fifo-reads-27-gbs-decode-56-64-gbs----the-wall-is-interpretation-not-the-pipe-bh-18-2026-08-19)
  - [§N+54 — AVX2 zone-block decode: 2.1x, SUSTAINED busy 5.8 -> 12.2 GB/s D2H, 667 Mzones/s (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n54--avx2-zone-block-decode-21x-sustained-busy-58---122-gbs-d2h-667-mzoness-bh-18-2026-08-19)
  - [§N+55 — Decode is memory-LATENCY bound: ALU trims are wall-time-neutral, two-pass scanning is a regression (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n55--decode-is-memory-latency-bound-alu-trims-are-wall-time-neutral-two-pass-scanning-is-a-regression-bh-18-2026-08-19)
  - [§N+56 — Cross-frame window prefetch REGRESSES 30-40%: bulk cold-line prefetch starves demand loads (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n56--cross-frame-window-prefetch-regresses-30-40-bulk-cold-line-prefetch-starves-demand-loads-bh-18-2026-08-19)
  - [§N+57 — AVX-512 emit measured: neutral-to-worse, as N+56 predicted; not kept (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n57--avx-512-emit-measured-neutral-to-worse-as-n56-predicted-not-kept-bh-18-2026-08-19)
  - [§N+58 — Live-extent packed frames: NIU-gathered packing at the socket push; 3.0x fewer bytes at 29% fill, exact everywhere (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n58--live-extent-packed-frames-niu-gathered-packing-at-the-socket-push-30x-fewer-bytes-at-29-fill-exact-everywhere-bh-18-2026-08-19)
  - [§N+59 — Op-perf CSV consumer vs the classic device profiler: semantics match; the capture costs this workload ~8-10% (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n59--op-perf-csv-consumer-vs-the-classic-device-profiler-semantics-match-the-capture-costs-this-workload-8-10-bh-18-2026-08-19)
  - [§N+60 — Per-lane STICKY_PROG: exact op attribution structurally; supersedes N+59's window method (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n60--per-lane-sticky_prog-exact-op-attribution-structurally-supersedes-n59s-window-method-bh-18-2026-08-19)
  - [§N+61 — ResNet-50 validation: perf-debug ops CSV vs classic, non-trace and trace (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n61--resnet-50-validation-perf-debug-ops-csv-vs-classic-non-trace-and-trace-bh-18-2026-08-19)
  - [§N+62 — Per-op repeatability under capture: perf-debug jitters 3-4x more than classic, medians unbiased (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n62--per-op-repeatability-under-capture-perf-debug-jitters-3-4x-more-than-classic-medians-unbiased-bh-18-2026-08-19)
  - [§N+63 — Raw fallback: per-frame cheapest-encoding selector; mover ceiling 45 GB/s at max load (bh-18, 2026-08-19)](STREAMING_PROFILER_FINDINGS.md#n63--raw-fallback-per-frame-cheapest-encoding-selector-mover-ceiling-45-gbs-at-max-load-bh-18-2026-08-19)
  - [§N+64 — DRISC code-region overflow at max instrumentation: one u64 division cost a 956 B soft-div (bh-18, 2026-08-20)](STREAMING_PROFILER_FINDINGS.md#n64--drisc-code-region-overflow-at-max-instrumentation-one-u64-division-cost-a-956-b-soft-div-bh-18-2026-08-20)
  - [§N+65 — kimi_k2 8-chip producer stalls: the DRAM ring FILLING was the mechanism; ship threshold + predictive valve + CV-first take 71,446 → 0 (yyz 8xp150, 2026-08-21)](STREAMING_PROFILER_FINDINGS.md#n65--kimi_k2-8-chip-producer-stalls-the-dram-ring-filling-was-the-mechanism-ship-threshold--predictive-valve--cv-first-take-71446--0-yyz-8xp150-2026-08-21)
  - [§N+67 — DMA-mover landed: rings on the mover's banks, GDDR-DMA reads, deterministic per-frame seq verification (yyz 8xp150, 2026-08-21)](STREAMING_PROFILER_FINDINGS.md#n67--dma-mover-landed-rings-on-the-movers-banks-gddr-dma-reads-deterministic-per-frame-seq-verification-yyz-8xp150-2026-08-21)
  - [§N+68 — The host decoder was the hidden half of every stall story: atomic zones had silently halved it (yyz 8xp150, 2026-08-21/22)](STREAMING_PROFILER_FINDINGS.md#n68--the-host-decoder-was-the-hidden-half-of-every-stall-story-atomic-zones-had-silently-halved-it-yyz-8xp150-2026-08-2122)
  - [N+69: The cleanup-surgery corruption was a hardcoded experimental read path](STREAMING_PROFILER_FINDINGS.md#n69-the-cleanup-surgery-corruption-was-a-hardcoded-experimental-read-path)
  - [N+70: The kimi stall regression is per-STREAM host-ack starvation, not device-side (yyz 8xp150, 2026-08-22)](STREAMING_PROFILER_FINDINGS.md#n70-the-kimi-stall-regression-is-per-stream-host-ack-starvation-not-device-side-yyz-8xp150-2026-08-22)
  - [§N+71 — Direct push now BEATS the ring design: gather-READ packing, one PCIe write per frame (bh-lb-120, 2026-08-25)](STREAMING_PROFILER_FINDINGS.md#n71--direct-push-now-beats-the-ring-design-gather-read-packing-one-pcie-write-per-frame-bh-lb-120-2026-08-25)
  - [§N+72 — Adaptive raw sweeps, the scan unroll, and what Mo's sub-10 filler knee actually is (bh-lb-120, 2026-08-25/26)](STREAMING_PROFILER_FINDINGS.md#n72--adaptive-raw-sweeps-the-scan-unroll-and-what-mos-sub-10-filler-knee-actually-is-bh-lb-120-2026-08-2526)

## 1. Overview and usage

The streaming profiler drains device zones to the host and hands decoded records to any callback you
register. It is Blackhole only (it needs the DRAM cores' DRISCs), and it needs a Tracy-enabled build of
tt-metal even when the Tracy sink itself is left off.

### 1.1 Three device-profiler modes (mutually exclusive)

| mode | `TT_METAL_DEVICE_PROFILER` | `TT_METAL_STREAMING_PROFILER` | device kernels | DRAM profiler (L1 → DRAM → per-op dump) | streaming (SPSC producer + DRISC relays + host receiver) | real-time profiler |
|---|---|---|---|---|---|---|
| 1 | unset | unset | no profiler code | off | off | on |
| 2 | set | unset | legacy `kernel_profiler.hpp` (`-DPROFILE_KERNEL`) | on | off | on |
| 3 | unset | set | streaming producer (`-DPROFILE_KERNEL=1 -DPROFILE_STREAMING=1`) | off | on | disabled |

Setting both variables is a `TT_FATAL` at `MetalContext` construction, before any device opens.
Mode 3 is Blackhole only (it needs DRISC drainers); on Quasar it is a `TT_FATAL`.

Device-side selection is one header: `tt_metal/tools/profiler/kernel_profiler.hpp` is main's DRAM producer
verbatim, wrapped so that `-DPROFILE_STREAMING` swaps in `tt_metal/tools/profiler/kernel_profiler_streaming.hpp`.
Shared streaming constants live in `hw/inc/hostdev/streaming_profiler_common.h`;
`hw/inc/hostdev/profiler_common.h` is the DRAM profiler's own. Not supported on the streaming
producer: sum zones (`DeviceZoneScopedSumN*`) and `DeviceRecordEvent` (both compile to nothing).

### 1.2 Environment variables

Fourteen variables are new in `tt_metal/llrt/rtoptions.cpp`, all read once at `MetalContext` construction.
Everything except `TT_METAL_STREAMING_PROFILER` itself applies in mode 3 only.

**Mode switches**

| variable | default | effect |
|---|---|---|
| `TT_METAL_STREAMING_PROFILER` | off | Boots the streaming profiler (resident DRISC relays + host receiver) at `MeshDevice` bring-up and compiles kernels with `-DPROFILE_STREAMING`. Does **not** set `profiler_enabled`, so nothing of the DRAM profiler is active, and the real-time profiler is disabled (it reads the same L1 rings). Fatal together with `TT_METAL_DEVICE_PROFILER`. |
| `TT_METAL_DRISC_PROFILER` | off | Streaming sub-option: arm the streaming producers but do **not** boot the built-in relays/receiver; the caller supplies its own DRISC drainer. Ignored without `TT_METAL_STREAMING_PROFILER=1`. |
| `TT_METAL_DEVICE_PROFILER_SYNC_EVENTS` | off | Compiles the CB/semaphore sync-event hooks (`-DPROFILE_SYNC_EVENTS`) for the critical-path tool. Streaming only: in mode 2 the JIT define is never emitted. |

**Streaming pipeline sizing**

| variable | default | effect |
|---|---|---|
| `TT_METAL_STREAMING_PROFILER_NRELAYS` | 0 (auto) | Forces the number of DRISC relays, one per DRAM view, in `[1, 8]`. 0 leaves it to bring-up, which takes `min(relay cap, DRAM views)`; a forced value above the view count is clamped there. |
| `TT_METAL_STREAMING_PROFILER_DRAM_MB` | 128 | Per-relay GDDR spool ring, MiB. Non-zero makes each relay DMA frames into a ring in its own DRAM bank and forward them to the host FIFO from a non-blocking pump, so the service loop never touches the PCIe tile and host-side pressure lands in spool occupancy instead of in the sweep interval. **0 selects direct push.** Capped at 4095 (32-bit ring arithmetic). |
| `TT_METAL_STREAMING_PROFILER_FIFO_MB` | 64 | Host FIFO per D2H socket, MiB, `[1, 3584]`. The pipeline's only elasticity in a direct-push run. Plain mmap + IOMMU host RAM reached by a full 64-bit NoC/PCIe address: costs no TLB window and has no channel cap. The 3.5 GiB cap is the socket's 32-bit byte size and the device's wrap-safe 32-bit credit arithmetic. |
| `TT_METAL_STREAMING_PROFILER_RING_MB` | 512 | Host-side verbatim-frame ring the receiver thread fills and the decode threads drain, MiB. The capture's elastic buffer; at ~9.8 wire bytes per zone the default holds ~55 M zones per stream. |
| `TT_METAL_STREAMING_PROFILER_DECODE_THREADS` | 2 | Decode threads per device, clamped by bring-up to the number of relay streams. |
| `TT_METAL_STREAMING_PROFILER_SHIP_MIN_PCT` | 25 | A relay defers shipping a live core until its fullest lane holds at least this percent of its own ring, unless the core aged out. 0 ships every live core every sweep; values past 50 are capped by the kernel's half-ring lane trigger. Per-lane, not per-span: the producer that blocks is always a lane. The measured stall-free band ends between 30 and 35. |
| `TT_METAL_STREAMING_PROFILER_WRITER_TIMEOUT_S` | 120 | How long the receiver waits for a stalled consumer before reporting it, seconds. |

**Consumers, all off unless set**

| variable | effect |
|---|---|
| `TT_METAL_STREAMING_PROFILER_TRACY=1` | Attach the built-in Tracy sink. Off by default because the primary consumers are the registered ones and Tracy is one more, expensive, consumer. |
| `TT_METAL_STREAMING_PROFILER_OPS_CSV=<path>` | Per-op CSV consumer (one row per op; see §1.7). |
| `TT_METAL_STREAMING_PROFILER_ZONE_CSV=<path>` | Per-zone CSV consumer (one row per zone). |
| `TT_METAL_STREAMING_PROFILER_STALL_CSV=<path>` | Producer-stall timeline CSV consumer (PRODUCER-STALL zones). |

A plain `TT_METAL_STREAMING_PROFILER=1` run therefore drains and decodes but writes nothing; that is the configuration the
knee sweeps use, since every consumer adds host-side cost.

**Outside rtoptions**: `TT_METAL_STREAMING_PROFILER_FULL_MESH=RxC` is read by a raw `getenv` in
`test_streaming_profiler_zones.cpp` only, to open the whole mesh for the test workload. It is not a profiler option.

**Mode 2 only**: `TT_METAL_DEVICE_PROFILER_DISPATCH`, `TT_METAL_DEVICE_PROFILER_NOC_EVENTS`, `TT_METAL_PROFILER_SYNC` and
the other legacy DRAM-profiler options ride on `profiler_enabled` and are unchanged from main.

### 1.3 Enable (Blackhole)

```
export TT_METAL_STREAMING_PROFILER=1
```

One switch — it also arms the device-side markers. Leave `TT_METAL_DEVICE_PROFILER` unset.

### 1.4 Register a callback

From `tt_metal/tools/profiler/streaming_profiler_consumer.hpp`:

```cpp
auto h = streaming_profiler::register_consumer("my-sink",
    [](const streaming_profiler::StreamingProfilerRecordBatch& b) { /* ... */ });
// later: streaming_profiler::unregister_consumer(h);
```

Register any time — before the device opens or mid-capture. Your callback runs on its own
thread; if you're slow you drop only your own records (`b.dropped_delta`), never anyone
else's. The batch span is only valid during the call, so copy what you keep.

### 1.5 What you get

Zones arrive **whole**: a zone is one record with a start and a duration. On the wire the device
ships most zones atomically (one 3-word packet at scope close, carrying end + duration); the kinds
that still ship as start/end pairs (the producer-stall zone, the >3.2 s long-zone fallback, DRISC
relay self-zones) are paired for you on the host. Either way you never see halves.

```cpp
enum class StreamingProfilerRecType : uint32_t {
    Zone = 1,   // a complete zone: data.zone = {start, duration}
    Data = 3,   // point marker with payload: data.ts; payload follows via Ext (+ Cont)
    Event = 4,  // point marker, no payload: data.ts; complete in itself
    Ext = 5,    // Data continuation: id = payload word count, data.ext = payload words 1-2
    Cont = 6,   // one uint64 of Data payload (words 3 and up): data.payload
};

struct StreamingProfilerRecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;  // which (core, RISC) stream: lane = core_index * 5 + risc
    uint32_t dev : 3;    // device index into the capture context
    StreamingProfilerRecType type : 3;
};

struct StreamingProfilerRec {
    // The active member is decided by meta.type.
    union {
        struct {
            uint64_t start;     // device timestamp of the zone open
            uint64_t duration;  // device cycles
        } zone;
        uint64_t ts;       // Data / Event
        uint64_t ext;      // Ext
        uint64_t payload;  // Cont
    } data;
    uint32_t id;            // structural zone id -> resolves to the zone's name
    StreamingProfilerRecMeta meta;
    uint32_t prog;          // runtime host-id of the op this lane is executing (0 = none yet)
};
```

Ordering: cross-lane interleaving is arbitrary — key any state you keep by
`(meta.dev, meta.lane)`. A zone is delivered when it **ends**, so under nesting
`data.zone.start` isn't monotonic within a lane; `start + duration` is complete information
either way.

### 1.6 The call pattern

The ops-CSV consumer in `tt_metal/tools/profiler/streaming_profiler_ops_csv.{hpp,cpp}` is the full working
reference.

```cpp
void MyConsumer::operator()(const streaming_profiler::StreamingProfilerRecordBatch& batch) {
    names_.refresh();  // ZoneNameMirror member: names arrive as kernels JIT, refresh once per batch
    for (const auto& r : batch.records) {
        if (r.meta.type != StreamingProfilerRecType::Zone) continue;
        std::string_view name = names_.lookup(r.id);                          // zone name
        const auto& lane = batch.context->devices[r.meta.dev].lanes[r.meta.lane];  // chip, core x/y, risc, role
        // ... aggregate: e.g. per-op rows keyed on r.prog, using r.data.zone.start / .duration ...
    }
}
```

### 1.7 Try it end to end

The built-in CSV consumer:

```
TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_OPS_CSV=/tmp/ops.csv python your_model.py
```

One row per op (kernel start/end unions, per-core and per-RISC splits), joinable against a
classic `ops_perf_results` CSV on GLOBAL CALL COUNT.
`tests/ttnn/tracy/test_streaming_profiler_ops_csv.py` runs exactly this;
`tests/ttnn/tracy/test_streaming_profiler.py` is the end-to-end pytest for the profiler itself.

The synthetic zone workload used throughout §6 is
`tt_metal/programming_examples/profiler/test_streaming_profiler_zones` (built by `./build_metal.sh` with
the programming examples). A clean run decodes exactly `iters × 6000 + 600` records on a 12×10 grid
(10 zones × 600 lanes + 1 trailing per lane); a stalling run's record surplus equals the device L1 stall
counter to the unit (§4.2).

A built-in callback pushes zones to a Tracy timeline; enable it with
`TT_METAL_STREAMING_PROFILER_TRACY=1`. What the three device primitives look like there is §4.1.

### 1.8 Two rules

Don't register/unregister from inside a callback, and don't block in it.

## 2. Architecture (current design)

The design as it stands on the `streaming-profiler-v3` branch (2026-09). Everything in §3 is a design this
one replaced; the measurements behind every choice here are in §6, and the section pointers below name
them.

### 2.1 The pipeline

```
worker RISC ──2/3/5-word zone packets──▶ L1 SPSC marker ring, one per RISC (512 words), heads + tails in a
    │                                     per-core control vector.  A producer BLOCKS on a full ring: the
    │                                     pipeline is lossless end to end (the stall is itself a zone).
    ▼
DRISC relay — one per DRAM view (≤ 8), resident on the bank's spare DRISC from device open to teardown
    CV pass    read every core's five tails (32 B/core), decide the ship set (SHIP_MIN_PCT / half-ring
               lane trigger / age); idle backoff grows the poll gap to a 5 µs ceiling
    gather     read each live run STRAIGHT to its packed wire offset in a staging slot, so the staged slot
               IS the frame's wire image (16 B src≡dst congruence pads, spsc_span_pack_pad())
    ship       ├─ direct push (DRAM_MB=0):   ONE PCIe write per frame into the relay's own D2H socket FIFO
               └─ GDDR spool  (DRAM_MB>0, default 128 MiB): GDDR-DMA the frame into a ring in the relay's
                  own bank; a non-blocking pump drains spool ▶ bounce slot ▶ host FIFO, so the sweep never
                  touches the PCIe tile and host pressure lands in spool occupancy, not in the sweep
    release    posted head write-backs hand ring space back to the producers
    ▼
D2H socket host FIFO, one per relay (FIFO_MB, default 64 MiB; plain pinned mmap, no TLB window)
    ▼
host receiver — one ingest thread per socket stream: poll + frame + copy + ack fused; frames are
    NT-streamed VERBATIM from the FIFO into a per-stream BroadcastRing (RING_MB, default 512 MiB) and
    acked as they land.  The ring, not the FIFO, is the capture's elastic buffer; one ring per stream
    keeps each ring single-writer.
    ▼
consumers — every consumer thread decodes and pairs for itself before its callback: the registered
    callbacks (§1.4), the Tracy sink (opt-in), ops / zone / stall CSV, and an internal audit consumer that
    decodes every stream for the wire-integrity report ingest (a pure copier) cannot produce.
```

### 2.2 The relay

`tt_metal/tools/profiler/kernels/streaming_profiler_relay.cpp`. Resident on a DRAM bank's spare DRISC, it
polls its slice of the worker SPSC rings, gathers live runs into wire frames, spools them in its own GDDR
bank and pumps them to the host FIFO over a D2H socket.

- **One kind of DRISC.** There are no longer fillers and movers (§3.1): every relay owns a slice of the
  worker grid *and* its own D2H socket. Direct push with gather-read packing measured faster than the
  DRAM-frame-ring design it replaced (§N+71), and the GDDR spool re-adds device-side elasticity without
  a second core — the relay's own bank, its own DMA engine, no NoC read of another core's ring.
- **Placement.** One relay per DRAM view, `boot_device` refusing two relays on one physical core (two DRAM
  views can resolve to the same NoC core, §N+40). The NIU has to be in stream mode to initiate NoC traffic
  at all; all relays' NIUs are flipped in **one** launch of `kernels/drisc_niu_mode.cpp` before any relay is
  resident (§N+34 — one launch per flip ran a `dram_barrier` across an already-stream-mode core and hung
  the box).
- **Launch.** `detail::LaunchProgram(..., force_slow_dispatch=true)`, outside the command queue: a
  DRAM-only program touches no fast-dispatch resource, so it stays resident across every workload, while
  going through the CQ would deadlock the first `Finish()`.
- **Reads on the NoC the writes do not use** (`kReadNoc = NOC_INDEX == 0 ? 1 : 0`), a static VC for PCIe
  pushes spread across relays by the host, and `NOC_MAX_BURST_SIZE` chunking on every host write.
- **Ship decision, per lane.** A core ships when its fullest lane holds ≥ `SHIP_MIN_PCT` of its ring, when it
  aged out, or at the half-ring lane trigger; a head only reaches a producer on a ship, so the idle backoff
  ceiling (5 µs) stays below a lane's fill time at high rates. The CV pass's tails are authoritative: a frame
  claims exactly `[mirror, tail-at-the-CV-read)`, which is safe at any gather lag because a producer only
  appends *past* a published tail (§N+71).
- **Frames.** Prefix + heads/tails/`SPSC_CORE_XY` control words synthesized from the mirror and the CV
  read (the only control words the decoder reads) + each RISC's live window packed flat with 0–3 pad
  words per run bringing the wire offset to its ring phase. A full span of five full rings and their pads
  always fits a slot, so the lane walk has no room gate. Frames are page-rounded (64 B pages); pad and
  page-tail words are never written and the host walks past them.
- **Spool mode** (`DRAM_MB` ≠ 0, the default). Two GDDR-DMA TX streams — stream 0 staging → spool, stream 1
  spool → bounce — with two bounce slots carved out of the staging arena and up to 15 DMA writes
  outstanding. A light workload that never reaches the occupancy bands is still shipped within ~50 ms
  (`kSpoolFreshCycles`), so host staleness is bounded.
- **Teardown** talks to a relay through its stop word: 1 = quiesce (every wait holds while the last frames
  drain, up to 1 s), 2 = kill switch (abandon waits, free the NIU). The relay publishes `0xD09E****` in its
  done word once its last page is out. Producers boot unarmed and are armed (`PROFILER_ARMED`) only on the
  cores a relay drains, once every relay is up; a path where a relay does not come up leaves them unarmed, so a
  missing relay can never wedge the workload (§N+24).
- **Not instrumented itself.** The relay opts out of the producer instrumentation (no drainer serves a DRAM
  core, so the ring would be write-only dead weight — §N+72); the DRISC code region is 11,264 B and every
  feature has had to be fitted into it (§N+43, §N+64).

### 2.3 The host

- `tt_metal/tools/profiler/streaming_profiler.{hpp,cpp}` — control plane: one `StreamingProfiler` per
  `MeshDevice`. Constructing it boots the relays on every eligible local Blackhole device and starts the
  receiver; destroying it (or `stop()`) quiesces the relays, drains the receiver, verifies capture
  completeness (every worker lane's tail against the receiver's consumed mirror) and leaves the resident
  idle FW alone. It also owns the host↔device clock sync: a least-squares fit of 100 `(host_time,
  device_cycles)` samples spaced 500 µs apart, one **origin per relay core measured on that core** (the
  DRAM-tile and Tensix wall clocks share a zero at chip reset but not a duty cycle, so a worker anchor can be
  minutes wrong for a DRAM core — §N+46) and **one shared frequency** for every context on the chip
  (per-core slopes are biased and fan the rows apart — §N+47/§N+48). Residual measured with a common
  trigger: 0.18 µs (§N+48); cross-domain offset constant to 0.1 ppm while both domains are active (§N+49).
- `streaming_profiler_receiver.{hpp,cpp}` — ingest and rings as in §2.1. A lagging consumer drops its own
  oldest lines (counted per consumer); decode recovers from a device-side drop by head adoption, resync
  counters and re-anchoring timestamps at the next absolute zone. Decode is vectorized (AVX2 zone blocks,
  §N+54/§N+68) and memory-latency bound (§N+55–§N+57).
- `spsc_marker_decode.hpp` (the decode single source of truth), `spsc_packet.h` (plain-C packet
  constants shared with the device's `ppfmt`), `hw/inc/hostdev/streaming_profiler_common.h` (span/frame
  geometry, pack-pad rule, `SPSC_SPAN_RAW_FLAG`).
- Consumers: `streaming_profiler_consumer.{hpp,cpp}` (`register_consumer`, `ZoneNameMirror`),
  `streaming_profiler_ops_csv`, `streaming_profiler_zone_csv`, `streaming_profiler_stall_csv`,
  `streaming_profiler_tracy_consumer` + `streaming_profiler_tracy_handler` (the Tracy sink; per-relay
  anchors, k-way merge of a context's lanes by timestamp so Tracy's 2^31-tick unwrap heuristic never fires —
  §4.2).
- Device producer: `kernel_profiler_streaming.hpp`, selected by `-DPROFILE_STREAMING`. Zone ids are 27-bit
  structural ids resolved to names per ELF on the host; every RISC emits its own `STICKY_PROG` at launch so
  `rec.prog` is exact on every lane (§N+60).

### 2.4 Sizing and where the numbers come from

| knob | default | what it buys | measured |
|---|---|---|---|
| relays | one per DRAM view | the sweep is O(cores per relay); fewer cores per relay is the only lever on the device-side knee | §N+28, §N+40, §N+71 |
| `SHIP_MIN_PCT` | 25 | fewer, fuller frames; stall-free band ends at 30–35 | §N+65, §N+71 |
| `FIFO_MB` | 64 | direct-push elasticity; 3 GiB holds a whole 150k-iteration capture and takes the host out of the knee | §N+71 |
| `DRAM_MB` | 128 | spool runway per relay; a ring absorbs a running deficit, not bursts — it is runway, not headroom | §N+39, §N+65 |
| `RING_MB` | 512 | ~55 M zones per stream; the host ring, not the device, is the first thing to lose data as volume grows | §N+39, §N+52 |
| `DECODE_THREADS` | 2 | the pipeline knee at 2 threads is host-decode-bound (~90 % duty); 6 threads exposes the device-side knee | §N+53–§N+57, §N+71 |

Headline measurements on the current shape (bh-lb-120, 150k iterations, 11×10 grid, single device):
pipeline knee `--delay` **106** with 2 decode threads (count-exact at 957,000,550 records, 0 anomalies);
device-side (filler) knee **75** with the host wall removed by FIFO or threads (§N+71, §N+72). End-to-end
overhead on real models: ResNet-50 batch 16 +2 % to +6 %, batch 32 +2.64 %, DistilBERT +5.8 %, Mistral-7B
decode +3.7 %, Llama-3.1-8B decode +2.5 % — a roughly fixed per-op cost that dilutes as the op grows
(§N+44, §N+45, §N+50). Op-level agreement with the classic device profiler: medians within ~1 % on
ResNet-50, +8–11 % on a zone-dense matmul microbenchmark (the per-zone scope cost) (§N+59–§N+63).

## 3. Design history (superseded designs)

**Nothing in this section describes the current code.** It is kept because the measurements in §6 were
taken on these designs and read wrong without them. Each sub-section states what it was and when it was
superseded; where the text names files or knobs that no longer exist, the mapping table below (and the
larger one at the top of §6) gives the current name. The text itself is not rewritten.

| era | design | measured in | superseded by |
|---|---|---|---|
| 2026-07 → 08-04 | one DRISC "conduit" drainer sweeping the whole grid, fused 10,496 B span read, raw span frames to one D2H socket; a Tensix-BRISC drainer arm for comparison | §6 (pre-§N), §N, §N+1 | two drainers |
| 2026-08-08 | `kNSockets = 2`: two drainers on the only two DRAM cores measured safe for host-facing duty (`0-0`, `9-0`) | §N+29 … §N+38 | the role split |
| 2026-08-11 | **fillers + movers**: 4 fillers sweep 30 cores each into DRAM frame rings; 2 dual-ring movers pull frames from the rings into their sockets — the design §3.1 describes | §N+39 … §N+65 | DMA mover |
| 2026-08-21 | DMA mover: rings co-located on the mover's banks, read by the GDDR-DMA engine, per-frame sequence verification | §N+66 … §N+70 | direct push |
| 2026-08-25 | **direct push**: fillers own the sockets, movers deleted, gather-READ packing, one PCIe write per frame — the plan in §6.2 | §N+71, §N+72 | the relay |
| 2026-09 | **the relay** (§2): one kernel, one per DRAM view, direct push or per-relay GDDR spool | — | current |

Names in the historical text and what they are today:

| in the text | today |
|---|---|
| `drisc_profiler_drain.cpp` (the filler/mover kernel, role by `kRole` compile arg), `drisc_profiler_filler.cpp` | `tt_metal/tools/profiler/kernels/streaming_profiler_relay.cpp` |
| `drisc_drain_common.hpp`, `test_kernels/misc/drisc_drain_frame.h`, the streaming constants that were in `profiler_common.h` | `tt_metal/hw/inc/hostdev/streaming_profiler_common.h` (the DRAM profiler keeps `profiler_common.h`) |
| `perf_debug_profiler.{hpp,cpp}`, `PerfDebugProfiler` | `tt_metal/tools/profiler/streaming_profiler.{hpp,cpp}`, `StreamingProfiler` |
| `PerfDebugTracyHandler`, `perf_debug_tracy_handler` | `StreamingProfilerTracyHandler`, `streaming_profiler_tracy_handler.{hpp,cpp}` |
| the host "writer"/"decoder" threads, `D2HSocket::read()` memcpy path, receiver v2 | `streaming_profiler_receiver.{hpp,cpp}` |
| host record ring of 24 B `StreamingProfilerRec` (`BroadcastRing`, `RING_RECS`) | per-stream `BroadcastRing` of verbatim frames (`RING_MB`); records are decoded per consumer |
| `drisc_niu_mode.cpp` | `tt_metal/tools/profiler/kernels/drisc_niu_mode.cpp` (same job) |
| `test_perf_debug_zones` | `tt_metal/programming_examples/profiler/test_streaming_profiler_zones` |
| `TT_METAL_PERF_DEBUG_*` | `TT_METAL_STREAMING_PROFILER_*` — full table at the top of §6 |
| `TT_METAL_STREAMING_PROFILER_{DRISC_ZONES,ROLE_RING_MB,SHIP_REPEAT}` as they appear in §3.1 and §3.3 | mechanical renames of `TT_METAL_PERF_DEBUG_*` knobs of the fillers/movers kernel; removed with it, no current equivalent |
| `ARCHITECTURE.md`, `HANGS.md`, `README.md`, `DIRECT_PUSH_PLAN.md`, `FINDINGS.md` | §3.1, §3.2, §3.3, §6.2, §6 of this document |
| `tests/tt_metal/tt_metal/api/test_dram_kernels.cpp` (`DramKernelDRISCScatterFixture`, `*DRISC*` gtests), `test_kernels/misc/drisc_{rdrbench,adaptive_drain,service_workers}.cpp`, `misc/socket/drisc_{d2h_egress,drain_to_host}.cpp`, `profiler_zone_producer{,_compute}.cpp` | deleted 2026-09 with the streaming profiler landing; their measurements stay in §6 as history |

### 3.1 Fillers + movers and the DRAM frame ring (2026-08-11; superseded 2026-08-25)

*Formerly `tools/drisc_drain/ARCHITECTURE.md`.* This is the six-DRISC design: four **fillers** sweeping
30 worker cores each into a DRAM frame ring, two **movers** each draining two rings into a D2H socket, with
optional self-profiling of the drainers. The direct-push plan (§6.2) deleted the movers and the rings; the
relay (§2) is what replaced the fillers. Its buffer-stack table and cost model are still the best
explanation of why a frame is 2,640 words and why a whole span fits one NoC burst.

Measured on bh-26 (Blackhole, single card), branch `mo/drisc_drain_fast`, 2026-08-11.

#### Naming, because "ring" means four different things here

Use these names. Confusing them has already cost real debugging time.

| name used in this file | lives in | holds | who writes | who reads |
|---|---|---|---|---|
| **L1 marker ring** | worker L1, one per RISC | 2-word markers | the worker RISC (producer) | the FILLER's bulk span read |
| **DRAM frame ring** | device DRAM, one per FILLER | whole frames | its FILLER | its MOVER |
| **socket FIFO** | host RAM, one per MOVER | 64 B pages | its MOVER, over PCIe | the host writer thread |
| **host record ring** | host RAM, one shared | 24 B `StreamingProfilerRec` | the decoder threads | the consumer thread -> Tracy |

So: `ring_ensure_room` / `PROFILER_STALL_ZONE` / `SPSC_STALL_COUNT_0` are about the **L1 marker ring**.
`ring-room waits` and `head`/`tail` are about the **DRAM frame ring**. `reserve_pages` / credit-wait /
`fifo_pages` are about the **socket FIFO**. `BroadcastRing` / `dropped` is the **host record ring**.

#### The six DRISCs

A DRISC only exists on a DRAM core, and metal exposes exactly ONE per core
(`bh_hal_dram.cpp` registers a single DM processor class, "DRISC0"; `DramConfig` has no `.processor`).

| DRISC | role | bank (DRAM view) | core | NoC row | job |
|---|---|---|---|---|---|
| 0 | FILLER | 5 | `9-9` | y!=0 | sweep worker cores `[0,30)` -> DRAM frame ring on bank 1 |
| 1 | FILLER | 6 | `9-5` | y!=0 | sweep `[30,60)` -> ring on bank 2 |
| 2 | FILLER | 4 | `9-2` | y!=0 | sweep `[60,90)` -> ring on bank 4 |
| 3 | FILLER | 1 | `0-3` | y!=0 | sweep `[90,120)` -> ring on bank 5 |
| 4 | MOVER | 0 | `0-0` | **y==0** | rings of fillers **0 and 2** -> socket FIFO 0 -> host |
| 5 | MOVER | 3 | `9-0` | **y==0** | rings of fillers **1 and 3** -> socket FIFO 1 -> host |

With `TT_METAL_STREAMING_PROFILER_DRISC_ZONES=N` **every one of the six also becomes a PRODUCER of its own zones**,
framed as a worker span and pushed down the path it already owns -- a filler into its own DRAM frame ring, a
mover into its socket FIFO. Each therefore gets its own Tracy row at its own NoC coords, and the lane space
grows by `n_drisc * 5` (lanes `[600,630)` at 120 cores). Default OFF; see "Self-profiling" below and
FINDINGS §N+41.

**Why 4 + 2 and not 4 + 4.** The knee is the FILLER's scan over its slice (FINDINGS §N+28), so fillers are
the thing to multiply: 4 of them at 30 cores each moved the knee from delay 40-60 to delay 15 (§N+40).
Movers cannot follow, for two independent reasons: only NoC row y==0 is measured safe for HOST-FACING duty
(§N+29: y==0 -> 1/75 failures, y!=0 -> 16/125), which is exactly two cores on this part, and the socket
FIFO's TLB budget is `kNSockets * nwin <= 16` with 12 MiB costing nwin=7. So each mover drains TWO rings
instead. It has the headroom: a mover is ~97% idle even after doubling (idle sweep 0.6-0.8 us, ~330 busy
sweeps of ~200,000).

Mover m takes fillers m, m+2 -- **strided, not adjacent**, so each socket carries one low-half slice and
one high-half slice of the grid rather than both halves of one end.

Two 25-run blocks on bank 5 -- §N+29's worst core at 5/25 as a full-job drainer -- cleared y!=0 for FILLER
duty: 0/25 with the NIU merely held in stream mode, 0/25 doing filler-only work. Every DRISC needs its NIU
in stream mode; in the default NOC2AXI mode it cannot initiate NoC at all.

**Ring placement is no longer disjoint from drainer channels, deliberately.** With 6 drainer channels and 4
rings against 7 allocator banks, the old "a ring shares a channel with no drainer" insurance is unreachable.
What is kept is the part with evidence: a ring is never on a MOVER bank (0, 3), because host-facing duty is
where the §N+29 hazard was measured. Rings land on banks 1, 2, 4, 5 -- three of which also host a filler.
Measured with that overlap in place (25 runs): 0 ring-room waits, 0 impossible-head reads, staged == moved
on all four rings, and the knee improved. A ring's traffic terminates at its channel's PREFERRED WORKER
endpoint while a drainer sits on that channel's unused subchannel, so even a shared channel means a
different core and a different NIU, and ~1.4 GB/s each way is a rounding error against a GDDR channel.

Also enforced now: **two DRISCs must never land on the same core.** `pick_unused_dram_logical_core()` takes
a DRAM VIEW and knows nothing about other views resolving to the same physical port -- §N+29 records views
0 and 7 both as NoC core `0-0` -- so `boot_device` `TT_FATAL`s on a duplicate. Two resident kernels sharing
one L1 would overlap staging, socket config, results and handshake with no counter noticing.

#### Buffer stack and sizes

| level | where | unit | size | count |
|---|---|---|---|---|
| **L1 marker ring** | worker L1 | 2-word (8 B) marker | 512 words = **2,048 B** -> **256 markers** | 1 per RISC, 5 per core |
| control vector | worker L1 | heads + stall counters | 64 words = **256 B** | 1 per core |
| **span** (unit of transfer) | worker L1 | 64 ctrl + 5 x 512 ring words | 2,624 words = **10,496 B** | 1 per core |
| **staging slot** (frame) | DRISC L1 | 16-word prefix + span | 2,640 words = **10,560 B** = **165 pages** | **7** per DRISC (73,920 B) |
| **self-frame slot** | DRISC L1 | one staging slot, reused as a frame the DRISC authors | **10,560 B** (of which 2,368 B carry data) | 1 per DRISC, only with self-profiling on |
| **DRAM frame ring** | device DRAM | one frame | **64 MiB = 6,355 frames** | 1 per FILLER (**4**) |
| **socket FIFO** | host RAM | 64 B page | 196,608 pages = **12 MiB** | 1 per MOVER (2) |
| host read chunk | host RAM | pooled buffer | <= 1,024 pages = **64 KB** per read | pool <= 4,096 |
| **host record ring** | host RAM | 24 B `StreamingProfilerRec` | 4 Mi default = 96 MiB (runs use 16 Mi = **384 MiB**) | 1 shared |

Geometry notes that are load-bearing, not incidental:
- 2,640 words is a whole number of 64 B pages, so a frame never needs padding, and the bulk span read
  lands at `slot + 64 B` so prefix and span are contiguous and **one** NoC write ships the frame.
- DRAM frame rings are sized a whole multiple of 165 pages, so **a frame never wraps** and the device
  never needs a split write.
- A span fits ONE NoC burst: 10,496 B < `NOC_MAX_BURST_SIZE` = 256 x 64 = 16 KB.
- Only **3** of the 7 staging slots are ever live (`kGenSlots = nstage/2`), so just 3 cores' reads are
  in flight and the 7th slot is unused. That is why the sweep is read-latency bound and cannot be
  widened without more DRISC L1.
- **DRISC L1 has no room for an eighth slot**, which is why self-profiling takes the seventh rather than
  adding one: UNRESERVED is 86,448 B, and 7 slots plus the 8 KB socket-config reserve, 4 KB head scratch and
  the misc block leave **1,792 B**. With the feature on the kernel is handed `nstage - 1` slots and uses index
  `kNStage` for its self frame, so L1 is unchanged and a MOVER's largest batch drops 7 -> 6 (a filler's is
  bounded by `kGenSlots = 3` either way).
- A full span carries at most 1,280 markers (2,560 ring words / 2) across 165 pages = **7.80
  markers/page** against a host batch bound of 8.00 -- a 2.5% margin, which is why that batch
  occasionally fills (handled by flush-and-continue, never by dropping).
- The **socket FIFO cannot be enlarged**: it is mapped through device TLB windows,
  `kNSockets * nwin <= 16`, and 12 MiB already uses 14. That is the whole reason staging moved to DRAM.
- A dual-ring MOVER does **not** split its 7 staging slots between its two rings. It visits them
  sequentially, each with the whole staging area, separated by the write barrier that staging reuse needed
  anyway -- so `max batch` stays **7 per peer** (measured on all four peers, every run). Splitting the slots
  would only pay if the two pushes could overlap, and they cannot: both go into ONE socket.

#### Activity, per role

**FILLER**, per worker core: one bulk NoC read of the whole span into a staging slot; reads go out on
`kReadNoc` (the NoC the writes do not use: `NOC_INDEX == 0 ? 1 : 0`, so reads on NoC 1 and writes on
NoC 0 by default). Software-pipelined across two staging generations -- generation G reads while G^1
ships -- with the read barrier issued **LAST**, which is what stops read and ship serialising. Then
inspect the control vector, patch the prefix, write the frame into its DRAM frame ring, and publish
`head` only after a *flushed* barrier. At exit it waits (bounded) until `tail == frames_staged`, so it
never reports a stale mirror -- `inflight = frames_staged - *hs_tail` IS the ring-room predicate.

**MOVER**: no worker grid at all, hence `proc 0.0 us` in its phase line. **For each of its two peers in
turn**: NoC-read that FILLER's `head` out of that FILLER's L1, pull up to 7 frames from that ring into its
own staging, write `tail` back to release ring space, ship to the socket FIFO in `NOC_MAX_BURST_SIZE`
chunks, then one write barrier -- which covers the PCIe push, the tail write, and the reuse of the scratch
word the next peer's tail write sources from. Per-peer state is strictly separate (`mv_tail`, `mv_moved`,
`mv_max_n`, `ring_hi`, the probe words and the live head/tail telemetry): one shared `mv_tail` across two
rings would ack frames on one ring that were only read from the other. Observed `max batch 7` per peer.

**EITHER ROLE, with self-profiling on**: emits its own 2-word markers into a 512-word ring inside its
self-frame slot, stamped with timestamps the drain loop has ALREADY read (`t_batch0`/`t_issue`, `stage_run`'s
`t0`/`t1`/`t2`, the barrier's `t_b0`) rather than fresh clock reads -- so a zone's duration is the same
quantity the matching `out[]` phase counter accumulates. At the end of a captured sweep it sets its own
`SPSC_CORE_XY` / head / tail in the frame's control vector and ships the slot through the same
`emit_run(kSelfSlot, 1)` the payload uses, then a bounded write barrier. Nothing about the wire format, the
ring protocol or the host decoder changes: the frame IS a worker span, and only ring 0 of its five is live
(`myRiscID == PROCESSOR_INDEX == 0` on a DRISC), so the host's per-RISC walk yields nothing from the rest.

#### Self-profiling: what the drainers say about themselves

Zone tree per drainer row: `DRISC-SWEEP` (depth 0) with `DRISC-READ`, `DRISC-READ-WAIT`, `DRISC-PROC` and
`DRISC-WR-BARRIER` as children, and `DRISC-CREDIT-WAIT` / `DRISC-WRITE` inside `DRISC-PROC` on a filler
(directly under `DRISC-SWEEP` on a mover, which has no proc phase). Per-occurrence means, bh-26:

| role | SWEEP | READ | READ-WAIT | PROC | CREDIT-WAIT | WRITE | WR-BARRIER |
|---|---|---|---|---|---|---|---|
| FILLER | 16.4-17.0 us | 128 ns | 67-69 ns | **825-864 ns** | 54 ns | 138-140 ns | 64-120 ns |
| MOVER  | 11.5-11.9 us | 1,425-1,774 ns | - | - | **2,698-2,781 ns** | 831-1,065 ns | 359-525 ns |

The two roles are bottlenecked on different things and their own zones say so: a filler's per-batch PROC
dwarfs everything else it does, while a mover's largest phase is the socket **credit wait** -- the quantity
§N+38 identified as setting the knee, now visible per occurrence, and ~1.6x its next-largest phase. The same
named phase differs by three orders of magnitude between the roles (54 ns of DRAM ring room on a filler
against 2.7 us of host FIFO credit on a mover), which is why one number for "the drainer" never meant anything.

**Superseded by full tracing (FINDINGS §N+41): `DRISC_ZONES=1` traces every sweep in one contiguous
window at sweep level, which measured CHEAPER than sampling because the per-sweep publish, not the markers,
was the cost. The sampler is described below for why it existed.** It triggered on discovered WORK, not on sweep number, because both roles are >99% idle (a filler
moves frames in ~114 of ~25,000 sweeps, a mover in ~350 of ~230,000). A sweep-number rule captured 1 working
sweep out of 55 on a filler and 11 of 64 on a mover. A mover arms before issuing its DRAM read, so its busy
visit is captured whole; a filler arms at the end of the first batch with live cores, so that sweep's earlier
batches are not recovered. An instrumented sweep that turns out idle is rewound or abandoned for free, and a
captured sweep is bounded to one ring (past that it is truncated, counted, and excluded from the counter
cross-check). Cost: **0.28-0.52% of a drainer's egress**, +4% on a filler's sweep time, 0 producer stalls at
delay 60 and 49-140 at delay 15 (a knee crossing).

**Sampling IDLE sweeps is off by default and should stay off** unless the question is specifically what an idle
poll costs. A drainer is resident from device open, so its idle sweeps span the whole process (~190 ms measured)
while a workload is a ~1.9 ms sliver: sampling them puts DRISC zones 187 ms before the first worker zone and
makes the rows unreadable. They are also not inert -- a filler's self frame is a real frame in its DRAM ring, so
its mover ships it, manufacturing mover credit-wait/write zones outside the workload window (measured: all 13
such zones followed a peer filler's publish by 1.7-2.4 us) and diluting the mover's credit-wait figure from
2.70-2.78 us down to 1.6-1.8. Work-triggered only, all six drainers start within 8-67 us of the first worker
zone; the fillers end just before the last one and the movers trail it by 2.5-2.9 ms, which is the ring's drain
tail.

#### Measured costs and occupancy

Per-sweep costs at 120 cores, delay 20, and they halve with the slice as the model predicts (2 fillers at 60
cores each -> 4 at 30):

| | 2 fillers x 60 cores | 4 fillers x 30 cores |
|---|---|---|
| FILLER idle sweep | 16.7 us | **8.2-8.5 us** |
| FILLER busy sweep | 27.8-28.0 us | **13.1-14.0 us** |
| FILLER worst sweep | 39.0-40.3 us | **17.2-19.7 us** |
| of which `proc` (the scan) | 15.0-15.3 us | **7.5 us** |
| MOVER idle sweep | 0.3-0.4 us | 0.6-0.8 us (two head reads) |
| MOVER busy sweep | 5.2-5.5 us x ~515 | 13.5-13.6 us x ~330 |
| MOVER worst sweep | 33.8-37.5 us | 51.9-52.9 us (credit-wait 28-34) |

FILLER blocking wait is **0.5 us** (it was 50-97 us of socket-FIFO credit-wait before the split). A MOVER is
still ~97% idle after doubling: 191,357 idle sweeps x 0.8 us = 153 ms of a 161 ms wall.

DRAM frame ring high-water at 5k zones/RISC: **328-770 of 6,355 frames per ring** across all four rings,
against 73-1,266 for the 2-ring configuration because each ring now carries a quarter of the grid. `ring-room
waits 0` everywhere, including at delay 5 where producers stall ~11,000 times. That is diagnostic and it is
the whole basis of the 4-filler change: the FILLERs are never blocked by the DRAM frame ring or by egress,
so what limits low delays is the SWEEP.

#### DRAM cost, and how big the ring actually needs to be

Measured, not estimated -- see FINDINGS §N+39 for the two sweeps and §N+40 for the 4-ring cost.

**The rings live in the HAL's per-bank DRAM PROFILER region** (channel-relative `0x40`), not in a
`MeshBuffer`. That region is reserved at the same offset in EVERY bank, and
`get_profiler_dram_bank_size_for_hal_allocation()` sizes it to
`max(old_profiler_size, streaming_profiler_dram_region_bytes_per_risc())` -- so `TT_METAL_STREAMING_PROFILER_ROLE_RING_MB`
is one knob for both the region and the ring, and the ring adapts to whatever was actually reserved
(`frames = region_bytes / 10,560`).

The consequence that matters: **the cost is per BANK and does not depend on how many rings you place.**
bh-26 has **7** allocator DRAM banks (8 channels, 1 harvested; confirmed on silicon via the ring-bank
validation message), so a 64 MiB setting reserves **448 MiB** whether 1 ring or 7 rings sit in it. Going from
2 rings to 4 therefore cost **zero additional DRAM** -- it moved 128 MiB of the reservation from "region no
ring uses" to "region carrying a ring", nothing more. There are still 3 unused banks, so a 5th, 6th and 7th
ring would also be free. For scale, the OLD profiler's region is 4.58 MiB/bank = **32.0 MiB** total, so 64
MiB puts us at **14x** the profiler we are replacing.

Which bank a ring lives in is IRRELEVANT to footprint. The only lever is
`TT_METAL_STREAMING_PROFILER_ROLE_RING_MB` itself.

**The earlier claim here -- that the ring "should come down a long way" because under a fifth is ever
used -- was wrong, and wrong for an instructive reason.** The 8.4%-18.9% high-water it rested on was
measured at 5,000 zones/RISC only. The ring does not have a steady-state occupancy: its high-water
tracks TOTAL VOLUME and never plateaus (14% at 5k zones/RISC, 49% at 10k, 92% at 15k, 100% at 20k).
The movers drain permanently slower than the fillers stage, so **the ring is runway, not headroom** --
it buys a fixed number of zones before back-pressure reaches producers, and 64 MiB buys ~16-17k
zones/RISC. A low high-water means the workload ended first, not that the ring is oversized.

So the ring size is a function of the capture VOLUME you intend to support:

| ring | frames | DRAM (x7 banks) | stall-free at 5k zones/RISC |
|---|---|---|---|
| 64 MiB (default) | 6355 | 448 MiB | yes (runway ~16-17k zones) |
| **12 MiB** | 1191 | **84 MiB** | **yes, 3/3 -- smallest that is** |
| 10 MiB | 992 | 70 MiB | 2 of 3 |
| 9 MiB | 893 | 63 MiB | no (0.7-1.7k stalls) |
| 6 MiB | 595 | 42 MiB | no (~1.8k stalls) |

12 MiB is the right default for 5k-zone captures -- 84 MiB, 2.6x the old profiler instead of 14x -- with
the env var as the escape hatch for high-volume runs. Undersizing costs producer perturbation but never
correctness: every size down to 6 MiB dropped **zero** records with **zero** timestamp regressions,
because the ring fills, the filler waits for room, and the producers stall. Nothing is lost.

Two structural notes for anyone tuning this. The lanes are **asymmetric** -- filler 1 fills first in
every undersized run while filler 0 sometimes never fills at all (at 8 MiB: 794/794 with 36 ring-room
waits vs 648/794 with 0), so the required size is set by the slowest lane, and balancing them buys
more than any sizing change. And a bigger ring cannot rescue a long capture: the host record ring is the
real volume ceiling (drops begin at 15k zones/RISC, one sweep point BEFORE producers stall), so doubling
the ring just doubles the runway.

**Not re-measured at 4 rings:** the sizing table above (12 MiB smallest stall-free at 5k zones/RISC, and the
20k zones/RISC volume knee) was taken with 2 rings. Four rings each carry a quarter of the grid, so per-ring
high-water roughly halved at 5k zones -- which suggests the same total runway spread over twice as many
rings, i.e. a smaller per-ring minimum. That is an inference, not a measurement; re-run §N+39's Sweep A
before lowering `ROLE_RING_MB` on the 4-ring configuration.

### 3.2 Blackhole device/host hang runbook (2026-08-07 … 08-08)

*Formerly `tools/drisc_drain/HANGS.md`.* Written against the single/dual-drainer and fillers+movers
kernels, but the failure taxonomy (WEDGE / TEARDOWN / DEGRADED / VM FREEZE), the classification and
recovery rules are about the card, the link and the box, and they hold for any resident DRISC. The harness
scripts it names (`drisc_hang_harness.sh`, `drisc_hang_compare.sh`, `drisc_reclassify.py`, `drisc_wedge_watch.sh`) were removed with the DRISC experiments in 2026-09; the listings below are kept as the record of how the runs were scored.

The **device- and host-level** hangs seen on Blackhole IRD boxes: how to tell them apart, how to
reproduce each, and what actually recovers them. Scope is card/link/host only — profiler-pipeline
wedges (full SPSC rings, two consumers on one ring, drainer boot) and tooling wedges (tracy-capture,
leftover processes) are deliberately out.

#### TL;DR (read these first)

- **Classify by CARD STATE, never by exit code.** Pooling these failures produced *and then destroyed*
  four findings in a single day. A run can exit 0 having hit a hanging condition.
- **There are three card/link failures plus one host failure.** They have different signatures,
  different causes, and different recoveries — `tt-smi -r` fixes one of them and nothing else.
- **Read the ROOT PORT (`0000:00:01.1`), not the endpoint.** A wedged endpoint's own sysfs reads
  all-ones and cannot tell you anything. `Unknown|63` is an all-ones artifact, not downtraining.
- **Read `DevSta`, not just AER.** The endpoint carries a real `UnsupReq+` while the AER capability
  reads clean. Every earlier "AER is all zero, so no PCIe errors" conclusion missed this.
- **The knee is irrelevant to the wedge.** Rate is flat across producer delays.

| failure | signature | recovery |
|---|---|---|
| **WEDGE** | card `Unknown\|63`, endpoint config space all-ones, **root port still linked 32 GT/s x16** | `tt-smi -r` — seconds, box survives, full health |
| **TEARDOWN** | `wait_until_cores_done` never completes, card perfectly **healthy** | process dies or is killed; card is fine |
| **DEGRADED** | ack-write ~13× slower (~185 ns → ~2300 ns) | **cold power cycle**; a warm reboot can make it *worse* |
| **VM FREEZE** | whole reservation VM stops, IRD watchdog reboots it; physical host never reboots | reboot is automatic; often leaves the card DEGRADED |

---

#### 1. WEDGE — PCIe endpoint stops completing TLPs

**Signature.** Card reports `Unknown|63`; endpoint config space reads all-ones; the root port is still
linked at 32 GT/s x16. Host hangs forever with no bus traffic and no AER.

**Why it hangs silently.** The completion-queue predicate compares the host read pointer against a
*device-written* pointer fetched via `read_cq_host_ptr<true>` → `read_sysmem` — that is **host DRAM**,
not device MMIO. A dead endpoint can never update it, and the loop has no timeout, no yield, and no
device access. Silent by construction.

**Repro**

```bash
tools/drisc_drain/drisc_hang_harness.sh    # scores card state + duration + masked-signature + rc
tools/drisc_drain/drisc_hang_compare.sh
tools/drisc_drain/drisc_reclassify.py      # re-derives classes from logs: a scoring bug costs no re-runs
```

- Rate is **~2–3% per run** with **no delay dependence** (125/150/500 alike), so budget ~50–100 runs
  for a couple of hits.
- **Randomize arm order, never alternate.** The wedge straddles run boundaries — the failure surfaces
  in the run *after* the one that seeded it, so alternation turns that into systematic bias. It voided
  two comparisons.

**Per-run PCIe probe.** `DevSta` bits are RW1C, so clearing them before each run turns the register
into a per-run probe:

```bash
sudo setpci -s 01:00.0 CAP_EXP+0A.w=0x000f   # clear, then read back after the run
# 0x8 = UnsupReq, 0x1 = CorrErr.  DevSta is PCIe cap + 0x0A — a DIFFERENT register from AER.
```

**The 220 ms MMIO stall is the root-port completion timeout firing** (`DevCtl2: Completion Timeout:
65ms to 210ms, TimeoutDis-` on `00:01.1`). The failing access is not slow — the endpoint never
completes it and the root port abandons it at ~210 ms. No 2 ms UMD retry budget can ever cover that.

**Not a fix:** arming `TT_METAL_OPERATION_TIMEOUT_SECONDS` gave 4 wedges vs 0 over 400 classified
randomized runs — p ≈ 0.12, not significant. It also fails to reliably rescue teardown.

---

#### 2. TEARDOWN — core-wait never completes, card healthy

**Signature.** `wait_until_cores_done` never returns while the card is perfectly healthy.

**Slow dispatch: SOLVED (2026-08-07) — it was a harness grid bug, not a device fault.** `--gx 0` gave a
12×10 producer grid against an 11-column poll list. Always pass `--gx 11 --gy 10`. Slow cells are now
10/10 clean, and every previously recorded slow-dispatch cell is void.

Two corrections worth keeping:

- `14-2..14-11` are **Tensix workers** (one full worker column), **not** the DRAM/DRISC column. That
  misidentification pointed the investigation at the drainer instead of the producers.
- The Tensix arm did not hang on ten cores — it hung on **all 120**, because its drainer sits in the
  very column the stray producers landed on.

**Fast dispatch.** Still rare, and hits a single core (`14-3`) where the throw *is* caught, so the run
**exits 0**. This is why armed runs can look clean while having hit the hanging condition.

---

#### 3. DEGRADED — ~13× MMIO latency

**Cannot be triggered directly.** It follows a **box freeze + watchdog reboot** (§4), not a card hang.
~510 runs and 9 hangs in one day produced zero degradation, because every reboot was clean.

**Confirm the freeze happened:**

```bash
last -x | head      # a session ending in "crash" + a boot with NO preceding shutdown = it froze
cat /proc/uptime    # small value = it went down
```

**A reboot does not fix it, and a warm reboot can make it worse.** Measured on bh-26 across two
reboots:

| | link | ack-write (posted) | device-read (non-posted) |
|---|---|---|---|
| healthy | 32 GT/s | ~185 ns | ~790 ns |
| after watchdog reboot | 32 GT/s | **2306 ns** | 2940 ns |
| after a clean warm reboot | **2.5 GT/s (downgraded)** | **183 ns** | **2738 ns** |

So there are at least **two distinct bad states**, and a warm reboot converted one into the other
rather than clearing anything. The Gen1 downtrain splits cleanly along posted vs non-posted: writes are
fire-and-forget so link rate barely matters, reads pay a round trip.

- **Include `current_link_speed` in the health check**, not just latency probes: `LnkSta: Speed 2.5GT/s
  (downgraded)` against `LnkCap: 32GT/s`, on **both** endpoint and root port. Without it, this state
  reads as "same degradation" when it is a different one.
- **`sock-read` GB/s is not a PCIe health indicator** — it measures the host reading the D2H socket out
  of host DRAM. It showed 17.28 GB/s on a Gen1 x16 link whose ceiling is ~4 GB/s. Check the arithmetic
  against the link rate first.
- Needs a **cold power cycle**; a warm reboot will not retrain the link.

---

#### 4. VM FREEZE → IRD watchdog reboot

**Signature.** Presents as "the box rebooted" with SSH keys gone, but the **physical host never
reboots** — the reservation VM hard-freezes and the IRD/hypervisor watchdog restarts it. Kernel-silent:
two independent processes stop writing at the *same instant*, there are zero
`tenstorrent`/AER/DPC/MCE/hang/panic lines in the pre-reboot ring buffer, and `efi_pstore` is empty.

**Mechanism** (confirmed by stack-sampling): a host CPU stalls on an **uncompleted MMIO read or ioctl
to the card**, which cannot be software-timed-out → VM lockup → watchdog reboot.

**Repro.** Not deterministic. Reproduced on **bh-05 (2026-08-06)** during a DRISC
streaming profiler sweep: box went down mid-sweep, `last -x` showed `crash` with no `shutdown` in the
following boot. Run a long DRISC sweep and watch for the signature.

**Catch the mechanism instead of guessing:**

```bash
# ptrace_scope=0; gdb at /usr/local/bin/gdb
# sample the pid holding /dev/tenstorrent every 0.4 s, writing to /localdev (NOT /tmp — see below)
```

The last completed backtrace before the freeze names the stuck op. Identical stacks across consecutive
samples = genuinely hung, not transient.

**It costs more than the run in flight:** the bh-05 freeze came back **DEGRADED and stayed degraded**
through two later clean reboots (§3).

**Recovery gotchas after the VM reboot / container recreate**

- `/home/$USER` comes back **root-owned and unwritable**, which breaks both the venv (its uv-python
  interpreter lived in ephemeral `/home/.local`) and the tt-metal JIT firmware cache
  (`~/.cache/tt-metal-cache`). Repoint the venv `python` symlink at `/usr/bin/python3.10` and fix
  `pyvenv.cfg home=/usr/bin`, then `export HOME=/localdev/$USER` for runs.
- **`/localdev` persists; `/tmp` and `/home` do not.** Put diagnostics on `/localdev`.
- Password auth survives the recreate; the **injected pubkey does not**, so `tt run` / `tt git` (key
  auth) break until it is re-added.
- Repeated freezes can put the box into a reboot cycle. **Stop hammering it and let it settle.**

---

#### Ruled out — do not resurrect

Egress bandwidth (saturates ~17 GB/s) · ingest/producer delay · cumulative runs · config churn · NoC
choice · host poll pressure (yield injected directly: 4/100 vs 3/100 over 300 randomized runs) · the
periodic device read (that comparison was masked teardowns) · static-TLB immunity to degradation ·
degradation-follows-a-hang · knee-as-safety-limit · `TT_METAL_OPERATION_TIMEOUT_SECONDS` as a fix.

**IOMMU page faults — DEAD (2026-08-07).** Decorrelated in both directions on one day: the last
`AMD-Vi IO_PAGE_FAULT` burst landed during a fully clean block, and two wedges logged no faults at all.
Faults without wedges, wedges without faults. The four near-zero IOVAs are real and unexplained but are
**not** the wedge.

#### Method rules that would have prevented every wrong conclusion

- Every error came from a **coarse observable standing in for the real one** — exit code for failure
  mode, wall-clock for card health, arm label for causation, one block for a rate. The fix was never
  more runs; it was always a finer discriminator.
- **Randomize arm order, never alternate** (see §1).
- **Never `pkill -f <pattern>` on an IRD box.** `tt run` wraps commands in a `bash -c` whose command
  line contains your pattern, so pkill matches and kills your own ssh session. Kill by PID, filtering
  the list with `grep -av "bash -c"`.
- A "killed" run can survive as a `tt run` process tree and keep holding the device. The next *build*
  then deadlocks in `precompile_fw` (it opens the device) at 0% CPU, looking like a hung build. Check
  `ps -eo pid,args` before blaming the build.

#### Source of record

The findings record (§6, formerly `FINDINGS.md`) — read §N+21 first, it carries a status banner. It holds
the full audit trail, including five claims retracted in one day with the reasoning. This runbook is the
short version: if the two ever disagree, the findings win.

### 3.3 The DRISC hang-investigation harness (2026-08)

*Formerly `tools/drisc_drain/README.md`.* The scripts (`drisc_hang_harness.sh`, `drisc_hang_compare.sh`,
`drisc_reclassify.py`, `drisc_wedge_watch.sh`, `drisc_2x2_rerun.sh`) were removed with the DRISC experiments in 2026-09; they
were written against the superseded kernels' knobs (`SHIP_REPEAT`, `ARMED`, `DISPATCH`), and the scoring
rules below are the durable part.

The findings record (§6, formerly `FINDINGS.md`) is the running record. Read the newest section first;
earlier conclusions are superseded in place and several are explicitly retracted.

#### Harness

Two scripts, both driven over ssh from a Mac against an IRD box.

| script | purpose |
|---|---|
| `drisc_hang_harness.sh` | single-condition runs, classified |
| `drisc_hang_compare.sh` | armed-vs-unarmed, randomized order, per-arm breakdown |

```sh
TAG=v1 DELAY=125 N=40 ARMED=0 STOP_ON_WEDGE=1 ./drisc_hang_harness.sh
TAG=cmp DELAY=125 N=40 ./drisc_hang_compare.sh          # N = runs PER ARM
```

Env: `TT_HOST` `TT_PORT` `TT_REMOTE` `OUT_DIR`. Writes `summary.txt` and `runs.csv`
(`k,delay,armed,rc,dur_s,card,class`).

#### Why it scores four axes

Every one of these corresponds to an error that produced a wrong finding earlier in this
investigation. Do not remove them.

- **CARD STATE is authoritative, never exit code.** "Hang" pooled two distinct failure modes —
  a genuine PCIe endpoint wedge (`Unknown|63`, all-ones config space) and a teardown
  `wait_until_cores_done` that never completes on a perfectly healthy card.
- **DURATION.** A 9x slowdown (45 s vs the 2.1 s baseline) hid inside ~280 runs that all exited 0.
- **MASKED.** `TT_METAL_OPERATION_TIMEOUT_SECONDS` also bounds the teardown core-wait; the
  exception is caught and the run exits 0. Such a run is *not* clean — unarmed it would hang.
  Scoring these as clean is what produced, and then destroyed, the §N+18 "periodic read prevents
  the wedge" claim.
- **rc**, last and least.

#### Method rules learned the hard way

- **Randomize arm order, never alternate.** The wedge straddles run boundaries: the failure
  surfaces in the run *after* the one that seeded it. Under strict alternation that always lands
  on the same arm, which is a bias, not noise — more runs make the wrong answer look stronger.
  This voided a `num_hw_cqs` comparison and, earlier, a Tensix-vs-DRISC one.
- **Watchdog the local ssh client.** A remote process can exit while the local client hangs,
  stalling a whole block silently. Hence `ServerAliveInterval`, `timeout -k`, and a local kill.
- **bash 3.2 on macOS has no associative arrays.** Tally from filenames, not in-script counters —
  string subscripts silently collapse to index 0 and every arm reports the same number.
- Recovery: try `tt-smi -r` first (seconds, and it does clear a wedged card), fall back to a host
  reboot only if the link stays `Unknown`.

#### Reclassifying after the fact

`drisc_reclassify.py <run_dir>` re-derives every run's class from its log and writes
`runs_reclassified.csv`. Use it on any block collected before a classification rule was fixed —
the raw logs hold everything, so no block ever needs re-running for a scoring bug.

The discriminator is **how far the log got**, not the exit code and not a log signature:

| ending | card | class |
|---|---|---|
| `Cluster destructor completed` | — | CLEAN |
| stops at the profiler teardown block | healthy | TEARDOWN |
| stops earlier | `Unknown\|63` | WEDGE |
| reaches the end, but logged a caught timeout | healthy | MASKED |

The signature-only rule was wrong because `waiting for physical cores to finish` is emitted **only
when the timeout is armed**. Unarmed runs hang at the identical place in silence, so every unarmed
teardown hang was landing in OTHER.

#### Egress amplifier as a candidate fast repro

`REPEAT=<n>` sets `TT_METAL_STREAMING_PROFILER_SHIP_REPEAT`, re-shipping each staged frame n times so egress
stops being bounded by producer rate. The payload becomes duplicate frames, so it is a STRESS tool,
never a capture (`NO_DECODE` is already on).

```sh
TAG=amp8 DELAY=500 N=60 ARMED=0 REPEAT=8 ./drisc_hang_harness.sh
```

**Calibrate before trusting it as a faster repro.** The historical basis is a single cell -- repeat=8,
delay 500, 1 hang in 12 runs. That is ~8%, but against the ~2.5% base rate measured at delay 125/150/500
it is p ~ 0.3, i.e. consistent with no difference at all. And the 84-run monitored churn used a full
SHIP_REPEAT ladder {1,2,4,6,8,12,16} and produced zero wedges. Measure the rate with the classifying
harness before spending a campaign on it, and check that what it produces is a WEDGE
(card `Unknown|63`) rather than a TEARDOWN -- a faster repro of the wrong failure mode is worse than
no repro, because it looks like progress.

## 4. Zone primitives and the wire format

Three device-side primitives — `DeviceZoneScopedN`, `DeviceTimestampedData`, `DeviceFlag` — and the
variable-width packet family that carries them. §4.1 points at the illustrated intro to the three;
§4.2 is the wire format and what it measures like. Consumers see complete
zones either way and none of §4.2 changes that contract.

### 4.1 Zones and point markers in the Tracy GUI

The three primitives are introduced, with a GIF of each in the Tracy GUI, in the short standalone
[`STREAMING_PROFILER.md`](STREAMING_PROFILER.md) (zone scope, timestamped data, flag). It is kept
separate on purpose: it is the first thing to read, and it does not depend on anything below.

### 4.2 The zone wire format, and what it measures like

*Formerly `STREAMING_PROFILER_ZONES.md`.*

This documents the variable-width zone packet family the SPSC streaming profiler puts on the wire
(producer: `kernel_profiler.hpp`; decoder: `spsc_marker_decode.hpp`; plain-C constants:
`spsc_packet.h`). For how to consume the stream, see
[§1](#1-overview-and-usage) — consumers see complete zones either way and none of
this changes that contract. For what zones and point markers look like in the Tracy GUI,
see [§4.1](#41-zones-and-point-markers-in-the-tracy-gui).

#### Zone packets

Every zone ships whole, in one packet, emitted at scope **close** — the RAII scope object carries the
start timestamp (`start_hi`/`start_lo`, 8 B of member state), so the open touches nothing but the wall
clock. Packets are sized by need. `word0 = type(5) | id27` in all of them; the id is the full 27-bit
structural zone id (`tu_id(17) << 10 | local(10)`), ELF-name-resolved on the host.

| type | name | words | payload after word0 | expresses |
|---|---|---|---|---|
| 3 | `ZONE_S` | 2 | `end_delta16 << 16 \| dur16` | end within 2^16 cycles (~48 µs @1.35 GHz) of the lane cursor and duration ≤ 2^16 cycles |
| 2 | `ZONE_ATOMIC` ("M") | 3 | `end_lo32`, `dur32` | duration < 2^32 cycles (~3.2 s), any gap; **re-anchors the cursor** |
| 4 | `ZONE_L` | 5 | `end_lo`, `end_hi`, `dur_lo`, `dur_hi` | anything — two full 64-bit values, self-contained |
| 0/1 | `ZONE_START`/`END` pair | 2+2 | `timer_lo` each | no worker emits it; the decoder still accepts it |

##### The lane cursor

Producer and decoder each keep, per lane, a 64-bit **cursor = the end of the last S or M zone**. Zones
are emitted at close, and closes happen in end order — the same per-lane monotonicity invariant the
host's order-regression check relies on — so end-to-end deltas are unsigned. This is why the delta is
**end-relative, never start-relative**: a closing *parent* zone's start lies before its already-closed
children's ends, so start deltas go negative; ends never do. Start is always reconstructed as
`end − duration`.

- Only S and M move the cursor, identically on both sides. The M packet's absolute `end_lo` is the
  re-anchor (`cursor = sticky_hi << 32 | end_lo`).
- L (and the paired form) leave the cursor alone — a stale cursor is merely conservative: the next
  zone's delta overflows 16 bits and falls back to M, which re-anchors.
- The producer invalidates its cursor (`hi = ~0`) at `init_profiler()` and on the idle-launch rewind,
  which makes the S class test fail arithmetically — the first zone after any launch is always an
  absolute re-anchor, with no extra branch.
- There is deliberately **no sticky-lo**: a separate re-anchor packet can never beat a 3-word M that
  also carries a zone. `STICKY_TIMER` (the 27-bit high half, ~one per 3.2 s) applies to M/`PP_DATA`/
  `PP_EVENT`, which carry absolute low words; S never needs one — the 64-bit cursor add crosses the
  2^32 lo-wrap for free.

The producer's class test is one OR-tree into one branch, laid out as the fall-through:
`(((c_lo_d | lo_d) >> 16) | c_hi_d | hi_d) == 0`.

##### The stall zone (PRODUCER-STALL)

Pinned to **M with a saturating duration**, written straight into the ring's stall reserve with no
room check — a room check from inside the full-ring path recurses into another stall scope, which is
exactly what the reserve exists to prevent. The stall open writes nothing (member state, like every
zone), so the reserve covers only the close: **4 words** (3-word packet + 1 sticky). S buys zero
reserve, since the reserve must cover the deterministic worst case, and L grows it to 5; a ≥2^32-cycle
stall is a wedged relay rather than a measurement, so saturating the duration loses nothing real.

##### ZONE_L and the >3.2 s fallback

A zone whose duration overflows 32 bits ships as one self-contained ZONE_L — no stickies, no cursor.
The decoder normalizes it to a synthetic START/END pair for the delivery-side pairing stack (a 64-bit
duration cannot ride the 32-bit dur argument); the in-the-past synthetic START trips the per-lane
order-regression diagnostic once per nesting parent, which is kept as wedge visibility.

The decoder **normalizes at the emit boundary** — S emits as wire-type ATOMIC with the cursor-resolved
end, L as the synthetic pair — so the receiver, every consumer, and the stall classifier see only the
types they always saw.

#### Measured behavior (Blackhole, 1.35 GHz)

##### Which classes real workloads use

The receiver's per-stream decode-path line breaks records down by class (`decode paths: ... zoneS16 +
zone8 + atomic16 ...`):

| workload | S | M |
|---|---|---|
| dense synthetic (test_streaming_profiler_zones, 10 zones/iter) | ~100% | per-lane first-zone re-anchors, trailing zones |
| op-level model capture (FW/kernel-granularity zones) | 0% | 100% |

Zones at FW/kernel granularity have end-to-end deltas — dispatch gaps and kernel durations — far
beyond the 16-bit/48 µs S window. **S is a dense-instrumentation feature; op-level captures ride M
entirely.** If S should ever matter for models, the lever is the delta width or denser in-kernel
instrumentation, not the duration field.

##### Producer cost per class (`--empty` calibration, 1×1 grid, 500 iters, median dur+gap)

| class | cycles | ns | ring words |
|---|---|---|---|
| S | 50–51 | ~37 | 2 |
| M | 45–47 | ~34 | 3 (+1 sticky, rare) |
| L | 74 | ~55 | 5 |

**ZONE_S is not a producer-cycle win**: its cursor bookkeeping (two RAM stores + a 64-bit delta + the
class test) slightly outweighs the one saved L1 store. What S buys is **wire volume** — and volume is
what the pipeline scales with: 2 words per zone instead of 3 is a third off a dense capture, and the
producer-stall onset knee moves with the volume, not with per-zone cycles.

Accounting identities worth keeping for verification: a clean run decodes exactly
`iters × 6000 + 600` records (10 zones × 600 lanes + 1 trailing per lane), and a stalling run's
record surplus equals the device L1 stall counter to the unit, one atomic stall zone per stall.

##### Rendering long zones in Tracy

Tracy's server carries an unwrap heuristic for wrapping GPU timestamp counters
(`TracyWorker.cpp ProcessGpuTime`): a backwards jump > 2^31 ticks in one context's GpuTime stream is
read as a counter wrap, and everything after it is shifted up by a power of two, cumulatively. So the
timestamps pushed for one GPU context must be monotone: a sink that flushes lane by lane jumps back
to capture start at every lane boundary, and any capture whose per-lane span exceeds **2^31 ticks
(~1.6 s)** then staggers its RISCs by huge power-of-two offsets. Each lane's bracket sequence is
already non-decreasing in ts, so merging a context's lanes by timestamp before pushing is a correct
k-way merge and keeps the heuristic from firing. `tracy_ctx_inspect` (with `CTX_ALL_THREADS=1`)
prints per-thread zone spans to check exactly this.

## 5. Dev tools: reading a `.tracy` without the GUI

Three small offline tools under `tools/drisc_drain/`, one directory each, built by the normal
`./build_metal.sh` into the build directory; each binary is named after its directory. They link the Tracy
server library and walk a saved capture, which is how every alignment and count claim in §6 was checked —
`tracy-capture`'s "Zones:" headline counts CPU zones only, so a device capture is judged by these, never by
the headline (§N+43).

| tool | what it reports | used for |
|---|---|---|
| `tracy_ctx_inspect` | every GPU (tt_device) context in the file with its zone count, thread count, calibration flag and name; `CTX_ALL_THREADS=1` adds per-thread zone spans | verifying device-context creation (one context per (chip, core)), context-index → core-name mapping, per-lane spans for the Tracy 2^31-tick unwrap check (§4.2). A correctly paired lane is depth ≤ 2; a lost END grows a deep staircase |
| `tracy_zone_csv` | every device zone in pre-order as `ctx,tid,risc,seq,depth,name,start,end`, enough to rebuild the nesting tree offline (a row's parent is the nearest preceding row on the same `(ctx,tid)` with `depth-1`) | zone-window arithmetic, per-phase means, the counter cross-checks in §N+41, §N+47–§N+49 (compute the windows in Python — `awk` overflows INT32 on these nanosecond values) |
| `tracy_plot_check` | the worker window and the drainer window (first/last device-zone time), then per plot: sample count, span, interval distribution (p50/max) and the fraction of samples inside the worker window | proving that decoded plot samples land on the device timebase, not at decode time (§N+43, §N+46). Plots only: it takes the zone windows from `tracy_zone_csv`, because walking `GetGpuData()` in-process for zones segfaults on three separate paths |

Both `tracy_ctx_inspect` and `tracy_zone_csv` iterate `GetGpuData()` in the same order, so the context
index joins their outputs.

## 6. Findings and benchmarks

The complete dated record of the findings and benchmarks — formerly `tools/drisc_drain/FINDINGS.md` and
`DIRECT_PUSH_PLAN.md` — lives in the companion file
[`STREAMING_PROFILER_FINDINGS.md`](STREAMING_PROFILER_FINDINGS.md). It is a separate file only because the
pre-commit large-file check caps a file at 500 KB; its sections keep the 6.x numbering used throughout this
document:

- [6.1 How to read this, and the name mapping](STREAMING_PROFILER_FINDINGS.md#61-how-to-read-this-and-the-name-mapping)
- [6.2 Direct host push: fillers own D2H sockets, movers deleted (plan, 2026-08-25)](STREAMING_PROFILER_FINDINGS.md#62-direct-host-push-fillers-own-d2h-sockets-movers-deleted-plan-2026-08-25)
- [6.3 The record (`FINDINGS.md`, 2026-07 → 2026-08-26)](STREAMING_PROFILER_FINDINGS.md#63-the-record-findingsmd-2026-07--2026-08-26)

It is an audit log, not a summary: later sections retract earlier ones in place and the reasoning behind each
retraction is kept on purpose. When a claim matters, read the newest section that touches it.
