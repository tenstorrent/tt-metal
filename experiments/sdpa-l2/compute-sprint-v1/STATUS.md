# Compute sprint status

**Complete:** see [consolidated final report](REPORT.md). Verified compute-only
time reductions: D 7.28%, C 16.93%, B 2.405%, A 0.199%, E 2.080%, G 2.171%.
All retained outputs match their frozen baselines bit-for-bit on their
qualification suites. No numerical recipe or production dataflow changed.
The chronology below preserves preliminary/rejected observations and should
not be mistaken for the final retained-candidate status.

## Environment verified, 2026-09-18 14:11 UTC

- IRD reservation **223862**, `bh-lb-08`, eight Blackhole P150-class devices,
  requested eight hours. Container:
  `bh-lb-08-special-cglagovich-for-reservation-223862`.
- Remote source: `/localdev/cglagovich/flux2-frontier-20260915/tt-metal`.
  Existing prebuilt libraries/dependencies reused; device kernels compile via
  JIT. No fresh whole-repository host build claimed.
- Logical device 0 exact BF16 matmul smoke passed, compute grid **12x10**.
  Firmware 19.13.1.0; KMD 2.9.0; Python 3.10.19; Torch 2.11.0+cpu.
  UMD emits harvesting-metadata warnings; preserve these in run logs rather
  than interpreting the warning alone as a hardware failure.
- Idle telemetry sampled around 14:10 UTC showed 800 MHz; this is NOT an
  active benchmark clock. Nominal fmax is 1350 MHz. Active clocks must be
  distinguished from the nominal useful-FLOP ceiling.
- All device jobs use one machine-wide lock, including the same lock name
  as safe_pytest. Independent agents edit/compile isolated candidate paths;
  queued device sessions serialize. No concurrent accelerator measurements.
- Canonical recipe, fullchip compute entry, FP32 streaming header and private
  Q256 correction hashes match between local and remote working copies.

The accidental initial allocation 223858 on bh-lb-01 hit an IRD device-file
mapping error and was released. No test ran there. The expired earlier
reservation was not reused. A private FP32 wrapper include typo caused the
first candidate compile to fail; device closed cleanly, no work was launched
by waiting agents, and coordinator cleared the dirty marker under lock after
verifying no remaining device processes. No hardware reset was necessary.

## Tracks

| Agent track | Variants | Initial implementation candidates | Status |
|---|---|---|---|
| FP32 | D / C | Four-tile identity-state copy, batched FP32 state unpack, blocked denominator pack | Canonical/private-copy D smoke bitwise matched; other candidates in progress |
| BF16 | B / A | Cache unchanged packer-MOP width; audit redundant packer-state configuration | Device smoke queued |
| Low precision | E / G | Reuse compensation macro setup; batch redundant unpack setup | Exact canonical preparation/baseline checks queued |

The table above records launch state. A proposed truncation of BF16 denominator
work was rejected during source review: its intermediate state uses every
column, so that shortcut would change numerical behavior.

## First results, approximately 14:20 UTC

All values below are preliminary, same-allocation, one-core compute-loop
measurements. They are not full-chip throughput or final qualification.

| Variant / candidate | Baseline ms | Candidate ms | Time change | Numerical checks so far |
|---|---:|---:|---:|---|
| D / four-tile identity copy + batched FP32 state unpack | 90.621 | 87.472 | -3.47% | Resident outputs/replay exact; distinct scaled-QK exact |
| C / same state scheduling | 73.659 | 70.480 | -4.32% | Resident outputs/replay exact; distinct scaled-QK exact |
| B / packer-width cache | 345.627 | 345.846 | +0.063% | Resident and changing-max outputs exact; reject as no speedup |

D/C screen uses 8 Q repetitions, 128 K chunks, 5 warmups and 5 measured
replays, sequential candidate screening. B uses 16 Q repetitions, 512 K
chunks, 12 warmups and 10 measured replays, alternating candidate order.
Do not compare the times between these different loop lengths. All retain
Q256/K512/D128; paired longer-loop D/C confirmation is pending.

E and G numerator plane-layout batching both pass distinct-input, changing-max
checks against the canonical implementation, including exact preparation and
two trace replays. Timing is pending. C's same-coefficient exp-refiner macro
schedule also passes an initial bitwise smoke; timing is pending.

[Telemetry](telemetry-first-round.json) has 31 samples from 14:16:16 to
14:19:12 UTC. Sampled clocks were 800 MHz idle or 1350 MHz active, with maximum
observed temperature 55 C. This sampling does not prove every timed replay's
clock; alternating controls and device profiles remain important.

## Second screen and rejected candidates

- C's same-coefficient cubic-exp loadmacro schedule combined with state
  batching screens at **64.693 ms versus 73.642 ms** (about 12.15% less time,
  0.933 to 1.062 TFLOP/s/core). Normal resident output SHA256 matches exactly;
  broader qualification and paired timing are pending. Polynomial coefficients,
  rounding and subtraction are unchanged.
- E's numerator plane-layout batching measures **293.051 to 291.940 ms**
  (0.379% less time; 1.8760 to 1.8831 TFLOP/s/core), seven alternating paired
  samples. This remains a preliminary small gain, pending repetition.
- A pack-width caching also failed to improve throughput: **275.796 to
  275.908 ms**. B's redundant L1-mode toggles were likewise neutral/slower:
  **345.568 to 345.603 ms**. These changes preserve output but are not retained
  as performance improvements.
- A separate G cross-group macro-setup cache **failed** on distinct
  Q2048/K8192 with changing maxima (147,367 value mismatches). Rejected, not
  an acceptable numerical tradeoff. Its device session closed cleanly; the
  dirty guard paused subsequent jobs until coordinator review/clearance.
  No hardware reset occurred. This failure does not invalidate the separate
  plane-layout candidate, which passed its initial exact checks.

## Recovery and third screen

At 14:27 UTC an A candidate job failed during **device initialization**:
Ethernet core 31-25 did not become active. The candidate was not executed.
All agents paused; no remaining test process was present. Coordinator reset
the eight explicitly identified, exclusively reserved PCI BDFs under the
global lock. [Reset log](reset-01.log). A fresh exact BF16 matmul smoke passed
at 14:31:14 UTC ([log](post-reset-smoke-01.log)); testing resumed with fresh
paired controls. No firmware, power or clock settings were changed.

C's earlier exp/state candidate passed all eight distinct-input pairs,
including genuine 256K normal KV and seven 32K stress distributions. A parent
audit independently checked its saved raw-output hashes, along with sample
B/E evidence: [audit](audit-first-pass.json). This does not qualify subsequent
new scheduling flags automatically.

Latest **screening** candidates:

- D additionally defers unused sub-init on the existing identity-max path and
  batches the same maximum-comparison loop: **90.606 to 84.091 ms**, about
  **7.19% less time**, 0.817 TFLOP/s/core. Resident hashes/replays match.
- C combines those changes with the same-coefficient exp schedule and hoisted
  macro setup: approximately **73.63 to 61.22 ms**, about **16.9% less time**,
  1.123 TFLOP/s/core. Final combined-candidate paired/stress checks are pending.
- E combines plane-layout packing with adjacent-column correction-vector
  reuse: **293.034 to 289.732 ms**, **1.127% less time**, 1.8761 to
  1.8975 TFLOP/s/core, ten alternating paired samples with reversed initial
  order. All output/preparation/replay gates pass for this timing case.
- A batches unchanged approximate-exp replay instructions over its existing
  eight DST tiles: **275.795 to 275.231 ms**, **0.204% less time**. Small gain,
  pending repeat/qualification. No coefficient or rounding changes.

The final report will distinguish retained improvements from these screens.
