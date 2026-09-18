# 16K prefill capacity and selected native transfer

**PASS — 18 September 2026.** Both endpoints allocated the actual 16,384-token,
two-slot BFP8 cache on full 32-chip Galaxies. The source used the real H2D/model
producer across all 32 layers. All 16 K/V configurations loaded; each endpoint
installed 524,288 read entries and 524,288 write entries.

| Prompt tokens | Source → destination slot | Exact selected range | Packed pages | Packed bytes |
|---:|:---:|---|---:|---:|
| 16,384 | 0 → 1 | [15,360, 16,384) | 16,384 | 71,303,168 |
| 16,352 | 1 → 0 | [15,328, 16,352) | 16,384 | 71,303,168 |
| **Total** | Both crossed mappings | **2,048 selected tokens** | **32,768** | **142,606,336 (136 MiB)** |

The second prompt is shorter than its 16K allocation and its selected range
crosses the 15,360-token chunk boundary. The test produces both complete prompts,
but copies and compares only the selected 1K range from each.

## Work, exactness and lifetime

The two real requests perform **32 full32 chunk calls and 1,024 real layer
acknowledgements**. Their selected ranges require 32 and 64 native layer commands
respectively: **96 total**. A separate 2-call compile warmup plus 32 capacity-width
warmup calls finishes before either native client starts. Those **34 warmup calls
emit no native acknowledgements**. There are **66 source calls in total**.

Selected source bytes are captured before their acknowledgements, matched
exactly at the passive destination across all configurations/layers, and checked
again after native shutdown. Source selected bytes also remain unchanged after
shutdown. Untouched-page checks cover **4,608 then 5,632 samples**, with **5,120
final samples**, including adjacent/outside-range and other-slot locations.
These samples do not establish preservation of every untouched cache page.

Both exact native-stop proofs passed before cache release. All 32 chips closed
cleanly on each endpoint, cleanup reports were empty, job steps were empty and
physical locks were available at handback. Controller, both dispatches, verifier,
both native managers and both bridges exited 0. All 411 source pins were unchanged.

## Resource observations

| Manager | Observation | RSS (MiB) | HWM (MiB) |
|---|---|---:|---:|
| Source | after tables | 688.21 | 688.21 |
| Source | after transfer | 689.07 | 689.07 |
| Passive | after tables | 688.41 | 688.41 |
| Passive | after transfer | 689.00 | 689.00 |

RSS is resident memory at that observation; HWM is the same process's lifetime
high-water mark through that observation. These are whole-manager values, not
index-only allocations, whole-job memory peaks, or post-shutdown measurements.
The reviewed limit was **16 GiB per manager**. Host admission required at least
128 GiB available memory and 32 GiB shared free disk. The calculated cache payload
is 2,176 MiB per endpoint (68 MiB per chip); it is not a measured allocator peak.

## Operational elapsed times

| Saved milestone | Seconds from dispatch |
|---|---:|
| Source allocation receipt | 905.38 |
| Source capacity warmup finished | 1290.16 |
| Both managers ready | 1312.71 |
| First selected range verified at destination | 1531.07 |
| Second selected range verified at destination | 1802.54 |
| Both native-stop receipts | 1993.80 |
| Verification artifact | 2119.74 |

These saved-receipt milestones include setup, cold compilation, warmups,
coordination and readback/validation as applicable. They are not prompt latency,
native-copy duration, bandwidth or a performance benchmark.

## Scope and status at acceptance

This is selected-range transport evidence at a 16K allocation. It adds no HF/KV
golden, PCC, full-model numerical-accuracy, performance or decoder claim. It does
not establish a full 16K-prefix copy or exhaustive untouched-cache preservation.

**At acceptance, 32K was authorized; 64K pending; 128K deferred.** This records
the workflow state at acceptance, not live progress. Existing accepted
model/performance evidence is separate and was not remeasured here. New commits
stay local; no public push is pending. This report and page are local artifacts.

The [machine-readable summary](migration-prefill-capacity-16k.json) includes exact
receipt, model-source, manifest and binary hashes, plus scoped resource/timing
fields. Root acceptance SHA:
6c0d107024092f1b3ce2dfc485b67838818ebf4dc9abf7a34b864586f77663c5.
Raw logs, weights, binaries and private deployment paths remain outside this
report.
