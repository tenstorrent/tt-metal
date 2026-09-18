# 8K prefill capacity and selected native transfer

**PASS — 18 September 2026.** Both endpoints allocated the actual 8,192-token,
two-slot BFP8 cache on full 32-chip Galaxies. The source used the real H2D/model
producer across all 32 layers. All 16 K/V configurations loaded; each endpoint
installed 262,144 read entries and 262,144 write entries.

| Prompt tokens | Source → destination slot | Exact selected range | Packed pages | Packed bytes |
|---:|:---:|---|---:|---:|
| 8,192 | 0 → 1 | [7,168, 8,192) | 16,384 | 71,303,168 |
| 8,160 | 1 → 0 | [7,136, 8,160) | 16,384 | 71,303,168 |
| **Total** | Both crossed mappings | **2,048 selected tokens** | **32,768** | **142,606,336 (136 MiB)** |

The second prompt is shorter than its 8K allocation and its selected range
crosses the 7,168-token chunk boundary. The test produces both complete prompts,
but copies and compares only the selected 1K range from each.

## Work, exactness and lifetime

The two real requests perform **16 full32 chunk calls and 512 real layer
acknowledgements**. Their selected ranges require 32 and 64 native layer commands
respectively: **96 total**. A separate 2-call compile warmup plus 16 capacity-width
warmup calls finishes before either native client starts. Those **18 warmup calls
emit no native acknowledgements**. There are **34 source calls in total**.

Selected source bytes are captured before their acknowledgements, matched
exactly at the passive destination across all configurations/layers, and checked
again after native shutdown. Source selected bytes also remain unchanged after
shutdown. Untouched-page checks cover **4,608 then 5,632 samples**, with **5,120
final samples**, including adjacent/outside-range and other-slot locations.
These samples do not establish preservation of every untouched cache page.

Both exact native-stop proofs passed before cache release. All 32 chips closed
cleanly on each endpoint, cleanup reports were empty, job steps were empty and
physical locks were available at handback. Controller, both dispatches, verifier,
both native managers and both bridges exited 0. All 405 source pins were unchanged.

## Resource observations

| Manager | Observation | RSS (MiB) | HWM (MiB) |
|---|---|---:|---:|
| Source | after tables | 514.52 | 514.52 |
| Source | after transfer | 515.41 | 515.41 |
| Passive | after tables | 514.46 | 514.46 |
| Passive | after transfer | 515.04 | 515.04 |

RSS is resident memory at that observation; HWM is the same process's lifetime
high-water mark through that observation. These are whole-manager values, not
index-only allocations, whole-job memory peaks, or post-shutdown measurements.
The reviewed limit was **16 GiB per manager**. Host admission required at least
128 GiB available memory and 32 GiB shared free disk. The calculated cache payload
is 1,088 MiB per endpoint (34 MiB per chip); it is not a measured allocator peak.

## Operational elapsed times

| Saved milestone | Seconds from dispatch |
|---|---:|
| Source allocation receipt | 705.41 |
| Both managers ready | 907.22 |
| First selected range verified at destination | 1130.29 |
| Second selected range verified at destination | 1310.88 |
| Both native-stop receipts | 1473.41 |
| Verification artifact | 1580.50 |

These saved-receipt milestones include setup, cold compilation, warmups,
coordination and readback/validation as applicable. They are not prompt latency,
native-copy duration, bandwidth or a performance benchmark.

## Scope and status at acceptance

This is selected-range transport evidence at an 8K allocation. It adds no HF/KV
golden, PCC, full-model numerical-accuracy, performance or decoder claim. It does
not establish a full 8K-prefix copy or exhaustive untouched-cache preservation.

The accepted run is successor002 with an absolute plan path. Historical
attempt001 stopped before device execution because of a relative plan path and
remains preserved separately.

**At acceptance, 16K was authorized and preparing; native 32K and 64K were
pending; 128K was deferred.** This records historical workflow state, not live progress. Existing accepted model/performance evidence is separate and was not
remeasured here. By the user's choice, fixture commits 70c1765 and 1e0c3754 remain
local; no public push is pending. This report and page are also local artifacts.

The [machine-readable summary](migration-prefill-capacity-8k.json) includes exact
receipt, model-source, manifest and binary hashes, plus scoped resource/timing
fields. Root acceptance SHA:
e3169d23a7bad2555e35110bd0ff2812199966a737d7f2d3b3175730934742a2.
Raw logs, weights, binaries and private deployment paths remain outside this
report.
