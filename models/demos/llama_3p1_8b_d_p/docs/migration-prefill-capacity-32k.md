# 32K prefill capacity and selected native transfer

**PASS — 18 September 2026.** Both endpoints allocated the actual 32,768-token,
two-slot BFP8 cache on full 32-chip Galaxies. The source used the real H2D/model
producer across all 32 layers. All 16 K/V configurations loaded; each endpoint
installed 1,048,576 read entries and 1,048,576 write entries.

| Prompt tokens | Source → destination slot | Exact selected range | Packed pages | Packed bytes |
|---:|:---:|---|---:|---:|
| 32,768 | 0 → 1 | [31,744, 32,768) | 16,384 | 71,303,168 |
| 32,736 | 1 → 0 | [31,712, 32,736) | 16,384 | 71,303,168 |
| **Total** | Both crossed mappings | **2,048 selected tokens** | **32,768** | **142,606,336 (136 MiB)** |

The second prompt is shorter than its 32K allocation and its selected range
crosses the 31,744-token chunk boundary. The test produces both complete prompts,
but copies and compares only the selected 1K range from each.

## Work, exactness and lifetime

The two real requests perform **64 full32 chunk calls and 2,048 real layer
acknowledgements**. Their selected ranges require 32 and 64 native layer commands
respectively: **96 total**. A separate 2-call compile warmup plus 64 capacity-width
warmup calls finishes before either native client starts. Those **66 warmup calls
emit no native acknowledgements**. There are **130 source calls in total**.

Selected source bytes are captured before their acknowledgements, matched
exactly at the passive destination across all configurations/layers, and checked
again after native shutdown. Source selected bytes also remain unchanged after
shutdown. Untouched-page checks cover **4,608 then 5,632 samples**, with **5,120
final samples**, including adjacent/outside-range and other-slot locations.
These samples do not establish preservation of every untouched cache page.

Both exact native-stop proofs passed before cache release. All 32 chips closed
cleanly on each endpoint, cleanup reports were empty, job steps were empty and
physical locks were available at handback. Controller, both dispatches, verifier,
both native managers and both bridges exited 0. All 417 source pins were unchanged.

## Resource observations

| Manager | Observation | RSS (MiB) | HWM (MiB) |
|---|---|---:|---:|
| Source | after tables | 1025.65 | 1025.65 |
| Source | after transfer | 1026.52 | 1026.52 |
| Passive | after tables | 1035.69 | 1035.69 |
| Passive | after transfer | 1036.31 | 1036.31 |

RSS is resident memory at that observation; HWM is the same process's lifetime
high-water mark through that observation. These are whole-manager values, not
index-only allocations, whole-job memory peaks, or post-shutdown measurements.
The reviewed limit was **16 GiB per manager**. Host admission required at least
128 GiB available memory and 32 GiB shared free disk. The calculated cache payload
is 4,352 MiB per endpoint (136 MiB per chip); it is not a measured allocator peak.

## Test-harness elapsed milestones

| Saved milestone | Seconds from dispatch |
|---|---:|
| Source allocation receipt | 1070.87 |
| Source capacity warmup finished | 2047.06 |
| Both managers ready | 2108.81 |
| First selected range verified at destination | 2553.64 |
| Second selected range verified at destination | 3004.62 |
| Both native-stop receipts | 3166.89 |
| Verification artifact | 3282.14 |

These saved-receipt milestones include setup, cold compilation, warmups,
coordination and readback/validation as applicable. They are not prompt latency,
native-copy duration, bandwidth or a performance benchmark.

## Scope and status at acceptance

This is selected-range transport evidence at a 32K allocation. It adds no HF/KV
golden, PCC, full-model numerical-accuracy, performance or decoder claim. It does
not establish a full 32K-prefix copy or exhaustive untouched-cache preservation.

**At acceptance, 64K was pending timeout budget review; 128K was deferred.** This
records the workflow state at acceptance, not live progress. No 64K launch is
claimed here. Existing accepted model/performance evidence is separate and was
not remeasured. New commits stay local by user choice; no public push is pending.
This report and page are local artifacts.

The [machine-readable summary](migration-prefill-capacity-32k.json) includes exact receipt, model-source,
manifest and binary hashes, plus scoped resource/timing fields.
Root acceptance SHA:
75b62b2804df93cafedb2845039b23c2ba09064271d7b0a1ea6603181049cd8c.
Raw logs, weights, binaries and private deployment paths remain outside this
report.
