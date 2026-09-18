# 4K prefill capacity and selected native transfer

**PASS — 18 September 2026.** Both endpoints allocated the actual 4,096-token,
two-slot BFP8 cache on full 32-chip Galaxies. The source used the real H2D/model
producer across all 32 layers. All 16 K/V configurations loaded; each endpoint
installed 131,072 read entries and 131,072 write entries.

| Prompt tokens | Source → destination slot | Exact selected range | Packed pages | Packed bytes |
|---:|:---:|---|---:|---:|
| 4,096 | 0 → 1 | [3,072, 4,096) | 16,384 | 71,303,168 |
| 4,064 | 1 → 0 | [3,040, 4,064) | 16,384 | 71,303,168 |
| **Total** | Both crossed mappings | **2,048 selected tokens** | **32,768** | **142,606,336 (136 MiB)** |

The second prompt is shorter than its 4K allocation and its selected range
crosses the 3,072-token chunk boundary. The test produces the complete two
prompts, but copies and compares only the selected 1K range from each.

## Work, exactness and lifetime

The two real requests perform **8 full32 chunk calls and 256 real layer
acknowledgements**. Their selected ranges require 32 and 64 native layer commands
respectively: **96 total**. A separate 2-call compile warmup plus 8 capacity-width
warmup calls completes before either native client starts; those **10 warmup
calls emit no native acknowledgements**. There are 18 source calls in total.

Selected source bytes are captured before their acknowledgements, matched
exactly at the passive destination across all configurations/layers, and checked
again after native shutdown. Source selected bytes also remain unchanged after
shutdown. Untouched-page checks cover **4,608 then 5,632 samples**, with **5,120
final samples**, including adjacent/outside-range and other-slot locations.
These samples do not establish preservation of every untouched cache page.

Both exact native-stop proofs passed before cache release. All 32 chips closed
cleanly on each endpoint, cleanup reports were empty, job steps were empty and
physical locks were available at handback. Controller, both dispatches, verifier,
both native managers and both bridges exited 0. All 398 source pins were unchanged.

## Resource observations

| Manager | Observation | RSS (MiB) | HWM (MiB) |
|---|---|---:|---:|
| Source | after tables | 427.76 | 427.76 |
| Source | after transfer | 428.62 | 428.62 |
| Passive | after tables | 427.52 | 427.52 |
| Passive | after transfer | 428.12 | 428.12 |

RSS is resident memory at that observation; HWM is the same process's lifetime
high-water mark through that observation. These are whole-manager values, not
index-only allocations, whole-job memory peaks, or post-shutdown measurements.
The reviewed limit was **16 GiB per manager**. Host admission required at least
128 GiB available memory and 32 GiB shared free disk. The calculated cache payload
is 544 MiB per endpoint (17 MiB per chip); it is not a measured allocator peak.

## Operational elapsed times

| Saved milestone | Seconds from dispatch |
|---|---:|
| Source allocation receipt | 665.57 |
| Both managers ready | 759.12 |
| First selected range verified at destination | 953.97 |
| Second selected range verified at destination | 1135.33 |
| Both native-stop receipts | 1297.43 |
| Verification artifact | 1403.94 |

These saved-receipt milestones include setup, cold compilation, warmups,
coordination and readback/validation as applicable. They are not prompt latency,
native-copy duration, bandwidth or a performance benchmark.

## Scope and stopping point

This is selected-range transport evidence at a 4K allocation. It adds no HF/KV
golden, PCC, full-model numerical-accuracy, performance or decoder claim. It does
not establish a full 4K-prefix copy or exhaustive untouched-cache preservation.

**Device testing stopped after 4K at the user's request.** Native 8K, 16K, 32K and
64K gates are paused until explicit greenlight; 128K remains deferred. Existing
accepted model/performance evidence is separate and was not remeasured here.

The [machine-readable summary](migration-prefill-capacity-4k.json) includes exact
receipt, model-source, manifest and binary hashes, plus scoped resource/timing
fields. Root acceptance SHA:
40d0d1afaebcb78cd857c63196538477780e88fcb6ff9e13c8c08f906ebb0aa3.
Raw logs, weights, binaries and private deployment paths remain outside this
public report.
