# Compact GDN: full GPQA qualification and profile handoff

The frozen compact candidate completed **177/198 GPQA Diamond (89.3939%)**
on October 10 at **21:20:55 UTC**, passing the unchanged 177-correct gate.
Measured evaluation time was **3592.93 seconds (59m 53s)**. All 198 questions
remain in the denominator. Five responses exhausted the **65536-output-token
budget** and received no credit; none was stopped by the configured 262144-token
model-context limit. There were 193 natural-stop responses.

The in-place audit matched all 198 private text/reasoning hashes and token-usage
records to their scored receipts. Both the harness score and the stricter
completed-response score are 177/198. Private response text remains on the host.
This is the corrected `preserve_scientific_notation_v1` protocol, not a claim
that the separate OpenBench/OpenRouter run or Tau qualification has passed.

Configuration: `precision_single_step_compact_gdn_bfp8_all.json`, BFP8 weights
and KV, BF16 activations, FP32 recurrent state; eight TP4 replicas, up to 16
users each. Sampling was temperature 1, top-p 0.95, top-k 20, seed 42, thinking
enabled. Checkpoint revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
the dataset and harness revisions are retained in the protocol receipt.

The eight-replica G0 and API checks also passed. Source audit verifies the
G0 and serving model hashes against the exact frozen manifest and their
precision policies match. The evaluated source is
`/home/ttuser/qwen38-artifacts-20261007/compact-gdn-source-v3`; the current
development branch additionally contains later default-off experiments, so
do not describe its entire HEAD as the exact evaluated source.

| Measurement | Result | Scope |
|---|---:|---|
| Native B16/32K decode | 20.035 tokens/s/user | One TP4, three repeats, before/after controls and matching output hashes |
| Native B16/16K decode | 22.751 tokens/s/user | Same protocol |
| GPQA mean request decode rate | 21.639 tokens/s | Variable context and active batch; not the fixed B16/32K microbenchmark |
| GPQA aggregate output rate | 804.216 tokens/s | Entire eight-replica eval including prefill and completion tail; not a saturated Galaxy decode ceiling |
| GPQA mean TTFT | 17.941 seconds | Includes serving/queueing, variable prompts |

The prior fusion run scored the same 177/198 with six output-budget cutoffs.
It used 65m54s; stochastic output length and scheduling differ, so the change
in evaluation duration is not an isolated kernel speedup measurement.
Thirty native TSU still requires approximately 16.58 ms less than the measured
49.91-ms B16/32K step. Prefix-cache/SSD integration, current container/Shield
qualification and the remaining performance goal are not proved by this run.

After serving shutdown, the bounded profile launcher observed its dirty marker,
completed its existing reset procedure, and loaded the full model. The
unprofiled restored-cache test subsequently passed with clean teardown and
three steps of 50.0873, 50.0951 and 50.0795 ms. This independently agrees with
the natural-prompt native timing but is a separate workload. The profiled
capture is running; projection/prefill/long-horizon/recurrence/gate experiments
remain ordered behind it.

## Dynamic profile accounting

Added a CPU-only inventory that infers operator counts from the actual capture,
includes unknown operation types, and verifies all four ranks and three
replays. It partitions each device's firmware span by family, with overlaps
and gaps explicit, and reports the longest rank for each replay. Family medians
and separate chip timelines are not summed as an end-to-end budget. Reader,
writer and compute timings include waiting and do not establish utilization.

Matrix-byte estimates include every padded BFP8 DRAM weight shape, not a fixed
list of old projection types. These assume one weight read and 512 GB/s/chip;
they are not physical counters. Profile overhead is calibrated against the
matching unprofiled run with identical source, precision, inputs and outputs.

Frozen CPU preflight: **601 passed, one skipped, 91 subtests passed**. A replay
of the previous completed P0 CSV exactly reproduced all 38 operator timings,
5001 records/rank/replay and 7,111,516,160 encoded weight bytes/chip. Its measured
whole-step profiling overhead remains 7.4637%. These old counts are validation
results, not assumptions imposed on the upcoming compact graph.

`qwen38-operator-profile-v1-20261010.service`, invocation
`725ee5aec59b4e32a96822a627a58686`, waits for either the normal completed profile
or its verified export recovery. It accesses no physical devices, retains the
frozen analysis source, has 16-GiB/four-CPU/28-hour bounds and survives a client
disconnect, not reboot. Its report will be at
`/home/ttuser/qwen38-artifacts-20261007/operator-profile-v1/report/INVENTORY.md`.
