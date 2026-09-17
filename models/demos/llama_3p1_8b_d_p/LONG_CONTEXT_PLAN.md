# Long-context prefill milestones

## Scope
These milestones come before runtime KV-readiness, cache migration, and Blaze handoff work.
Use one user-assigned Blackhole Galaxy, SP=4 and TP=8.
K means 1,024 tokens. Keep 1,024-token compute chunks and two independent cache slots.
Extend prefill capacity through 131,072 tokens. The user added 128K coverage after the original 64K plan. The initial SC4 migration fixture remains separately limited to 2,048 total tokens.

## Milestone order

| Milestone | Input tokens | Required result |
| --- | ---: | --- |
| Repair and verify the baseline | 2,048 | Full32 correctness in both slots; BF16 diagnostic and BF8 target caches; complete golden KV report |
| First extended context | 4,096 | Full forward, bounded writes, exact slot isolation, positions and chunk boundaries |
| Extended context | 8,192 | Same checks, including rotary positions near 8,192; the 16K case also covers positions beyond it |
| Extended context | 16,384 | Same checks, including new cache pages and SP block ordering |
| Extended context | 32,768 | Same checks, including late-prefix attention and resource reuse |
| Extended context | 65,536 | Same checks, with documented memory bounds and full valid-prefix attention |
| Full requested context | 131,072 | Same checks at the Llama-3.1 context limit, before claiming migration support at this length |
| Performance and publication | All lengths above | Measured table, raw per-chunk data, reproducible commands, and logical commits |
| Runtime and migration | After the milestones above | Continue the existing native tt-d-gen integration plan |

No length is complete because allocation succeeds. Every length must execute all 32 layers on all 32 chips and pass independent numerical and structural checks.
Use a bounded or disk-backed golden reference for long lengths; memory cost must not silently remove reference coverage.
Freeze long-context accuracy criteria before hardware experiments. Preserve the canonical 2K
contract in tests/full_model/NATIVE_INPUT_ACCURACY.md: strict same-input layer/KV checks and
final raw-global logits/token checks, with accumulated raw hidden/KV errors reported separately.
Do not reinstate the rejected assumption that every late-layer FP32 trajectory must stay inside
a single-layer precision envelope.
Do not infer long-context numerical correctness from the 2K reference alone.
Do not drop a failing case or lower a threshold to complete a milestone.

## Design work required
1. Establish one requested prefill capacity shared by input validation, model construction, cache allocation, attention, and indexed RoPE.
2. Derive local cache sequence length from capacity/SP. Preserve two slots, 32 layers, 32-token cache pages and 256-token SP blocks.
3. Replace the attention reorder that currently names exactly eight blocks. Derive the complete rank-major to sequence-major permutation.
4. Audit the gather size, mask shape, position tables, SDPA loop bounds and L1/DRAM requirements at each capacity.
5. Keep Llama3 frequency scaling and the Meta-interleaved K frame. Do not import GPT-OSS-specific model math.
6. Bound golden-reference memory. The current oracle retains about 48 GiB per 64K prompt and 96 GiB per 128K prompt before logits and temporary tensors. Stream or store reference data by layer; compare against the current independent oracle at 2K before relying on the new path.
7. Use bounded selected logit positions, including every tested boundary. Do not let a fixed stride produce unbounded full-vocabulary host tensors.
8. Re-run 2K regression tests after capacity generalization, including invalid requests, partial tails, overlap, input retention and both slots.

Affected source files: tt/config.py, tt/input.py, tt/model.py, tt/weights.py, tt/kv_cache.py, tt/attention.py and their tests.
Indexed RoPE already accepts capacity arguments; extend its position/boundary coverage. Llama3 frequency scaling applies at all positions; 8,192 is not a position where the RoPE formula switches.
The 1,024-wide K/V projection features and fixed 1,024-token physical input chunk are not sequence-capacity constants.

## 2K KV golden report
Compare post-RoPE K and raw V against independent raw-checkpoint Hugging Face/PyTorch values.
Use all 32 layers, eight KV heads, both slots and all 2,048 valid token positions.
Report K PCC and V PCC separately for each layer/head/slot, for BF16 and BF8 caches.
Include normalized L2 and maximum absolute error to show the magnitude of any discrepancy.
Provide a compact summary with worst layer/head/slot and link the complete CSV/JSON.
Earlier per-chip/per-chunk PCC values must be labelled as such; do not average them and call the result whole-cache PCC.

## Cost check for the 2K KV report

The existing full-context oracle already computes and retains K/V for all 32 layers. Reuse those values and the cache readbacks from correctness checks; do not run a second full golden model just to produce the report.

A bounded CPU-only check on 2026-09-16 measured the existing PCC and normalized-L2 calculation. It ran 1,024 comparisons: 32 layers × eight KV heads × two slots × K/V, with 2,048 × 128 values per comparison.

| Metric-only measurement | Result |
| --- | ---: |
| Three rounds, four CPU threads | 12.36–13.89 seconds |
| Median | 13.60 seconds |

This is a synthetic timing measurement with a reused tensor pair. It excludes golden generation, device work, readback, SP reconstruction, cold full-cache memory traffic and report output. It is not a device accuracy result or an end-to-end report-time measurement. Source script and raw results are in evidence/task-8-full-prefill/kv-report-cost-001.

Keep the full 32-layer 2K comparison. A one-layer check is useful during debugging but cannot
show error accumulated through the stack. Completed diagnostics now contain both raw-global and
same-native-input rows for every layer. All 9,792 local rows passed per dtype; some accumulated
raw hidden/V rows crossed the old limits, as they also did in stock-HF BF16 controls.
The canonical boundary/baseline/held-out suites passed all six cases; independent review returned clean-pass.

## Timing contract
Publish timing as an accepted-model result only after correctness. The user authorized measuring
the unchanged 2K production code during diagnosis and asked not to repeat that measurement for
test-policy or small kernel changes. Preserve that baseline and its exact source identity.
Use the actual device forward with all 32 layers.
Keep model/weight loading, compilation/warmup, golden calculations and correctness readbacks outside warm timing.
Record cold setup and compile/warmup costs separately.
Synchronize before starting and after completing the measured forward.
Use pre-tokenized input; report whether packing/H2D is included in the wall boundary.
Record one active user per measurement with two allocated cache slots; throughput is valid prompt tokens divided by synchronized wall seconds.
Use at least three warm runs and report median and spread. Keep dtype, math fidelity, chunk size, trace/eager mode, node, commit, checkpoint and timing boundary beside the results.
The saved 2K baseline includes final normalization and the LM head. Use the same path and timing
boundary for the comparable length table: upload, forward and synchronization for each sequential
chunk. Report forward-only and upload-inclusive wall intervals separately. Logit readback is excluded.
A KV-only measurement or a prompt measurement without intermediate synchronization is a separate
optional series; do not mix its numbers into this baseline or imply they were already measured.
Do not remeasure 2K solely because its tests or numerical-comparison policy changed.
Reset request/cache state outside warm forward timing; report request setup costs separately.
For each instrumented run, save every chunk's start/end positions, valid tokens and synchronized elapsed time.
Report per-chunk first/median/p95/last/max values; retain the full CSV because attention cost grows with the prefix.

| Input tokens | Chunks | Warm wall, seconds | Throughput, tokens/s/user | Chunk first/median/p95/last/max, ms | Correctness |
| ---: | ---: | --- | --- | --- | --- |
| 2,048 | 2 | 0.432562 / 0.431713 (slot0 / slot1 medians) | 4,734.58 / 4,743.89 | Full per-chunk data recorded; see the 2K validation report | Six canonical cases passed; independent review clean-pass |
| 4,096 | 4 | Not measured | Not measured | Not measured | Not run |
| 8,192 | 8 | Not measured | Not measured | Not measured | Not run |
| 16,384 | 16 | Not measured | Not measured | Not measured | Not run |
| 32,768 | 32 | Not measured | Not measured | Not measured | Not run |
| 65,536 | 64 | Not measured | Not measured | Not measured | Not run |
| 131,072 | 128 | Not measured | Not measured | Not measured | Not run |

Commit the final table with the runnable full-prefill benchmark and commands on divanovic/llama31-8b-disagg.
Use separate logical commits for capacity support, long-context tests/reference support, and measured benchmark/report work.
Do not present the current correctness-run wall times, which include diagnostic readbacks, as performance numbers.

## Current evidence

Embedding, final norm/head, one-layer wrappers, resource tests and both short full32 chat cases passed.
The original first-chunk layer11 raw-FP32 mismatch remains archived; the accurate SDPA exponential
fix did not by itself resolve it. Complete diagnostics subsequently showed strict local hidden/K/V
agreement at all 32 layers and passing sampled final logits/token checks in both cache types.
Stock-HF BF16 controls reproduced intermediate raw-global mismatches.

The explicit comparison contract now preserves accumulated-error characterization while enforcing
all local and final semantic/structural checks. Fourteen CPU contract and mutation tests passed.
Canonical boundary, baseline and held-out device suites passed all six cases. Independent stage review
returned clean-pass. See [the 2K report](docs/validation-2k.md). Longer contexts and migration remain untested.

Saved performance evidence: evidence/task-8-full-prefill/performance-preparation-002/attempts/attempt-002-bfp8-baseline.
There was one warmup per slot and three measured requests per slot, executed sequentially.
All 13 model production source hashes match the current canonical model. The table does not claim
simultaneous-user throughput, network TTFT, decode performance or measured long-context performance.
