# Stage 03 precision evidence

Snapshot: 2026-09-26 11:40 UTC. Decisions are provisional; final defaults,
combined policies, final contracts and device profiling remain pending.
The unchanged PCC threshold is **0.995**; no recorded position is excluded.

Per the user's instruction, synthetic/random-input PCC failures cannot veto
configurations that win on real checkpoint weights and actual recorded layer
inputs. Gaussian failures and repair controls remain unchanged in
`AUTODEBUG_stress.md`, `AUTOFIX_stress.md` and their original reports/logs.
They are diagnostic evidence, not erased results or production-selection gates.

## Input and measurement provenance

`actual_text_fixture_manifest.json` records the pinned model/tokenizer revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, generator command and source hashes.
The corpus is actual text from `README.md`,
`docs/realtime_profiler_architecture.md`, and `docs/L1_ACCUMULATION_FP32_ANALYSIS.md`.
Layer 0 receives scaled checkpoint embeddings; layer 5 receives genuine HF
layers 0..4 outputs. Upstream computation is FP32 eager with FP32 KV and original
layer scalars. The chosen FP32 embedding scale is 53.06599807739258; the
recorded BF16 scale control is 53.0.

`actual_text_layer{0,5}_{4096_128,1025_512}.json` supplies fixture hashes and
shapes. Raw FP32 boundaries and exact BF16-rounded transport inputs are retained;
HF and TT checks consume the same transported inputs. Continuations are
teacher-forced corpus tokens. These are layer-only checks on this corpus, not
full-model generation or benchmark accuracy.

All times below are **warmed host-wall microseconds**, computed as medians of
`traced_decode_host_us` or `warmed_prefill_host_us`. Decode timing uses five
groups of 30 replays at the final checked position; prefill timing uses three
synchronized samples when present. They are not device-profiler durations.
Compare times only within the same workload. Small differences alone do not
establish a robust gain. A dash means not recorded; failed-candidate times are
observations, not accepted wins. All listed completed runs report deterministic
first-position replay, clean runtime audits and enabled program-cache guards.

## Original defaults and Gaussian-repair controls on actual inputs

The recorded original policy uses two QKV terms/16 lanes/HiFi4; decode expert
BFP8 gate/up and BFP4 down/LoFi; shared BFP8/LoFi; prefill BFP8 gate/up and
BFP4 down sliding/BFP8 down full/LoFi; guarded sharded hidden norms
(input+common sliding, post+common full); and BF16 cache.

| Exact result file | Workload/control | Prefill PCC | Minimum decode PCC | Decode median, us | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `actual_text_fused_layer0_4096.json` | Sliding fused, 4096/128 | 0.999903756 | 0.999792063 | 5067.964 | Pass |
| `actual_text_original_default_layer0_4096.json` | Sliding original, 4096/128 | 0.999252531 | 0.995388274 | 2138.290 | Pass |
| `actual_text_fused_layer5_4096.json` | Full fused, 4096/128 | 0.999922551 | 0.999708259 | 5519.467 | Pass |
| `actual_text_original_default_layer5_4096.json` | Full original, 4096/128 | 0.999931761 | 0.998328714 | 2270.984 | Pass |
| `actual_text_original_default_layer0_1025.json` | Sliding original, 1025/512 | 0.999205096 | 0.995526104 | 1997.579 | Pass |
| `actual_text_original_default_layer5_1025.json` | Full original, 1025/512 | 0.999922478 | 0.998456538 | 1806.089 | Pass |
| `actual_text_repaired_layer0_4096.json` | Sliding three terms/BFP8 down/grid110, 4096/128 | 0.999252531 | 0.999802708 | 2210.008 | Pass |
| `actual_text_stress_down4_layer0.json` | Sliding three terms/BFP4 down/grid110, 1025/512 | 0.999205096 | 0.995533782 | — | Pass |
| `actual_text_stress_down8_layer0.json` | Sliding three terms/BFP8 down/grid110, 1025/512 | 0.999205096 | 0.996275731 | — | Pass |

Original defaults pass all actual HF checks in both workloads. The Gaussian
repair therefore does not mandate BFP8 down or a third QKV term. The repaired
headline also changes QKV geometry; its timing cannot be attributed to down
dtype alone. Exact commands/return codes are in `actual_text_first_commands.json`,
`actual_text_second_commands.json`, and `actual_text_stress_sliding_commands.json`.

## Expert gate/up and prefill controls

| Exact result file | Workload/control | Prefill PCC | Minimum decode PCC | Decode median, us | Prefill median, us | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `actual_text_gate4_layer5_4096.json` | Full decode gate/up BFP4, 4096/128 | 0.999931761 | 0.995955897 | 2267.558 | — | Pass |
| `actual_text_gate4_stress_layer5.json` | Full decode gate/up BFP4, 1025/512 | 0.999922478 | 0.996907513 | 1804.839 | 51777.432 | Pass |
| `actual_text_gate4_layer0_4096.json` | Sliding decode BFP4, 44 workers/K11; three-term QKV/grid110 | 0.999252531 | 0.992109831 | 2198.449 | — | Fail: 4130, 4138 |
| `actual_text_gate4_grid11_layer0.json` | Sliding decode BFP4, 11 workers/K11; original QKV | 0.999252531 | 0.992200217 | 2214.788 | 226089.081 | Fail: 4130, 4138 |
| `actual_text_gate4_grid22_layer0.json` | Sliding decode BFP4, 22 workers/K11; original QKV | 0.999252531 | 0.992200217 | 2141.376 | 226555.872 | Fail: 4130, 4138 |
| `actual_text_gate4_k22_layer0.json` | Sliding decode BFP4, 44 workers/K22; original QKV | 0.999252531 | 0.992200217 | 2109.606 | 226179.573 | Fail: 4130, 4138 |
| `actual_text_prefill_gate4_layer0.json` | Sliding prefill gate/up BFP4, down BFP4; 4096/128 | 0.998768810 | 0.995388274 | 2135.548 | 224789.305 | Pass |
| `actual_text_prefill_gate4_layer5.json` | Full prefill gate/up BFP4, down BFP8; 4096/128 | 0.999495761 | 0.998328714 | 2270.136 | 204308.587 | Pass |

Full decode gate/up BFP4 is eligible despite the older Gaussian
`headline_expert4_retry_layer5.json` failure (.993126197). Sliding BFP4 fails
actual inputs in all four recorded configurations; BFP8 remains supported
there unless another actual-input control repairs the failure. These are not
a complete grid/K Cartesian sweep, and the 44-worker/K11 row also changes
QKV. Both prefill BFP4 gate/up candidates pass the actual headline; combined
and longer checks remain pending. Full prefill down remains BFP8 in that
control. Commands: `actual_text_second_commands.json` and
`actual_text_expert_geometry_commands.json`.

## Attention and cache candidates: selection pending

The following completed headline artifacts were available at this snapshot.
Longer actual-text stress and combined native/cache policies are pending.

| Exact result file | Workload/control | Prefill PCC | Minimum decode PCC | Decode median, us | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `actual_text_cache8_layer0.json` | Sliding BFP8 cache, precise attention; 4096/128 | 0.999252531 | 0.995293268 | 2100.032 | Pass |
| `actual_text_cache8_layer5.json` | Full BFP8 cache, precise attention; 4096/128 | 0.999930508 | 0.998335777 | 2253.695 | Pass |
| `actual_text_native_sdpa_layer0.json` | Sliding native SDPA/full-DST sync/BF16 cache; 4096/128 | 0.999252531 | 0.995161226 | 1384.416 | Pass |
| `actual_text_native_sdpa_layer5.json` | Full native SDPA/full-DST sync/BF16 cache; 4096/128 | 0.999931761 | 0.998370139 | 1495.726 | Pass |

The probe's `attention_probe.cache_bfp8` identifies cache controls: the base
factory's `precision_policy.kv_cache` still says BF16 because the wrapper
changes cache storage afterward. Read both fields and the exact commands in
`actual_text_attention_commands.json`. Native rows have `native_sdpa=true`,
`sdpa_full_sync=true`, and do not also enable BFP8 cache. Runtime/probe source
hashes remain in each report; the campaign spans source snapshots.

Older Gaussian native full-sync minima were .971596656/.987119347 and BFP8
cache minima .968496996/.985093361 for layers 0/5
(`headline_sdpa_fullsync_layer{0,5}.json`,
`headline_precision_cache8_layer{0,5}.json`). Those also use older cumulative
policies. They remain diagnostics and cannot veto the actual-input passes.
Native SDPA must retain the independently established full-DST synchronization
requirement. Test native and combined BFP8-cache policies on actual 1025/512
inputs before selecting the fastest correct topology.

## Per-group provisional decisions

| Group/boundary | Actual-input evidence | Gaussian diagnostics retained | Current decision, not final |
| --- | --- | --- | --- |
| Decode QKV terms | Two terms pass both kinds/workloads; three terms pass sliding | Three terms repair large sliding direct misses | Two-term candidate remains eligible |
| Decode expert down | BFP4 passes both kinds/workloads; BFP8 sliding also passes | BFP8 repairs small sliding direct misses | BFP4 remains eligible; no upgrade from Gaussian alone |
| Decode expert gate/up | BFP4 full passes headline and 512; BFP4 sliding fails 4130/4138 across tested geometry | Older BFP4 failures | Full BFP4 eligible; sliding BFP8 supported by actual failures |
| Prefill expert gate/up | BFP4 passes both actual headlines | Older candidate-specific sweeps retained | BFP4 candidate; combined/stress pending |
| Prefill expert down | BFP4 sliding/BFP8 full pass original-default checks | Older full BFP4 rejection retained | Current per-kind pair eligible; actual full down-only BFP4 control pending |
| Shared MLP | BFP8 passes original-default actual checks | `shared_bfp4_dram1_sliding33.json` fails .991661616 | BFP8 supported; no actual shared-BFP4 result in this snapshot |
| Attention weights | BF16 passes actual checks | Reduced QKV/O Gaussian failures retained | BF16 supported; actual reduced-weight frontier pending |
| Hidden norms/residual | Original guarded sharding passes both actual workloads | Full common-only norm repairs Gaussian preservation | Original policy eligible; Gaussian repair is diagnostic |
| KV cache | BFP8 passes both actual headlines | Gaussian BFP8 failures retained | Actual stress, combined policy and contract checks pending |
| Decode SDPA | Full-sync native passes both actual headlines with lower recorded host times | Gaussian full-sync misses retained; half-DST defect remains | Active candidate; final selection pending |

This ledger was assembled from saved artifacts on CPU. No hardware was
executed and no runtime was edited for this evidence task.
