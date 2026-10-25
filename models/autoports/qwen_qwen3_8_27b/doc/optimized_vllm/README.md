# Optimized vLLM serving — Qwen/Qwen3.8-27B

Primary single-user vLLM TTFT **447.393ms**, decode **36.204tokens/s/user**
(4096 input / 128 output tokens, batch1, concurrency1, max-num-seqs1).
Before:445.241ms /36.217tokens/s/user on the same workload. These single-run
results are effectively unchanged; no speedup is claimed. Optimized full-model
traced token-out decode is36.171tokens/s/user on4096/128/B1 (serving ratio1.0009).

The selected change skips unused final norm/LM-head/logits merge work on
intermediate eager prefill pieces. The final emitted piece still runs the
canonical split sampler. Exact reduced controls prove identical final logits,
sampled IDs and subsequent state; full real-vLLM mixed traffic validates the
adapter path. Decoder math, native262144 context, non-aligned prompt support,
precision, cache ownership and trace signatures are unchanged.

## Measurement contract

All headline and CI performance runs use the real TT plugin and
`tt/generator_vllm.py`, 1x4 Blackhole (P300x2), `head_bfp8_hifi2`, greedy explicit
generation, `sample_on_device_mode=all`, decode-only traces, async scheduling,
block4096 and max-model-len262144. Primary is the shared `run_vllm_server` random
4096/128 workload with ignore-EOS, one request/concurrency1, max-num-seqs1 and
scheduler budget8192. Before/after argv/environment/source manifests are retained.
Tracking is disabled only for timing; separate identical-source correctness runs
include trace allocations and program-cache allocations. No serving profiler,
Tracy, tt-perf-report, adapter profiler or ReadDeviceProfiler was used. Device
latencies/rooflines remain unknown; HTTP and host event timing are not device time.

| Workload / server capacity | Phase | TTFT median ms | TPOT mean ms | ITL median / p99 ms | Aggregate output tokens/s | Decode tokens/s/user |
|---|---|---:|---:|---:|---:|---:|
| 4096/128,1 request,concurrency1,max-seqs1 | Before | 445.241 | 27.6114 | 27.616 /28.345 | 32.3853 | 36.2169 |
| 4096/128,1 request,concurrency1,max-seqs1 | Selected | 447.393 | 27.6216 | 27.614 /28.370 | 32.3574 | 36.2035 |
| CI100/100,32 requests,unlimited admission,max-seqs32 | Before | 2395.195 | 112.0602 | 109.197 /204.297 | 237.2145 | secondary only |
| CI100/100,32 requests,unlimited admission,max-seqs32 | Selected | 2394.809 | 112.0499 | 109.071 /204.695 | 237.2383 | secondary only |
| 4096/128,1 request,concurrency1,max-seqs32 | Before | 15088.975 | 133.1592 | 133.176 /133.935 | 3.9999 | 7.5098 |
| 4096/128,1 request,concurrency1,max-seqs32 | Selected | 15081.533 | 133.1156 | 133.162 /134.004 | 4.0015 | 7.5123 |

All rows complete with exact requested input/output counts and zero failures.
Decode tokens/s/user is1000/meanTPOT; aggregate output throughput includes prefill
and request overhead. CI burst is capacity/nightly-parity evidence, never the
headline. `comparisons.json` contains exact values and immutable raw paths.
B16 selected4096/128,16 requests/concurrency16: TTFT14400.397ms, TPOT73.8735ms,
aggregate86.1170tokens/s, decode13.5366tokens/s/user. Standalone B16 same token
counts:13.8597tokens/s/user (97.7%); its TTFT7305.290ms has different cache/admission
boundaries and is not a serving TTFT baseline. No tail-population inference is
made from the single-request headline.

## Capacity and mixed traffic

`batch_configurations.json` records B1/B8/B16/B32. Selected B8 budget32768 and B16
budget65536 allow one4096-token scheduling slice per user; B32 budget131072 and
long-prefill threshold4096 retain the previously tuned scheduling regime. B1/B8/B16
physical prefill chunk4096; B32 physical chunk1024. Prior B32 chunk1280 failed
allocation and is not selected. All supported decode batches fit their tested
workloads. The289-block B32 pool cannot hold the planning workload of32 simultaneous
32768-token prefixes; this is a configured-pool shortfall, not a reduction of the
native262144 context contract. See the agentic memory ledger and Stage09 allocation
controls for the distinction.

The selected `after_b32_lifecycle.json` passes112 synthetic mixed-shape requests:
non-aligned137/138/139 prompts, chunked prefill with changed starts/row order while
other requests decode, and32 two-turn conversations (4130 then4163 input tokens).
Four trace IDs survive unchanged, zero retirements/evictions and zero host-logits
reads. Prefix reuse is131072/132160 previously sent tokens (99.1768%). Host workload
255.249s, TTFTp50 4972.055ms/p95 32628.208ms/p99 32629.142ms; SSE pausep99 426.075ms,
max31916.428ms. These are mixed synthetic traffic metrics, not headline decode.
Long eager-prefill pauses remain. `mixed_traffic_timeline.svg` is a host event
visualization, not a device profile. `doc/agentic_serving.json` records memory,
prefix-hit limits and these latency findings.

## Trace, async and sampling decisions

Decode retains persistent token, position, RoPE, page-table, cache and sampler
inputs. The existing generator executes model/sampler traces nonblocking. The
adapter returns device tensors for `read_from_device=False`; deferred sampled-ID
readback transfers one replica, with plugin synchronization after the async
boundary. `after_adapter_contract.json` verifies delayed reads, changed tokens and
positions, changed/unchanged tables, exact subsequent state and stable captures.
Full-server logs show overlap exercised; supports_async_decode remains true.
Performance manifests omit host compatibility; shutdown/event audits show zero
host-logits reads. No host greedy argmax, full-vocabulary readback or eager greedy
sampling was introduced. Existing generator-wide preparation remains a superset
of the suppressed prefill path; no new allocation scope or trace exemption.

Full sampling profile:76 passed,74 skipped,0 failed at B8/native262144
(`sampling_verdict.json`). Explicit host compatibility is used only for unsupported
sampling parameters/logprobs in this conformance run; it is not a performance
configuration.71 skips concern unconfigured speculation;3 opt-in chunk-budget
cases are covered by the actual tracked mixed-chunk lifecycle.13 focused host
contract tests pass. The reduced real-layer terminal-suppression probe passes
128+128+1,64+64+65,123+6,128+1 boundaries, exact final logits/sampling/state and three
subsequent decode steps. Its first decode input is teacher-forced, so token-feedback
proof comes from the separate adapter test. Watcher10 passed with Ethernet watcher
excluded after its debug kernel exceeded Ethernet code capacity; compute watcher
remained enabled. See `anomaly_ledger.md` for recovery and warning controls.

Qualitative: all18 outputs from six packaged and three shared prompts (greedy and
temperature0.7/top-k4) were reviewed. All nine greedy strings exactly match the
existing full-model controls. Final-answer content is coherent and on topic.
Raw reasoning remains in API content because no reasoning parser is configured;
the structured answer after `</think>` is valid JSON, but the entire response is
not JSON-only. The256-token cap truncates three packaged greedy answers exactly
as in the controls, sampled story/thermodynamics and the shared sampled code
example. This is capped smoke coverage, not proof that those answers complete.
`qualitative_verdict.json` and `quality_control_comparison.json` record the scope.

## Rejected options and limitations

Batched recurrent snapshots reduced metadata uploads75→15 for the same39 saves and
34 restores, but mixed workload255.249→255.381s and B16 performance were flat. The
additional kernel/metadata implementation was rejected. Source and probes are
archived in `rejected_snapshot_batch/`; misleadingly named `final_b16_*` and
`final_b32_*` are this rejected experiment's historical artifacts, not acceptance.
`selected_source_restoration.json` verifies exact runtime equivalence with selected
terminal-only B8/B16/B32 evidence. The earlier `superseded_b1_*` run is excluded
because experimental source edits overlapped its lifetime; final B1 is rerun.

Unchecked page-table refresh removal would break scheduler allocator-growth
correctness and was rejected. Off-grid continuation still uses the existing
teacher-forced bridge trace, including its terminal work, before reaching the next
aligned eager segment. No extra trace variants were added for that path. Tiny
single-run CI/mixed latency differences do not establish a speedup. B16 serving
prefill includes vLLM admission/cache/snapshot work and uses native-context external
cache; standalone prefill timing is not an equivalent execution boundary.

## Reproduction and evidence

Inside `qwen38-functional-build`, use `tests/serve_vllm.sh`; it sets runtime/cache
paths and the real plugin adapter override. Primary command:

```sh
TT_METAL_TRACE_ALLOC_TRACKING=0 bash models/autoports/qwen_qwen3_8_27b/tests/serve_vllm.sh --stages serve,benchmark,qualitative --max-num-seqs 1 --no-benchmark-ci-serving
```

B32 CI command (also runs the separate one-request physical-B32 benchmark):

```sh
TT_METAL_TRACE_ALLOC_TRACKING=0 bash models/autoports/qwen_qwen3_8_27b/tests/serve_vllm.sh --stages serve,benchmark --max-num-seqs 32 --additional-server-args='--language-model-only --mamba-cache-mode align --mamba-block-size 4096 --max-num-batched-tokens 131072 --long-prefill-token-threshold 4096 --max-logprobs -1 --async-scheduling'
```

For trace acceptance use tracking1/program-cache-skip0 and
`QWEN38_SERVING_EVENTS`, then `tests/run_serving_lifecycle.py --max-concurrency 32
--agent-turn-length 4130`. Exact runtime argv/env/source hashes live in command
manifests; workload args and token counts live in raw benchmark JSON.
`artifact_index.json` maps immutable raw/normalized copies because normalized
runner paths originally referenced mutable `readiness_vllm` locations.
`work_log.md`, `topology_audit.md`, `optimization_checklist.md` and the final
independent review record acceptance and limitations. Commits remain local.

Independent review: **clean-pass**, no required work (`stage_review.md`). Stage-owned changes are checkpointed locally; commit receipts are in `work_log.md`. No push.
