# GPT-OSS-20B optimized multichip decoder

Status: complete; independent stage review is `clean-pass`.

This stage optimizes the completed `openai/gpt-oss-20b` decoder in place on
the fixed `MeshShape(1, 4)` Blackhole P300c ring.  Every performance and PCC
number below is from `tt/multichip_decoder.py` using TP4 attention and EP4
gate-selected active experts.  Single-chip timing is retained only as the
pre-existing scaling reference.  No full-model or vLLM work is included.

## Selected default

- TP4 packed QKV, local paged SDPA, row-parallel O projection, then a BF16
  ring all-reduce.
- EP4 whole-expert ownership: eight experts per rank and exactly the global
  top four selected experts per token.  Dense all-expert execution is not
  used.
- Separate gate/up sparse projections with BFP8 weights; BFP4 down weights;
  BF16 sparse outputs, activations, and collectives.  The sparse kernels use
  explicit LoFi exact math; the dense attention/router kernels retain HiFi4.
- EP programs: `9x10`, `in0_block_w=45`, subblock 1 for S=128 prefill; `5x9`,
  `in0_block_w=45`, subblock 2 for decode.
- BF16 paged KV cache with two local KV heads per device.
- Replicated BF16 `[1,1,S,2880]` input/output residual.  No gather, reshard,
  or all-reduce is required between adjacent decoder layers.

The final output of a layer may be passed directly to the next layer.  The
next layer begins from the same replicated BF16 contract and performs its own
input RMSNorm.  The two remaining ring all-reduces are internal: one restores
the row-parallel attention O projection and one combines disjoint EP rank
contributions.  Full-model bringup should preserve this boundary rather than
inserting an inter-layer collective or conversion.

## Final correctness

The acceptance threshold is PCC >= 0.99 against the completed single-chip
optimized decoder.  `final_current_pcc_ep.junit.xml` contains the current
default rerun and every component value is also printed by the suite.

| Layer kind | Prefill PCC | Decode PCC, positions 128-130 | Attention PCC | Router PCC |
| --- | ---: | ---: | ---: | ---: |
| sliding attention | 0.995640 | 0.994720-0.995832 | 0.999994-0.999994 | 0.999994-0.999998 |
| full attention | 0.994173 | 0.994415-0.995403 | 0.999904-0.999941 | 0.999975-1.000000 |

The consolidated final-default suite is `final_default_suite.junit.xml`: nine
tests pass.  It covers both meaningful layer kinds, exact active top-4 routing,
S=17 and S=33 non-aligned logical prefill behavior, local KV heads and arbitrary
page mappings, traced mutable decode positions 128-130, trace recapture for a
new physical block, full 131072-token cache allocation and position 131071,
configuration parsing, and the static/runtime fallback audit.  Public inputs
remain logically unaligned; the decoder owns tile padding, chunk padding,
masking, and slicing.

## Final performance

Both measurements use real weights at S=128 on the target four-card mesh.
Prefill is warmed and averaged over 20 repeats.  Decode is warmed, captured,
and averaged over 500 trace replays.

| Path | Before | Final default | Improvement |
| --- | ---: | ---: | ---: |
| multichip warmed prefill | 22.648058 ms/layer | 20.583611 ms/layer | 9.12% (1.100x) |
| multichip traced warmed decode | 0.598642 ms/layer | 0.503275 ms/layer | 15.93% (1.190x) |

The final numbers are reproduced by the current defaults, not copied from an
earlier candidate.  Provenance is in `final_default_perf_seq128.json` (SHA256
`d01c2883c039800d6a6fd55e5f0dae8734481e29974b183e16fedd552529f00a`)
and `final_default_perf_seq128.junit.xml`.  The fresh before artifacts are
`baseline_perf_seq128.json` and `baseline_pcc_ep.junit.xml`.

## Operation-topology audit and actions

| Family | Audit finding | Action and measured evidence | Decision |
| --- | --- | --- | --- |
| repeated same-input matmuls | gate and up consume the same hidden activation and sparsity | packed the BFP8 gate/up weights into one 5760-wide sparse matmul; PCC was unchanged; subblock 2 measured 22.0785 / 0.64053 ms and subblock 1 measured 25.0251 / 0.67932 ms | reject: decode regression; keep the tested rewrite off by default |
| packed/fused projections | QKV was already one packed projection; gate/up was the remaining repeated projection | retained packed QKV and separate gate/up; packed gate/up artifacts are `graph_rewrite_packed_gate_up_pcc.junit.xml` and `packed_gate_up*_perf_seq128.json` | selected topology minimizes total warmed latency |
| residual layout | attention and EP both naturally produce partial H=2880 contributions; layer boundary was replicated | first compared replicated all-reduce + local norm/router/expert to padded RS + distributed norm/router + whole-expert gather (0.59346 vs 1.29431 ms); then carried the real EP partial through persistent RS, a 736-wide residual, distributed RMSNorm, and fused AG with a real packed QKV consumer | BF16 was 0.452406 vs 0.539213 ms and BFP8 was 0.452890 vs 0.526221 ms; both were highly accurate but 19.19%/16.19% slower, so retain replicated boundary |
| collective placement | restoring before the whole-expert input costs two internal all-reduces but avoids a boundary conversion | retained collectives immediately after O and EP down; no inter-layer movement remains | selected stack contract |
| async CCL + persistent buffers | CCLManager exposes preallocated output buffers, persistent semaphores, and ring async RS/AG | correct 2880-to-2944 RS+AG family passed PCC and trace safety; 20.58585 / 0.53381 ms | reject vs 0.503275 ms final decode |
| fused matmul-CCL | O is the applicable row-parallel matmul followed by a reduction | exact M=32/128, K=1024, N=2944 fused matmul+RS passed at ~0.999987 PCC; first rank-3 prefill failure was adapted to owned 4D reshape/padding and retried end to end | correct but 20.71160 / 0.65654 ms; reject and retain diagnostic switch |
| fused all-gather+matmul | attended activations are already head-fractured before O, and a fractured EP residual can feed the next packed QKV | implemented persistent fused attended-AG + rank-local O at natural N=720 and padded N=736; the first padded rank-3 failure was localized, then natural720 passed all ranks at PCC 0.999997 and both layer kinds. BF16 measured 20.77186 / 0.642176 ms; BFP8 isolated PCC was 0.999937 but full-attention prefill missed the gate at 0.989274. The carried EP-to-QKV BF16/BFP8 family above also passed | reject both applicable fused families with adapted correctness and timing evidence |
| DRAM-sharded decode matmuls | profiler advice identified QKV/O weight movement | built exact local QKV 2880x1280 and O 1024x2880 DRAM-sharded weights/programs, preserving bias and restoring layout only at the family boundary; BF16 22.6946/0.72015, BFP8 22.6681/0.66808, BFP4 22.7420/0.66790 ms | reject: activation reshards dominate |
| activation sharding | starting QKV and norms are L1 width-sharded; sparse programs distribute N across cores | crossed QKV/O DRAM activation sharding, EP 90/45/30-core geometries, and the fractured residual/distributed-norm path | retain 10-core QKV/norm and 45-core decode sparse geometry |
| lower precision | expert traffic dominated the first profile | all-BFP4 failed prefill PCC at 0.97895/0.97741; BFP4 gate/up failed at 0.98278/0.98133; BFP4 down alone passed and became default | select BFP8 gate/up + BFP4 down |
| activation/CCL dtype | original EP sparse outputs were BFP8 before collective | BF16 activation/CCL with mixed weights measured 20.59978/0.50348 in search and 20.58361/0.503275 finally; BFP8 activation was about 21.59/0.5853. BFP8 crossed with persistent padded RS+AG passed both layer kinds but measured 21.56382/0.62738 | select BF16 activation/CCL |
| fidelity | dense and sparse fidelity are separate controls | global HiFi2 measured 21.5635/0.58131 but changed a near-tied top-k route (router PCC 0.81211); global LoFi measured 21.6022/0.63194. The actual sparse default was made explicit as LoFi; sparse-only HiFi2 and HiFi4 both pass both layer kinds but measure 20.68978/0.504293 and 20.98461/0.509997 | retain explicit sparse LoFi and dense HiFi4 |
| interleaved attention precision | final packed QKV/local-O weights remained BF16 outside the rejected DRAM-sharded family | materialized the final interleaved topology directly at BFP8 and BFP4; BFP8 missed full-attention prefill PCC at 0.985071 and BFP4 missed sliding/full prefill at 0.858267/0.843126 | retain BF16 attention weights |
| KV dtype/layout | BFP8 could halve cache traffic and capacity | first fill dtype error was adapted by prefill casts; paged decode update requires BF16/FP32 and internally repacks, so decode stayed BF16; retry measured 21.5713/0.63686 | retain BF16 KV; context unchanged |
| sparse program geometry | decode uses one physical M tile and prefill S=128 | 90-core, 45-core/subblock-2, 30-core/subblock-3, in0 45/90, and 45/90-core prefill variants were measured | select 45-core/subblock-2 decode and 90-core/subblock-1 prefill |
| fused MoE | `moe_compute` supports exact clamped SwiGLU but has a rolling expert-buffer output contract | adapted a rank-local 8-expert, H=I=2880, top-4, biased compute-only probe; it passes, but validation exposes only the rolling double-buffer experts rather than arbitrary local expert 0-7 outputs needed by this EP path | reject for incompatible output ownership; production remains sparse active-expert execution |

The shard advisor was run on the exact local dense shapes.  It reported zero
L1 spills and advised a 40-core QKV output, 90-core residual/O/router/expert
layouts, and 10-core norms.  Its closest legal QKV geometry (`30,3,1,1`, since
the nominal 23-core input did not divide 90 K tiles) measured 22.62038 / 0.60963
ms and was rejected.  The retained report and final IR have SHA256
`e7a72adffe59ebcad22af4121ca83800226ed1e4dbc399557da7cae9c41e7886`
and `77aa7643fefa1f24a742c9f9a72cc334385f0c4336ce304b35ef73433a8b1d61`.

## Profiler evidence and advice disposition

The final Tracy capture is `perf/final/ops.csv` (SHA256
`7842a736b3def283deb524c374a7f46dae5f4d4438a94ce2af63e72037e65668`).
`tt-perf-report` tables, CSV summaries, PNGs, command logs, and actionable
advice are retained beside it.

- Prefill device time is 21.7655 ms in the instrumented run.  The three sparse
  matmuls total 11.9065 ms (54.70%); unary is 3.3736 ms (15.50%); reshape is
  3.2625 ms (14.99%); binary is 1.2283 ms (5.64%); and reduce-scatter work is
  1.2027 ms (5.53%).
- The decode profile range contains three trace replays and totals 1.473187 ms
  of device operations, or 0.491062 ms/device replay.  Its same-run wall time
  is 0.574001 ms/replay, leaving a named 0.082939 ms instrumentation/dispatch/
  synchronization gap.  The exact 143,397,376-byte compulsory-read roofline is
  0.070018 ms/token; the full accounting is `perf/final/perf_accounting.json`.
- The DRAM-sharded-matmul advice was tried at BF16/BFP8/BFP4 and rejected with
  complete-family timings above.  Dense fidelity advice was tried at
  HiFi2/LoFi; the former failed routing and the latter was slower.  Sparse
  HiFi2/HiFi4 advice was separately measured and rejected as slower.  Host-gap
  advice is addressed by the captured/replayed final decode path.

## Context, stress, fallback, and watcher

`../context_contract.json` records the unchanged maximum context of 131072.
Mixed expert weights save about 0.74 GiB/device across 24 layers; BF16 KV
layout is unchanged.  The largest logical cache position 131071 was physically
allocated and updated on all four ranks.  Internal padding never becomes a
public alignment precondition.

The runtime audit rejects host tensor conversion and CPU/PyTorch fallback in
the measured methods.  Active-expert tests verify four routes/token rather
than dense all-expert execution.  Worker/Tensix watcher instrumentation passes
real-weight EP prefill/decode for both layer kinds, attaches/checks/detaches
all four devices cleanly, and is captured in `final_watcher_ep.log`,
`final_watcher_ep.junit.xml`, and `final_watcher_device.log`.  Full Ethernet
watcher remains physically unavailable on this machine because instrumentation
grows ACTIVE_ETH to 27920 bytes beyond the 25600-byte firmware kernel buffer;
this is a hardware-tool limitation, not a decoder failure.
`final_tt_smi_post_gates.log` records four healthy resettable P300c boards
after the final PCC and latency runs.

## Reproduction commands

Run from the repository root in the configured environment.

```bash
RUN_MULTICHIP_DECODER_PERF=1 \
MULTICHIP_DECODER_PERF_SEQ=128 \
MULTICHIP_DECODER_PREFILL_REPEATS=20 \
MULTICHIP_DECODER_TRACE_REPLAYS=500 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/final_default_perf_seq128.json \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf -s

pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py \
  -k 'test_real_weight_prefill_decode_matches_single_chip_optimized and ep' -s

RUN_MULTICHIP_CONTEXT=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py \
  -k 'test_multichip_runtime_contract_and_fallback_audit or test_multichip_perf_candidate_parsing or test_warmed_trace_replay_updates_hidden_position_and_paged_cache or test_full_context_cache_allocation_and_last_page_update or (test_active_prefill_uses_exactly_four_routes_per_token and ep) or (test_synthetic_non_aligned_prefill_matches_optimized_and_cache_is_head_local and ep) or (test_real_weight_prefill_decode_matches_single_chip_optimized and ep)'
```

See `work_log.md` for the exact candidate commands, retry history, artifact
map, profiler invocation, watcher command, and checkpoint SHAs.
