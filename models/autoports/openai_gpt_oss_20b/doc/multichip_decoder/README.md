# GPT-OSS 20B multichip decoder

Status: implementation and frozen-source hardware validation complete.
Independent review is recorded in `stage_review_round*.md`.

This stage implements one GPT-OSS decoder layer on the complete mesh exposed
by this host. It starts from `tt/optimized_decoder.py` and ends at a
stack-ready layer boundary. It does not include full-model or vLLM work.

## Hardware and strategy decision

The final strategy was chosen before the production path was frozen:

- four local Blackhole P300c devices;
- fixed `MeshShape(1, 4)` with `FABRIC_1D_RING`;
- H=2880, 64 query heads, 8 KV heads, head dimension 64;
- 32 experts, top-4 routing, expert intermediate H=2880; and
- 31.875 GiB allocator view per device.

The selected implementation is TP4 attention plus EP4 whole experts:

- 16 query heads and 2 KV heads per rank;
- 8 complete experts per rank;
- replicated BF16 residual, norm, and router boundaries;
- one ring all-reduce after the local O projection;
- one ring all-reduce after the local expert sum; and
- exactly gate-selected active-expert execution via three sparse matmuls.

Dense all-expert execution and replicated expert weights are absent.

## Tensor, activation, cache, and collective plan

| Boundary/tensor | Global logical shape | Mapping | Per-rank shape and physical contract |
| --- | --- | --- | --- |
| layer input/output | `[1,1,S,2880]` BF16 | replicated | same logical shape; tile/chunk padding is internal and sliced away |
| norm weights | `[2880]` BF16 | replicated | optimized 10-core decode norm layout |
| packed QKV weight | `[2880,5120]` BF16 | column/head TP4 | `[2880,1280]`: Q=1024, K=128, V=128 |
| packed QKV bias | `[5120]` BF16 | column/head TP4 | `[1280]` |
| query heads | `[1,64,S,64]` | head TP4 | `[1,16,S,64]` |
| key/value heads | `[1,8,S,64]` | head TP4 | `[1,2,S,64]`, local through cache/attention |
| O weight | `[4096,2880]` BF16 | row/K TP4 | `[1024,2880]`; real bias on rank 0, zero bias elsewhere |
| O partial | `[1,1,S,2880]` BF16 | partial sum | ring all-reduce to replicated residual |
| router | weight `[2880,32]`, scores `[S,32]` | replicated | FP32 scores/global top-4 before ownership filtering |
| expert gate/up/down | 32 complete H×H projections | expert-axis EP4 | 8 complete BFP8 experts/rank, BF16 activations/bias |
| expert partial | `[1,1,S,2880]` BF16 | disjoint expert ownership | ring all-reduce to replicated output |
| page table/position | `[1,ceil(L/64)]`, `[1]` INT32 | replicated | arbitrary physical-page permutation, trace-stable buffers |
| K or V cache | global `[ceil(L/64),8,64,64]` BFP8 | KV-head TP4 | `[ceil(L/64),2,64,64]` per rank |
| RoPE cos/sin | each `[1,1,L,64]` BF16 | replicated | device-side decode views |

The logical decode all-reduce payload is 2,880 BF16 values, or 5,760
bytes/token/rank, at each partial boundary. Tile-padded M=1 storage is 32
rows, or 184,320 bytes/rank. At S=128 the logical payload is 737,280
bytes/rank per boundary.

Logical sequence length is never required to be tile-, chunk-, or page-
aligned. Prefill pads/chunks internally, cache writes use the logical page
table, and outputs are sliced to exact S.

## Context and per-device memory

The advertised context remains 131,072 tokens. With 64-token pages, each
rank owns 2,048 blocks and local K/V shape `[2048,2,64,64]`.

- one local BFP8 K or V cache: about 17 MiB;
- K+V per layer: about 34 MiB;
- K+V for 24 layers: about 816 MiB/device;
- eight local BFP8 experts: about 201.7 MiB/layer;
- total layer weights including TP4 attention and replicated small tensors:
  about 214.5 MiB/layer, or 5.03 GiB/device;
- replicated full-context RoPE: about 0.75 GiB/device for 24 layers; and
- model/cache/RoPE plus embedding/head and a conservative 4 GiB runtime
  reserve: about 11.8 GiB/device.

This fits the 31.875 GiB allocator view. No physical limit justified reducing
the context contract, and the endpoint hardware gates pass both layer kinds.

## Precision and attention policy

AutoFix localized and retained the minimum layer-aware policy that matches
the current one-device optimized baseline:

- sliding prefill: native attention, HiFi2 active experts;
- full S=128 prefill: TP-local exact FP32 attention;
- full prefill: HiFi4 active experts;
- decode experts: LoFi;
- sliding long-decode QKV/O: HiFi2; and
- full long-decode QKV/O: HiFi4, needed to preserve an endpoint top-4 route.

Expert weights remain BFP8; all paths remain sparse and gate-selected.

Below position 127, decode uses native paged attention. At and above position
127, the native paged control is not sink-correct on this model (PCC
0.01579/0.20854 at position 128 for layers 12/13), so the implementation uses
a device-only FP32 sink-aware path:

- one fixed capture per 64-token page bank;
- mutable device position and masks within the bank;
- three gathered pages for sliding attention; and
- a fixed prefix through the bank end for full attention.

Recapture is required only at a 64-token bank boundary, not at every token.

## Correctness and cache results

All acceptance comparisons use real checkpoint weights and isolated current
`OptimizedDecoder` references. Required PCC is 0.99.

The frozen default suite reports 34 passed and 18 expected opt-in skips.
It covers both meaningful layer kinds, arbitrary pages, stacked layout,
fallback audit, S=17/127/128/129/2048, following decode, and both trace
contracts.

### Real prefill and following decode

Across S=128, 129, and 2048 for both layers:

- minimum attention residual PCC: 0.999994;
- minimum prefill routing PCC: 0.992617;
- minimum final prefill PCC: 0.997193;
- minimum prefill selected-expert top-4 agreement: 96.899%;
- every following-decode top-4 set: exact;
- following-decode PCC: at least 0.997193; and
- logical prefill K/V PCC: 1.0.

At S=17, final decode PCC is 0.998110 for sliding layer 12 and 0.998843 for
full layer 13.

### Page/sliding boundary

For positions 127-131:

| Layer | S=127 prefill PCC | Min attention PCC | Min routing PCC | Min final decode PCC |
| --- | ---: | ---: | ---: | ---: |
| 12 sliding | 0.999791 | 0.999397 | 0.999961 | 0.997467 |
| 13 full | 0.999308 | 0.999051 | 0.999961 | 0.997569 |

Reverse page mappings verify logical K/V gathers and exact appended local
cache rows. Synthetic S=129 crosses both a 64-token physical page and the
128-token internal work chunk. Direct layer-12-to-layer-13 device handoff
preserves replicated BF16 `[1,1,S,2880]` without host conversion.

### Full advertised endpoint

At position 131071, both layers allocate per-rank K/V
`[2048,2,64,64]`, use a reverse page table, and write physical page 0 offset
63 on every rank.

| Layer | Attention PCC vs exact manual | Routing PCC | Top-4 | Final PCC vs exact manual | Final PCC vs default native |
| --- | ---: | ---: | --- | ---: | ---: |
| 12 sliding | 0.999791 | 0.999967 | exact | 0.998325 | 0.997989 |
| 13 full | 0.999947 | 0.999994 | exact | 0.999319 | 0.999263 |

Endpoint K/V PCC is at least 0.999857/0.999875.

## Trace, determinism, and fallback audit

- One short native capture mutates hidden and position across 17-19.
- One long page-bank capture advances through 128-131 and 191, then one
  recapture covers 192-193.
- Both layers match eager with PCC 1.0 for every replay.
- Local physical K/V pages match eager exactly.
- Five repeated replays are deterministic.
- The runtime audit covers collectives, paged gathers, manual attention,
  norms, routing, active experts, prefill, decode, and candidate branches.
  It forbids Torch/CPU conversion and inherited single-chip fallbacks.

## Topology alternatives and decision

Current frozen-source correctness passes all eight candidate cases. S=128
whole-layer timing is:

| Candidate | L12 prefill/decode ms | L13 prefill/decode ms | Result |
| --- | ---: | ---: | --- |
| ring all-reduce + EP4 | 21.7227 / 1.0581 | 25.2078 / 1.0417 | selected |
| padded RS+AG + EP4 | 21.7068 / 1.0857 | 25.2011 / 1.0705 | decode 2.60%/2.76% slower |
| attended AG/local O + EP4 | 21.7161 / 1.1911 | 25.1914 / 1.1789 | decode 12.57%/13.17% slower |
| ring all-reduce + TP4 experts | 36.4951 / 1.1111 | 39.4058 / 1.0937 | prefill 68.00%/56.33%, decode about 5% slower |

The carried residual alternative pads H=2880→2944, reduce-scatters to
736/rank, and reaches distributed RMSNorm, a row-sharded router, and the next
real packed QKV. It passes with minimum PCC 0.999888 and exact top-4 but is
only 0.7824x/0.7820x the replicated boundary.

Blackhole fused matmul+reduce-scatter is source-rejected because issue #46181
guards an M-tiles=32 race. Dense all-expert execution is rejected by
contract and was not benchmarked as a candidate.

## Warmed performance and profiler

All wall rows use separate 1x1/1x4 processes, 20 warmed prefills, and 500
trace replays.

| Layer | S | 1x1 prefill | 1x4 prefill | Prefill speedup/eff. | 1x1 decode | 1x4 decode | Decode speedup/eff. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 sliding | 17 | 3.6884 | 5.0955 | 0.724x / 18.10% | 0.8736 | 0.6317 | 1.383x / 34.58% |
| 13 full | 17 | 3.7451 | 5.1102 | 0.733x / 18.32% | 6.7482 | 0.6347 | 10.632x / 265.80% raw |
| 12 sliding | 128 | 12.8805 | 21.7227 | 0.593x / 14.82% | 0.7968 | 1.0581 | 0.753x / 18.83% |
| 13 full | 128 | 10.2699 | 25.2078 | 0.407x / 10.19% | 6.7544 | 1.0417 | 6.484x / 162.10% raw |

The full-layer baseline has a distinct ~6.75 ms path; its superlinear raw
ratios are not mesh scaling claims. Prefill and S=128 sliding decode are
known regressions on this mandatory fixed mesh.

The frozen `tt-perf-report` profile attributes decode to 35.16% ring
all-reduce, 31.63% sparse active-expert matmul, 7.08% ordinary matmul, 0.90%
paged SDPA, and 0.79% paged updates. Decode modeled DRAM is 26.4%/135 GB/s.
Prefill is 41.65% sparse matmul and 18.19% ring all-reduce, with a merged
107.5%/550 GB/s modeled roofline. Exact tables, CSVs, commands, and
interpretation are in `perf/perf_report.md`.

## Watcher and device health

Full `TT_METAL_WATCHER=10` reproduces a platform-only failure before model
execution: the instrumented ACTIVE_ETH program is 27,920 bytes versus the
25,600-byte kernel configuration buffer.

With `TT_METAL_WATCHER_DISABLE_ETH=1`, worker/Tensix watcher passes boundary
and sequential page-bank trace tests for both layers: 4 passed, all four
devices attached, no watcher/NoC assertion, deadlock, illegal access, hang,
or timeout.

Repository `check_arc` reports about 10 heartbeats/s on all four devices.
`check_eth_status` cannot read on this installed UMD because its Python call
uses an incompatible `noc_read` signature; the log records the skip and is
not represented as a pass. Coverage is compensated by the CCL-heavy watcher
run, repeated successful ring collectives/traces, and a final `tt-smi`
listing of four P300c boards.

Known non-fatal host warnings are low `/dev/shm` margin and missing
`B850M-C` discovery metadata.

## Reproduction

```bash
# Frozen default suite.
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py

# Current topology correctness controls.
RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 \
MULTICHIP_TOPOLOGY_TRACE_REPLAYS=100 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_decode_collective_rs_ag_pad64_matches_selected_all_reduce' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_fused_o_projection_all_gather_natural_bf16_candidate' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_tp4_gate_selected_experts_match_ep4_candidate' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_carried_ep_residual_to_distributed_norm_router_and_qkv'

# Both 131072 endpoint cases, after isolated 1x1 reference capture.
RUN_MULTICHIP_CONTEXT=1 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_current_single_chip_full_context_endpoint_reference'
RUN_MULTICHIP_CONTEXT=1 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_current_single_chip_manual_full_context_endpoint_reference'
RUN_MULTICHIP_CONTEXT=1 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_full_context_cache_allocation_and_last_page_update'

# Worker/Tensix watcher; full ETH watcher is the documented size-limit control.
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_boundary_positions_match_current_optimized' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_warmed_long_position_trace_replay_matches_eager'
```

## Authoritative artifacts

- source/provenance: `evidence_manifest.json`
- independent review: `stage_review_round2.md` (`clean-pass`)
- AutoFix investigation: `AUTODEBUG.md`, `AUTOFIX.md`
- frozen default suite: `logs/final_frozen_source_suite.junit.xml`
- precision: `logs/precision_multichip_all_lengths_selected_final.junit.xml`
- endpoint: `logs/final_context_capture_131072.junit.xml`,
  `logs/autofix_h5_manual_endpoint_capture.junit.xml`,
  `logs/autofix_h5_endpoint_final_postformat.junit.xml`, and the four hashed
  `logs/current_optimized*_context_endpoint_layer*.pt` control tensors
- trace: `logs/precision_postmerge_h2_trace_regression.junit.xml`,
  `logs/autofix_h5_boundary_trace_regression.junit.xml`
- topology: `logs/final_topology_candidates_correctness.junit.xml` and
  `logs/final_perf_candidate_*.junit.xml`
- accepted timing: `logs/current_optimized_perf_layer*_seq*.json`,
  `logs/final_selected_ar_ep4_layer*_seq*.json`
- profile: `perf/perf_report.md`, `perf/*_final_frozen*.csv`
- watcher/health: `logs/watcher_eth_attempt_final_frozen.log`,
  `logs/watcher_final_frozen.log`, `logs/watcher_final_frozen.junit.xml`,
  `logs/triage_eth_arc_final_frozen.log`, `logs/tt_smi_final_frozen.log`

## Limitations

- Batch one and this fixed 1x4 P300c ring only.
- All measured prefills and S=128 sliding decode are slower than 1x1.
- Long decode recaptures once per 64-token page bank.
- Full Ethernet watcher instrumentation exceeds a platform kernel buffer;
  worker/Tensix watcher and complementary fabric evidence are clean.
- This stage ends at the decoder layer and does not start full-model or vLLM
  integration.
