# AutoFix report

The repair loop tested every round-1 hypothesis in isolation, retained
proven fixes, rejected unsupported alternatives, and reran the complete
frozen-source gates.

## H1: precision — fixed

Root cause was split across full-prefill attention and layer-aware compute
fidelity:

- full S=128 native attention routing PCC was 0.982516;
- TP-local FP32 manual attention raised it to 0.998064;
- HiFi4 full-prefill active experts raised final PCC from 0.998063 to
  0.999240; and
- full long-decode QKV/O needed HiFi4 to prevent an endpoint top-4 route
  flip. Sliding long decode remains HiFi2 and decode experts remain LoFi.

The retained production policy uses BFP8 expert weights and only
gate-selected EP4 execution. Real S=128/129/2048 tests for both layer kinds
pass with minimum attention PCC 0.999994, minimum final prefill PCC
0.997193, and following-decode PCC at least 0.997193. Prefill top-4 agreement
is at least 96.899%; every following-decode top-4 set is exact.

Primary artifacts:

- `logs/precision_capture_s128_bfp8_hifi_split_final.junit.xml`
- `logs/precision_capture_s129_s2048_bfp8_hifi_split_final.junit.xml`
- `logs/precision_multichip_all_lengths_selected_final.junit.xml`
- `logs/final_frozen_source_suite.junit.xml`

## H2: sequential long trace — fixed

The proposed native paged path was refuted: at position 128 it produced PCC
0.01579 for layer 12 and 0.20854 for layer 13. The retained device-only
manual attention uses fixed 64-token page banks and a device position tensor
to generate causal/sliding masks. Sliding gathers three pages; full attention
gathers the fixed prefix through the bank end. One capture is reusable within
the bank.

Both layers replay positions 128-131 and 191, then recapture once for
192-193. Output PCC is 1.0 versus eager, per-rank K/V writes are exact, and
five deterministic replays pass.

Primary artifacts:

- `logs/precision_postmerge_h2_trace_regression.junit.xml`
- `logs/autofix_h5_boundary_trace_regression.junit.xml`
- `logs/watcher_final_frozen.junit.xml`

## H3: provenance — fixed

`MULTICHIP_PERF_RESULT_PATH` isolates profile JSON from accepted wall timing.
Final source hashes are recorded in `evidence_manifest.json`. The frozen
suite, all 20/500 timings, candidate timings, Tracy source CSVs, watcher
runs, context gates, and health checks were regenerated afterward.

## H4: topology — selected all-reduce + EP4

Current frozen-source correctness passed for every candidate. Whole-layer
S=128 timing shows:

| Candidate | L12 prefill/decode ms | L13 prefill/decode ms | Decode delta versus selected |
| --- | ---: | ---: | ---: |
| all-reduce + EP4 | 21.7227 / 1.0581 | 25.2078 / 1.0417 | selected |
| padded RS+AG + EP4 | 21.7068 / 1.0857 | 25.2011 / 1.0705 | +2.60% / +2.76% |
| fused attended AG/local O + EP4 | 21.7161 / 1.1911 | 25.1914 / 1.1789 | +12.57% / +13.17% |
| all-reduce + TP4 experts | 36.4951 / 1.1111 | 39.4058 / 1.0937 | +5.01% / +4.99% |

The carried 736-wide residual reaches distributed RMSNorm, the row-sharded
router, and the next real packed QKV with minimum PCC 0.999888 and exact
top-4, but is 0.7824x/0.7820x the replicated boundary. Blackhole fused
matmul+reduce-scatter is source-rejected because issue #46181 documents an
M-tiles=32 race.

Primary artifacts:

- `logs/final_topology_candidates_correctness.junit.xml`
- `logs/final_perf_selected_ar_ep4_seq128.junit.xml`
- `logs/final_perf_candidate_{rs_ag_ep4,fused_o_ag_ep4,ar_tp4}_seq128.junit.xml`
- `logs/candidate_*.json`
- `logs/final_candidate_*.json`

## H5: context endpoint — fixed

Both layer kinds allocate local K/V `[2048,2,64,64]` on all four ranks with
reverse page mappings. Position 131071 writes physical page 0 offset 63.

Layer 12 endpoint final PCC is 0.998325 versus the exact-manual control and
0.997989 versus the default-native control. Layer 13 is 0.999319 and
0.999263. Exact-manual attention PCC is 0.999791/0.999947, routing PCC is
0.999967/0.999994, top-4 is exact, and K/V PCC is at least
0.999857/0.999875.

Primary artifacts:

- `logs/final_context_capture_131072.junit.xml`
- `logs/autofix_h5_manual_endpoint_capture.junit.xml`
- `logs/autofix_h5_endpoint_final_postformat.junit.xml`
- hashed default-native and exact-manual endpoint controls:
  `logs/current_optimized_context_endpoint_layer{12,13}.pt` and
  `logs/current_optimized_manual_context_endpoint_layer{12,13}.pt`

The pre-fix `final_context_multichip_131072` run is superseded: it exposed the
layer-13 HiFi2 endpoint route flip that H5 fixed and is intentionally excluded
from the final evidence manifest.

## H6: watcher — platform limit reproduced and compensated

Full watcher fails before model execution with the exact platform limit:
ACTIVE_ETH program 27,920 bytes versus a 25,600-byte kernel configuration
buffer. With only ETH instrumentation disabled, all four boundary/long-trace
tests pass under worker/Tensix watcher with four devices attached.

`check_arc` reports approximately 10 healthy heartbeats/s on all four
devices. `check_eth_status` is present but its reads are skipped by a current
UMD Python binding signature mismatch; this is recorded rather than claimed
as a pass. Compensation consists of the CCL-heavy watcher pass, repeated
successful 1x4 ring collectives/traces, and a final `tt-smi` listing of four
healthy P300c boards.

Primary artifacts:

- `logs/watcher_eth_attempt_final_frozen.log`
- `logs/watcher_final_frozen.log`
- `logs/watcher_final_frozen.junit.xml`
- `logs/triage_eth_arc_final_frozen.log`
- `logs/tt_smi_final_frozen.log`

## Closure

The exact frozen implementation passes 34 default tests with 18 expected
opt-in skips; candidate correctness separately passes 8/8. No AutoFix
hypothesis remains open. Independent stage-review closure is recorded in the
latest `stage_review_round*.md`.
