# GPT-OSS 20B multichip decoder work log

## 2026-07-23: initialization and strategy lock

- Branch: `mvasiljevic/models/v2/openai-gpt-oss-20b`
- Starting HEAD: `e78e3cd1108`
- Starting worktree: clean
- Skills: `multichip`, `tt-device-usage`
- Scope: decoder layer, tests, context contract, and
  `doc/multichip_decoder`; no full-model or vLLM work

Hardware discovery found four Blackhole P300c devices. A serialized mesh
smoke opened `MeshShape(1,4)` with `FABRIC_1D_RING`, 11x10 worker grid, and
closed cleanly.

The pre-code decision selected TP4 attention plus EP4 whole experts:
16 local query heads, 2 local KV heads, 8 complete experts/rank, replicated
BF16 residual/norm/router, and ring all-reduces after O and experts. The
131072 context remained advertised because the calculated per-device model,
KV, RoPE, and runtime reserve is about 11.8 GiB versus 31.875 GiB available.
Exact shapes, padding, bytes, and rejected strategies are in `README.md`.

## 2026-07-23: implementation

Created `tt/multichip_decoder.py` as a real fixed-1x4 subclass of the current
`OptimizedDecoder`. Direct functional-base initialization avoids replicated
dense expert allocation.

- QKV `[2880,5120]` → four `[2880,1280]` column/head shards
- O `[4096,2880]` → four `[1024,2880]` row/K shards
- rank 0 owns real O bias; ranks 1-3 own zero bias
- local cache `[ceil(L/64),2,64,64]` BFP8
- 8 complete BFP8 experts/rank
- exactly global top-4 active routes, three sparse matmuls
- replicated BF16 `[1,1,S,2880]` stack boundary
- arbitrary positive logical S with internal padding/chunking

The runtime audit includes production and candidate collectives, paged
gathers, manual attention, norms, routing, active experts, prefill, and
decode. It rejects Torch/CPU conversions and inherited one-device fallbacks.

## 2026-07-23: first independent review

`stage_review_round1.md` returned `more-work-needed` with six findings:

1. preserve the optimized baseline's precision-sensitive full-prefill policy;
2. make long trace advance sequential positions;
3. regenerate source-provenanced timing/profile/context/watcher evidence;
4. remeasure topology alternatives on current code;
5. qualify the full-attention 131072 endpoint; and
6. explain or compensate for disabled Ethernet watcher.

The `autodebug` wrapper could not start because `bubblewrap` is unavailable.
A fresh-context source audit produced `AUTODEBUG.md`, then the authorized
AutoFix loop tested each hypothesis.

## 2026-07-23: AutoFix H1 precision

Full S=128 native attention produced routing PCC 0.982516. TP-local FP32
manual attention raised it to 0.998064. HiFi4 full-prefill active experts
raised final PCC from 0.998063 to 0.999240. BFP8 EP4 expert weights remained
qualified; dense experts were unnecessary.

Final policy:

- sliding prefill experts HiFi2;
- full prefill experts HiFi4;
- full S=128 TP-local FP32 manual attention;
- decode experts LoFi;
- sliding/full long projection HiFi2/HiFi4.

Real S=128/129/2048 across both layer kinds passes with minimum attention PCC
0.999994, final prefill PCC 0.997193, routing PCC 0.992617, prefill top-4
agreement 96.899%, and exact following-decode top-4.

Artifacts:

- `logs/precision_capture_s128_bfp8_hifi_split_final.junit.xml`
- `logs/precision_capture_s129_s2048_bfp8_hifi_split_final.junit.xml`
- `logs/precision_multichip_all_lengths_selected_final.junit.xml`

## 2026-07-23: AutoFix H2 sequential long trace

The native paged control was refuted at position 128: PCC 0.01579/0.20854
for sliding/full layers. The retained manual path uses fixed 64-token trace
banks, device-mutable position/masks, three sliding pages, and a fixed full
prefix through the bank end.

One capture advances 128-131 and 191; one recapture advances 192-193.
Both layers match eager with PCC 1.0, exact per-rank K/V writes, and five
deterministic replays.

Artifacts:

- `logs/precision_postmerge_h2_trace_regression.junit.xml`
- `logs/autofix_h5_boundary_trace_regression.junit.xml`

## 2026-07-23: AutoFix H4 topology

Implemented off-default current controls for padded RS+AG, fused attended
all-gather/local O, TP4 gate-selected experts, and a carried 736-wide
residual. Blackhole fused matmul+reduce-scatter is source-rejected due the
M-tiles=32 race in issue #46181.

Current correctness command:

```bash
RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 \
MULTICHIP_TOPOLOGY_TRACE_REPLAYS=100 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_decode_collective_rs_ag_pad64_matches_selected_all_reduce' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_fused_o_projection_all_gather_natural_bf16_candidate' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_tp4_gate_selected_experts_match_ep4_candidate' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_carried_ep_residual_to_distributed_norm_router_and_qkv'
```

Result: 8 passed. RS+AG PCC is 0.999807/0.999773. Fused O PCC is
0.999877/0.999880. TP4 versus EP4 expert PCC is 0.999728/0.999732 with
exact routes. Carried-residual minimum downstream PCC is 0.999888 with exact
routes, but latency is only 0.7824x/0.7820x the replicated boundary.

Frozen S=128 whole-layer 20/500 timing:

| Candidate | L12 prefill/decode ms | L13 prefill/decode ms |
| --- | ---: | ---: |
| selected all-reduce + EP4 | 21.7227 / 1.0581 | 25.2078 / 1.0417 |
| padded RS+AG + EP4 | 21.7068 / 1.0857 | 25.2011 / 1.0705 |
| fused attended AG/local O + EP4 | 21.7161 / 1.1911 | 25.1914 / 1.1789 |
| all-reduce + TP4 experts | 36.4951 / 1.1111 | 39.4058 / 1.0937 |

The original all-reduce + EP4 topology remains selected.

## 2026-07-23: AutoFix H5 full context

Both layer kinds allocate local `[2048,2,64,64]` K/V on every rank and write
reverse-mapped physical page 0 offset 63 at logical position 131071.

Sliding/full final output PCC is 0.997989/0.999263 against the default-native
current baseline and 0.998325/0.999319 against exact manual controls. Exact
attention PCC is 0.999791/0.999947; routing PCC is
0.999967/0.999994; top-4 is exact.

Artifacts:

- `logs/final_context_capture_131072.junit.xml`
- `logs/autofix_h5_manual_endpoint_capture.junit.xml`
- `logs/autofix_h5_endpoint_final_postformat.junit.xml`
- four hashed default-native/exact-manual endpoint `.pt` controls in `logs/`

The earlier `final_context_multichip_131072` run failed the layer-13 route at
0.934595 PCC and is superseded by the post-format H5 artifact above. It is not
part of the authoritative evidence set. No advertised capability changed.

## 2026-07-23: source freeze and complete suite

Frozen hashes:

- `tt/multichip_decoder.py`:
  `25fee42d80fb2bcf4a8b8136cb133775ab3ff824d60d31c885191c598cbd9d64`
- `tests/test_multichip_decoder.py`:
  `3df7e045c066b5beefb790ec8ff4998e9cf912b6c7f198de966f07ed75471c13`
- `doc/context_contract.json`:
  `f3797d184f7239fe481b5216db6f2b16d88c90f65b07bef46c48b1f127e3f1b2`

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/final_frozen_source_suite.junit.xml
```

Result: 34 passed, 18 expected opt-in skips, 0 failures in 313.19 seconds.

Static checks also pass:

```bash
python -m black --check \
  models/autoports/openai_gpt_oss_20b/tt/multichip_decoder.py \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py
python -m py_compile \
  models/autoports/openai_gpt_oss_20b/tt/multichip_decoder.py \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py
python -m json.tool models/autoports/openai_gpt_oss_20b/doc/context_contract.json
git diff --check
```

## 2026-07-23: frozen timings and profiler

All accepted rows use isolated processes, 20 prefills, and 500 trace
replays.

| Layer | S | 1x1 prefill | 1x4 prefill | 1x1 decode | 1x4 decode | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 sliding | 17 | 3.6884 | 5.0955 | 0.8736 | 0.6317 | 1.383x |
| 13 full | 17 | 3.7451 | 5.1102 | 6.7482 | 0.6347 | 10.632x raw |
| 12 sliding | 128 | 12.8805 | 21.7227 | 0.7968 | 1.0581 | 0.753x |
| 13 full | 128 | 10.2699 | 25.2078 | 6.7544 | 1.0417 | 6.484x raw |

The full-layer one-device path explains the superlinear raw ratios; they are
not scaling claims. All measured prefill and S=128 sliding decode regress.

Tracy wrote a profile-specific JSON and current raw op CSV. `tt-perf-report`
shows:

- decode: 35.16% ring all-reduce, 31.63% sparse experts, 7.08% matmul,
  modeled DRAM 26.4%/135 GB/s;
- prefill: 41.65% sparse experts, 18.19% ring all-reduce, modeled merged DRAM
  107.5%/550 GB/s.

Exact commands and CSV tables are in `perf/perf_report.md`.

## 2026-07-23: watcher and health

Full Ethernet watcher attempt:

```bash
TT_METAL_WATCHER=10 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_boundary_positions_match_current_optimized[blackhole-1x4-12]'
```

It failed before model execution with the expected platform limit:
ACTIVE_ETH program 27,920 bytes versus a 25,600-byte buffer.

Maximal worker/Tensix watcher:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_boundary_positions_match_current_optimized' \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_warmed_long_position_trace_replay_matches_eager'
```

Result: 4 passed with no watcher/NoC fault.

`python tools/tt-triage.py --run=check_eth_status --run=check_arc --dev=all`
reports healthy ARC heartbeats (~10/s) on all four devices.
`check_eth_status` is explicitly skipped because its installed UMD
`noc_read` Python signature is incompatible. Repeated ring runs and a final
`tt-smi -ls` show four available P300c boards.

Artifacts:

- `logs/watcher_eth_attempt_final_frozen.log`
- `logs/watcher_final_frozen.log`
- `logs/watcher_final_frozen.junit.xml`
- `logs/triage_eth_arc_final_frozen.log`
- `logs/tt_smi_final_frozen.log`

## 2026-07-23: closure sequence

The source/test/context hashes remained unchanged through every final
hardware run. `evidence_manifest.json` ties them to the authoritative
artifacts. Fresh independent review round 2 returned `clean-pass` after the
reviewer re-audited the corrected endpoint provenance chain; see
`stage_review_round2.md`. The local stage commit SHA is appended after commit.
No push is performed.

## Local commits

- starting checkpoint: `e78e3cd110810695ca1172002deffcc6ddb97e43`
- multichip decoder stage: `99388e9c5f7e184e246eb0325cd0f5c7f0accdb8`
