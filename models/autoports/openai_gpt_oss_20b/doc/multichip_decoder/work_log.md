# GPT-OSS-20B full-mesh multichip decoder work log

Final runtime policy: fixed 1x4 TP4 attention + EP4 whole active experts,
replicated layer boundary, QKV `(10,9,2,2)`, 9x10/subblock-1 gate/up and down
prefill grids with post-sparse BF16, context 131,072.

## 2026-07-18: stage start and pre-implementation decision

- Scope is restricted to `tt/multichip_decoder.py`, its tests, and stage/docs.
  Full-model and vLLM work are not started.
- Baseline: optimized decoder commit `9949cb70f3f`; archived TP2 checkpoint at
  current HEAD `51757f775a6` is used only as a seed.
- Required references read: `$multichip`, `$tt-device-usage`, `$optimize`,
  `$tt-enable-tracing`, `$autofix`, `$stage-review`,
  `tech_reports/LLMs/llms.md` section 3.3, optimized-decoder evidence, and
  `doc/functional_decoder/multichip_provenance.json`.
- `timeout 60 tt-smi -ls --local`: exit 0, four Blackhole P300c devices.
- Serialized mesh smoke: set `FABRIC_1D_RING`, `ttnn.get_num_devices()==4`,
  opened and closed `MeshShape(1,4)` with exit 0 and `MESH_SMOKE_OK`.
- Mesh-open warnings to classify in final evidence: only about 17.5 MiB free
  in `/dev/shm` for a requested 16 MiB MPI segment; unknown `B850M-C`
  motherboard falls back to bus IDs; inspector could not replace a
  permission-owned `generated/inspector/programs_log.yaml`. None prevented
  mesh open/close. Hardware tests will determine whether they affect the path.
- Chosen target: fixed full 1x4 ring matching compiler provenance. Planned
  default is TP4 attention and TP4 gate-selected experts with 2,880->2,944
  internal intermediate padding. Mandatory compiler-prior candidate is TP4
  attention plus whole-expert EP4 (eight experts/rank). Both retain exact
  top-4 execution and a replicated stack boundary.
- Initial context calculation retained 131,072 tokens with 1.5 GiB KV/device
  for 24 layers, about 5.04 GiB weights, 1.2 GiB embedding/head allowance, and
  4.0 GiB runtime/trace reserve. Final audit added a conservative 1.5 GiB for
  the per-layer replicated tiled prefill RoPE pair and row-major decode pair,
  making the corrected total about 13.24 GiB versus 31.875 GiB DRAM.
- Detailed tensor/activation/cache/program table and collective topology table
  were written to `README.md` before implementation.

## Commands

```bash
timeout 60 tt-smi -ls --local
timeout 90 python - <<'PY'
import ttnn
print(ttnn.get_num_devices())
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1000000)
print(mesh.shape)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK')
PY
```

## Stage progression

The implementation and hardware gates below supersede the initial pending
state. Independent review and checkpoint records are appended at the end.

## First TP4 focused experiment

- Command: synthetic TP4 non-aligned prefill test, isolated 1x4 session.
- Result: attention/cache setup reached the first active gate sparse matmul,
  which rejected a 24-core `per_core_N=1` grid because the padded local expert
  width has 23 tiles and therefore only 23 cores had work.
- Interpretation: verified program-geometry error, not a sharding or numerical
  failure. The gate/up seed is changed from 3x8 cores to 3x4 cores; the builder
  then assigns `per_core_N=2`, so all 12 rectangular cores have work. No other
  policy changed.
- Artifact: `logs/synthetic_tp4.junit.xml` (negative pre-fix evidence).

## First EP4 focused experiment

- Command: synthetic EP4 non-aligned prefill test, isolated 1x4 session.
- Result: the first runtime `mesh_partition([S,32], dim=1)` rejected rank
  starts at expert 8/16/24 because a tiled width slice must start on a 32-wide
  boundary.
- Interpretation: the compiler prior records a logical `[S,32]->[S,8]`
  partition but not the required physical layout adaptation. Routing scores
  are now converted to row-major before partition; each local `[S,8]` result
  stays row-major as sparse metadata and is separately tilized for route
  weighting. Expert ownership and collective placement are unchanged.
- Artifact: `logs/synthetic_ep4.junit.xml` (negative pre-fix evidence).

## First real-weight TP4 experiment

- Sliding attention passed: prefill PCC `0.9981549`, attention PCC
  `0.99999397--0.99999457`, routing PCC `>=0.9999939`, and final decode PCC
  `0.9991766--0.9992319` at positions 128--130.
- The initial full-attention seed 20260718 reached attention PCC `0.99992784`
  at position 129 but swapped a near-tied fourth-place expert (routing PCC
  `0.72514`). This is a top-k discontinuity stress anomaly, not evidence for a
  broad precision change. The main correctness gate is being recaptured with
  the optimized-stage canonical seeds (sliding 10010, full 20260717); the
  20260718 anomaly remains negative evidence for an exact-attention component
  control.
- Artifact: `logs/real_tp4_correctness.junit.xml`.

## Correctness completion and final strategy selection

- Both TP4 and EP4 passed synthetic non-aligned S=17/S=33 prefill against an
  isolated 1x1 optimized TTNN artifact, with prefill PCC at least
  `0.9999994139`. Arbitrary page-table local K/V comparisons are PCC 1.0 on
  every rank.
- Both strategies passed real canonical sliding-attention seed 10010 and
  full-attention seed 20260717. The final EP4/QKV10 confirmation records:
  - sliding prefill `0.99947939798`; attention `0.9999937943`,
    `0.9999944066`, `0.9999943094`; routing at least `0.9999942775`; final
    decode `0.9994587921`, `0.9994009754`, `0.9994291157`;
  - full prefill `0.99808843089`; attention `0.9999036544`,
    `0.9999345434`, `0.9999408975`; routing at least `0.9999747856`; final
    decode `0.9992363002`, `0.9990928377`, `0.9991240644`.
- Direct output layout is replicated `[1,1,1,2880]` and was consumed through
  the next decoder contract without host fallback.
- Controlled S=17 routing records 68 global active entries for each of gate,
  up, and down. TP4 has 68 on every fractured rank; EP4 rank-local masks sum
  to 68. Every sparse call uses exact sparse metadata, and EP4's three calls
  use `nnz=None` because rank-local counts vary from zero to four.
- Final artifacts: `logs/final_ep4_qkv10_correctness.junit.xml` and
  `logs/active_expert_tp4_ep4.junit.xml`.

## Trace, cache, context, topology, and stress

- Warmed trace capture/replay passed sliding and full layers at mutable
  positions 128--130, PCC 1.0 against eager. Reverse page-table K/V pages are
  bit-identical on all four ranks. Five repeated trace replays are bit
  deterministic. Artifact: `logs/final_ep4_qkv10_trace.junit.xml`.
- The selected default allocates per-rank K and V `[2048,2,64,64]` caches and
  updates logical position 131071 through a reversed page table at physical
  page 0, offset 63. All eight rank/cache checks are nonzero. Artifact:
  `logs/full_context_131072.junit.xml`.
- The coherent sharded-stream candidate included O reduce-scatter, width-local
  residual, distributed RMSNorm, row-sharded router, 32-logit all-reduce, and
  a full-hidden gather for experts. Norm/router PCCs are `0.9999925766` and
  `0.9999975049`. It took `1.2716548983 ms` versus `0.6087731104 ms` for the
  selected replicated O-all-reduce boundary, so replicated is `2.0888815x`
  faster. Artifacts: `logs/residual_topology_candidate.json` and
  `logs/residual_topology_probe.junit.xml`.
- Near-tie seed 20260718 reproduced a fourth/fifth route swap at attention PCC
  `0.99992783899`. Replacing only TP attention with the exact baseline restores
  routing PCC 1.0 and gives final PCC `0.9993818206`; this isolates the stress
  discontinuity without weakening canonical correctness. Artifacts:
  `logs/near_tie_baseline.junit.xml`, `logs/near_tie_tp4_stress.junit.xml`.

Commands:

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/final_consolidated_suite.junit.xml

RUN_MULTICHIP_CONTEXT=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_full_context_cache_allocation_and_last_page_update \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/full_context_131072.junit.xml

RUN_MULTICHIP_TOPOLOGY_PROBE=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_sharded_residual_topology_candidate \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/residual_topology_probe.junit.xml
```

The pre-review consolidated default run finished `17 passed, 4 skipped in
139.70s`. The final post-remediation suite is recorded below and supersedes
this count after two opt-in CCL parametrizations were added.

## Fixed-policy performance sweep

Every candidate retained the optimized baseline dtype/fidelity policy. Timings
used 10 warmed S=128 prefill iterations and 100 warmed trace replays:

| Strategy / QKV geometry | Prefill ms | Decode ms | Result |
| --- | ---: | ---: | --- |
| TP / `(30,3,2,2)` | 39.8186 | 0.656294 | rejected |
| EP / `(30,3,2,2)` | 26.7193 | 0.604859 | EP wins |
| EP / `(30,3,1,1)` | 26.7182 | 0.609944 | rejected |
| EP / `(30,3,4,4)` | 26.7303 | 0.609892 | rejected |
| EP / `(18,5,2,2)` | 26.7117 | 0.599635 | rejected |
| EP / `(10,9,2,2)` | 26.7505 | 0.598815 | selected |
| EP / `(45,2,2,2)` | 26.7347 | 0.618945 | rejected |

The production reconfirmation used 20 prefill iterations and 500 trace
replays. Isolated 1x1 optimized baseline: `13.4082631 ms` prefill and
`0.9758344321 ms` decode. Selected 1x4: `26.6769041 ms` prefill and
`0.5986405937 ms` decode, for decode speedup `1.630083964x`, efficiency
`40.7521%`, and prefill speedup `0.50261691x`. The prefill regression is
explicitly accepted only as a limitation of this decode-focused stage.

Key artifacts:

- `logs/single_chip_perf_reference_seq128.json`
- `logs/final_perf_ep4_qkv10_seq128.json`
- `logs/perf_tp4_default_seq128.json` and all `logs/perf_ep4_*.json`
- matching JUnit XML provenance

Representative commands (the baseline capture ran in a separate 1x1 process):

```bash
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PREFILL_REPEATS=20 \
MULTICHIP_DECODER_TRACE_REPLAYS=500 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_single_chip_optimized_perf_reference

RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PREFILL_REPEATS=20 \
MULTICHIP_DECODER_TRACE_REPLAYS=500 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/final_perf_ep4_qkv10_seq128.junit.xml
```

## Profiler audit

Profiler and watcher were never enabled together. The successful profiler
command was:

```bash
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PREFILL_REPEATS=1 \
MULTICHIP_DECODER_TRACE_REPLAYS=3 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/profile_perf.json \
python -m tracy -r -p \
  -o models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/perf/tracy_final \
  -n gpt_oss_20b_ep4_qkv10 -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf
```

- Decode: AllBroadcast 12.81%, sparse matmul 16.76% combined, QKV 3.76%,
  router 4.22%, SDPA 3.08%; aggregate modeled DRAM roofline 40.7%.
- Prefill: sparse matmul 53.04%, typecast 13.86%, reshape 11.60%, unary
  5.96%, fill-pad 5.68%, reduce-scatter 5.45%.
- The communication/topology alternative, TP/EP candidates, dynamic rank-local
  nnz constraint, QKV sweep, optimized-stage sharding prior, and inherited
  dtype/fidelity rejections are dispositioned in `perf/perf_report.md`.
- Retained checkpoint tables: `perf/cpp_device_perf_report_ep4_qkv10_autofix.csv`,
  `perf/decode_report_autofix.csv`, `perf/prefill_report_autofix.csv`, summary
  CSVs, and summary PNGs. The 4.48 MiB unmerged op dump remains locally but is
  omitted from the checkpoint under the repository's 500 KiB file limit.
- The raw `perf/tracy_final` directory was 1.4 GiB and was deleted after the
  compact tables were retained. It is not recoverable except by rerunning the
  recorded command.

## Watcher, fallback, and final health

The runtime-method audit passes with no `torch`, `from_torch`, `to_torch`,
`get_device_tensors`, or CPU path in prefill/decode/trace execution. The final
watcher command was run separately from the profiler:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
script -q -e -c "pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_prefill_decode_matches_single_chip_optimized -k ep --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/watcher_final_ep4_qkv10.junit.xml" \
models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/watcher_final_ep4_qkv10.log
```

Result: 2 passed, 2 deselected. No watcher/NoC assertion, deadlock, illegal
access, hang, or timeout appears. Ethernet watcher was disabled for installed
firmware 19.8.0; Tensix watcher attached to and detached from devices 0--3.
The final `tt-smi -ls --local` exits 0 with all four P300c devices present and
resettable; see `logs/final_device_health.txt`.

## Superseded failures and recovery

Historical negative artifacts remain for auditability:

- `logs/synthetic_tp4.junit.xml`: invalid 24-core receiver for 23 output tiles;
  fixed by the 3x4 program grid.
- `logs/synthetic_ep4.junit.xml`: tiled width partition at expert offsets
  8/16/24; fixed by row-major routing-score partition.
- `logs/real_tp4_correctness.junit.xml`: near-tied seed route swap; isolated by
  the exact-attention control and covered by canonical end-to-end seeds.
- `logs/final_trace_replay_mutable_page_table.junit.xml`: replay after changing
  page-table contents without recapture fails for both layer kinds, proving
  that this TTNN paged-attention trace treats the table as capture-static. The
  valid successor preserves populated mappings and passes after explicit
  release/warm/recapture in
  `logs/final_trace_replay_page_table_recapture.junit.xml`.

After each hardware failure the devices were reset/reopened and a mesh smoke
passed before the next experiment. None of these XMLs is a current gate; the
final acceptance artifacts and consolidated suite are green.

## Independent review and commits

Round 1 returned `more-work-needed`; see `stage_review_round1.md`. Its three
required findings were handled through `$autofix` as follows.

### Persistent/alternate decode CCL

- Implemented a focused policy A/B at both decode reductions. The candidate
  explicitly pads 2,880 to 2,944, runs persistent-semaphore minimal-async
  reduce-scatter plus async all-gather on ring axis 1, and slices to 2,880.
- Correctness passes sliding/full layers; final PCC is at least
  `0.9990923269`. Mutable-position trace replay, paged-cache equivalence, and
  repeat determinism also pass.
- 20/500 wall timing: candidate `0.638599288 ms`, current all-reduce
  `0.598640594 ms`; candidate is 6.7% slower and rejected.
- A dedicated Tracy run confirms six
  `ReduceScatterMinimalAsyncDeviceOperation` rows (`152.28 us`) and six
  `AllGatherAsyncDeviceOperation` rows (`88.28 us`), plus explicit pad/slice.
  Compact device/report CSVs and human-readable command logs are retained; the
  4.63 MiB unmerged op dump remains locally but exceeds the repository's 500
  KiB file limit. The reproducible 1.4 GiB raw directory was removed.
- Fused matmul + minimal RS is blocked specifically on Blackhole by GPT-OSS
  race #46181. Gathered-input/local-output O needs another all-gather to
  recover the replicated stack boundary; the coherent sharded-stream family
  is already measured 2.089x slower.
- Artifacts: `logs/autofix_ccl_rs_ag_correctness.junit.xml`,
  `logs/autofix_ccl_rs_ag_trace.junit.xml`,
  `logs/autofix_perf_ccl_rs_ag_pad64_seq128.json`,
  `logs/profile_perf_ccl_rs_ag_pad64.json`, and
  `perf/*ccl_rs_ag_pad64*`.

### EP4 prefill remediation

All tests held the QKV10 policy, exact top-4 routes, dtypes/fidelity, and public
layout constant unless the candidate name states otherwise.

| Candidate | S=128 prefill ms | Disposition |
| --- | ---: | --- |
| reviewed 3x5 gate / 5x6 down | 26.6769 | superseded |
| gate 5x6 | 24.9871 | improved |
| down 5x9 | 26.7416 | rejected |
| down 9x10 | 26.4462 | improved |
| gate 5x6 + down 9x10 | 24.6387 | improved |
| chunk 64 / 32 | 27.0177 / 27.2026 | rejected |
| post-sparse BF16 | 24.8080 | improved |
| 5x6 + 9x10 + BF16 | 22.8116 | improved |
| 5x9 + 9x10 + BF16 | 22.9150 | rejected |
| 9x10 + 9x10 + BF16 | 22.6662 | selected |

The selected rewrite materializes gate/up/down sparse outputs as BF16, then
returns only the local weighted partial to BFP8 immediately before the ring
collective. This removes four prefill fill-pad calls and eight DRAM typecasts
while keeping the collective and residual contract unchanged. Canonical real
sliding/full and exact-route synthetic tests pass after promotion. Final
20/500 production timing is `22.6186355 ms` prefill and `0.598718766 ms`
decode, versus isolated one-chip `13.4082631/0.975834432 ms`: decode speedup
`1.629871x`, efficiency `40.7468%`, prefill speedup `0.592797x`. This improves
the reviewed multichip prefill by 15.2%.

Fresh profiler attribution: prefill device-op total `23,971.90 us`; sparse
`12,840.49 us` (53.56%), typecast `2,580.63 us` (10.77%), reshape `3,260.31
us` (13.60%), fill-pad `251.43 us` (1.05%), and RS/AG `1,399.05/51.44 us`.
The final compact files use the `*_autofix` suffix; both reproducible raw
Tracy captures were 1.4 GiB and were deleted after verifying retained CSVs.

### Full-watcher closure and recovery

The exact full-Ethernet command was run with
`TT_METAL_WATCHER_DISABLE_ETH` unset. Both parametrizations fail during mesh
setup, before model execution, with the same hard physical instrumentation
limit:

```text
TT_FATAL: Program size (27920) too large for kernel config buffer (25600) on ACTIVE_ETH
```

Console, JUnit, and generated watcher logs are retained as
`watcher_full_eth_*`. Devices detached cleanly, but the approved recovery was
still applied: `timeout 180 tt-smi -r` reset IDs 0--3; `tt-smi -ls --local`
showed all four devices; and `post_full_eth_reset_mesh_smoke.log` records four
devices plus `MeshShape([1,4])` and `MESH_SMOKE_OK`.

The maximal physically legal worker/Tensix watcher run then used
`TT_METAL_WATCHER_DISABLE_ETH=1` on the promoted production default. Sliding
and full attention both pass (`2 passed, 2 deselected`) with clean watcher
attach/detach on devices 0--3 and no NoC/assert/hang/timeout. Artifacts:
`watcher_final_ep9x10_bf16.log`, its JUnit, and generated watcher log.
Profiler and watcher were never combined.

### Final gates

- Final consolidated suite: `17 passed, 6 skipped in 139.91s`; skips are the
  two CCL, context, topology, and two performance opt-ins, all with dedicated
  passing artifacts.
- Promoted correctness: sliding prefill PCC `0.9992127929`, full prefill PCC
  `0.9978150772`; attention/decode PCC values remain those recorded above.
- Promoted warmed trace: both layer kinds PCC 1.0 with mutable positions,
  bit-identical reverse-page cache, and five deterministic repeats. A focused
  page-table audit proves that buffer-content changes are capture-static for
  this paged-attention op. After preserving populated mappings, swapping two
  future physical allocations, releasing/warming/recapturing, both layers pass
  eager PCC 1.0 at position 192 and update the selected page. Artifact:
  `logs/final_trace_replay_page_table_recapture.junit.xml`.
- Final `tt-smi -ls --local`: exit 0, devices 0--3 present/resettable; artifact
  `logs/final_device_health.txt`.
- `$autofix` succeeded; its hypothesis/result mapping is in `AUTOFIX.md`.

Round 2 returned `clean-pass` with no required work; see
`stage_review_round2.md`. The review accepted the physically limited
full-Ethernet watcher result, capture-static page-table contract, remaining
prefill slowdown, and fixed batch-one/1x4 scope as documented residual risks.

Local stage-owned checkpoint SHAs are appended after checkpoint creation. No
push will be performed.
