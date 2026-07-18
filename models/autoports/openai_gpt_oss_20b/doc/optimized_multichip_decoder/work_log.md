# GPT-OSS-20B optimized multichip decoder work log

## 2026-07-18: stage start

- Scope: optimize the completed fixed `1x4` Blackhole path in
  `tt/multichip_decoder.py` in place.  Do not begin full-model or vLLM work.
- Starting HEAD: `b311eae4485`; the unrelated pre-existing edit in
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` is excluded from
  this stage.
- Skills: `$optimize`, `$tt-device-usage`, mandatory `$graph-rewrite` and
  `$shard-advise`; `$autofix` for hard failures and independent `$stage-review`
  before checkpointing.
- Hardware health: `timeout 60 tt-smi -ls --local` found four resettable
  Blackhole P300c devices.  A serialized `FABRIC_1D_RING` `MeshShape(1,4)`
  open/close passed with `MESH_SMOKE_OK`.
- Accepted starting contract: TP4 packed QKV/local paged SDPA/row-parallel O,
  EP4 whole gate-selected experts, two ring all-reduces, replicated BF16
  `[1,1,S,2880]` inter-layer residual, BFP8 sparse expert weights, BF16 KV
  cache, arbitrary public logical prefill lengths, and 131072-token context.

## Starting operation-topology audit

| Boundary | Current measured sequence | Material repetition/movement | Candidate family and constraints | Initial action |
| --- | --- | --- | --- | --- |
| decode input/QKV | replicated BF16 DRAM -> 10-core sharded RMSNorm -> width reshard -> one packed `2880x1280` QKV linear -> interleaved head creation | one reshard; BF16 weight traffic; QKV row is advised for DRAM sharding | DRAM-sharded local QKV crossed with BF16/BFP8/BFP4 and HiFi4/HiFi2/LoFi; preserve packed QKV and head ordering | measure precision-locked DRAM geometry candidates |
| cache/attention | device-indexed RoPE -> paged K/V update -> explicit-mask paged SDPA -> concat heads | explicit mask build and cache traffic; no host boundary | BFP8 KV, explicit/default SDPA program, activation dtype; retain non-aligned logical contract | rerun reduced-cache and explicit-program candidates on final topology |
| attention output | local `1024x2880` O linear -> L1 conversion -> ring all-reduce -> replicated residual | one material BF16 collective per token | local-MM+AR; MM+RS with fractured residual; fused MM+RS; fused AG+local-output MM; BFP8 CCL; persistent buffers | compare as coherent residual/CCL families through the next norm/router/expert consumer |
| post-attention/router | replicated residual -> 10-core sharded RMSNorm -> DRAM restore -> FP32 router matmul -> top-k/softmax/scatter | sharded-to-interleaved transition; router matmul is one-core in the starting profile | keep fractured residual through distributed norm/router or tune replicated router/L1 placement | measure stack-compatible family and router candidates |
| EP4 gate/up | full hidden replicated to every rank -> mesh-partition routes -> separate same-input sparse gate and up matmuls -> separate biases | repeated same-input sparse matmul and broadcast work | packed sparse gate+up weight and output split; exact GPT-OSS SwiGLU semantics required | implement as the first graph-rewrite A/B and verify real-weight PCC/perf |
| SwiGLU | clamp gate/up -> scale -> sigmoid -> multiply -> add-one -> multiply | primitive elementwise chain | generic `ttnn.swiglu` is not equivalent (no GPT-OSS clamp/alpha/up+1); fused `moe_compute` implements exact form but requires packed dispatch contract | test packed projections first; adapt the exact fused-MoE family rather than substituting the non-equivalent generic op |
| EP4 down/output | sparse down -> bias -> route weighting -> expert sum -> BFP8 ring all-reduce -> replicated residual | one material expert collective; DRAM/intermediate movement in prefill | L1 decode intermediates, geometry/dtype cross-product, persistent/async CCL, fused MoE compute/combine | profile and sweep as a family; preserve active-expert execution |
| inter-layer boundary | replicated BF16 output accepted directly by next decoder | no gather/reshard between layers; collectives remain inside the layer | fractured residual may remove an AR only if EP whole-expert input and the next layer consume it without immediate restoration | keep only a whole-stack-compatible winner; document the final contract |

Starting evidence is the committed completed-stage default: warmed S=128 prefill
`22.6186 ms/layer`, traced warmed decode `0.598719 ms/layer`, and PCC above
0.9978 prefill / 0.9990 final decode for both meaningful layer kinds.  A fresh
same-checkout baseline run is collected before edits below.

## Fresh baseline

The stage-local baseline used the completed default with 20 warmed S=128
prefill repeats and 500 warmed trace replays:

```bash
RUN_MULTICHIP_DECODER_PERF=1 \
MULTICHIP_DECODER_PERF_SEQ=128 \
MULTICHIP_DECODER_PREFILL_REPEATS=20 \
MULTICHIP_DECODER_TRACE_REPLAYS=500 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/baseline_perf_seq128.json \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf -s
```

Result: 22.6480578 ms warmed prefill and 0.5986421 ms traced warmed
decode.  Real-weight EP4 PCC passed for sliding and full attention in
`baseline_pcc_ep.junit.xml`.

All performance candidates below used the same command shape, changed only by
the named `MULTICHIP_*` environment control, and wrote a candidate-specific
JSON/JUnit file.  Search runs used at least 15/400 samples for prospective
winners and 20/500 for the final reproduction.

## Graph rewrite and sharding advice

- `$graph-rewrite` first targeted the two same-input sparse matmuls.  Added an
  off-by-default packed gate/up weight/bias representation, a single wider
  sparse matmul, device split, and the existing exact clamped GPT-OSS SwiGLU.
  Both layer kinds passed PCC in `graph_rewrite_packed_gate_up_pcc.junit.xml`.
  `MULTICHIP_EP_GATE_UP=packed` measured 22.07848/0.64053 ms with subblock 2
  and 25.02510/0.67932 ms with subblock 1.  The 2.5% prefill gain did not
  compensate for the 7.0% decode regression, so separate projections remain
  default and the correct rewrite remains available as a diagnostic.
- `$shard-advise` captured a dense shape-only representation of the exact
  per-rank local operations (QKV 2880x1280, O 1024x2880, router, packed
  experts, and down).  `shard_advise/advise_multichip_gpt_oss.py` emitted
  `report.json`, `report.txt`, `final_ir.mlir`, pipeline log, and decision
  trace.  The report had zero spills.  Its nominal 23-core QKV input did not
  legally divide K=90 tiles, so the closest legal adapted seed was 30 cores:
  `MULTICHIP_QKV_AB=30,3,1,1` measured 22.62038/0.60963 ms and was rejected.

## Coherent topology and collective families

### Residual layout

`RUN_MULTICHIP_TOPOLOGY_PROBE=1` compared the old contract through the real
next EP consumer, rather than restoring immediately after reduce-scatter:

- replicated all-reduce -> replicated RMSNorm/router: 0.593460 ms;
- padded width-sharded RS -> distributed RMSNorm -> row-sharded router plus
  32-logit all-reduce -> full-hidden gather required by each rank's whole
  active experts: 1.294307 ms.

The candidate norm/router PCC was 0.999993/0.999998.  It was 2.181x slower in
`residual_topology_candidate.json`; therefore the replicated family remains.
No collective or reshard is present between decoder layers.

### Async CCL, persistent buffers, and placement

`MULTICHIP_DECODE_COLLECTIVE_AB=rs_ag_pad64` used CCLManager's persistent
semaphores and preallocated buffers.  It owns H=2880 -> 2944 padding, async
ring reduce-scatter, async all-gather, and slicing.  Both real layer kinds
passed PCC in `collective_rs_ag_pad64_pcc.junit.xml`; the family measured
20.58585/0.53381 ms in `collective_rs_ag_pad64_perf_seq128.json`.  It loses to
the final 0.50322 ms decode, so the two production all-reduces stay placed at
the O projection and EP output boundaries.

### Fused matmul + CCL

The first exact-shape probe used M=32/128, K=1024, padded N=2944, TP4 fused
matmul+RS, Ring topology, and one link:

```bash
RUN_MULTICHIP_FUSED_MM_RS=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_fused_o_projection_reduce_scatter_shape_candidate -s
```

Both shapes passed at about 0.999987 RS PCC.  The first real-path attempt then
failed because prefill supplied rank-3 input.  The candidate was not rejected:
the decoder added owned 4D reshape plus H padding and retried.  Sliding EP PCC
passed, then `MULTICHIP_FUSED_O_RS=1` measured 20.71160/0.65654 ms.  The fused
matmul+RS plus necessary trailing gather is correct but slower; artifacts are
`fused_mm_rs_exact_shape_first.junit.xml`, `fused_o_rs_pcc_ep_first.junit.xml`,
`fused_o_rs_pcc_ep_retry1.junit.xml`, and `fused_o_rs_perf_seq128.json`.

This initial audit conclusion was incomplete: attended activations are already
head-fractured before O, so fused attended-AG + local-output O is applicable;
and a fractured EP residual can feed the next packed QKV.  Both were added and
measured during the independent-review remediation recorded below.  This old
conclusion is retained here only as provenance for the review finding.

## Expert topology, geometry, precision, and fidelity

| Candidate | Prefill / decode ms | PCC or disposition |
| --- | ---: | --- |
| BFP8 gate/up/down, original BFP8 activation | approximately 22.65 / 0.599 | starting default |
| BFP4 gate/up/down | 19.4903 / 0.57702 | fails prefill: sliding 0.97895, full 0.97741 |
| BFP4 gate/up only | not promoted | fails prefill: 0.98278 / 0.98133 |
| BFP4 down only, original geometry | 21.5293 / 0.59116 | passes both kinds at approximately 0.994-0.995 |
| all BF16 expert weights | 29.8030 / 0.57965 | decode improves but prefill regresses materially |
| mixed BFP8 gate/up + BFP4 down, BF16 activation, selected geometry | 20.59978 / 0.50348 search; 20.58625 / 0.50322 final | selected; all PCC >= 0.99417 |
| same mixed weights, BFP8 activation/CCL | approximately 21.59 / 0.5853 | reject both paths |
| global HiFi2 dense control (sparse remained implicit LoFi) | 21.5635 / 0.58131 | sliding position 130 near-tied route flips; router PCC 0.81211 |
| global LoFi dense control (sparse remained implicit LoFi) | 21.6022 / 0.63194 | slower |

Decode sparse geometry was crossed as complete gate/up/down families: 90
cores, 45 cores/subblock 2/in0 45, 30 cores/subblock 3/in0 90, and 45
cores/in0 90.  The corresponding decode timings were 0.59116, 0.58531,
0.59720, and 0.58940 ms before the BF16-activation winner.  Prefill in0 90
measured 21.7429 ms, forced 45 cores measured 21.6377 ms, and the retained
90-core/in0 45 family won.  Final defaults use 45-core decode and 90-core
prefill programs.

The production EP path continues to infer rank-local `nnz` dynamically and
execute only gate-selected experts.  Exact-top-4 tests cover non-aligned S=17.

### Fused MoE retry and AutoFix

The generic fused `moe_compute` candidate was shape-adapted rather than
rejected at the API boundary:

```bash
RUN_MULTICHIP_FUSED_MOE=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_fused_moe_compute_ep_local_candidate -s
```

The single-card compute-only H=I=2880, eight-local-expert, top-4, bias, exact
SwiGLU probe passes (`fused_moe_ep_local_retry.junit.xml`).  `$autofix` then
verified that slot 4 aliases a two-expert ping-pong buffer and the upstream
validator checks only terminal experts 6/7.  Arbitrary local expert 0-7
outputs cannot survive.  Eight slots would require 1,474,560 bytes/core before
other CBs.  Full 1x4 mode can consume each result, but a replicated batch-1
decode token must be duplicated into four source tokens (4x selected-route
work), and the kernel fixes expert weights to BFP4, already below the stage's
real PCC gate.  Older `moe_gpt` and the DeepSeek unified op have incompatible
expert/core and SwiGLU/bias contracts.  `AUTOFIX_FUSED_MOE.md` records source
lines, predictions, and refutations.  No dense all-expert fallback was added.

## Attention, activation sharding, and KV family

The local QKV and O weights were converted to exact DRAM-sharded layouts and
paired with DRAM-sharded activation/program configs.  Bias was applied after
the DRAM matmul and the family restored L1 only at the collective boundary.
Results:

| Attention weight dtype | Prefill / decode ms | Decision |
| --- | ---: | --- |
| BF16 | 22.6946 / 0.72015 | reject |
| BFP8 | 22.6681 / 0.66808 | reject |
| BFP4 | 22.7420 / 0.66790 | reject |

This directly tries the profiler's DRAM-sharding advice.  Weight traffic fell,
but activation reshards dominate.  QKV remains the original packed BF16
projection with 10-core input/20-core output sharding and the O output remains
90-core width-sharded before collective.

For KV BFP8, the first prefill failed the paged-fill dtype contract.  Prefill
K/V was explicitly cast and retried.  Decode paged update then exposed its
BF16/FP32-only input contract and internal repack behavior; decode input stayed
BF16 and the complete family was retried again.  It passed and measured
21.5713/0.63686 ms, so BF16 KV remains.  The three attempts are retained as
`selected_kv_bfp8_perf_seq128*.junit.xml` and the successful JSON.  No cache
dtype/layout or context reduction was accepted.

## Final profiler

Correct Tracy invocation (the earlier invocation containing a standalone
`--` selected no tests and is not provenance):

```bash
RUN_MULTICHIP_DECODER_PERF=1 \
MULTICHIP_DECODER_PERF_SEQ=128 \
MULTICHIP_DECODER_PREFILL_REPEATS=1 \
MULTICHIP_DECODER_TRACE_REPLAYS=3 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/profile_final_perf_seq128.json \
python -m tracy -r -p -v -o /tmp/gpt_oss_optimized_multichip_final \
  -n optimized_multichip_final -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf
```

The raw v2.1 multi-device ops CSV was copied to `perf/final/ops.csv`.  Reports:

```bash
tt-perf-report models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/perf/final/ops.csv \
  --start-signpost PERF_PREFILL_MULTICHIP_EP \
  --end-signpost PERF_PREFILL_MULTICHIP_EP_END \
  --active-experts 4 \
  --csv models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/perf/final/prefill_report.csv \
  --summary-file models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/perf/final/prefill_summary.csv \
  --no-color

tt-perf-report models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/perf/final/ops.csv \
  --start-signpost PERF_DECODE_MULTICHIP_EP \
  --end-signpost PERF_DECODE_MULTICHIP_EP_END \
  --active-experts 4 \
  --csv models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/perf/final/decode_report.csv \
  --summary-file models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/perf/final/decode_summary.csv \
  --no-color
```

Those numbers were the pre-review final capture.  The current-source refresh
below supersedes them while retaining this paragraph as provenance.

## Final gates

Final wall timing command used 20/500 and produced
`final_default_perf_seq128.json`:

```bash
RUN_MULTICHIP_DECODER_PERF=1 \
MULTICHIP_DECODER_PERF_SEQ=128 \
MULTICHIP_DECODER_PREFILL_REPEATS=20 \
MULTICHIP_DECODER_TRACE_REPLAYS=500 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/final_default_perf_seq128.json \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/final_default_perf_seq128.junit.xml -s
```

The pre-review reproduction was 20.5862457 ms warmed prefill and 0.5032223 ms
traced warmed decode.  The post-review current-source final reproduction below
supersedes it.

Final real-weight PCC was rerun with current defaults and plain-text logging:

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py \
  -k 'test_real_weight_prefill_decode_matches_single_chip_optimized and ep' \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/final_pcc_ep.junit.xml -s
```

Sliding prefill PCC is 0.995640 and its three decode PCCs are
0.994766/0.994720/0.995832.  Full prefill PCC is 0.994173 and decode is
0.995403/0.994589/0.994415.  Attention and routing are >=0.99990 and
>=0.99997, respectively.

The consolidated default suite selected nine tests and passed in 67.88 s:
runtime fallback/config audits, both real layer kinds, both trace paths,
active-expert routing, non-aligned S=17/33, local paged KV, trace recapture,
and full-context position 131071.  Artifact: `final_default_suite.junit.xml`.

Watcher command:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py \
  -k 'test_real_weight_prefill_decode_matches_single_chip_optimized and ep' -s
```

Both layer kinds passed; device-side `watcher.log` records clean attach/check/
detach on devices 0-3.  `final_tt_smi_post_gates.log` records four resettable
P300c cards after the final timing and PCC gates.
Full Ethernet watcher cannot fit the machine's ACTIVE_ETH firmware buffer
(27920 bytes generated versus 25600 available); worker/Tensix watcher is clean.

`doc/context_contract.json` now identifies this stage, the mixed expert policy,
the unchanged replicated boundary, final CCL decisions, 12.50 GiB/device
capacity estimate, and physical validation of the 131072-token contract.
Public non-aligned logical inputs remain supported.

Static final checks: `python -m compileall` passes, both JSON documents parse,
and `git diff --check` is clean.  Independent `$stage-review` and local
checkpoint SHAs are appended below after the review gate.

## Independent-review remediation and current-source final refresh

The first independent `$stage-review` returned `more-work-needed` and wrote
`stage_review.md`.  Its three blockers were: the implicit sparse LoFi runtime
contradicted the HiFi4 prose; fused attended-AG/local-O and a fractured
EP-output-to-real-QKV family had not been measured coherently; and the roofline,
device time, and wall time were not from one profiled run.  `$autofix` reports
`AUTOFIX_SPARSE_FIDELITY.md` and `AUTOFIX_MISSING_TOPOLOGIES.md` record the
implementation and retry history.

### Sparse fidelity and final attention/CCL precision

Sparse fidelity is now an explicit, isolated control at all four sparse call
sites.  The selected default is LoFi with exact math, BF16 destination, and L1
packer accumulation.  Both higher-fidelity retries preserved both meaningful
layer kinds but were slower:

| Sparse control | Prefill / traced decode ms | Correctness |
| --- | ---: | --- |
| LoFi current default | 20.583611 / 0.503275 (20/500 final) | both kinds pass |
| HiFi2 | 20.689783 / 0.504293 (15/400) | both kinds pass |
| HiFi4 | 20.984613 / 0.509997 (15/400) | both kinds pass |

The final interleaved packed-QKV/local-O topology was materialized directly at
BFP8 and BFP4 rather than relying on the rejected DRAM-sharded family.  BFP8
passed sliding attention but failed full-attention prefill at 0.985071.  BFP4
failed sliding/full prefill at 0.858267/0.843126.  BF16 remains selected.
Artifacts are `sparse_hifi{2,4}_*` and `interleaved_attention_bfp{8,4}_*`.

BFP8 activation/collective dtype was also crossed with the lower-movement
persistent padded RS+AG topology.  Both layer kinds passed, but
`rs_ag_bfp8_ccl_perf_seq128.json` measured 21.563820/0.627380 ms versus the
BF16 default, so it was rejected with complete-family evidence.

### Fused AG+local O and carried inter-layer residual

The new off-default `MULTICHIP_FUSED_O_AG` path uses
`all_gather_minimal_matmul_async` from local attended K=1024 to rank-local O.
The first padded-N=736 full-layer run failed PCC; the isolated diagnostic
localized this to rank 3 (0.982617) while ranks 0-2 were 0.999997.  Retrying the
smallest legal natural N=720 shape passed every rank at 0.999997 and both full
layer kinds.  BF16 then measured 20.771860/0.642176 ms and was rejected.  BFP8
natural720 passed the isolated kernel at 0.999937 but failed full-attention
prefill at 0.989274, so it was rejected before timing.

The exact lower-movement boundary probe keeps the real active-expert EP partial
fractured through pad2944, persistent minimal reduce-scatter, a local736
residual, distributed RMSNorm with logical-width stats correction, and
persistent fused all-gather plus a real packed QKV weight.  The retained fixture
contains only layer 12, so that real layer's weights model the downstream
consumer of the preceding output; the geometry and operation contract are the
same for every decoder layer.  The first run exposed the absent-layer-13
fixture lookup and the next exposed an invalid diagnostic pad on a width-shard.
Both were adapted without changing candidate math/CCL, then retried:

| Payload | Minimum residual / norm / QKV PCC | Replicated / fractured trace ms | Decision |
| --- | ---: | ---: | --- |
| BF16 | 0.999996 / 0.999993 / 0.999993 | 0.452406 / 0.539213 | reject, 19.19% slower |
| BFP8 | 0.999898 / 0.999878 / 0.999859 | 0.452890 / 0.526221 | reject, 16.19% slower |

Artifacts are `fused_o_ag_*`, `fractured_ep_to_qkv_bfloat16.json`,
`fractured_ep_to_qkv_bfloat8_b.json`, and their JUnit retry provenance.  These
measurements close the coherent residual, persistent-buffer, CCL-dtype, and
fused matmul-CCL families without immediately restoring the old residual.

### Current default, profiler accounting, and final gates

The post-remediation default was rerun with the same 20/500 command above.
`final_default_perf_seq128.json` now records 20.583611284 ms prefill and
0.503275194 ms traced decode: 9.115% and 15.931% faster than the fresh baseline.
Its SHA256 is
`d01c2883c039800d6a6fd55e5f0dae8734481e29974b183e16fedd552529f00a`.
`final_current_pcc_ep.junit.xml` passes sliding/full prefill and decode with the
same component values in the final suite.

The final Tracy command was rerun from current defaults, with no candidate
environment variables:

```bash
RUN_MULTICHIP_DECODER_PERF=1 \
MULTICHIP_DECODER_PERF_SEQ=128 \
MULTICHIP_DECODER_PREFILL_REPEATS=1 \
MULTICHIP_DECODER_TRACE_REPLAYS=3 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/profile_final_current_perf_seq128.json \
python -m tracy -r -p -v -o /tmp/gpt_oss_optimized_multichip_final_current \
  -n optimized_multichip_final_current -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf
```

The raw CSV was copied byte-for-byte to `perf/final/ops.csv` (SHA256
`7842a736b3def283deb524c374a7f46dae5f4d4438a94ce2af63e72037e65668`)
and both documented `tt-perf-report` commands were rerun.  Prefill contains
21.765528 ms of device ops; explicit LoFi sparse rows total 11.90653 ms.  The
three-replay decode range contains 1.473187 ms of device ops (0.491062 ms per
replay), while the same profiled wall run is 0.574001 ms/replay.  The named
0.082939 ms gap and exact 0.070018 ms compulsory-read roofline are recorded in
`perf/final/perf_accounting.json`; the 500-replay 0.503275 ms headline remains
separately identified rather than substituted into the same-run accounting.

The refreshed consolidated command selected exactly nine tests and passed in
66.29 s.  It includes fallback/parser audits, exact active top-4 routing,
non-aligned S=17/33, both layer kinds, traced mutable positions/page recapture,
and full context position 131071.  `final_default_suite.junit.xml` is the
artifact.  Worker/Tensix watcher was rerun for both real layer kinds and is
clean in `final_watcher_ep.junit.xml` and `final_watcher_device.log`; the latter
contains 2243 watcher lines with no error/assert/hang/failure match.  The final
`tt-smi -f .../final_tt_smi_post_gates.log` snapshot shows four healthy P300c
boards after all device gates.

The fresh independent rereview overwrote `stage_review.md` with
`Verdict: clean-pass`.  It found no remaining required-work blocker after
checking the current default, first-review remediation, topology/fidelity
matrix, profiler/accounting provenance, stress/fallback/watcher evidence, and
scope.  No hardware or implementation was changed by the reviewer.
