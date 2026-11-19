# Full-model mesh and topology plan

Stage7 mesh decision and evidence map. All20 acceptance jobs and the final
collector pass (`hard_gates.json`); independent review returned clean-pass
(`stage_review_round1.md`).
Source lineage, including the standalone-loader import-only repair, is explicit
in `import_resolution_proof.json`. No vLLM integration or datatype frontier is
part of this stage. Historical evidence remains identified below.

## Target and unchanged policy

The constructor in `../../tt/model.py` requires Blackhole TP4, mesh 1x4. The
64 distinct layers comprise 48 GDN and 16 full-attention layers. Axis 1 owns
tensor parallelism; `_BlackholeCCLManager` selects Linear collectives. Preserve
the matching fabric setup used by the existing bounded test runners.

`../../tt/multichip_decoder.py::_STAGE5_DEFAULTS` and
`../optimized_multichip_decoder/final29_selected_overrides.json` agree on the
selected overlays. The constructor supplies the remaining defaults: BFP8
attention/MLP weights, BF16 residual/CCL payload, BFP8 paged KV, and hidden-sharded
residuals. GDN attention uses HiFi4, GDN MLP HiFi2, and full-attention projections
LoFi; norm/general arithmetic uses the existing HiFi4 configuration. Narrow GDN
correction remains BFP8 with its existing FP32 combine behavior. Persistent GDN
state is BF16; bounded prefill holds recurrence in FP32 until publication.
Embeddings, LM head and logits remain BF16; final norm weights remain FP32 and
the head uses `model.io.ckc` (HiFi4/FP32 destination). No policy is changed here.

Do not undo the GDN HiFi4 repair: the rejected inherited LoFi contract has
actual-HF contribution PCC 0.9860137708810336 at S31, documented by
`../optimized_multichip_decoder/final_validation28_contract_l0_tp4.json`,
`final28_contract_stage4_l0.json`, `AUTOFIX_contract29.md` and the passing final29
contract. BFP4 KV failed actual-text/cache checks; faster BFP4 weight policies
remain datatype-stage candidates, not this stage's defaults. Minimal distributed
norm remains off because its 64-owner L1 resource test failed. The rejection and
untried ledgers remain authoritative in
`../optimized_multichip_decoder/final29_candidate_decisions.md` and
`final29_handoff.md`.

## Full-path tensor and program map

Here H=5120, vocabulary V=248320, physical decode rows T=32, logical batch B<=32.
Padding is internal. The existing nonaligned prefill path remains supported.

| Boundary | Global / logical tensor | Per-rank storage or work | Program/layout contract |
| --- | --- | --- | --- |
| Token embedding | `[V,H]`, token IDs replicated | BF16 row-major `[248320,1280]`; output `[B,S,1280]` | Embedding shards hidden dimension; no entry collective |
| Residual stack | H-wide residual | Prefill `[B,S,1280]`; compact decode `[1,B,1280]` | BF16 hidden shard passes directly between layers |
| Full attention | 24 Q heads, 4 KV heads, head256 | 6 Q heads and 1 KV head/rank | Packed QKVG; local SDPA; output projection row-parallel then RS |
| GDN projections/state | Packed QKV/Z/BA | Packed primary local width4160, narrow correction2624; recurrent `[B,12,128,128]`, conv `[B,3,2560]` | Accepted packed correction, local convolution/recurrence and row-parallel output |
| MLP | Gate/up output split, down input split | TP4 intermediate ownership | Packed gate/up and selected SwiGLU; row-parallel down restores hidden shard |
| Decode matmuls | Role-specific K,N | N padded to `32*lcm(8*readers,cores)`; DRAM width-sharded over8 banks | Defaults cores8/readers2/block2; input shard `[32,K/8]`; per-core N from padded width. Historical geometry evidence below |
| Prefill matmuls/RS | B*S logical rows | Internal alignment and chunks<=512 | Fused MMRS for selected `256 < rows <= 32768` range, chunk_rows16384; separate small/decode path |
| Paged attention state | Logical page tables `[B,rounded_context/32]` | Local K,V `[physical_pages,1,32,256]`, BFP8 | Page32; immutable backing storage and changed-only decode page updates |
| RoPE/position | Absolute request positions | Shared per-model cos/sin tables on each rank; persistent decode indices | Device increment in decode; eager chunk inputs upload outside capture |
| Final norm | Hidden shard1280 -> normalized5120 | Decode gathers hidden then existing local norm; B1 prefill uses distributed stats then gathers normalized hidden | Terminal-only hidden gather; no replicated inter-layer stream |
| LM head | `[H,V]` | BF16 `[5120,62080]`, vocabulary-sharded | Existing interleaved matmul, HiFi4; sampled columns retain rank offsets |
| Sampler-ready logits | Logical `[B,62080]` per rank | Persistent `[1,1,32,62080]`; sampler pads local width to65536 with invalid columns masked | Local physical top32, semantic greedy k1/p0/temp1; same path supports k/p sampling |
| Sampling exchange | 32 candidates/rank/row | Values BF16, local indices UINT16 at this width; gathered128 candidates/row | Two small candidate gathers; generator-owned sample trace writes `tt_out_tok` |
| Generator boundary | Token/position/page/cache state | Persistent inputs and sampler logits allocated before captures | Resident window removes per-step reads; P128 trace retains state ownership. `performance_final.json` and strict `full_trace_tracking.json` pass |

The decoder's selected matmul geometry and dtype are confirmed by prior final29
profile artifacts, with their known tool annotation limitation: tt-perf-report
assumes8 workers although the selected two-reader configuration uses16. Do not
use those Cores/FLOPs percentages as measured utilization. See
`../optimized_multichip_decoder/final29_profiler_limitations.md`.

## Terminal topology alternatives before knob tuning

Byte figures are payload models, not measured link traffic. For one physical
32-row BF16 tile, a hidden shard is81920 bytes and gathered H is327680 bytes.
Each rank needs245760 bytes from peers for that hidden AG. Routing, retransmit,
semaphores and padding beyond the described tile are excluded. At T32 the two
candidate tensors total4096 local bytes (values BF16 + UINT16 indices), requiring
12288 peer bytes/rank for both AGs. A full-vocab BF16 AG would instead require
11919360 peer bytes/rank. Logical B1 arithmetic is smaller, but these operators
carry tile-padded rows. Profiler `sampling.csv` confirms the UINT16 index CCL.

| Family | Residual before/after; next consumer | Collective payload and persistent plan | Evidence / disposition |
| --- | --- | --- | --- |
| Current terminal decode AG -> norm -> vocab head -> candidate AG | Sharded residual -> terminal replicated normalized H -> vocab-sharded logits | Hidden AG245760 peer bytes/rank plus candidate AG12288. Existing per-signature persistent CCL buffers; sample adapter clones borrowed AG results because common sampler deallocates owned outputs | Retained; refreshed `profile.json`, `perf/analysis.md`, runtime dtype rows and `performance_final.json` |
| Distributed final norm -> AG -> same head | Sharded normalized H, then head consumes gathered H | Same hidden AG plus distributed stats exchange; unchanged BF16 CCL. Allocate norm/stat and AG storage before capture | Already used by B1 prefill; decode alternative untried stop-rule handoff. No empirical rejection |
| Distributed norm -> fused AG-matmul | Sharded normalized H -> vocab logits | Retains hidden exchange volume, could remove materialized gather/dispatch. Requires supported BF16/H5120/N62080 packing and preallocated fused outputs | Untried terminal stop-rule handoff. Decoder module analogues do not prove terminal API/layout compatibility |
| Hidden-sharded head partials -> RS vocab logits | Sharded residual remains local to head; vocab-sharded logits after RS | Requires local weight `[1280,248320]`, same element count as current head, but reduction over full-vocab partial logits: nominal11919360 bytes/rank at T32; persistent RS scratch/output | Untried; much larger communication model than hidden AG. No empirical rejection claimed |
| Local head partials -> all-reduce/full replication | Sharded hidden -> replicated full-vocab logits | Full-vocab reduction/all-gather; more data than candidate exchange, plus full output storage | Untried comparison, not selected; no reason to introduce full-vocab exchange when current sampler needs only candidates |
| Current terminal + DRAM-sharded head | Same hidden and logits ownership | No new network collective; padded per-rank head weights, input/output shard conversion and local concat counted | Actual same-policy split2_r2_k2 and whole_r3_k1 lose complete-terminal timing; whole_r2_k1 failed L1 then adaptations ran. Not a rejection of every possible geometry |
| Force-argmax sampler | Vocab logits -> global greedy token | Common force-argmax route; compare semantically identical greedy and retain device feedback | Fresh serial `sampler_split.json`0.512781ms vs `sampler_argmax.json`2.725168ms,200 iterations; unique/tied maxima pass. Retain split |
| No-readback resident split replay | Same model/sample graphs and persistent state | No per-token host transfer; nonblocking queue submissions, one end synchronization/read boundary | Contract fix retained; `performance_final.json`32.258325ms/step,30.999750t/s/u, zero per-token refresh/readback; seeded greedy/stochastic endpoint and eager/trace parity pass |
| P128 first-chunk trace | Same sharded stack and terminal, persistent existing bounded-prefill inputs | No topology/dtype change; additional trace storage, no new model weights or cache | Retain; same-process median eager86.855779 -> trace86.650346ms (0.2365%). Strict tracking passes. Other lengths/B>1/resumes retain eager prepared path |
| Cached identical sampler parameters | Same physical k/p/temperature tensors; RNG reset remains explicit | Skips identical parameter uploads; slot remaps invalidate signature | Retain for deterministic upload reduction; false86.730679 -> true86.746810ms TTFT (0.0186% slower median, no measured speedup) |

Current persistent CCL buffers are owned per decoder/signature and consumed on
one command queue before reuse. `SamplingCCL.line_all_gather` returns clones so
the common sampler cannot free decoder-owned storage. No broad trace allocation
exemptions are introduced. `full_trace_tracking.json` passes all64 ordering with
tracking1/skip-program-cache0; `trace_contract.json` passes the broader reduced
request/changed-page battery and retains four captures/program count. Separate
`trace_positive_control.log` proves tracker activity. `runtime_tracking.json`
and `runtime_audit.md` record executed host-boundary sentinels and their scope.

## Exact inherited performance evidence

- `../full_model/perf_terminal/lm_head_decision.json` and
  `../full_model/lm_head_uninstrumented.json`: current head+sample+read2.203154ms,
  split2/two-reader2.349247ms, whole/three-reader2.689188ms; medians5x100,
  serialized uninstrumented confirmation, same BF16/HiFi4 policy. Exact
  logits/PCC/greedy and stable program/trace checks accompany the trials.
- `../full_model/perf_terminal/{decode,sampling,prefill,logits_only}.csv` and
  `kernel_accounting.json`: head about1.55–1.57ms, sampling device about0.453ms.
  These are reduced layers[0,3], not all64 timings. Inter-window profiler-drain
  gaps are excluded; see `profiling_analysis.md`.
- `../full_model/layer_stack_lower_bound.json`: best observed
  `48*0.509915878 + 16*0.411218298 = 31.055455ms`; median31.074964ms.
  Layer probes use prefix512/repeated weights, unlike P128 full-model decode;
  the difference is not an exact terminal subtraction.
- `../full_model/performance_final.json` and `performance_summary.json`: prior
  all64/P128/G128/C262144 median TTFT86.924ms and token-out32.292ms including
  every-token reads. They are the prior accepted baseline, not the new resident
  benchmark. Fresh `baseline.json` passes: readback-inclusive TTFT87.031086ms,
  30.976371t/s/u. `performance_final.json` final resident path reports
  TTFT86.850814ms,30.999750t/s/u (32.258325ms/step); sampling/token feedback are
  included, per-step reads are excluded. These different readback boundaries
  must not be represented as a same-boundary speedup. Final `readback_final.json` reports86.888559ms TTFT and30.956937t/s/u;
  its repeat ranges overlap baseline. `teacher_forcing.log` reports31.04t/s/u
  with reference updates/readback, separately from resident autoregression.

Context262144 and the shared aggregate KV capacity contract remain unchanged.
`performance_final.json` includes32769 nonaligned prompt tokens,65 internal
chunks, and final position32770; existing four traces,24456 programs and their
stable addresses survive. Its B1/C262144 final memory snapshot records the new
trace configuration. A new max-capacity run is unnecessary because KV dtype and
layout are unchanged. The existing context proof and B32 pool are in
`../context_contract.json`, `../full_model/max_context_final.json` and
`../full_model/slots32_all64_pool_fixed.json`. New reduced B32
`slots32_tracking.json` passes inactive/remap/resident assertions; all64 `slots32.json` also passes complete preparation,128 resident steps
and inactive/remap/release at context1024 (31 active users,16.237918t/s/u). B32 never enables the new B1-only prefill trace.

## Optimization stop and limits

The last two retained optional changes are the P128 trace and cached identical
sampling parameters. Their final same-process median TTFT changes are0.2365%
lower and0.0186% higher, respectively, both below3%. Stop Rule2 therefore bounds
further optional search; it is not evidence of two statistically resolved gains.
Cache retention is justified by eliminating redundant uploads with equal outputs,
not by a latency win. Resident no-readback decode is a required contract fix,
not a claimed large optimization gain. Required hard gates pass; independent review returned clean-pass.
Untried families are enumerated in
`checklist.md` under `Not tried (stop rule)`; none is relabeled as rejected.
