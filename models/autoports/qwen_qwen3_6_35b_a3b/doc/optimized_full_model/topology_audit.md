# Stage 7 full-model topology audit

CPU-only source/artifact inspection on 2026-11-16, starting HEAD `ece17d228671dbc1375e45c6e73f90156707aa30`. This is a candidate/contract audit, not a performance acceptance report. No device was opened by this auditor. Paths below are relative to this directory unless noted. Stage 7 retains the completed stage-6 generator and optimizes the complete TP4/EP4 path on 1x4 Blackhole; no vLLM implementation or broad datatype sweep belongs here.

## Preserved arithmetic and distribution

The selected residual is **replicated BF16**, not hidden-sharded. This is the selected distribution from earlier measured family comparisons, not a stage-7 fallback. All ranks execute their attention/expert partitions. There is no wrapper gather/reduce between decoder layers. The decoder stack has 30 linear-attention and 10 full-attention layers, H=2048, V=248320, E=256, k=8, I=512. Indexed TP4 at B1 has I/TP=128 (four tiles); grouped EP is used for batched work. At B32 the expected selected-expert union is `1-(1-8/256)^32 = 0.637945`; this supports retaining per-token grouped routing rather than union-masked dense work.

| Contract | Selected implementation/policy | Exact inherited evidence |
|---|---|---|
| Residual | B1 BF16 TILE replicated H2048, L1 width shards over 8x4 cores, physical [32,64] per core; B>1 DRAM singleton tuple | `../optimized_multichip_decoder/residual_contract.md`; `../full_model/preserved_policy.md` |
| Full/shared projections | BFP8 weights, selected HiFi2; BF16 activations; full packed QKVG | `../optimized_multichip_decoder/optimization_decisions.md`, precision/geometry rows |
| Sensitive linear projections/router | Ordered BF16 weight scans with FP32 products/recurrence; selected compensated norm and exact SiLU; full router FP32 | `../../tests/multichip_functional.py`; `../../tt/multichip_decoder.py`; `../optimized_multichip_decoder/AUTOFIX_functional.md` |
| Routed experts | BFP8/LoFi active top-8; indexed TP4 B1, grouped EP4 B>1; deterministic grouped-decode rounding | `../full_model/AUTODEBUG_MIXED.md`, `../full_model/AUTOFIX.md`, `../full_model/batched_teacher/result.json` |
| CCL | Selected BF16 with FP32 linear-partial exception, shared serial CollectiveWorkspace | `../optimized_multichip_decoder/residual_contract.md` |
| Cache | BF16 TILE paged KV, page32, head ownership [0,0,1,1]; FP32 recurrent state and BF16 convolution history | `../context_contract.json`; `../../tt/cache_contract.py` |
| Terminal | BF16 hidden-sharded embedding; FP32 offset RMSNorm weight (`1+weight`); untied BFP8 vocabulary head, HiFi2 FP32 accumulation; BF16 logits | `../../tt/model.py::{__init__,embed,terminal}` |
| Sampling | Common SamplingGenerator graph inside generator-owned trace, physical local top32, semantic k=1/p=0/temp=1 for greedy | `../full_model/sampler_comparison.md`; `../../tt/generator.py::_sampling_forward` |

The closed rejection ledger is `../optimized_multichip_decoder/optimization_decisions.md`, `residual_contract.md`, `AUTOFIX.md`, `AUTOFIX_functional.md`, and `native_restore_verification.json`. In particular, `batch_sharded_l{0,3}.json`, `autofix_fused_rs_l3.json`, `repaired_fused_bfp8_l3.json`, `sharded_bfp8_l0.json`, `attention_bfp4_l3.json`, `repair_fp32_ar_{exact,native}_l0.json` and layer34 controls are not accepted policies. A faster rejected native projection/norm/recurrence family must not become the full-model default. BFP8 KV/LoFi screens do not authorize a new precision default here. The user explicitly asks to preserve this policy and rejection ledger.

## Full-model shape, memory and movement table

Shapes are local to one rank unless stated. `P(R)=32*ceil(R/32)`, B is configured capacity, C is configured aggregate context. Payload bytes exclude allocation padding unless stated. For AG, an S-byte source delivers 3S new bytes to each rank (12S aggregate); these are payload formulas, not measured link traffic.

| Boundary | Local physical shape/config | Bytes and interfaces |
|---|---|---|
| Embedding weight | BF16 ROW_MAJOR DRAM [248320,512], hidden-sharded | 254279680 bytes/rank; full replicated weight alternative would be 1017118720 (+762839040) |
| Embed to first layer | BF16 TILE [P(R),512] → AG [P(R),2048]; Linear axis1, two links | Singleton source32768 bytes/rank, 98304 newly received; B singleton gathers in fixed-slot decode. Prefill uses64 rows: source65536, received196608 |
| Residual stack | Selected carrier above, 40 serial layers | No added inter-layer collective; shared workspace must not overlap independent forwards |
| Final norm | Input converted to DRAM, BF16→FP32→RMSNorm→BF16, [1,P(B),2048] | FP32 norm gamma logical8192 bytes/rank (tiled262144). No CCL; output feeds local head |
| LM head | BFP8 [2048,62080]; BF16 [1,P(B),62080], HiFi2/FP32 accumulation; baseline uses automatic linear geometry | 135086080 stored weight bytes/rank (`17/16` bytes/element); local physical32-row output3973120 bytes; no logits gather |
| Stable sampler logits | BF16 TILE [1,1,32,62080] | 3973120 bytes/rank; copied by model trace, consumed by sampling trace |
| Candidate exchange | Values BF16 and local IDs UINT16 [1,1,32,32] each; gather to width128; Linear axis1, one link requested by common sampler | Combined source4096 bytes/rank, received12288, aggregate49152; local IDs widened after AG. Full-vocab physical32-row AG would receive11919360 bytes/rank |
| Token/position/RoPE | Persistent UINT32 tokens [1,1,1,32]; one INT32 position/activity value per slot; shared BF16 RoPE [C,64] each sin/cos | 256*C RoPE bytes/rank; token feedback via `tt_out_tok`; position/RoPE advance inside model trace. Host refresh at actual scheduling boundaries only |
| Cache | Ten pools: each K/V [ceil(C/32),1,32,256] BF16, rounded to attention read extent; per-slot linear state | Full KV10240*C bytes/rank at aligned C; recurrent15728640 + conv368640 bytes per slot/rank across30 layers. Padded page-table aliases do not enlarge logical extent |
| Prefill staging | Three replicated BF16 ROW_MAJOR [ceil(C/32)*32,2048] | `3*ceil(C/32)*32*2048*2`; 3221225472 bytes/rank at C262144; valid row writes hide all padding |
| Generator state | Persistent tokens/logits, changed-only tables, request snapshots, output leases, common sampler params/seeds | Prepared before first capture; output aliases and shared workspace need allocation lifetime review. Counter deltas must separate request setup from resident replay |

Baseline `SamplerCCL.line_all_gather` accepts `buffer_key` but does not use it to bind an explicit persistent output. Captured allocation reuse is not proof that an explicit-buffer candidate is enabled. This is a precise candidate for the sampling/embedding CCL audit, not a diagnosed correctness bug. Preserve preallocation and every live trace's ownership if adding buffers.

## Coherent topology choices to measure or resolve

| Family | Producer → next consumer and residual contract | Movement/config implications | Status at stage entry |
|---|---|---|---|
| Existing hidden-sharded embedding + AG | Hidden512 → replicated H2048 first decoder norm | One entry AG only; two links; serial shared semaphores | Correct baseline; measure explicit-buffer/link and batched embedding alternatives only with same entry contract |
| Replicated embedding weight | Local lookup directly produces replicated H2048 | Removes entry AG, costs +762839040 DRAM bytes/rank at full vocabulary; preserve first norm arithmetic | Unmeasured full-model candidate; evaluate memory census/context, not only op latency |
| Existing vocabulary-parallel head | Replicated terminal norm → local [2048,62080] GEMM → local candidates | No reduction/full-vocab gather; tune DRAM-sharded/head splitting geometry under BFP8/HiFi2 | Highest-priority new matmul role; stage5 geometry does not cover N62080 |
| Row-parallel head | Hidden512 × full vocabulary partial → AR or RS → sampler | Produces very wide partials; AR BF16 physical32-row global logits is far larger than candidate exchange. RS restores vocabulary ownership but adds reduction and changes accumulation | Structural alternative, not measured rejection. Any attempt must keep next sampler-ready vocabulary shards and include the full collective |
| Distributed terminal norm + AG-MM | Hidden-sharded norm stats → gather/fused head | Decoder already returns replicated hidden; introduces terminal fracture then gather unless head topology changes coherently | No inherited proof of win; no permission to change all layer residuals |
| MM+RS / fused MM+RS / residual-sharded decoder families | Next norm consumes true H512 output, not immediate AG to old contract | Existing stage5 coherent families measured through next consumer | Reuse exact rejection scope; do not reopen merely because embedding is sharded |
| Split greedy vs power-of-two local padding vs force-argmax | Same resident head logits → common candidate graph → persistent token input | Semantically greedy k1 for all; physical top32 keeps legal tiles; optional width padding must mask invalid vocab | Prior real-terminal0.4883/0.5218/2.6647 ms. Refresh on selected terminal path; generic sampled k32 is not comparison |
| Prefill chunk/transport | Embed64; linear attention64; full attention1 or32..256; selected MoE chunk; final logical row | Eager host dispatch and DRAM row transport; public starts/lengths may select singleton paths. Changing grouping can alter ordered numerics | Profile material rows and keep chunk sizes/numerics unless matched real-weight evidence permits change; nonaligned tails mandatory |

The prior sampler is ~1.95% of all40 queued decode, not dominant. Reduced profile sampler23.6% is a two-layer attribution and must not be extrapolated as the full40 share. The existing v4 native profile contains real kernel inventories/CSV and is a useful prioritization source; changed paths require fresh profile evidence.

## Two distinct performance budgets

The isolated stage5 layer-stack estimate is `30*0.719504474 + 10*0.340836807 = 24.99350229 ms/token` (40.0104 t/s). The stage6 queued token-out25.081887921 ms differs by0.088385631 ms (0.354%). This is a cross-workload comparison estimate, not an additive timing decomposition: separate controls measure model24.600408906 and sampling0.487787433 ms. No >10–15% wrapper gap is established by these historical numbers. Recompute against the fresh same-regime stage7 before/after runs and named terminal costs.

The **FUNCTIONAL 10% useful-byte target is different**, and remains unmet by those historical numbers. Reusing the exact stage5 `tests/multichip_functional.py::byte_model` with30 linear +10 full layers and positions128..255 gives972432320 mean useful bytes/rank/token. Add LM head `2048*62080*17/16=135086080`, embedding one useful local row `512*2=1024`, and norm weight `2048*4=8192`: total1107527616 bytes/rank/token. At512 GB/s/rank, ideal2.163139875 ms; 10% ceiling **21.63139875 ms/token**, minimum **46.22909556 t/s/u**. For the existing127-step128-in/128-out convention (positions128..254), ceiling21.63129875 ms and minimum46.22930928 t/s/u. This model excludes intermediates, CCL, physical padding and rereads consistently with stage5; do not add sampler padding traffic merely to raise the target denominator.

Historical25.081887921 ms achieves8.6243% of that useful-byte roofline, and needs13.76% lower latency to meet10%. The inherited measured layer-stack alone is slower than the full-model10% ceiling. Terminal-only work cannot plausibly close that gap while leaving the layer stack unchanged. The stage5 owner decision (`../optimized_multichip_decoder/owner_decision.md`) accepts only that stage's linear0.7195 ms/7.2% result and explicitly grants no later-stage exemption. Stage6 acceptance does not waive this new stage7 floor. If the fresh baseline also misses, preserve policy and investigate a compatible decoder/kernel repair through AutoDebug/AutoFix; do not claim acceptance merely because the wrapper gap is small or the optimization Stop Rule fires.

Measured maximum-context capacity remains C262144, aggregate pool and up to32 fixed slots, not32 simultaneous262K requests. Stage6 census is25372641280 allocated and8606073856 free bytes/rank; it includes live prepared state and must not have trace reservation added again. A replicated embedding candidate can fit the arithmetic headroom but still needs measured census and the required32K–64K smoke. No max-context rerun unless KV dtype/layout changes.
