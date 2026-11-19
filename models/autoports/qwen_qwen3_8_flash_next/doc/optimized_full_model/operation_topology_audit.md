# Full-path operation topology audit

Current acceptance disposition (2026-11-16): the [owner decision](owner_decision_2026_11_16.json) accepts the measured decode speed as a run-specific performance limitation and prohibits further tuning for this floor. Numerical floor failure and round-1 findings below are historical evidence; this amendment supersedes their acceptance-blocking disposition only. Independent [round-2 review](stage_review_round2.md) returned `clean-pass`; stage 7 is complete under the amended owner contract. Checkpoint sources are `checkpoint_sources_v5.json`; measured sources remain v4.

The selected mesh is four Blackhole ASICs, serialized 1x4, with TP4 attention,
EP4 routed experts and one complete 2560-feature hyperconnection stream per
rank. This audit starts from stage6 `de432e04` and distinguishes inherited
measurements, current source, source-derived bytes and final matched-source profiles.
Current selection includes the native causal scorer, head20/3/2, exact residual
write/local rank sum, RHS hyperconnection norms and R16 Q1 GR-up/GDN-output.
It preserves precision, residual ownership and synchronous CCL. Sampler
async-barrier gathers were measured slower and rejected. Native GR compaction
was rejected by integrated mixed-prefill comparison and its dispatch reverted.

[Final resident v1](final_resident_v1.json) measures the selected exact-fusion
bundle at103.064960ms/token,9.702619t/s/u and4705.950625ms TTFT; its caller-read
control preserves all128 baseline tokens. The public decode_forward resident
boundary is103.034054ms,9.705529t/s/u; caller-read decode is103.173001ms.
These boundaries remain separate from tracked teacher-forcing readiness. [Restored-source serving Watcher v2](final_serving_watcher_v2.json)
passes all16 reduced serving/state checks. Current [full48 validation v2](final_validation_v2.json)
completed PASS at14:24:33UTC: AIME prefill/teacher forcing both top5/top100=1,
nonaligned resume,32769-token context, autoregression and full-body runtime
audit. The [fresh qualitative verdict](final_validation_v2/qualitative/verdict.json)
passes its matched-control margin. [All48 B32 HF/control v1](final_batch32_hf_v1.json) now passes its C512
correlated-prompt scope. [Current sources v4](final_sources_v4.json) adds only
the scorer standard-tile rejection after these runs; supported kernels/math
are unchanged. Final resident, [component traces](component_trace_v2.json) and
[decode attribution](terminal_profile_analysis_v2.json) match all72 model and3
shared hashes in v4. [Full48 B2 mixed/resumed prefill](final_prefill_signatures_v1.json)
passes all checks on v4. [Prefill-profile analysis](prefill_profile_analysis_v1.md)
and semantic review are complete; B32 clean throughput passes; independent review confirms the failed mandatory floor is the only Required Work item; earlier v1 profile is historical. The final
strict6.223772% FUNCTIONAL fraction still fails10%. The owner confirmed the4h
wall-cap Stop Rule at15:09:21UTC; this does not waive the floor or required
gates. No overall stage pass is claimed.

## Tensor, program and communication contract

Shapes below are **per rank**, unless marked global. `T` is the logical number
of rows; tiled operations physically use `P=32*ceil(T/32)` rows. Decode B1 has
`T=1, P=32`. `C=262144` is allocated capacity, independent of live position.
Payload bytes are uncompressed tensor bytes, excluding fabric headers,
semaphores, packet padding and local copies. For four-rank all-gather of a
local `S`-byte shard, output is `4S` bytes and each rank must receive `3S` bytes;
`3S` also describes the ideal ring send volume. This is an algorithmic bound,
not a measurement of this implementation's links or overlap. BF16 physical
padding counts even when only one logical row is active.

| Boundary | Global meaning → per-rank tensor/layout | Program, weights and local shard | Collective payload at B1; ownership after boundary |
|---|---|---|---|
| Token embedding | Global `[248320,2560]` → `[248320,640]` BF16 row-major DRAM; lookup `[1,1,T,640]` → `[1,1,T,2560]` | Hidden-axis TP4 table, setup-time shard; lookup returns tiled BF16 consumer | Gather local `P*640*2=40960 B`, receive `122880 B`; identical initial values on four distinct stream owners. This is an entry cost, not a per-layer gather. |
| Inter-layer residual | Global four streams `[1,4,T,2560]` → `[1,1,T,2560]` BF16 tiled DRAM | One **whole stream**, not one 640-feature slice, per rank | **Zero boundary CCL** between all 48 layers. Each next norm sees its complete 2560-feature normalization domain. |
| GR / terminal normalization | Independent per-stream RMSNorm; FP32 learned `(1+weight)`; BF16 normalized `[1,1,T,2560]` | Q1..32 local2560 uses `merged_rms(...,rhs=True)` in MeshResidual.read and FinalMixer; square/mean/eps/rsqrt, FP32 learned weight and BF16 result boundaries retained; other shapes fall back. Normalized streams gathered to `[1,1,T,10240]` | Local `163840 B`, receive `491520 B`. This gather mixes distinct streams, not replicated copies. |
| GR / terminal down projection | Global `10240→320`; rank computes logical 80 outputs, padded to 96 | FP32 weights `[96,10240]`, exact elementwise-product/reduction projection, default block32; output BF16 `[1,1,T,96]` | Gather physical 96 columns: local `6144 B`, receive `18432 B`; slice each rank's 16 padded columns **after** gather, concatenate to logical320. |
| GR / terminal up and weighted sum | Replicated320 → rank-local mixing vector2560; multiply this rank's normalized stream | MeshResidual.up Q1: R16 exact streaming projection, zero-once buffers, exact BF16→FP32 input widening and resident FP32 `[2560,320]` weights; explicit BF16 result. Q>1 and FinalMixer.up retain inherited block256 projection. SiLU/sigmoid boundaries unchanged | BF16 contribution local `163840 B`, receive `491520 B`; unchanged gather then native exact local FP32 rank reduction for eligible Q1..32; caller scale1/4 and BF16 cast unchanged. Output mixed2560 is replicated; persistent residual remains rank-owned. |
| GR injection and residual write | Global10240 → rank scalar; block2560 × rank injection → rank stream update | BF16/HiFi4 projection, cores/readers/K-block `8/3/5`, scalar output padded32; eligible local2560 Q1..32 uses `exact_residual_write`: FP32 product→BF16 update→FP32 residual add→BF16 output, with original fallback beyond support | No extra CCL. Terminal mixer omits this injection/write entirely. Two GR boundaries occur per decoder layer. |
| GDN QKV and ZAB | Local4 key heads +12 value heads, D128; packed QKV2560, ZAB logical1560 | QKV BF16 exact native ordered projection; ZAB BF16/HiFi4 DRAM `8/2/2`, padding each packed part internally | No QKV CCL. Local recurrent state `[1,12,128,128]` FP32, convolution state local2560. |
| GDN output projection | Local1536 recurrence output → gather global6144 → exact local640 projection → gather2560 | Resident BF16 `[640,6144]` weights; Q1 uses R16 exact projection with zero-once buffers and explicit BF16 result, Q>1 retains inherited accurate projector; no row-parallel native matmul substitution | First local `98304 B`, receive `294912 B`; second local `40960 B`, receive `122880 B`. Replicated block result feeds rank-local GR write. |
| QSA QGate/K/V | Global24 Q heads /2 KV heads → local6 Q /1 KV, D256; KV duplicated across rank pairs; packed output3584=`3072+256+256` | BFP8/HiFi4 weight `[2560,3584]`, DRAM `8/3/2`; BF16 activations. SDPA HiFi2, FP32 DEST, approximation/packer off, grid8x8, q-chunk32/k-chunk128 | No QKV/SDPA CCL. Head ownership and query gates are preserved. |
| QSA paged cache/indexer | K and V each `[8192,1,32,256]` BF16; page table `[1,8192]` int32; position `[1]` int32; compressed index keys `[65536,128]` BF16 | Page32, selected2080 slots; four replicated index heads of128. Selected native scorer reads live-prefix bounds from device position. Q1 packs four heads into `[1,1,4,65536]` FP32 (physical32 rows), reuses key tiles across heads; Q2..32 keeps `[1,4,Q,65536]` | No cache CCL. Query row0 hardware broadcast replaces1024 reader stores with32; exact selected SFPU products, four sequential feature-tile folds and row reduction remain. Fixed output padding is zeroed; no host capacity bucket or recapture. Exact score/ordered-ID component evidence: causal_packed_v4.json. |
| QSA WO | Local6 heads×256=1536 → partial2560 | BFP8/HiFi4 `[1536,2560]`, DRAM `8/3/2`, BF16 partial | Local `163840 B`, receive `491520 B`; unchanged BF16 gather then eligible native exact FP32 rank sum. Replicated block result, rank-owned residual write. |
| MoE router | Replicated2560 → rank128 logits → global512 logits; top10 | BF16 exact `[128,2560]` projection; gathered output widened FP32 for router/top-k; scores BF16 after softmax | Local `8192 B`, receive `24576 B`. Same selected IDs/scores on ranks. |
| Routed experts | E512, top10, I640; rank owns128 **whole** experts, each gate/up2560×640 and down640×2560 | Selected inherited BFP4/LoFi, 10x8 compute grid, FP32 DEST/packer/approximation off; `deepseek_prefill.dispatch → moe_fused_swiglu → combine`, then specialized lone-row or native grouped reduction | Dispatch/combine group size1 means local packing, **not cross-rank all-to-all**. All logical row counts use grouped expert execution; there is no selected indexed lone-row expert kernel. EP contributions combine with shared contribution before one ordered rank sum. |
| Shared expert and MoE output | Replicated2560 → local160 gate/up → local down2560; separate replicated scalar gate | Packed BFP8/HiFi4 gate/up, DRAM `8/3/5`; BFP8 down `[160,2560]`; BF16 activations/CCL | Combined routed/shared local2560: `163840 B`, receive `491520 B`; one unchanged BF16 gather/native exact local FP32 rank sum restores eligible Q1..32 block result. No gather per active expert. |
| PLE lookup | 16 token-word IDs; table320001536 rows → rank80000384 rows (`id%4` ownership, `id>>2` address); 16 decoded160-feature embeddings → `[1,1,T,2560]` | uint32 packed20-word payload +4-word scales per row; native pipelined lookup, batch32; all16 local addresses looked up before ownership mask | Local lookup has no cross-rank dispatch; masked BF162560 result uses unchanged gather and eligible exact local rank sum: local163840 B, receive491520 B. Do not charge the entire resident table as bytes per token. |
| Final mixer → LM head | Mixer emits replicated2560; global vocab248320 → rank62080 logits `[1,1,T,62080]` | BF16/HiFi4/FP32 DEST, approximation/packer off; head `[2560,62080]`, bank-padded62208; selected DRAM `20/3/2` | Head has **no full-vocabulary CCL**. Logits retain vocab TP4 through sampler. |
| Sampler canvas | Persistent `[1,1,32,62080]` BF16 tiled DRAM; inactive rows filled -100 | Sparse row writes for small active sets; fixed canvas regardless of B1. Common local top32, actual semantic greedy k1, sampled top-k/top-p retained | Local top32 values `[1,1,32,32]` BF16 and local indices UINT16 each2048 B; each gather receives6144 B, total12288 B/rank. Global candidate128; indices widened/offset to UINT32 afterward. |
| Token feedback / generator | Persistent `tt_out_tok=[1,1,1,32]` UINT32, device seed/active state; fixed slot cache/position/prompt-length/batch metadata | Separate model and sampler captures; device position/RoPE advance, inactive-row token restoration, changed-only page tables, six prefill buckets1/2/4/8/16/32 | No token/position/logits readback or per-token synchronization inside resident replay. Caller-visible token read is a separate boundary and must be timed separately. Arbitrary logical lengths use internal padding and masks. |

Source contracts: [model.py](../../tt/model.py),
[multichip_decoder.py](../../tt/multichip_decoder.py),
[generator.py](../../tt/generator.py), and
[dram_projection.py](../../tt/dram_projection.py). For DRAM projections, the
actual L1 input shard is WIDTH_SHARDED, row-major core orientation, on `A`
storage cores, shape `[32,K/A]`; weights remain WIDTH_SHARDED across eight DRAM
banks with per-bank shape `[K,32*ceil(N/256)]`. Output is WIDTH_SHARDED L1 with
per-core `[32,32*ceil(N/(32*A))]`, then returned to interleaved DRAM. Program is
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, `per_core_M=1`,
`per_core_N=ceil(N/(32*A))`, role-specific `in0_block_w`, and readers per bank as
listed. Decoder DRAM roles retain A8. Selected head A20 uses input `[32,128]`,
weight `[2560,7776]` per bank, output `[32,3104]` per core, `per_core_N=97`,
three readers and K-block2. Weight bank padding remains62208; output-core
allocation covers62080 exactly. The prepared head weight byte count is unchanged.
Padding never changes vocabulary semantics or prompt-length support.
Exact helpers retain interleaved storage. Selected Q1 GR-up/GDN-output use
streamed native multiply/reduce and an assembler; remaining exact projections
and Q>1 use inherited product/reduction programs. No DRAM matmul config or
faster rejected accumulation policy is assigned to these exact helpers.

The selected scorer is wired in `MeshQSA._selected_ids(..., packed_decode=True)`.
[causal_packed_v4.json](causal_packed_v4.json) passes all36 real-operand cases,
Q1/3/17/32, positions from0 through near262144: live FP32 scores and ordered
selected IDs are exact. Q1 uses one quarter of the former causal dot-output
storage and one key-block read for all four heads. This is component evidence;
the artifact has `tracking=false`. The later serving Watcher v2 supplies
reduced integrated allocation/state evidence; [full48 v2](final_validation_v2.json)
now passes AIME/context/runtime-guard checks. [B32 v1](final_batch32_hf_v1.json),
[final resident](final_resident_v1.json) and [matched-source decode profile](terminal_profile_analysis_v2.json)
also pass their respective scopes. The subsequent scorer tile guard has25 CPU checks and does
not alter this supported standard-tile arithmetic. The explicit MeshQSA call is the
selection authority.

[head_sweep_v2.json](head_sweep_v2.json) passes ten head geometries with real
weights and **synthetic normalized activations**, Q1/Q32. Selected20/3/2 has
Q1 head-with-layout/output-copy0.706998 ms vs8/3/1 control0.718521 ms;
head-plus-split-greedy1.183168 vs1.193880 ms. Q32 head0.683270 vs0.693737 ms.
These are synchronized component traces, not full-model throughput, and the
artifact explicitly sets `acceptance_eligible=false`. Precision stays
BF16/HiFi4/FP32 DEST/packer-off. [head_sweep_v1.json](head_sweep_v1.json) records
8/1/1's real allocation rejection:3108800-byte static DFB region exceeds
1572864-byte L1. It is not an accuracy rejection. [Larger-block retries](head_geometry_followup_v1.md)
pass Q1/Q32 but are slower, including adapted40/80-core block10. Integrated
resident benefit is measured above; current [AIME v2](final_validation_v2.json)
passes prefill .97/1/1 and teacher forcing .96/1/1 top1/top5/top100. Final resident/TTFT and matched-source decode profile are complete; B32
HF/state controls pass. B32 clean throughput passes at 0.390840 tokens/s/user (12.506878 aggregate); [prefill analysis](prefill_profile_analysis_v1.md) is complete.

## Current exact-kernel selection and evidence boundary

| Source dispatch | Kept implementation / exact fallback | Evidence and remaining limit |
|---|---|---|
| [MeshResidual.write](../../tt/multichip_decoder.py) | [exact_residual_write](../../tt/exact_residual_write.py), local2560 Q1..32; original TTNN product/casts/add for unsupported shapes. One whole BF16 stream per rank. | [Watcher](exact_writer_watcher_v1.json), [clean](exact_writer_clean_v1.json): Q1 .039336→.012933ms; Q32 .039434→.013041ms. |
| MeshResidual.read and [FinalMixer](../../tt/model.py) norm only | [merged_rms](../../tt/fused_decoder.py) RHS mode for local2560 Q1..32; original RMS fallback. FinalMixer.up does not opt into native Q1 projection. | [Tracked norm](exact_norm_tracked_v1.json), [clean](exact_norm_clean_v1.json): exact FP32 scale/BF16 result; Q1 about .0689→.0632ms. |
| MeshResidual.up / MeshGDN.out through MeshExactProjection | [exact_projection_q1](../../tt/exact_projection_q1.py), R16/zero_once only Q1; GR-up widens BF16 input to FP32 once, weights stay FP32; GDN input/weights remain BF16. Separate FP32 multiply, sequential K-tile fold and selected SFPU row reduction, then explicit BF16 cast. Q>1 and long-prefill Q1 tails retain inherited path. | [Real-weight clean](exact_projection_real_clean_v3.json), [Watcher](exact_projection_watcher_v5.json): GR-up .060828→.047846ms; GDN-out .181423→.119231ms. Inputs synthetic; integrated gates remain necessary. GR-down/router native candidates slower and rejected. |
| Collectives.sum after the same synchronous gather | [exact_rank_sum](../../tt/exact_rank_sum.py), BF16 [1,4,Q,2560] Q1..32 on Blackhole1x4; FP32 selected H-tree including positive-zero terms. Original widen+sum fallback. Caller scale/cast stays outside; reaches eligible GR/final mixer, QSA, MoE and PLE consumers. | [Tracked](exact_rank_sum_tracked_v4.json), [clean](exact_rank_sum_clean_v2.json): Q1 .051082→.020202ms; Q32 .155644→.020196ms including original caller scale/cast. |
| GatheredProjection valid_width80 | **Original four slices plus concat remains selected.** [Compaction helper](../../tt/compact_gr_down.py) is experimental only after dispatch rollback. | [Component record](compact_gr_down_proposal.md) passes all patterns, but [integrated v1](final_serving_watcher_v1.json) fails mixed-prefill logits; [restored-source v2](final_serving_watcher_v2.json) passes all16 checks. No kept compaction speedup. |

The shared ownership boundaries above remain unchanged: weights are prepared
before capture, BF16 KV/activation/transport stays selected, local exact helpers
consume/produce the same residual layouts, and token/position/RoPE/page-table
inputs remain persistent. The earlier resident timing manifest is
[exact_integrated_sources_v1.json](exact_integrated_sources_v1.json). Current
[final_sources_v2.json](final_sources_v2.json) had71 implementation hashes;
validation also records `tt/kernels/router_topk_algorithm.hpp`. [v3](final_sources_v3.json)
adds that header and shared-source coverage without code changes. Current
[v4](final_sources_v4.json) changes only the scorer validator to reject custom
or face-transposed tiles; standard32x32 path math/kernels are unchanged. All
current implementation hashes match v4. [25 metadata checks](check_causal_standard_tile_guard.py)
pass on the installed validator, and [final code hooks v3](final_code_hooks_v3.log)
pass over45 stage-owned code files. B1/B32/serving artifacts retain their
original source bindings; they are not retroactively relabeled as post-guard
hardware execution. Final resident, component traces, terminal v2 profile and
full48 mixed-prefill signature checks provide separate v4 execution evidence. New native files retained solely as experiments
do not establish runtime dispatch. No new precision frontier is authorized.

## Reusable modules and exact hyperconnection mismatch

Common attention/MLP modules use a single hidden stream and conventional
hidden-sharded residual; this model uses **four independently normalized
2560-feature streams**. For rank r, let `z_r=RMSNorm_r(x_r)` and
`z=concat(z_0,z_1,z_2,z_3)`. Ignoring only the explicit casts shown in source,
`a=SiLU(W_down*z/4)`, `m_r=sigmoid(W_up,r*a)`,
`h=sum_r(m_r*z_r)/4`, `g_r=2*sigmoid(W_inject,r*z/4)`, and
`x'_r=x_r+g_r*block(h)`. The terminal mixer returns `h` without injection.
FP32 products/reductions and the BF16 boundaries are numerical contracts;
ordinary hidden RMSNorm over10240, averaging streams before RMSNorm, or treating
one rank as a640-feature slice changes the function.

The reusable families from `tt-transformers` attention_1d/mlp_1d and distributed
RMSNorm are still relevant: local heads/cache, setup-time column/row weight
partitioning, compact collectives, delayed gathers and next-consumer sharding.
They do not implement this exact GR down/up/injection computation, GDN's exact
recurrent projection/reduction, QSA's compressed-key selection, or the PLE
lookup. Model-local implementations preserve these contracts. Common
`models/common/sampling/generator.py` **is used directly** with its slot/token
contract; common `models/common/tt_ccl.py` supplies semaphore ownership through
`Collectives.prepare_sampling_collectives`. Merely instantiating that manager
does not establish that every sampler gather is asynchronous or preallocated.
The inherited split profile explicitly contains two `AllGatherDeviceOperation`
rows (BF16 and UINT16), while `_perform_all_gather` uses synchronous
`ttnn.all_gather` when this wrapper has no `line_all_gather`. [Stage7 async-barrier controls](sampling_ccl_v1.json) are slower than
synchronous gathers, with [tracked correctness](sampling_ccl_tracked_v1.json).
The persistent-output-buffer variant remains untried because its current
factory disables the replay barrier; it requires a separately safe design.

## Required collective-family comparison

For one2560-wide BF16 partial at physical32 rows, let `S=163840 B`. Figures
below are ideal per-rank network send/receive volume, not observed packet bytes.
FP32 transport doubles these sizes. Next-consumer scope includes GR/norm/write,
not an immediate artificial restore to the former stream contract.

| Family | Residual before → after; next consumer | Expected communication and dtype | Persistent-buffer plan and evidence/status |
|---|---|---|---|
| Local matmul + full gather / ordered sum (selected) | Whole stream2560 → whole stream2560; next GR norm local | Gather `3S=491520 B`; BF16 transport, FP32 selected ordered reduction | Static trace allocation; synchronous gather and native local exact rank reducer selected. Final v2 exact-kernel attribution is complete; collective payload/transport is unchanged. Weighted CCL transport6.130895ms excludes local reduction/copy/fill. |
| Local matmul + native all-reduce | Same whole-stream ownership; next GR norm local | Ring AR `2*(3/4)*S=245760 B` at BF16, or491520 B FP32 | Persistent native AR was measured in stage4; changed summation is not automatically exact. Lower per-request accuracy/small gain rejected. A custom exact-order AR is a distinct unmeasured implementation, not declared impossible. |
| Reduce-scatter + delayed gather | Four stream owners → hidden-sliced representation; distributed per-stream norm and GR consume it | RS `(3/4)*S=122880 B`; later gather same122880 B if full result is needed. Exact GR may need additional stream assembly | Stage5 whole QSA retains hidden slices through both GR blocks/MoE, then reports slower B32. This is real next-consumer evidence, not RS followed immediately by AG. Extending exact GDN/PLE remains unmeasured and user disallows replacing selected stream policy here. |
| Fused all-gather-matmul | Gathered activation fused into projection; proposed hidden-sliced consumer | Same logical gathered payload as unfused producer (`3S` for local shard S); fusion can remove DRAM materialization, not its information transfer | Whole QSA AG-MM hidden-layout candidate rejected on B32. GDN exact output projection cannot be substituted without preserving its sensitive reduction tree. |
| Fused matmul + RS / AR | Row-parallel producer directly emits hidden slice or reduced block; next GR adapted | Same RS/AR lower bounds above, possible overlap and avoided intermediate writes | Producer/next-GR components tested in inherited topology audit; distinct full-layer fused MM-RS combination remains hand-off, not a measured loser or selected path. |
| Rank-fractured residual (selected) | One full stream/rank remains one full stream/rank | Zero inter-layer boundary conversion; normalized-stream and compact mixing gathers as above | Selected full stack honors it. Generic replicated10240 residual is prohibited by this stage's explicit user contract. |
| Async/persistent same-order gather | Same selected residual and arithmetic | Same491520 B; buffering may reduce launch/allocation overhead, not payload | Stage5 measured preallocated async decoder CCL regressed. Retain rejection unless new matching full-path evidence overturns it. Sampler candidate buffers are a separate narrow scope. |
| Candidate-only split sampler vs full-vocab gather | Vocab shards → replicated128 candidates vs replicated248320 logits | Split total receive12288 B; full-vocab BF16 canvas local3973120 B, receive11919360 B | Correct frozen-logit stage7 [clean control](terminal_clean_v1.json): split .481979ms vs force-argmax2.765952ms. Current matched-source eager split .426224ms versus force-argmax2.680506ms is a distinct profiler boundary. Selected split is0.414% of103.064960ms resident; no host argmax, generic TopK or full-vocab gather selected. |

## Inherited rejection ledger and precision authority

The user's explicit instruction to preserve selected dtype/fidelity/KV/
activation/CCL policy and residual layout overrides a generic optimization
recommendation to explore lower precision or alternate replicated streams.
This stage is **not** the datatype frontier. Inherited routed BFP4/LoFi is the
accepted physical-fit exception: a fresh blanket BF16/BFP8 rule must not
silently replace it. Conversely a faster rejected attention/shared/KV/CCL
policy cannot become the optimized-full-model baseline. Exact candidate
artifacts and their adjacent provenance remain authoritative:

| Decision | Exact inherited evidence |
|---|---|
| Async/persistent/global BFP8 CCL rejected | [async_l0.json](../optimized_multichip_decoder/async_l0.json), [async-persistent_l0.json](../optimized_multichip_decoder/async-persistent_l0.json), [async-bfp8_l0.json](../optimized_multichip_decoder/async-bfp8_l0.json), matched [before_l0.json](../optimized_multichip_decoder/before_l0.json). Persistent B1/B32=2.27989/58.0905 ms; global BFP8=2.33470/59.6262 ms. |
| Hidden residual native RS / fused AG-MM rejected for measured B32 | [hidden_qsa_rs.json](../optimized_multichip_decoder/hidden_qsa_rs.json), [hidden_qsa_agmm.json](../optimized_multichip_decoder/hidden_qsa_agmm.json), [hidden_stream_control.json](../optimized_multichip_decoder/hidden_stream_control.json): +1.90%/+1.77% B32. Provenance names are recorded in [candidate_summary.csv](../optimized_multichip_decoder/candidate_summary.csv), including retry histories. |
| Native AR not selected | [mesh_l0_persistent_ar.json](../multichip_decoder/mesh_l0_persistent_ar.json), [residual_contract.json](../multichip_decoder/residual_contract.json), [AUTOFIX.md](../multichip_decoder/AUTOFIX.md). Two historical per-request misses; no recorded aggregate B32 PCC, so this is not claimed to fail the current whole-output gate. |
| BF16 KV retained | [real_bfp8_kv.json](../optimized_multichip_decoder/real_bfp8_kv.json): short decoder passes, B1+12.09%, B32+0.49%; no new maximum-context claim. |
| Real BFP4 attention rejected for accuracy | [real_qsa_bfp4.json](../optimized_multichip_decoder/real_qsa_bfp4.json): minimum short whole PCC0.96450335, B32whole0.98230147; faster timing cannot override failure. |
| BF16 CCL/activation retained | [real_ccl8_moe.json](../optimized_multichip_decoder/real_ccl8_moe.json), [real_ccl8_qsa.json](../optimized_multichip_decoder/real_ccl8_qsa.json); activation [shared](../optimized_multichip_decoder/real_activation8_shared_l0.json), [QSA](../optimized_multichip_decoder/real_activation8_qsa_l3.json), [WO](../optimized_multichip_decoder/real_activation8_wo_l3.json). Real trials pass, no material matched benefit. |
| Selected HiFi4 projection roles retained | [real_dram_qsa_HiFi2_native.json](../optimized_multichip_decoder/real_dram_qsa_HiFi2_native.json), [real_dram_qsa_LoFi_native.json](../optimized_multichip_decoder/real_dram_qsa_LoFi_native.json), [real_dram_gdn-compact_HiFi2_native.json](../optimized_multichip_decoder/real_dram_gdn-compact_HiFi2_native.json), [real_dram_shared_LoFi_bfloat4_b.json](../optimized_multichip_decoder/real_dram_shared_LoFi_bfloat4_b.json). Shared BFP4 is deferred precision evidence, not a newly selected stage7 policy. |

The complete [stage5 topology audit](../optimized_multichip_decoder/operation_topology_audit.md),
[anomaly ledger](../optimized_multichip_decoder/anomaly_ledger.md), and
[candidate CSV](../optimized_multichip_decoder/candidate_summary.csv) retain
other program-config/kernel attempts and exclusions. The unbound historical
`parallel-indexer_l3.json` experiment is explicitly excluded as optimization
evidence. Inheriting a rejection does not assert all untried variants are slow.

## Baseline scope, final accounting and remaining gates

The [stage4 clean layer comparisons](../multichip_decoder/performance_summary.json)
provide warmed single-chip/mesh B1: GDN3.0770/2.2645 ms (1.3589x,33.97%
efficiency), PLE/GDN4.2459/3.4315 ms (1.2373x,30.93%), QSA4.9553/4.7477 ms
(1.0437x,26.09%). These are **layer-baseline speedups**, not a claim that the
complete48-layer model and full KV capacity fit or were benchmarked on one ASIC.
Selected stage5 layers are faster than those earlier mesh controls; do not
combine stage4 single-chip latency with a different stage7 workload and label
it a matched full-model speedup.

Historical stage5 C128 weighted stack is
`35*2.081642866 + 1*3.136728048 + 12*2.858865023 = 110.300608635 ms`.
Stage6 full48 C262144 token-out138.754269 ms and reduced eager terminal1.552066 ms
would imply24.05% overhead, but those regimes are not interchangeable and this
is not a certified lower bound. Current [component traces](component_trace_v2.json)
use C262144, representative real frozen layer inputs and multiplicities35/1/12:
102.507076ms decoder stack +1.576551ms entry/terminal/sampler =104.083628ms.
This is a representative trace estimate, not a physical lower bound or a direct
all48 layer measurement. [Final resident](final_resident_v1.json)103.064960ms is
0.979% below it, so no positive10% orchestration gap is present. The resident
window has127 model and sampler replays, zero interior read/sync/refresh/capture/
release counters, and persistent token/sampler buffer identities; explicit
window synchronizations and final readback are outside the per-token boundary.

[Terminal v2](terminal_profile_analysis_v2.json) reconciles68,024 host/device/final
operation keys with complete signposts, [tables](terminal_final_perf_v2/model_perf_report.txt),
[CSV](terminal_final_perf_v2/ops_perf_results.csv) and provenance. Audited advice
uses raw runtime geometry and selected kernels. Weighted eager device work is
99.939984ms, including6.130895ms CCL transport; this excludes local arithmetic
from transport and is a separate regime from trace wall time. All72 model and3
shared hashes match v4 for profiler, component and resident artifacts.

Warmed public-generate TTFT for31/32/33/128/1024 tokens is respectively
1845.539/1202.206/1304.506/4705.951/38102.021ms, including reset/setup, complete
prefill, first device sample and caller read. The matched128-token baseline is
7035.653ms; other lengths are final-only, with no invented before/after delta.

The stage6 [terminal analysis](../full_model/terminal_final_analysis.md) and
[split report](../full_model/terminal_final_perf_v1/split_greedy_perf_report.txt)
are the source of inherited sampler/terminal numbers. Current [full48 v2](final_validation_v2.json)
passes AIME prefill/teacher forcing (top5/top100=1),257-token resumed/nonaligned
controls and32769-token smoke. QSA positions advance32770→32771 on the extra
device-feedback replay; terminal logits remain finite and191643648B/bank free.
The full-body runtime audit records12 prefill,2 decode,3 common-sampler and8
complete-capture calls without failures, positive controls and restored
bindings. Caller/setup/reference work is deliberately outside that audit.

The [fresh six-prompt qualitative verdict](final_validation_v2/qualitative/verdict.json)
uses66 new TT outputs,120 independently label-hidden sampled ratings and zero
failures per matched prompt/backend; all prompts stay within the20-point
margin. [AIME review](aime_autoregressive_review_v2.json) finds coherent matched
prefixes;128-token caps establish neither completed answers nor AIME scores.
[Prior comparison](final_validation_v2/qualitative/prior_candidate_comparison.json)
confirms all66 current token/completion rows equal the earlier candidate, but
fresh scoring is retained independently. Teacher-forcing8.790475t/s/u is a
tracked readiness/caller boundary, not clean resident token-out throughput.

Final B1 resident, decode attribution and representative-stack reconciliation
are complete. [Six-bucket prefill interpretation](prefill_profile_analysis_v1.md) completes89,740
operations and semantic advice review with all72 model+3 shared hashes matching v4.
[B32 clean performance](batch32_resident_v1.json) passes at 0.390840t/s/u (12.506878aggregate) with127 resident public model/sampler steps and no interior host boundary. Independent review returns `more-work-needed` for the floor alone; local checkpoint details are in the work log. The strict useful-weight+KV floor
is6.414528ms;103.064960ms achieves6.223772%, below10%. Reaching64.145280ms
requires removing38.919680ms; no graph-inflated denominator or correctness pass
waives this requirement. The confirmed wall-cap handoff and nonadditive current
candidate ceilings are in [Not tried (stop rule)](optimization_checklist.md#not-tried-stop-rule).

## Completed six-bucket prefill analysis

[Analysis/advice](prefill_profile_analysis_v1.md) and [JSON/provenance](prefill_profile_analysis_v1.json)
cover one warmed reduced0/1/3-layer chunk per logical1/2/4/8/16/32 bucket,
89,740 operations and12 signposts. Eager device sums are respectively
8.177249/17.827547/22.579935/34.278169/53.380053/87.538737ms; each bucket is
an alternative shape, not a sequential term to sum. Inputs precede the marker;
no sampling, live trace, checkpoint save or measured caller read is included.
Q32 ordered-indexer kernels cost24.091004ms (27.52% of that reduced body),
while head matmul stays about .644ms. These identify research priorities, not
achievable savings or an all48 denominator. Generic kernel groups span roles.

All72 model+3 shared hashes match v4. Each bucket has equal per-rank operation
counts, finite active logits and positions matching logical length. Some later
CSV input descriptors remain32[2]; descriptor reuse is plausible but unproven,
so source runtime counts and driver state establish logical shape. Whole-body
markers do not identify individual layer classes; the full48 device-prefill
estimate remains unavailable. The independently measured4705.950625ms public
TTFT must not be compared by subtraction or48/3 scaling. Semantic advice review
is complete; multirow exact/indexer scheduling remains an unknown-gain handoff.

## Completed B32 HF and state-control scope

[final_batch32_hf_v1.json](final_batch32_hf_v1.json) passes all48 layers, B32,
C512 and shared paged KV. Baseline and migration-at-step32 runs each score2704
predictions:2645 top1 matches (.978180473),2704 top5/top100 matches. Exact
matched-history logits and greedy tokens, final request-state hashes, dormant
state and all-inactive no-op controls agree. Separate resident common-sampled
feedback passes four steps, including reduced active slots, coherent device
positions and unchanged inactive feedback. Its diagnostic reads are explicit,
and `performance_claim=false`; no B32 tokens/s claim follows.

These are correlated continuations of one203-token AIME24 chat-template prompt
(`source_prompt_count=1`), not32 independent prompts. The artifact reports
1241.019s internal wall time (owner-observed process roughly1246s). The existing
cross-geometry gate is unchanged and not adjudicated by this test. Hardware
validation predates the v4 scorer rejection-only guard; final resident and
full48 mixed-prefill checks separately execute the supported v4 standard path.

## Full48 mixed and resumed prefill on final source

[Final signature checks](final_prefill_signatures_v1.json) pass with tracking,
all48 layers, shared KV and maximum batch2. Mixed and reversed request order
produce exactly matching logits; retained-state resumes68→257 and31→265 also
match exactly. Retained trace IDs, bounded candidate traces, zero request-time
releases, decode submission order and no late compilation all pass. This is
correctness evidence, not a throughput claim. Compaction remains reverted:
its integrated mixed-prefill mismatch and later minimal reproduction fail;
conditioned eager real-operand comparisons pass, but do not resolve the native-only
serving-trace failure or prove its root cause.
