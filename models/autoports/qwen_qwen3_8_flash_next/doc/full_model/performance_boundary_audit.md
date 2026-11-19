# Full-model performance boundary audit

CPU source/evidence audit, 2026-11-14. No hardware was run for this audit. The
28.454 ms difference from the stage-5 weighted decoder estimate is **not a
measured wrapper cost**. The compared paths have different allocated context,
input histories, resident weights, and measurement boundaries. Current evidence
supports a capacity-dependent QSA scoring cost; its exact share remains unmeasured.

## Measured boundaries

| Evidence | Capacity / workload | Measured result |
|---|---|---|
| `acceptance_perf_final_v1.json` | Full48 B1, C262144, 128 input / 128 output, 127 steady decode steps, clean tracker0 | Token-out 138.754268995 ms, 7.206985466 tokens/s/user; TTFT 7038.672064 ms |
| Same artifact, separate model-only window | Same resident full48/C262144; fixed resident token, advancing positions; no sampler/readback | 138.152609609 ms, 7.238372137 tokens/s/user |
| Stage5 `../optimized_multichip_decoder/real_mesh_final_l{0,1,3}.json` | Separate representative layers, C128, length33; 32 warmed blocking trace replays | 35×2.081642866 + 1×3.136728048 + 12×2.858865023 = 110.300608635 ms |

Each stage5 matching `_provenance.json` explicitly records `--capacity 128
--length 33 --trace --batch 32 --repeats 32 --clean-timing`; this is not an
assumption from a default. The B1 metric is separate from that job's B32 check.
The weighted sum is a historical estimate, not an end-to-end trace or a bound.
It omits full-model embeddings, final mixing, vocabulary projection, assembly,
sampling and caller token reads, and does not capture all48 residency effects.

Token-out minus model-only is 0.601659386 ms; model-only minus the stage5 sum is
27.852000974 ms. The first difference is consistent with a small sampler/read
boundary, but the windows execute different token streams and are not additive
component accounting. The full-model steady window records127 model replays,
127 sampler replays and127 token reads, with zero token/position/page refreshes,
captures, releases or resets. Thus per-token host token-feedback, page-table
rebuilding, trace capture or reset cannot explain this particular measured gap.

## Production QSA source chain and capacity arithmetic

The actual selected path is `tt/multichip_decoder.py:MeshQSA.build/forward_chunk`
→ `tt/fused_decoder.py:FusedChunkedQSA._selected_ids`
→ `FusedQSA.indexer_scores` / `fused_indexer_scores`, followed by
`fused_selected_token_ids` and `local_selected_paged_kv`. It is neither the
baseline `qsa.py` scorer nor TTNN's unrelated experimental DSA indexer operator.
`MeshQSA.build` selects `score_sum=True`, `merge_score_heads=True`, and
`selector=True`; its existing precision/fidelity settings are unchanged.

State construction allocates compressed keys `[1,1,C/4,128]`. The scorer casts
**all** those keys to FP32 and computes four-head products before any position
mask. For Q1, `block=65536`, so both capacities execute one Python block; the
physical tensor work changes rather than the number of scorer calls.

| Per rank, per QSA layer, Q1 | C128 | C262144 |
|---|---:|---:|
| Compressed keys K=C/4 | 32 | 65536 |
| BF16 compressed-key storage | 8192 B | 16777216 B |
| FP32 key-cast logical data | 16384 B | 33554432 B |
| FP32 product `[4,1,K,128]` | 65536 B | 134217728 B |
| Final FP32 score logical elements | 32 | 65536 |

This is2048× scoring width, independent of the short live history. Product
arithmetic alone covers1.5 GiB across12 QSA layers per rank per full decode
step at C262144, versus768 KiB at C128; these are tensor-size sums, **not measured
DRAM traffic or simultaneous live memory**, and do not model read/write reuse,
padding, scheduling or sustained bandwidth. The accurate FP32 reduction then
processes that width. Static trace capture preserves those prepared tensor
shapes; it does not shorten scoring to the current position at replay.

The native selector has a different boundary: `_QSA_SELECTION_KERNEL` computes
`complete=(position+1)/4` and loops only while `start<complete`. Its heap therefore
scans causally complete history, **not all allocated keys**. At current128-input
steady decode positions128..254 this is32..63 complete groups, versus roughly8
at stage5 length33. It always writes2080 selected slots. The subsequent selected
K/V gather and SDPA have fixed2080 selected-token geometry. Do not attribute
2048× capacity scanning to the native heap, selected-cache gather, or SDPA.

Larger cache/resident allocation and changed logical history can also affect
memory behavior. Source establishes which operations scale; it does not prove
that QSA accounts for precisely28 ms or exclude other model-only costs.

## Named full-model-only costs and existing evidence

| Boundary | Current implementation / evidence | Attribution limit |
|---|---|---|
| Embedding and stream formation | `QwenModel.embed`: hidden-width-sharded lookup, CCL gather and residual construction | Not present in stage5 layer-only estimate; needs current op profile |
| Final norm/hyperconnection mixing | `FinalMixer.__call__`: RMS, exact down/up projections, gather/sum and activation boundaries | Not a decoder-layer cost; needs current op profile |
| Vocabulary projection | `QwenModel.logits`, direct DRAM subgroup block1 BF16/HiFi4/FP32 DEST, packer disabled | `lm_head_dram_v1.json` isolated real-weight synthetic-input Q1 projection incl.layout/output copy:0.718163 ms vs native0.868638 ms; not current whole-model attribution |
| Physical sampler row assembly | `_decode_body`: native fill plus valid-row write for B1; concat only above24 active rows | `terminal_rows_v2.json` isolated traced B1 fill+write0.047810 ms vs concat0.537668 ms, exact logits/greedy tokens; synthetic input and separate windows |
| Split greedy, RNG bookkeeping and device token feedback | Common sampler plus current `DeviceTokenFeedback`; fixed device buffers | `terminal_timing_v2.json` historical C128 reduced split0.541401 ms versus force2.813899 ms; token-out9.680536 vs11.981420 ms, same14 tokens; predates latest feedback/workspace factoring |
| Caller token read and orchestration | `_read_tokens`; no full-logit read on measured token-out path | Historical reduced isolated read0.029564 ms; current full-model separate-window difference0.601659 ms is not exact sampler/read attribution |
| Prefill workspace/checkpoints/reset | Prefill/transition lifecycle, outside steady decode | May affect residency/TTFT; no per-token checkpoint restore/reset recorded |

Historical `terminal_perf_v3/coverage.json` proves23048 host/device/final ops
matched with six signposts and no skipped rows. Its old C128 reduced eager model
merged-kernel sum was9.440300 ms, split0.467541 ms and force2.939592 ms. Split was
4.72% of that reduced model-plus-split kernel sum; force was6.29× split. The old
head/row assembly has since changed. These figures justify the existing common
sampler selection, not a current complete profile or additive full48 latency.
No new sampler, decoder arithmetic policy, context capability, or rejection-ledger
change is proposed by this audit.

## Final attribution capture

C128 remains useful for historical reproduction and isolated context-independent
head/sampler costs. It is insufficient for attributing the current C262144
model-only gap. The prepared final reduced capture now explicitly uses
`--capacity 262144`, retaining layers0/1/3, B1, the same six signposts, one measured
iteration, two warmups, no serving-prefill preparation and no live traces.
The CLI default remains128; its parser validates128 multiples through262144.
Report and generator construction both consume the selected capacity. The
post-capture recipe requires reported capacity262144 before accepting its CSV.

`terminal_final_commands.md` and `run_terminal_final_v1.sh` record explicit
checkpoint, packed PLE cache, visible devices, mesh and profiling environment.
The8192 operation-support ceiling is retained: Q1 scorer uses one merged-head
block at both capacities, and causal selector iteration occurs inside one native
kernel. Increasing C therefore does not multiply Python-dispatched op count.
Historical full capture had5762 ops/rank across warmup/setup/regions, below8192,
and the new preparation omits prefill. This is source-based sizing, not a
promise of profiler capacity; strict coverage must still pass, and failed raw
captures must be retained.

The reduced capture exposes the large-capacity QSA scoring operations and the
current full-model terminal boundaries. It still cannot establish exact12-layer
critical-path scaling, full48 residency effects, or the cost at positions near
262144: its decode starts from reset and has a very short live history. Do not
multiply eager kernel sums into an advertised full-model throughput. Reconcile
its named terminal operators and advice with the existing clean all48 trace
measurement and preserved precision policy before deciding whether work remains.

Validation after the authorized profiler-only change: five actual-helper CPU
scope/parser/propagation tests pass (`python_env/bin/python -m unittest
models.autoports.qwen_qwen3_8_flash_next.tests.test_terminal_profile_scope_cpu -v`),
pinned Black23.10.1/isort5.13.2/autoflake2.3.1 checks pass, Python compilation
and shell syntax validation pass. No production model or generator source was
modified. No capture has been executed by this audit.

## Completed C262144 attribution evidence

The authorized final reduced capture is now complete. Its coverage was
independently recomputed:21488 exact host/device/final operations, six signposts,
no skipped rows, matching positive durations. All captured source hashes still
match. `terminal_final_analysis.md` and `.json` record operation boundaries,
hashes, advice, dtype/fidelity and rank-merging limits.

Current merged kernel sums are11.256808 ms model,0.429263 ms split greedy and
2.886563 ms force-argmax; both candidates return true greedy token 243975 on the
same frozen logits. Split is3.673% of model-plus-split kernel sums, so these
results do not establish sampler dominance. Named full-model costs are
embedding/gather0.036766 ms, final mixer0.372330 ms, LM head with layout0.668237 ms
and native row assembly0.045470 ms. The head's HiFi4/BF16/FP32 DEST/block1 policy
is recorded by the actual profiler row, with no head or sampler advice remaining.

The existing QSA scorer is2.387150 ms at C262144 versus0.025220 ms in historical
C128 v3, with metadata confirming65536 versus32 compressed keys. Native causal
selection remains0.010177 versus0.010167 ms. This substantiates the capacity
cost mechanism; it is not an exact additive attribution of the all48 gap.
Some BinaryNg cached shape annotations conflict with source/adjacent metadata;
the analysis discloses that limitation and makes no FLOP/bandwidth claim from
those fields. Larger-context scoring and current full-model-only costs are now
named measured operators, while full48 critical-path decomposition remains
unmeasured. Historical failures/raw captures and advertised capacity are unchanged.
