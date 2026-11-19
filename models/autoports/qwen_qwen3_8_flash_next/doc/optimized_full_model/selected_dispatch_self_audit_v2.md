# Selected dispatch self-audit v2

CPU-only self-audit on 2026-11-14; not an independent stage review. No devices or production edits by this audit. At the original audit, all **71/71** listed source hashes matched [final_sources_v2.json](final_sources_v2.json). Subsequent guard hardening is recorded below under [current v4](final_sources_v4.json). Experimental compaction is not imported or selected by production.

**Conclusion:** no new confirmed bug on the selected Blackhole 1x4, standard-tile model path. The original unsupported-input guard gap is preserved below and now **resolved by rejection-only validation**. [B32 HF/control v1](final_batch32_hf_v1.json) subsequently passed; final resident/performance evidence remains pending. This self-audit is not independent stage review.

## Original concern outside the selected input contract — resolved below

At the original audit, `tt/causal_indexer_scores.py::_causal_indexer_dots.validate` checked logical/padded shapes, TILE layout, dtype, allocation, DRAM placement, and device equality, but did not check `get_tile()` dimensions or face-transpose flags. Its kernels hard-code 2048-byte BF16/4096-byte FP32 pages and standard32x32 face offsets. A nonstandard tile can share the checked padded dimensions, pass validation, and be addressed incorrectly. The exact projection, rank-sum, and residual-write helpers explicitly reject such tiles. Follow-up hardening should give the causal scorer the same explicit check; no precision or algorithm change is needed. This is not an observed model regression: the model's query/key/position producers and component fixtures use default standard tiles, and no nonstandard-tile model branch was found.

**Resolution after B32:** [v4 source manifest](final_sources_v4.json) records
only `tt/causal_indexer_scores.py` changed from v3. The common validator now
requires tile_shape32x32 with both face-transpose flags false for query, keys,
positions and packed/general outputs. [Patch](causal_standard_tile_guard.patch)
preserves every kernel, arithmetic operation, descriptor and allocation on the
supported path. [CPU metadata check](check_causal_standard_tile_guard.py) passes
25 cases against the installed validator without TTNN/device import; [code hooks
v3](final_code_hooks_v3.log) pass across45 stage-owned files. v3 itself only
added router-header/shared-source manifest coverage. B32 and earlier numerical
artifacts retain their original hashes and precede this hardening; final
resident is running as the post-guard standard-path check, with no result
claimed. The custom-tile guard finding is resolved; arbitrary position values
and untested architectures remain outside the helper contract.

The scorer also relies on its documented nonnegative integer position contract. Reader assertions bound positions to compressed capacity; writer bit decoding assumes the same valid values. Invalid fractional/negative/nonfinite positions are not a supported helper input and are not how inactive serving slots are represented: inactive slots skip model work. The public generator validates positions and capacity before submission. Do not describe arbitrary device position tensors as supported.

## Selected branches and ownership

| Component | Source assessment and evidence |
|---|---|
| Q1 exact projection | Only GR-up and GDN-out opt in. GR-up retains FP32 precast for BF16 activation/FP32 weight. Both use R16 and zero-once. Q2..32 use the inherited selected algorithm; Q>32 explicitly calls the base algorithm for every chunk, including a trailing Q1. Standard tile, O/K alignment, output non-alias, and mesh guards are present. Each scratch group and assembled output tile has one owner; scratch/raw allocations are invocation-local and passed to their consuming operations. Source and R16 branch were checked against [Watcher v5](exact_projection_watcher_v5.json) and [clean real-weight v3](exact_projection_real_clean_v3.json), with finite-operand scope retained. |
| Rank sum | Dispatch requires BF16 `[1,1,Q,2560]`, Q1..32, Blackhole1x4, standard gathered `[1,4,32,2560]` physical tiles. Synchronous gather and rank order remain unchanged. Helper allocates fresh FP32 output, owns complete feature tiles, and processes each token row independently using the selected positive-zero additions and rank tree. [Watcher v1](exact_rank_sum_watcher_v1.json) covers all Q1..32, poison padding, cancellation/signed zero, and changed-input traces. Sibling MoE/QSA/PLE consumers remain in scope, not only GR. |
| Residual writer | Dispatch is limited to matching Q1..32 BF16 local2560 residual/block and one-column injection. Helper validates standard tiles, padded dimensions, devices and allocations; fresh output has one owner per tile. Both BF16 rounding boundaries remain explicit. Q>32 uses inherited operations. [Watcher v1](exact_writer_watcher_v1.json) covers Q1/3/17/32 and changed-input replay; all intermediate Q values share the same physical one-row-tile kernel. |
| Causal scorer | Q1 selects packed four-head storage; Q2..32 uses the general four-head path. K is tile-aligned32..65536, covering the selected maximum262144-token context after4:1 compression. Every invocation rereads persistent device positions; both writers clear every physical output tile before filling live cells, including a shrinking prefix. Workers exclusively own full output tiles. Caller-provided outputs cannot alias query/key/position. No helper-global mutable cache or persistent allocation was found. [Packed v4](causal_packed_v4.json) reports36 exact-score/ordered-ID cases across Q1/3/17/32 and positions through262112, using real captured operands; these component records are clean timing evidence, not standalone tracking/Watcher evidence. |

The generic-op program hash excludes runtime values deliberately. Runtime positions, addresses, and Q are supplied through descriptors; a missing Q value in compile-time arguments alone is not proof of a cache bug. The selected integrated cross-request/cross-Q behavior has separate evidence below. No stale-position host cache or input/output alias was found in the reviewed selected helpers.

## Integrated evidence checked

[final_serving_watcher_v2.json](final_serving_watcher_v2.json) passes all16 reduced real-layer serving checks after compaction dispatch was reverted, including mixed3/33 prompts, resumed/non-aligned prefill, cross-request trace reuse, fixed/inactive slots, device token feedback, migration, and allocation-tracker positive control. It exercises retained Q1/Q2 traces after later Q32 work, unlike an isolated probe that releases each trace before switching Q.

[final_validation_v2.json](final_validation_v2.json) reports a full48-layer pass with prefill top5=100%/top100=100%, teacher-forcing top5=100%/top100=100%, resumed-prefill checks, autoregressive/qualitative phases, and a32769-token non-aligned context smoke. Its runtime audit covers all48 layers and actual prepared/captured prefill, decode, and split-sampling bodies, with no failed guarded calls. The context smoke advances all QSA positions across four ranks to32770. These findings support the selected default-tile path; they do not expand the helper contracts to custom tiles, arbitrary position tensors, or untested architectures.

Compaction's failed mixed-prefill run remains a rejected experimental branch. Its passing standalone bit-pattern probe cannot be transferred to selected production, and its unresolved trace/LLK-conditioning hypotheses are not findings against the reverted selected path.

[Final B32 HF/control v1](final_batch32_hf_v1.json) passes two2704-prediction
runs (baseline/migration), top1=.978180473 and top5/top100=1. Matched-history
logits/tokens and final request states agree exactly, with dormant/inactive
controls and resident sampled feedback passing. Scope is C512,32 correlated
continuations from one203-token AIME prompt; it makes no throughput or32-diverse-
prompt claim. Final resident timing, selected profiler/lower-bound refresh,
independent review and the strict performance floor remain separate gates.
