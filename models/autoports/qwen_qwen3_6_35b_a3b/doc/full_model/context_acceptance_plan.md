# Full-model context acceptance plan

Status: source-only candidate plan. No hardware execution or full-stack maximum-context capacity claim is made by this document. The advertised context remains **262144**, and `doc/context_contract.json` is unchanged. The final runner is `tests/run_full_model_context_acceptance.py`.

After short gates pass, the parent agent runs exactly one long acceptance pass, serialized on the assigned TP4 mesh:

```bash
timeout 14400 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_context_acceptance
```

The runner loads all 40 real layers and the actual batch-one generator at context262144, prepares and captures using short fixtures, then prefills262143 tokens once. That fills positions0..262142. A token produced by the sampling trace is consumed at position262143, the final valid position. The next-position sentinel becomes262144; no second decode is executed. Prefilling262144 tokens and then decoding would request an out-of-contract position. The repeated pinned AIME token sequence is a capacity fixture, not qualitative evidence.

An existing result file makes the runner refuse to start. There is no retry, smaller-context fallback, capacity sweep, or contract edit. A hardware failure produces an incomplete acceptance artifact for investigation. If a hard capacity limit is eventually demonstrated, the parent applies the separately required single bracketing pass/fail pair and evidence-backed contract update; this script never does that automatically.

## Arithmetic before measurement

These are **per-chip** logical/tile payload bytes at batch1, derived from the actual pinned text configuration (40layers, hidden2048, vocabulary248320,30linear/10full). They exclude bank rounding and temporary peaks. Each chip holds one full-attention KV head; the two HF KV heads are duplicated across TP4 ranks. Therefore KV bytes must not be divided by four again.

| Allocation | Per-chip bytes | Basis |
|---|---:|---|
| Full-attention K/V | 2,684,354,560 | 10layers ×2 ×8192pages ×32tokens ×1localhead ×256 ×2bytes |
| Page tables | 327,680 | 10 ×8192 ×INT32 |
| FP32 recurrent state | 15,728,640 | 30 ×8localheads ×128 ×128 ×4 |
| BF16 convolution history | 368,640 | 30 ×3historyrows ×2048localchannels ×2 |
| Reset sources | 16,097,280 | One recurrent/conv zero mirror; KV pools reset in place |
| Slot snapshots | 16,424,968 | Recurrent/conv/page-table snapshots plus position/activity scalars |
| Three prefill residual workspaces | **3,221,225,472 (3GiB)** | 3 ×262144 ×2048 ×BF16, replicated |
| Shared RoPE cosine/sine | 67,108,864 | 2 ×262144 ×64 ×BF16 |
| Embedding shard | 254,279,680 | vocabulary ×512localhidden ×BF16 |
| LM-head shard | 135,086,080 | BFP8 tiles, including64-byte exponent overhead per1024-value tile |
| Final norm tiled FP32 | 262,144 | Padded32 ×2048 ×4 |
| Sampler logits | 3,973,120 | 32rows ×62080localvocabulary ×BF16 |
| Padded zero-logit row | 3,973,120 | Logical singleton still occupies32tiledrows |
| Two logits/two token output snapshots | 7,946,496 | Current depth-two output pool |
| Reserved trace region | 200,000,000 | Explicit mesh-open setting; do not double-count in allocator census |

The inherited `full_model_plan.retained_decoder_weight_bytes_per_device` estimate is18,471,895,040bytes. It is historical arithmetic, not a substitute for current measured residency: the full model shares immutable SiLU LUTs and removes unreachable inherited routing buffers. The earlier plan also describes zero sources too broadly; current large KV pools have no full-size zero mirror. Recurrent/conv reset sources and serving slot snapshots are distinct allocations and both remain resident.

Additional storage includes retained decoder weights and lookup/helper buffers, prepared expert-dispatch workspaces, CCL scratch/semaphores, common sampler/penalties/logprob state, input/activity/seed vectors, prefill transport and RoPE-index helpers, trace storage, allocator alignment, and bounded temporary activations. The runner inventories actual root allocations by unique buffer ID, groups their page payload and aligned-page arithmetic, and avoids counting shared tensors twice. It records the complete allocator census after **all** preparation, serving-state/output-pool preparation, and both live traces. It separately records loaded weight/RoPE residency and the final census. The authoritative headroom is allocator free bytes at this prepared state; inventory payload sums are explanatory and must not be added to census totals.

The selected paged-cache contract is unchanged: BF16 backing, explicit prefill cast to destination dtype, and the decode update operation's native dtype/repack contract. FP32 recurrent state is separate. TP4/EP4, BF16 replicated inter-layer residuals, selected projection/expert fidelity and CCL policy stay in force; their runtime precision-policy dictionaries and source hashes are saved.

## Acceptance evidence

The compact JSON records exactly one long prefill and one acceptance decode, context positions, valid sampled tokens, finite final logits and recurrent/conv state, source hashes, actual configuration arithmetic, prepared DRAM capacity/free bytes, memory views, program-cache counts, trace IDs, and generator counter deltas.

All diagnostic readback shapes are warmed before capture. The final KV page is checked on every chip for all ten full-attention layers: rows0..30 must be finite and populated after prefill while row31 stays zero. After the sole decode, row31 must be finite/populated and rows0..30 must be unchanged. Decode runs under `device_only`, receives `tokens=None`, disables input refresh and token reads, and must perform one model/one sampler replay with zero token, position, page-table refreshes or in-loop readbacks. Position becomes262144. Cache/logit reads are explicit acceptance diagnostics outside the decode loop, never model computation or token feedback.

Full-model maximum-context numerical agreement is not inferred from these finite/state checks. Accuracy gates and matched HF controls are separate, previously required evidence. This pass validates the full stack's physical capacity, legal boundary position, nonaligned public prefill, and traced device feedback at that boundary.

Before first execution, independent CPU/source review checked the current model/generator APIs and evaluated the arithmetic against the pinned HF configuration. It strengthened final-page validation to require each of the31 visible rows to be populated, and failure recording now precedes cleanup. Cleanup failures are saved separately, both cleanup operations are attempted, and an original acceptance exception is preserved. No hardware run was performed for this review.
