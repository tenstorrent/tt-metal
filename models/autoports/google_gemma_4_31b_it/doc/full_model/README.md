# Full model

Full-model TP4 token-out autoregressive generation on four local Blackhole p300c ASICs, **4096 input / 128 generated tokens**. Batch 1 is first; values are medians of three warmed requests with diagnostics disabled. Decode includes model and sampling traces, device token feedback and caller token readback.

| Batch / concurrent requests | TTFT (ms) | Decode tokens/s/user | Decode ms/token | Evidence |
|---:|---:|---:|---:|---|
| 1 | 659.37 | 34.22 | 29.23 | [measurement](performance_b1.json) |
| 32 (agent target) | 20677.09 | 23.18 | 43.14 | [measurement](performance_b32.json) |
| 16 | 10327.03 | 29.13 | 34.33 | [measurement](performance_b16.json) |
| 8 | 5162.31 | 32.31 | 30.95 | [measurement](performance_b8.json) |

The separate AIME24 teacher-forcing workload is 161 input / 100 output tokens: **94.56 ms TTFT and 37.38 tokens/s/user**. It includes on-device sampling and explicit reference-token feedback; it is not logits-only timing or the headline workload.

## Implementation and inherited policy

`tt/model.py` assembles 50 sliding-attention and 10 full-attention layers using this run's optimized TP4 `MultichipDecoder`. The model, tokenizer and HF reference are pinned to revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. This implementation uses HF reference semantics and generic TTNN operations; it imports no removed or stock Gemma adapter. The standard `build_generator(model_dir, mesh_device, **kwargs)` readiness import is verified by [readiness_import.json](readiness_import.json). No vLLM integration is included.

The wrapper preserves BFP4/LoFi decoder projection weights, packed sliding QKV/full QK, HiFi4/FP32 normalization, BF16 residuals, BFP8 paged caches and Ring CCL. Prefill K/V stay BF16 until cache fill; decode uses the inherited BFP8 update path. Residuals remain replicated across devices. Decode uses local eight-core width sharding; prefill uses interleaved L1 residuals for the prepared chunks of at most 1024 rows, as selected by the optimized decoder. Its two L1 gather buffers are shared across the sequential stack, with no extra inter-layer collective. The selected decoder checkpoint is `ab700638cd4aa5bfe032a566c454b94fe848757f`; this stage started on `bdb236e49bdd887c9719e7a13b35af8bdc659964`. The prior rejection ledger remains authoritative: `../optimized_multichip_decoder/candidate_summary.json`, its work log, and `../multichip_decoder/mesh_plan.md`.

The only decoder API extension supplies prepared attention page views and device chunk offsets to the existing SDPA call; the scalar API and arithmetic policy remain intact. The wrapper embedding gather uses the inherited collective barrier contract. Controlled A/B evidence in [AUTOFIX.md](AUTOFIX.md) shows why this barrier is necessary for repeatable batched prefill.

Embedding lookup uses BF16 row-major hidden shards, a gather, and the pinned HF embedding scale. Final norm uses the HF weight directly. The tied vocabulary head uses a separate BFP8 tiled transposed vocabulary shard with HiFi2/FP32 accumulation and HF softcap. The two physical representations avoid runtime repacking; their measured allocation sizes and reason for not aliasing are recorded in [memory_plan.json](memory_plan.json).

## Context, memory and request ownership

The full stack preserves **262144 tokens**. [full_context_post_barrier.json](full_context_post_barrier.json) records public non-aligned 262143-token generation, exact 262144-token prefill, and decode at input position 262143 advancing the device position to 262144. These are execution/capacity checks, not a new long-context HF accuracy score. The short readiness reference supplies the full-model accuracy gate; earlier decoder artifacts retain their long-context HF comparisons.

The public generator pads physical tails to 32 and chunks prefill at 1024. Causal masking excludes later padded rows; readiness returns only valid logits. Normal token-out prefill selects the final valid hidden row on device before the vocabulary head. Tracked 1023/1024/1025-token tests exercise both sides of the chunk boundary.

The default full-attention pool holds 262144 / 262144 / 524288 / 1048576 tokens at batches 1 / 8 / 16 / 32. The target is 32 concurrent 32768-token working sets, with typical 4096-token inputs. The pool limits aggregate live tokens; it does not promise all 32 slots at maximum context simultaneously. Sliding layers retain 2048 physical tokens per fixed slot. Request admission reserves disjoint pages rounded to the decode attention read window, and inactive rows consume no full-attention pages.

The full-stack persistent allocator snapshots, physical tensor inventory, page tables, RoPE, trace reservation and reduced transient measurements are reconciled in [memory_plan.json](memory_plan.json) and `../context_contract.json`. The 768 MiB transient budget is a conservative bound based on a measured 406 MiB reduced-layer peak plus retained batch logit tiles. It is explicitly not a measured full-stack transient peak. The separately reserved 128 MiB trace region is already excluded from reported allocator capacity.

Low-level `prefill_forward` and `decode_forward` expose bound cache/page-table handles, tokens, positions and logical prompt lengths. External handles must be bound before preparation and retained while traces live. Page ownership must agree within each attention kind. `refresh_page_tables(revision=...)` copies changed mappings and skips unchanged revisions. Decode accepts explicit per-slot inputs; inactive rows use position -1.

Low-level prefill samples all fixed rows, including inactive rows. A scheduler prefilling a subset must preserve or explicitly restore live decode inputs through the explicit-input decode API. This stage validates fixed-slot and mixed-prompt behavior; it does not claim overlapping continuous serving. `reset` clears the whole bound cache in place, so external owners must prepare before filling live requests and avoid resetting unrelated live slots. Mesh teardown belongs to the caller.

## Sampling and trace contract

[Sampling comparison](sampling_choice.md) selects common `TTSampling` over `Sampling1D` for its state and greedy contracts. A semantically greedy, shape-correct force-argmax alternative was measured and rejected as slower. No custom sampler is used.

One model trace produces persistent sampler-ready logits and advances positions on device. A separate sampling trace writes `tt_out_tok` directly into the token tensor consumed by the next model replay and advances RNG seeds. Normal token-out generation has no host argmax, full-logit readback, Python token feedback, or per-token position/page-table rebuild. Caller-visible token reads are counted separately. `host_sampling=True` is an explicit greedy compatibility mode; its first and later tokens use host logits, and it is excluded from performance measurements.

Generator-wide preparation warms reset, all 32 tail shapes, both terminal modes, decode and sampling before either trace exists. No corruptible allocation scope or exemption is used. `generator_state_b*.json` records allocation-tracked B1/8/16/32 replay with program allocations included: token feedback, coherent positions, changed/unchanged maps, mixed/inactive slots, seeded sampling in every active slot, greedy recovery, stable program counts and the same two traces. [logit_determinism.json](logit_determinism.json) adds full-stack bitwise vocabulary-logit equality across all 32 slots and repeated requests.

Scalar sampling parameters broadcast to all configured slots; lists must match the batch. Stochastic top-k supports 1 through 32, with top-p applied within that candidate set. Wider or unrestricted top-k is rejected; HF-default stochastic-distribution parity is not claimed. For crowded exact ties the common candidate sampler may choose a different maximum-valued token from HF's lowest-index choice; semantic-greedy probes verify that selected values are maxima. Finished high-level outputs stop at EOS; physical slots remain allocated until the batch completes. Fixed-length timing ignores EOS explicitly.

## Accuracy and qualitative results

The fresh pinned AIME24 chat-template reference contains 161 prompt tokens and 100 reference output positions with top-100 candidates. Prefill top-1/5/100 is **94% / 100% / 100%**; teacher-forced decode is **93% / 100% / 100%**. Both required bars pass. [readiness_accuracy.json](readiness_accuracy.json) contains the installed runner results; [reference_metadata.json](reference_metadata.json) records the fresh control and exact revision.

The autoregressive runner uses the same rendered chat tokens and pinned HF control because its generic API does not expose revision/chat-template options. Both complete AIME outputs and all six shared HF/TT qualitative controls were read. They remain coherent, without mechanical collapse or wrong-language drift. Early free-running branching and 128-token truncation prevent task-accuracy claims. The thermodynamics wording anomaly is reproduced by pinned HF on the exact TT prefix. See [qualitative_verdict.md](qualitative_verdict.md), its control artifacts and `degenerate_check.log`.

Teacher-forcing timing uses the separate 161/100 workload and still includes on-device sampling plus explicit reference-token feedback. It is not a logits-only timing and is not substituted for the 4096/128 token-out result.

## Performance interpretation and audit

Clean full-model timings include request reset, prefill and first sampling in TTFT, followed by traced model/sampling and caller token readback in decode. Each measured request has 127 model replays, 128 sampler replays/readbacks, zero token/page refreshes, one initial position refresh and zero captures. Diagnostics are disabled for timing and enabled separately for trace acceptance.

The inherited per-layer profiles give nominal stack references of 29.039 / 43.599 / 35.502 / 32.191 ms at B1 / B32 / B16 / B8. These differ in profiling regime from clean full-model host timing and are not an additive overhead decomposition or strict lower bound. In particular, subtracting B1's 29.039 ms from full token-out timing does not measure wrapper cost. [layer_stack_comparison.json](layer_stack_comparison.json) records the calculation.

The real-shape reduced profile contains only layers 0 and 5. Complete device windows measure 1.255 ms for final norm/head/softcap and 0.519 ms for the canonical greedy sampler. Sampling does not dominate full-model decode. All five `tt-perf-report` scopes completed with advice enabled; [profile_analysis.md](profile_analysis.md) explains the advice and diagnostic boundaries. No all-layer device profile was collected, and reduced/host timings are not labeled full-model device latency or rooflines.

[Runtime audit](runtime_audit.md) covers cache ownership, fallback boundaries, sampling and reset behavior. Watcher passed with 14 dumps and the inherited Ethernet instrumentation limit recorded in [watcher.json](watcher.json). [Work log](work_log.md) records commands, repairs, exact artifacts and checkpoint SHAs. Independent review is recorded in [stage_review.md](stage_review.md).
