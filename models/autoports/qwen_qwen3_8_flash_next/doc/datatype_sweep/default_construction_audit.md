# Selected-default construction and final evidence audit

CPU-only source/planner audit, 2026-11-16. No runtime/source/config edits, TTNN imports, devices or heavy CPU jobs. At audit time `selected_precision_config.json` does not exist; no final-default execution is claimed.

**Concrete remaining compatibility issue:** candidates with two expert-layer BFP8 upgrades construct privately at full262144/B1, but the current public shared-cache planning expression rejects that same advertised profile. `e8_edges_lofi`, `e8_edges_hifi2`, `e8_two_hifi2` and `e8_two_hifi4` add629145600 persistent bytes/rank, leaving a conservative7100-block shared pool against8194 required blocks. A default validation or resident benchmark uses private caches and will not expose this planner rejection. If one of these policies wins, resolve the planner/measurement discrepancy using evidenced runtime allowance/shared-pool fit; do not lower advertised context or silently ignore the weight delta. This issue does not affect the inherited or tested memory-saving projection policies in the table below.

## Construction path and exact assertions

`build_generator(model_dir, mesh, max_seq_len=262144, max_batch_size=1)` without `precision_config` calls `load_policy(model_dir, None)`. That loader reads the selected JSON when present and validates every field; it falls back to inherited only when absent. The resolved policy is passed to `QwenModel`, which resolves layer exceptions before constructing all48 decoders. Direct default `QwenModel` construction likewise reads the selected file from the autoport root. No environment precision override exists.

After writing the final selected artifact and before running final jobs, require the following assertions in a compact host/default proof record (the existing harness summaries supply the measured runtime objects):

1. Selected JSON exists. Its complete validated dictionary equals the winning candidate's `runtime_precision.requested`, including config ID, all weight groups, every layer override/fidelity, activation/residual/CCL/KV/logits/sampling fields and fixed assumptions. Record selected-file SHA256 before and after each final job; require equality. Existing executable-source snapshots do not include this JSON.
2. Final commands/build call contain **no `precision_config` override**. Both validation `runtime_precision.requested` and benchmark `perf.runtime_precision.requested` equal the selected JSON in full. Equality of config ID alone is insufficient. Record all48 ascending layers and verify actual group tensor dtypes/compute configs against resolved policy as in `precision_propagation_audit.json`.
3. `generator.capacity == generator.model.config.max_position_embeddings == 262144`, `generator.max_batch_size == 1`, model layer indices are0..47. Policy `max_context` and context contract `hf_advertised_context`/`supported_context` are262144. Preserve separate historical layer-only capacity annotations; do not replace them with claims about the full model.
4. Actual cache declaration has `context=262144`, `slots=1`, `page_size=32`, `local_kv_heads=1`, `head_dim=256`, selected `paged_kv_dtype`; its12 QSA K/V tensors are distinct, `[8192,1,32,256]`, and their actual dtype matches the policy. Other recurrent/index state retains its fixed dtype. Neither current final harness serializes `cache_spec()`, so retain a separate construction snapshot if direct actual-cache evidence is wanted.
5. Compute `delta=static_weight_delta(selected)` and `static_bytes=KNOWN_STATIC_BYTES+delta`. Independently call the pure `plan_shared_kv_cache(262144,1,known_static_bytes=static_bytes,kv_cache_dtype=selected["kv_cache_dtype"])`. Serialize its ledger and verify the public `QwenGenerator.get_max_tokens_all_users(max_model_len=262144,max_num_seqs=1)` agrees with `plan.plugin_max_tokens_all_users()`. A private generator pass cannot substitute for this assertion. Preserve returned budget semantics below.
6. Compare measured loaded/prepared memory to the candidate's prior evidence with the known static delta and KV bytes. Keep helper/program residuals separate; `weight_memory_residual_audit.md` explains why a whole-allocator delta need not equal the tensor delta exactly.

The static public planner loads the selected artifact itself from the autoport root, then adds the actual weight delta and supplies its KV dtype. It has no explicit-policy argument: default-artifact checking is essential. Do not test a different explicit builder policy and assume the static planner used it.

## Pure-host full262144/B1 planning results

`KNOWN_STATIC_BYTES=28290535424`. All successful BF16 or BFP8 rows below return262176 tokens:262144 request tokens plus one32-token null page. The pinned external worker then adds one32-token output page, so the final8194 blocks must fit. This return value does **not** advertise a262176-token model context. `maximum_pool_blocks` is capacity before the method caps its return to the requested profile.

| Policy family | Actual persistent weight delta/rank | Planner static bytes/rank | Maximum pool blocks | Public return |
| --- | ---: | ---: | ---: | --- |
| inherited, proj8 fidelity switches | 0 | 28290535424 | 8700 | 262176 |
| shared4 inner layers | -39567360 | 28250968064 | 8800 | 262176 |
| QSA4 inner layers | -72089600 | 28218445824 | 8883 | 262176 |
| two BFP8 expert layers | +629145600 | 28919681024 | 7100 | **ValueError: cannot admit one full context** |
| GDN Z/A/B8 | -154828800 | 28135706624 | 9093 | 262176 |
| GR8 | -235929600 | 28054605824 | 9300 | 262176 |
| head8 | -149299200 | 28141236224 | 9079 | 262176 |
| KV8 | 0 | 28290535424 | 16376 | 262176; separately capability-rejected |

The two-upgrade rejection is conservative planning, not proof of physical impossibility: these candidates already have private full-context allocation/trace evidence. The planner reserves2GiB runtime separately and refuses admission rather than shrinking context. Its current source has no selected-policy-specific measured reserve input at the public method boundary. This discrepancy needs work only if such a policy is selected or its public planner capability is claimed; a P1 reference can still use its measured private full-context construction without claiming shared scheduler admission.

## Final validation and token-out coverage

`run_optimized_full_model_validation --stage datatype_sweep` constructs through the normal builder with full262144/B1 and no precision override; it now records actual `runtime_precision`. Use explicit `--accuracy --resume --autoregressive --qualitative --context-smoke` selectors if all are intended: the default runs accuracy only. Required results include overall `passed`, runtime audit `passed`, main AIME top1/top5/top100 gates, resume `status=pass`, preserved default resume thresholds, nonaligned32769-token smoke with finite logits and correct positions, plus separate prompt-correct matched-HF content/blind reviews. `passed=true` only certifies execution of qualitative generation; the harness deliberately records its content verdict as pending. The final accuracy phase's timing is not the warmed sweep ranking metric and it scores AIME only; retain both-reference candidate records for selection.

For unchanged BF16 KV/layout, its32769-token smoke satisfies the requested32K–64K selected context smoke while retaining full capacity. It does not rerun a262144-token prefill and must not be described that way. A selected changed KV/layout would require the user's full max-context acceptance and corresponding nonaligned controls; this driver explicitly does not perform max-context acceptance.

`benchmark_full_model_resident --public-resident --capacity 262144` routes through `run_full_model_acceptance`, which also uses the normal builder with no precision override. It records `perf.runtime_precision`, `memory_loaded/prepared/final`, exact executable source snapshots and two distinct resident windows:

- `perf.resident_token_out`: the original optimized-model comparator, B1 128 prompt/128 generated,127 traced model+sampling steps with device token feedback; use this for a like-for-like post-selection token-out comparison.
- `perf.serving_contract_resident_token_out`: public `decode_forward`, same workload, persistent inputs, `reload_inputs=false`, `reload_page_table=false`, `read_from_device=false`, device sampling enabled. This is useful public-boundary evidence; it includes public orchestration overhead and must retain its own name.

Both assert127 model and127 sampling replays, zero captures/releases/readbacks/uploads/cache resets during timing, stable trace/input identities, matching first/final tokens versus caller-read control, unchanged inactive slots and correct QSA positions. They synchronize only at the window boundaries. `perf.ttft_ms`/`warmed_ttft_by_length` are warmed public TTFT including first sample and caller read; do not relabel the entire TTFT as no-readback. Keep caller-read decode and logits-only diagnostic throughput separate from the resident token-out headline. Final selected reporting must include TTFT, original resident speed, public-resident speed if collected, workload and counters; use post-selection token-out for later comparisons, not teacher-forcing throughput.

Together these jobs can prove selected-default model construction, accuracy/resume/context-smoke and post-selection resident performance. They do **not** invoke the public cache-budget method, allocate a scheduler-owned shared pool, prove all-batch capability, complete qualitative judging, or replace P1/final stage-review. Add the pure planning/default proof described above, final P1 selected ID, final context contract, content verdicts and independent review.

No `generator_vllm.py` or tested vLLM adapter exists in this autoport. The public builder/cache interfaces are future integration entry points. Report their selected-policy behavior without claiming tested vLLM construction, serving or integration; no such work was performed here.
