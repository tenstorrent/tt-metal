# Stage09 cache adapter repair

Diagnosis: AUTODEBUG_cache.md confirms upstream scheduler blocks32/64 and shared tensor indices. The adapter incorrectly required32 and allocated one pool per layer.

Repair: generator_vllm.py now allocates one K/V pair per tensor_idx, retains tile-storage aliases, expands each global64 scheduler ID into two page32 IDs, and trims scheduler right padding before expansion. No plugin/vLLM edits. Full input reload now includes page tables as the pinned decode-reload contract requires; steady calls do not refresh them. Synchronous outputs are explicitly formatted as host tensors.

Checks: host_contract.json passes noncontiguous32/64 mapping, stale input avoidance, partial-row parameter padding, RNG preservation, full reload/page coupling and unchanged-table behavior. adapter_shared_cache.log and adapter_state.json pass device alias-address checks and stale token/position output equality on layers0/5. reduced_server_shared/server.log records successful startup at context262144, one shared arena for reduced layers0/5, and completed nonaligned requests129/130/131/2177. nonaligned_requests.json contains responses. The2177 request uses an intermediate host-compatibility prefill and final device sampling. These checks are reduced-model interface evidence only.

The server was stopped through its runner after all requests completed; runner exit0, API/EngineCore shutdown logged, subsequent /proc descriptor audit empty. No resets, profiler or external publication.

Remaining validation: full-model resume and serving, allocator-driven page growth, full batch matrix, trace allocation tracking, sampling/quality/benchmarks and independent clean-pass review. AutoDebug also noted upstream clears extra_retained_tokens; arbitrary-start continuation must not rely on that requested retention surviving planning.

## Resume accuracy investigation

Default-threshold full-model check failed; baseline results retained before strict-leading repair. AutoDebug_resume.md identifies a source-backed SDPA partition candidate, not a proven numerical explanation. Reduced prefill_geometry.json establishes an earlier fresh-prefix difference: layer0 V is exact across physical32/64/128/256 versus1024, while rotated K differs (max0.015625); hidden drift follows. This narrows producer investigation to Q/K normalization, RoPE or cache conversion, and refutes claiming that only resumed attention geometry is at fault. Further discriminating probes run before choosing a change. Precision policy and gates remain unchanged.

## Shape-dependent rotary correction

The qkv stage probe (prefill_qkv_stages.json) found bit-identical input, QKV linear, Q/K norms and rotary inputs/cos/sin, but unequal rotary outputs for fresh32 versus1024. The legacy rotary_embedding factory gives core group1 a default compute configuration and group2 the requested one. The HF rotary factory honors the requested configuration for both groups. Prefill now uses the same generic rotary_embedding_hf operation family as decode, without changing the selected policy. prefill_qkv_hf_rope.json verifies exact Q/K/V and hidden outputs for fresh32/64/128/256 first32 rows versus1024 on layers0/5. Hooks passed.

The reduced natural continuation gate remains failing after this isolated correction (prefill_resume_hf_rope.json): top1 agreement at starts32/64/1029/2496 remains below0.98. Attention partition control independently holds Q, cache and tables fixed: fullQ32/K32 versus slicedQ32/K32 is exact; fullQ128/K128 differs. attention_partition.json records both layer types and unchanged cache digests. Next discriminating arm tests Q128/K32 against Q32/K32 before choosing a consistent key partition geometry. These are reduced correctness experiments, never full-model performance evidence.

Fixed K32 with the existing alignedQ128/unalignedQ32 policy passes the isolated attention comparison, but does not pass reduced continuation as a whole (prefill_resume_hf_rope_k32.json). Further probe prefill_geometry_hf_k32.json identifies exact layer0 K/V but differing hidden output for resumes32/64/128, while resume256 is exact. This disproves declaring the SDPA partition hypothesis sufficient. A first-divergence probe is queued across raw attention, projections and collectives.

Adapter stale-token/current-position and page-only-refresh residency checks pass with HF RoPE/K32 (adapter_hf_k32.log, adapter_state.json), including deferred readback. supports_async_decode is enabled for the reduced overlapped server experiment; final served allocator-growth and non-overlapped control remain mandatory before completion.

The stage-level probe resume_stages.json now isolates the remaining first divergence to the collective output: layer0 Q/K/V, raw attention, projection input and local output-projection result are exact for fresh versus resume32; output after reduce-scatter/gather differs (max1.0), propagating through residual/MLP. This is a new verified causal boundary, not a justification to relax thresholds. Investigating stable reduction geometry under selected BFP8 CCL.

Reduced async first request failed solely in JSON event serialization of NumPy lengths before prefill device execution. Normalize slot/start/end to Python int in shared chunk iterator. AST host regression verifies NumPy129 logical tokens and160 physical tokens serialize; passed. Failure preserved in reduced_async/server.log; runner exited, /proc device owner audit empty. Retry uses reduced_async_intfix/.

## Absolute prefill collective alignment

The ring reduction alternates direction at8-output-tile granularity; this TP4 width has42 output tiles per tile row, so its rounding order repeats every8 tile rows (256 tokens). This predicts the observed exact resume256 and divergent resume32/128. resume_ccl.json compares all suffix rows using unchanged real layer0 projection data: leading zero padding by start%256 is exact at starts32/64/128/256, while direct reduction, fixedworkers2 and tailpadding fail at nonzero phases.

multichip_decoder.py now preserves this absolute phase for prefill BFP8 ring reductions and removes leading rows before gather. Decode and selected policy are unchanged. Generator preparation warms all pad/reduce/slice combinations before traces, sharing seven zero buffers across compatible layers (nominal5,117,952 bytes/chip before allocator alignment). prefill_resume_ccl_phase.json now passes all aligned offsets with top1=1; reduced unaligned132/1029 remain0.97665 below0.98. Negative controls fail as expected. Full60-layer check is running; no final gate is waived.

## Partial-page precision parity

Full60 check prefill_resume_full_ccl_phase.json passed aligned offsets at2560/16384 but failed unaligned132/1029 (top1~0.85), confirming one-token decode arithmetic was unsuitable for prefill continuation. Replace that path with one aligned32-token prefill tile: placeholder prefix inputs, real new rows, device dynamic slice of the existing KV page, BF16 where blend preserving prefix rows, and cast back to selectedcache dtype before fill. Ordinary prefill norms/attention/MLP/CCL then process the new rows; no sampled token feedback or host tensor readback is introduced. The shared row selector returns only newly ingested rows. Dynamic indices/masks are persistent and warmed on boundcache beforecapture.

dynamic_kv_slice.json passes exact extraction/blending and unchanged cache across changing pageIDs/masks on both KV geometries/allfourchips; no new programs after each initialsignature. prefill_resume_partial_merge.json passes all seven reducednatural offsets withtop1=1 and all negativecontrols failing as expected. Fullmodel rerun and serving validation remain pending. The former promptdecode helpers were removed. Tensor-indexed slice indexbuffers are now512uint32 entries so kernel fulltile reads remain in allocatedstorage.

## Sampler seed rollover and slot reload

Source audit found a real host/device state disagreement at the signed boundary. The shared sampler allocates `seeds_tt_tensor` as UINT32 (`models/common/sampling/tt_sampling.py`), and the generator advances it with `ttnn.plus_one` without a 31-bit mask. The adapter instead advanced and reconstructed seeds modulo2**31. A draw at seed0x7fffffff therefore left a different host mirror; a later slot remap or absolute-position reset could change the stochastic stream. The manual-seed kernel consumes the uint32 value, so its upper bit is not an ignored tag.

The adapter now normalizes external base seeds modulo2**31 before adding absolute positions modulo2**32, advances the mirror modulo2**32, and uploads remapped seeds as uint32. Generator per-slot seed validation/storage also uses uint32. No sampler operation, per-step kernel, or plugin source was added or changed. With the advertised262144 context and31-bit normalized base, the shared manual-seed0xffffffff sentinel is unreachable during a served request.

`python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.check_serving_host_contract` passes; `readiness_vllm/host_seed_rollover.json` records the CPU-only result. Checks cover signed-boundary advancement,32-bit wrap of the mirror, absolute-position reconstruction, negative/large external base normalization, generator uint32 uploads, and adapter remap uploads. `check_vllm_adapter.py` now includes a reduced real-layer stochastic control beginning at seedINT32_MAX, comparing steady, remapped, and reset execution and reading device seed counters for equality with the host mirror. That hardware check is prepared but has not been run for this change; the active serving device remains owned by the root agent.

Seed boundary device regression PASS: timeout1200 check_vllm_adapter, exit0, readiness_vllm/adapter_seed_rollover.log and adapter_state.json. Reduced real layers0/5 compare stochastic steady/remap/reset outputs and actual uint32 seeds across INT32_MAX; stale-host/page-only checks also pass. Devices close normally; no /proc device owners. No sampler kernel added.
