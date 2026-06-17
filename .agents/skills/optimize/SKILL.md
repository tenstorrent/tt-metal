---
name: optimize
description: Optimize per-device performance of runnable TTNN code, preserving correctness while improving layout, precision, sharding, program configs, data movement, and warmed latency with tt-perf-report evidence except in vLLM serving stages, where profiler collection is intentionally skipped.
---

# Optimize TTNN code

This skill assumes you have runnable TTNN code with passing correctness tests. If not, first use the appropriate bringup or debugging skill. This guide is written for autoregressive LLMs with prefill and decode phases. If the target model differs, map each requirement to the nearest equivalent path and record that mapping; do not drop correctness or performance evidence.

This guide does not explain how to make more efficient multi-device mesh layout decisions (e.g. mixtures of TP/DP/EP) but if you have multi-device TTNN code it will make every device run as fast as it can given the existing multi-device weight layout choices.

Read the advice in `tech_reports/LLMs/llms.md`, particularly section 4 "Best practices and optimizations". In this skill we will strive to optimize *on-device* performance. For decode it is required to always measure the performance of a traced execution run; untraced/eager decode performance is not acceptable optimized evidence. Teacher-forcing decode must also use the traced path. For complete model or serving paths, avoidable host gaps are part of the optimization target and must be removed rather than merely noted. Always perform optimization using real tensor shapes, sequence shapes, batch size, sharding, and dtypes. Do not shrink hidden sizes, head counts, sequence lengths, or weight shapes just to make evidence easier to collect.

Optimization must preserve the model's context-length contract. If a change affects KV-cache dtype, cache layout, trace buffers, activation memory, CCL buffers, or any other persistent allocation, update `models/autoports/<model>/doc/context_contract.json`. Do not improve performance by lowering `max_model_len`, benchmark context, or eval context. A smaller context is valid only with device-DRAM evidence and the largest feasible context recorded.

When direct traced generator decode is already fast but vLLM/serving decode is slower, treat the gap as orchestration overhead before retuning decoder math. First fix the adapter/generator path: async decode split, nonblocking trace replay, on-device traced sampling, host readbacks, page-table/input refreshes, and fallback sampling. Keep same-harness serving before/after metrics.

A note on the term "sharding" - tt-metal uses this to mean two things. On-device sharding means sharding across the cores or DRAM banks of one device, such as L1-sharded activations or DRAM-sharded weights. Multi-chip sharding means distributing tensors across devices in a mesh. On-device sharding is in scope for this skill. When `tt-perf-report` mentions sharding, it usually means on-device sharding.

Profile warmed prefill and decode separately. Use `tt-perf-report` to find bottlenecks and suggestions for decoder, module, and non-serving full-model optimization. Try applicable advice. Keep changes that improve the target without unacceptable correctness or complexity cost. Record why rejected advice was rejected. If advice seems wrong, incomplete, or misleading, call that out as a candidate improvement to `tt-perf-report`.

When this skill is invoked as part of vLLM integration or optimized-vLLM, do not collect Tracy, `tt-perf-report`, or `TT_METAL_DEVICE_PROFILER` metrics from the live vLLM server or serving adapter. The profiler/table requirement does not apply to vLLM serving stages. Use same-harness `run_vllm_server` benchmark JSON for before/after serving performance, plus sampling, qualitative, degenerate-output, async-split, stale-input, and no-host-fallback evidence. Use earlier full-model or reduced non-serving profiles for device-op context if available; if not, record that evidence gap instead of profiling the vLLM stage.

Do not run Tracy or device-profiler collection on a full-model stack with every layer present. Full-stack profiling can create multi-GB profiler dumps, overflow device-profiler buffers, and distort the measurement. For full-model profiling, build a reduced profiling variant with one real layer of each layer kind and the real surrounding path: embeddings or input projection, the representative layers, final norm, LM head, sampling or token feedback when relevant, real KV-cache/page-table shapes, and the same trace path. Capture one warmed traced decode replay, or the smallest signposted prefill/decode window that answers the question. Use this reduced-layer profile for `tt-perf-report`; use the complete model only for end-to-end timing and correctness.

Run watcher and profiler evidence as separate hardware runs for non-vLLM optimization. Do not combine `TT_METAL_WATCHER` with device-profiler collection. Do not run profiler evidence in vLLM serving stages at all. On T3K, the dangerous pattern seen in Phi-3.5 Mini experiments was: a vLLM/serving profiler failure or watcher failure, followed by a full in-process 32-layer serving-adapter profile under device-profiler env such as `TT_METAL_DEVICE_PROFILER=1`, `TT_METAL_PROFILER_CPP_POST_PROCESS=1`, `TT_METAL_PROFILER_MID_RUN_DUMP=1`, `TT_METAL_PROFILER_TRACE_TRACKING=1`, and `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000`, then explicit `ttnn.ReadDeviceProfiler(mesh)` readback. With signatures such as `Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, or ARC lock/readback waits, this can leave the T3K undiscoverable: `tt-smi -ls --local` hangs and `tt-smi -r` may hang. If this happens, stop profiler collection, preserve the logs, mark the evidence `hardware-profiler-limited`, and run the T3K reset recovery procedure below before declaring the optimization stage blocked.

## T3K Reset Recovery

ARC, ERISC, remote Ethernet, or `tt-smi` discovery/reset failures during optimization are recoverable infrastructure events until the recovery steps below fail. Do not mark the model implementation blocked just because a board is temporarily undiscoverable after watcher, profiler, serving, or reset trouble.

When you see signatures such as `Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, ARC lock/readback waits, `tt-smi -ls --local` hanging, or a failed `tt-smi` reset:

1. Stop only the risky or stale test/server/profiler processes for this run. Preserve `CODEX_HOME`, repo state, multigoal logs, work logs, README files, benchmark JSON, and reduced profiler outputs. Do not delete authenticated config or successful stage evidence.
2. Do not collect more Tracy, watcher, device-profiler, serving-adapter profiler, or `ttnn.ReadDeviceProfiler(mesh)` evidence while the card is unhealthy.
3. Run a bounded reset from the host:

```bash
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
```

4. If all eight devices are visible, verify a minimal source-backed mesh open/close before resuming optimization:

```bash
python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

5. If reset or the mesh smoke fails, ask the monitor/operator for a host reboot and reservation re-acquire. If you have direct experiment-monitor authority, reboot the host, reacquire the same or equivalent T3K reservation if needed, restore the run root or preserved `CODEX_HOME`, and repeat the device list plus mesh smoke check.
6. After recovery, resume the same optimization stage from the preserved run state. In the multigoal bringup flow use `--resume-stage <stage_number>` instead of restarting earlier completed stages. Verify the resumed objective still names the exact target model and expected stage skill.
7. Record the recovery in the stage work log: failure signature, commands run, whether reset or reboot was required, final `tt-smi -ls --local` health, mesh smoke result, and resumed stage/thread. This is infrastructure evidence, not a model correctness or performance result.

Keep large raw Tracy/profiler dumps out of copied-back artifacts after a recovery. Preserve compact evidence such as `*_perf_report.txt`, `*_perf_report.csv`, reduced summaries, benchmark JSON, logs, and READMEs. Raw multi-GB Tracy CSVs are not worth destabilizing the node or evidence copy.

For decoder or module-level optimization, do not use a blunt global dtype policy. Start with a named precision/fidelity policy and tune tensor groups separately: attention weights, MLP/expert weights, KV cache, activations/residuals, CCL communication, norms, logits, and layer exceptions. Use the fallback policy below as the starting point unless an earlier stage has already selected a faster correct policy. Move one tensor group at a time.

Then tune precision and fidelity one group at a time so regressions can be assigned. For precision tuning always use real weights and recorded input activations; synthetic weights and activations are not representative enough to veto a policy. A common fallback starting point, when no prior policy exists, is BF16 activations and norms, BFP8 attention/MLP weights, BFP8 KV cache if PCC allows it, and selective BFP4 trials for MLP/expert weights.

If a prior-good policy fails in the generated code, debug the mismatch before discarding it. Check loader grouping, tensor layout, KV-cache update math, scale/transpose handling, and whether the validation harness is exercising the same full-model policy. For KV-cache precision, compare cache shape and mapper as well as dtype. Local-head replicated caches, global-head sharded caches, page-table distribution, and `paged_fill_cache`/`paged_update_cache` input dtype restrictions are different contracts. Lower-precision cache fill should cast the prefill K/V fill tensors to the cache dtype before `paged_fill_cache`; decode update tensors should stay BF16/FLOAT32 for `paged_update_cache`.

When optimizing a complete full model in the repo-local autonomous bringup flow, keep the main focus on full-model parallelism, tracing, sharding, data movement, program configs, compute-kernel configs, and removing host boundaries. `$datatype-sweep` owns the final accuracy/performance frontier, but this pass must still try targeted precision/fidelity changes when the measured full-model decode is materially below a credible target and the decoder-layer roofline says reduced precision could be the difference. Do not reject such work as "datatype sweep" by default. Try small, evidence-backed policies such as MLP gate/up BFP4, selected layer exceptions, KV/cache/CCL dtype changes, or compute-fidelity changes, then validate on the same traced full-model token-out path. Leave broad Pareto exploration to `$datatype-sweep`.

Before finishing non-vLLM optimization, review a current `tt-perf-report` output. If an applicable optimization remains untried, try it. If it fails, debug the failure. If a TTNN op or runtime limitation blocks the optimization, keep a small repro or exact failure evidence. Do not leave a known optimization for a later stage unless another skill explicitly owns it. For vLLM serving optimization, review the serving benchmark and contract evidence instead; do not create a profiler run to satisfy this paragraph.

Sometimes you will encounter a ttnn limitation or a bug. If, for example, you try an optimization and find that L1 buffers overlap (insufficient L1 space) do not take this as an excuse to give up on that optimization entirely. Instead, dive in to the code of the op and its shapes and configs and understand how you can reduce the L1 requirements in this part of the model. Or perhaps your specific shapes is not supported by the op and you need another one. Or the op does not support padding -> change the model contract so the tensors are manually padded in torch before conversion - all these things are in scope. If the failure crosses several ops, kernels, layouts, or planner/runtime boundaries and you are not making progress, use `$autofix`; it will run `$autodebug` if needed, then verify or refute each proposed bug before keeping any fix. Solve problems. Be curious. Be tenacious. Be creative. Be brilliant!

## Performance Accounting

Every optimized non-vLLM decode result must reconcile three numbers from the same run:

1. Theoretical roofline: the bytes the measured path must move per token (weights at their stored dtypes plus KV-cache reads) divided by the aggregate DRAM bandwidth of the chips used.
2. Device-time decode: per-token device time from your own signposted `tt-perf-report` window.
3. End-to-end decode: warmed measured ms/token from the host.

Report all three and use the gaps to drive implementation work: end-to-end = device time + dispatch gap + host work. Remove avoidable non-device terms before accepting the result. "The device math is fast but the loop is slow" is an unfinished optimization, not a result; a large unexplained gap between device time and end-to-end usually means an untraced path, per-step synchronization, host readback, or input-refresh overhead. Only name a ttnn/runtime/API limitation after you have tried the targeted fix and have evidence that the limitation blocks the optimized path.

The roofline fraction achieved varies legitimately by architecture - modules built from many small ops sit lower - so the explanation, not a fixed percentage, is the requirement. Name the limitations precisely; they feed the ttnn improvement backlog.

When optimizing a complete model or serving path, also write `doc/<stage>/perf_summary.json` with this shape. For vLLM serving stages, set device-time fields to `null` and name the reason, for example `vllm_serving_profiler_disabled_to_protect_hardware`.

```json
{
  "workload": {"prompt_len": 128, "gen_len": 128, "batch": 1},
  "ttft_ms": 0.0,
  "decode_ms_per_token_e2e": 0.0,
  "decode_ms_per_token_device": 0.0,
  "roofline_ms_per_token_estimate": 0.0,
  "named_limitations": ["..."]
}
```

## Full-Model Decode Closure

For optimized full-model work, first compute a target budget from the best decoder-layer evidence:

- `layer_stack_ms = sum(layer_count[kind] * optimized_multichip_decode_ms[kind])`;
- `layer_stack_tps = 1000 / layer_stack_ms` for batch-1 single-user decode;
- `full_model_overhead_ms = measured_full_model_ms_per_token - layer_stack_ms`.

If the layer-stack estimate is already slower than the target, return to decoder optimization before spending time on generator orchestration. If the layer-stack estimate can meet the target but the full model cannot, optimize the overhead explicitly before changing the mathematical core: final norm, LM head, logits movement, sampling trace, token/current-position/RoPE/page-table refresh, trace replay blocking, synchronizations, host readbacks, cache management, and CCL buffer lifetime. For token/current-position/RoPE/page-table refresh specifically, the optimized steady-state loop should use persistent device tensors, `tt_out_tok` feedback, device-side position advance for fixed-step decode, and page-table copies only when the page table changes.

### LM Head And Sampling

For models with an LM head and token sampling, treat the terminal path as part of optimized decode. A fast decoder layer stack is not enough if final norm, LM head, logits movement, sampling, or token feedback add avoidable per-token work.

Before accepting full-model token-out decode performance:

- Profile a reduced full-model token-out trace that includes final norm, LM head, logits movement, sampling, and token feedback. Measure these terms separately from the decoder-layer stack.
- Treat the LM head as a real decode matmul, not as small postprocessing. It is a hidden-size by vocab-size projection and is usually DRAM-bound.
- Keep the hidden stream in the optimized sharded layout through final norm and into the LM head when possible. Do not gather a full hidden vector or full logits tensor merely because it simplifies the wrapper.
- Split the LM head over the mesh and vocab dimension when running on more than one device. Each device should compute a shard of the vocabulary rather than every device computing replicated full-vocab logits.
- Use `models.common.modules.lm_head.LMHead1D` if you can. It is already well optimized for 1D mesh LM heads. If, after debugging shapes, sharding, and weight loading, you cannot make it support the target model, use it as the template for how to optimize your implementation.
- Use the intended LM-head weight dtype, DRAM-sharded or ring matmul program configs, and output memory config for decode. Avoid a single replicated full-vocab BF16 matmul as the default terminal implementation.
- Pad LM-head weights when needed to make the vocab-sharded decode matmul legal and fast, including DRAM-sharded matmul shapes. Keep the real tokenizer vocab size as separate metadata for sampling; the padded LM-head width is the tensor shape, not the valid token range.
- Put logits into the layout the sampler expects. Design the LM-head and sampling boundary so full-vocab all-gather is not in the hot path.
- Mask padded vocab IDs inside each local logits shard before force-argmax or local TopK. Use very negative values for token IDs `>= vocab_size`. Zero-padding LM-head weights is not a sampling mask, because those padded columns produce zero logits that can beat negative real logits.
- Make the sampler's local TopK input width friendly to the fast TopK path. On current TTNN paths, a non-power-of-two local vocab shard can fall back to a slow single-core `TopKDeviceOperation`. After invalid vocab IDs are masked, pad each local logits shard to a power-of-two width before TopK when needed, using very negative values and invalid indices for the extra power-of-two tail. Do not accept a multi-ms, one-core TopK as a final decode path.
- Use `models.common.sampling.SamplingGenerator` for token-out decode. Enable its internal trace. Pass `tt_out_tok=<persistent decode token input tensor>` so the sampled token becomes the next token input on device.
- Keep sampling trace keys distinct for greedy, penalties, and log-prob modes. Warm and capture the active mode before measuring.
- For vocab-sharded greedy decode, keep the split-sampling tensors tile-shaped. A good default is local `topk(..., k=max_top_k)`, usually `max_top_k=32`, on each vocab shard; all-gather those candidates; then pass sampling params that are semantically greedy (`k=1`, `p=0`, `temp=1`). Do not build a physical `top_k=1` per-shard path if it creates a gathered width smaller than a tile or forces a fallback. This keeps greedy behavior while avoiding full-vocab all-gather plus global argmax.
- For greedy decode, use the vocab-sharded split-sampling path above. Do not replace it with full-vocab all-gather plus global argmax because an unpadded split path is slow; fix the split path first.
- The split-sampling greedy benchmark must be semantically greedy. Do not use a generic sampled `top_k=32` or top-p-capable path as the only comparison against force-argmax. If `top_k=1` or equivalent greedy split sampling fails because of sampler shape, layout, or tiling requirements, fix that contract or keep a minimal repro and leave the stage incomplete.
- If `ArgMaxDeviceOperation`, full-vocab all-gather, generic `TopKDeviceOperation`, or sampling trace replay dominates token-out decode, fix the LM-head/sampling contract before retuning decoder dtypes or CCLs. Do not mark the optimization complete with this bottleneck still in the measured path.
- Make vLLM reuse the same optimized terminal path. Do not add adapter-side host argmax, full-logits readback, or a separate fallback sampler for serving.

For vLLM serving performance, do not profile the live server or serving adapter to split these terminal costs. Reuse the full-model or reduced non-serving terminal evidence above, then prove the serving adapter uses that path with same-harness benchmark JSON and contract checks.

The same measured path must be used for before/after comparisons. A teacher-forcing or device-logit replay number is useful, but it does not prove a token-out generator or vLLM path is fast unless it includes the same sampling and token-feedback work. Record both when they differ.

If a decoder optimization was disabled in the full model because the stacked model hit L1, semaphore, trace, or CCL limits, do not accept the fallback as final until you have tried to reduce or pool that resource. Examples include persistent CCL buffers, output buffers, ring buffers, semaphores, trace input tensors, and page-table buffers. If it still cannot fit, record the exact allocation or runtime failure and the measured cost of the fallback.

Preserve the multichip decoder's data-layout contract across the stack. If the decoder was optimized around a sharded/fractured residual stream, do not insert a layer-to-layer all-gather merely to simplify the full-model wrapper. Try fused collective/matmul or sharded-output patterns first and find a way to make the performant solution work. $autofix can help you if you are running into bugs here.

## Evidence To Leave

Final optimized evidence checklist - these items MUST be completed:

- Functional checks still pass against the optimized path.
- Prefill and decode PCC remain at the functional acceptance bar, with any material delta explained.
- Paged KV-cache and warmed trace replay still behave correctly.
- Runtime fallback audit remains clean.
- Stress or repeated-run coverage appropriate to the risk of the changes.
- Warmed prefill and decode latency before/after optimization.
- `tt-perf-report` output with advice enabled and the main performance conclusions, collected from representative decoder/module tests or a reduced full-model profiling variant, not a full all-layer model trace. This requirement does not apply to vLLM serving stages.
- Watcher still clean. Watcher should be run by setting `TT_METAL_WATCHER=10`, don't skip asserts or anything. Keep watcher runs separate from device-profiler runs. If watcher/profiler collection produces remote Ethernet, ARC, or ERISC errors and `tt-smi` starts hanging, do not retry more profiler collection; preserve compact evidence, run T3K reset recovery, and resume the same stage if the node returns healthy.
- For vLLM decode-serving optimization: same-harness vLLM before/after metrics and proof the measured path used on-device sampling without host greedy argmax or full-logits readback.
- For vLLM decode-serving optimization: no Tracy, `tt-perf-report`, live-server device profiler, or serving-adapter profiler collection was attempted; if profiler evidence is absent, record that this is intentional.
- Optimization checklist:
-[ ] Decoder path fully traced with no host fallbacks
-[ ] Decode activations generally width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
-[ ] Prefill activations generally DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
-[ ] Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives where the target model fits their contracts.
-[ ] Explicitly configured `memory_config`, `program_config`, and `compute_kernel_config` for important ops.
-[ ] Shard specs and core grids that divide tensor dimensions cleanly into tiles where possible, code grids as large as this and the model/hardware allows.
-[ ] DRAM-sharded decode matmuls.
-[ ] Fused matmul-CCL ops used where possible (or profiled and discarded with evidence).
-[ ] For MoE models: optimized the routed active-expert path with `ttnn.sparse_matmul` where the model/hardware fits, following the GPT-OSS experts pattern for sparse gate/up/down projections, routing-score weighting, expert reduction, and no dense all-expert runtime path.
-[ ] For models with an LM head and sampling: final norm, LM head, logits movement, sampling, and token feedback are included in the optimized token-out path; terminal costs are profiled separately in full-model or reduced non-serving evidence, not in vLLM serving stages; LM-head weights are padded when needed for legal/fast DRAM-sharded or vocab-sharded matmuls; padded vocab IDs are masked in local logits shards before force-argmax or TopK; split-sampling TopK input widths are padded to avoid the slow single-core TopK fallback where possible; avoidable `ArgMaxDeviceOperation`, full-vocab all-gather, generic `TopKDeviceOperation`, host argmax, and full-logits readback have been removed. If a TTNN/runtime limitation blocks removal, the stage remains incomplete until there is a minimal repro or a lower-level fix.
-[ ] LM Head is optimized for DRAM-sharded matmuls if present.
-[ ] Reduced precision/fidelity experiments appropriate to this module-level optimization stage have been carried out and documented using real weights and input activations. For complete full-model top-k tuning, final datatype frontier selection is deferred to `$datatype-sweep`.
-[ ] Performance accounting reconciled: roofline estimate, device-time decode, and end-to-end decode reported from the same run; avoidable gaps optimized away, and any remaining gap named as a ttnn/runtime/API limitation only after a targeted fix attempt; `perf_summary.json` written when optimizing a complete model or serving path. For vLLM serving stages, use same-harness serving metrics and set device-time/profile fields to `null` with the no-profiler reason.

If this checklist is not completed, go back and perform those optimization steps. For decoder/module-level work the main focus is on-device performance. For complete model and serving work, host orchestration, synchronizations, readbacks, and input-refresh overhead are also in scope and must be driven out of the measured path where the runtime contract allows it.

# Useful Optimization Knowledge

Use this reference while optimizing functional TTNN code. It captures repo-local optimization patterns and the strongest current LLM guidance. If you are not optimizing an LLM, use your best judgement about what applies in your case.

## Code Paths Worth Reading

- `tech_reports/LLMs/llms.md`: LLM memory configs, matmul variants, DRAM-sharded matmul guidance, and perf-report interpretation.
- `models/common/modules/attention/attention_1d.py`: reusable attention configs with BFP8 attention weights, BFP8 KV cache, DRAM-sharded decode matmuls, SDPA configs, and L1-sharded decode residual paths.
- `models/common/modules/mlp/mlp_1d.py`: decode/prefill MLP split, DRAM-sharded decode matmuls, sharded outputs, and precision knobs.
- `models/common/modules/lm_head/lm_head_1d.py`: reusable LM-head output projection with vocab splitting, LM-head dtype, DRAM-sharded weight memory config, input/output memory configs, and decode program config.
- `models/common/tests/modules/lm_head/test_lm_head_1d.py`: expected LM-head construction, weight splitting, memory config, and PCC checks.
- `models/common/tensor_utils.py`: helpers to serialize program and compute-kernel configs for artifact reporting.
- `models/common/sampling/generator.py`: reusable on-device sampling, internal trace capture/replay, force-argmax trace keying, and `tt_out_tok` feedback.
- `models/demos/gpt_oss/tt/experts/README.md`, `models/demos/gpt_oss/tt/experts/decode.py`, `models/demos/gpt_oss/tt/experts/prefill.py`, `models/demos/gpt_oss/tt/experts/weights.py`, `models/demos/gpt_oss/tt/experts/config.py`, and `models/demos/gpt_oss/tt/topk.py`: default routed MoE active-expert path using `ttnn.sparse_matmul`.
- `models/demos/gpt_oss/tt/`, `models/demos/gemma4/tt/`, and `models/demos/deepseek_v3/tt/`: model-specific examples where common modules do not fully fit.

## Core Optimization Rules

- Your initial functional test suite remains the correctness floor. Rerun the same functional prefill, decode, PCC, paged KV-cache, determinism, stress, trace, and watcher checks against the optimized path before accepting performance wins.
- Avoid data movement before tuning math. A slightly smaller core grid can beat a faster individual op if it avoids resharding between ops.
- Decode activations should generally stay width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
- Prefill activations are usually large and often belong in DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
- Use SDPA/FlashDecode/FlashAttention ops instead of hand-built attention primitives when the target model fits their contracts.
- Explicitly configure `memory_config`, `program_config`, and `compute_kernel_config` for important ops. Defaults are often correct but suboptimal.
- Choose shard specs and core grids that divide tensor dimensions cleanly into tiles. Padding in sharded paths is a common source of bugs or wasted work.
- For DRAM-sharded decode matmul, weights should be width-sharded in DRAM and activations/outputs width-sharded in L1 on the matching core grid.
- Keep the optimization target single-user prefill/decode. For MoE decoders on non-Galaxy systems, preserve gate-selected active-expert execution and prefer the GPT-OSS `ttnn.sparse_matmul` path for sparse expert projections plus score weighting and expert reduction. Dense all-expert execution is a debug baseline, not the optimized target.

## Matmul Choices

- Decode matmuls with small activations and large weights are usually DRAM-bound. Use `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
- Prefill matmuls with large M and N are usually compute-bound. Use `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` over a large 2D grid.
- `in0_block_w` should be at least 2 when possible and must divide the tiled K dimension. Higher is usually better until L1 pressure or correctness fails. If the only valid `in0_block_w` is 1, try a different shard-spec core count that allows 2, even if it uses fewer cores. For DRAM-sharded matmuls, the compute core count is fixed by the op, so the input/output shard grid may be more flexible than it first appears. Padding weights can be worth trying when it enables a better block size, but changing the shard spec is usually preferable.
- Output subblock size should usually be at least `2x1` or `1x2` when legal.
- If any `in0_block_w` or output subblock sizes are <2 for a matmul that is a non-trivial percentage of the runtime, call them out explicity in your final output summary and list the exhaustive set of things you tried to enable a value >=2 and why they failed.
- If an op runs out of L1, first try to increase the core count. If that's not possible, reduce `in0_block_w`, `out_subblock_h`, or `out_subblock_w` and see which combination preserves the most performance whilst avoiding the L1 OOM issue.
- If `tt-perf-report` says a matmul is DRAM-bound and it is not DRAM-sharded, trying DRAM-sharded matmul is mandatory. You can usually figure out a way to make it work with resharding if necessary and it's also usually worth it. If you find it is not a performance win, record that you tried and why it was not in your final output summary.

## Precision And Fidelity

- Start optimized decoder with BF16 activations and BFP8 weights. Keep norms BF16.
- Try BFP8 KV cache. Keep it if PCC remains above threshold and perf/memory improve. If BFP8 KV fails, inspect the prefill fill-cache dtype path before concluding the dtype is invalid: cache-fill tensors should be explicitly typecast to the cache dtype, while decode `paged_update_cache` inputs should remain BF16/FLOAT32.
- Try BFP4 for MLP FF1/FF3; these often tolerate BFP4 well.
- Try BFP4 for FF2/down-projection, but expect it to be more sensitive. Fall back based on PCC evidence, not preference.
- For BFP8 weights, HiFi2 is the normal starting point. LoFi may work but needs PCC evidence.
- For BFP4 weights, LoFi is expected.
- For BF16 weights or numerically sensitive operations, use HiFi4 or FP32 accumulation where PCC demands it.
- Evaluate precision changes one group at a time so regressions can be assigned to the right tensor group.
- Activation size matters for CCLs. Try using BFP8 activations and see if PCC (and final top-1/top-5/benchmark eval scores if run) remain high enough.
- Prefer fused CCL + matmuls where possible.
- Otherwise use async CCLs but be careful to ensure there are sufficient semaphores - other models have some good examples of CCL helper classes that track these.
- Always test with watcher when using async CCLs, it's easy to make mistakes that end up in data corruption or hangs.

## Compute Kernel Configs

Common Wormhole-style starting points:

```python
compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

Use the architecture-appropriate config class when optimizing non-Wormhole targets.

## `tt-perf-report`

Do not use this section to collect vLLM serving-stage evidence. vLLM and optimized-vLLM stages intentionally skip Tracy, `tt-perf-report`, and device-profiler collection to protect T3K hardware.

Install and verify in the active tt-metal environment:

```bash
python -m pip install tt-perf-report
tt-perf-report --help
```

Generate a Tracy ops CSV for a signposted window:

```bash
python -m tracy -r -p -v -m pytest <test-path> -k "<selector>"
```

If Tracy collection fails, use the device-profiler fallback and post-process:

```bash
TT_METAL_DEVICE_PROFILER=1 pytest <test-path> -k "<selector>"
python tools/tracy/process_ops_logs.py --date
```

Known tooling-failure signatures and prescribed actions - do not burn hours rediscovering these:

- Tracy enrichment failing with "too many source locations" or dropped device markers on large models: the profile was too broad. Stop it, preserve the log, and rerun a reduced-layer probe with one layer per kind instead of the full stack. Do this reduced probe up front for full-model profiling rather than waiting for the full-stack profile to fail.
- Device-profiler-enabled serving appearing in a run: this should only happen accidentally, because vLLM serving stages must not use profiler collection. Kill leftover `EngineCore`/server processes and run at most one bounded health check. If `tt-smi -ls --local` or reset hangs, or if logs show remote Ethernet/ARC/ERISC failures (`Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, ARC lock/readback waits), stop profiler collection and run the T3K reset recovery procedure. Do not run a full serving-adapter profile followed by `ttnn.ReadDeviceProfiler(mesh)`. Preserve the logs as `hardware-profiler-limited`; reboot/re-acquire is a fallback after reset fails, not the first blocker conclusion.
- Watcher overflowing the ACTIVE_ETH kernel config buffer: retry with `TT_METAL_WATCHER_DISABLE_ETH=1` and record the scoped limitation.
- Transient CCL/fabric link errors immediately after a failed multi-device run: reset devices and retry once before treating it as hardware evidence.

Copy the final CSV into the artifact directory and run:

```bash
export ARTIFACT_DIR="models/autoports/<model>/doc/optimized_decoder"
cp <ops_perf_results_*.csv> "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --csv "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.csv" \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.console.log"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --no-summary \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.txt"
```

Use the same pattern for prefill with `PERF_PREFILL` signposts and `prefill_*` filenames. If your installed `tt-perf-report` version uses different flags, run `tt-perf-report --help`, use the equivalent flags, and record the exact command. You'll have to add these signposts to your code, of course.

The `*_perf_report.txt` file is for the human-readable table. Do not redirect the stdout from a `--csv` run into that filename; `--csv` mode prints command/status boilerplate such as "Writing CSV output..." rather than the rendered report table. Keep that chatter in `*_perf_report.console.log` if it is useful for provenance.

`tt-perf-report` runs should keep advice enabled. If you also need a compact no-advice table, run that as a secondary command with a distinct filename and keep the advice-backed table in your work log and final reports.

Check time units before computing latency. Filtered `tt-perf-report` CSVs may expose `Device Time` in microseconds; raw Tracy ops CSVs often expose `DEVICE KERNEL DURATION [ns]`.

## Advice Policy

For every actionable `tt-perf-report` recommendation:

- try it. If there is a good reason to reject it, record the reason;
- record before/after latency, PCC, and any watcher or correctness issue;
- keep it if it improves the target metric without unacceptable PCC or complexity;
- reject it only with evidence, then continue optimizing the rest of the decoder.

Avoid suppressing advice in the report used to guide optimization. When applicable advice remains untried, call that out as remaining work rather than implying the optimization pass is complete.

## Final Audit Checks

- No unnecessary `InterleavedToSharded`, `ShardedToInterleaved`, `reshard`, `tilize`, `untilize`, `to_torch`, or `from_torch` in the optimized runtime path.
- Decode trace replay still measures the optimized path, not a fallback path.
- Program configs and compute-kernel configs are described in the final report or a compact structured summary.
- If `supports_async_decode=True` is advertised, the split vLLM path has been exercised: `decode_forward(read_from_device=False)`, `read_decode_output(async_read=True)`, and `process_decode_output_host`.
- On-device sampling returns device tokens/logprobs through decode; host top-1 or argmax fast paths are removed or proven unused by the measured benchmark.
- PCC covers prefill and decode for every representative layer kind.
- Optimized stress runs and passes for every representative layer kind and exercised mode; skipped stress is not a passing optimized result.
- Final perf reports cover warmed prefill and warmed decode separately, except vLLM serving stages where profiler collection is intentionally skipped.
- Watcher-clean evidence exists for the optimized correctness run.
