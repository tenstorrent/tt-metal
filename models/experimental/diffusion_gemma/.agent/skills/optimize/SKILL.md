---
name: optimize
description: Optimize per-device performance of runnable TTNN code while preserving correctness. For DiffusionGemma, use traced denoise-step and per-block evidence plus the approved synchronized-op substitute when Tracy is unavailable; never use autoregressive per-token metrics.
---

# Optimize TTNN code

## DiffusionGemma adaptation

Load `diffusion-gemma` first; it overrides the autoregressive assumptions below for the text-diffusion path.

- The optimization unit is the **denoise step** over the 256-token canvas (≤48 steps/block) plus the commit — NOT per-token autoregressive decode. Map every "per token" metric (roofline, ms/token, gen_len, t/s/u) onto per-step / per-block; report tokens-per-block / blocks-per-second, never `1000/mean_tpot_ms`. `perf_summary.json` needs new fields (`ms_per_denoise_step`, `steps_per_block`, `ms_per_block`, canvas size); the profile is not `single_user_decode`.
- Replace the entire LM-Head / greedy-argmax / split-sampling / `tt_out_tok` apparatus: the terminal path is **entropy-budget acceptance** (sort → cumsum → scatter/inverse-permutation over the canvas). These ops have no generic guidance here — add a candidate table (program configs, sharding, DRAM-vs-L1 placement, tile-friendly widths for the 256 axis) for sort/cumsum/scatter/gather/entropy.
- Roofline changes fundamentally: there is **no incremental single-token KV read**; each of the ≤48 steps re-reads weights and recomputes over the full 256 canvas against the frozen prefix. Reconcile measured device time against per-step-weight-traffic × steps.
- Keep every captured trace **shape- and operation-static** (on-device cutoff mask, tensor-valued scatter indices, warmed program cache). The shipping default replays a fixed 48-step trace. The landed opt-in `DG_DENOISE_EARLY_HALT` path shortens execution by replaying a one-step/window trace and reading one halt scalar between replays; it does not branch inside a captured trace. Under #48291 it currently halts 0/5 prompts and adds ~2% no-halt overhead, so it remains default OFF. Token-feedback tests become **canvas-feedback tests**.
- **NEVER edit `models/demos/gemma4/`**; validate with the shared-directory gate using the actual `DG_BASE_REF`, not a stale local `main`. Optimize DiffusionGemma-local code and drive the backbone through existing knobs. Evidence goes under `models/experimental/diffusion_gemma/doc/<stage>/`.
- **Read the `DiffusionGemma denoise-step optimization playbook` below before tuning any knob.**
  The July-10 **18.844 t/s @48** row is historical warmed same-shape argmax trace replay with a
  prompt-only prefix; it is not current first-request vLLM TTFT or correct growing-prefix
  multi-block throughput. Current serving benchmarks must record explicit flags and use the
  `plan.md` Part-0 execution contract. Start with `README.md`, `perf_campaign_worklog.md`,
  `traced_serving.md`, and the newest dated evidence rather than selecting a headline by value.
  The older full-canvas L1 study is historical/ineligible default evidence:
  `DG_NORM_FULLCANVAS=1` reached **20.68 t/s (+15.8%)** but remains OFF because it is not
  bit-identical, and `DG_MOE_L1` was a wash.

### Current benchmark guardrails (2026-07-17)

- A plain vLLM launch is not optimized: `DG_SPARSE_MOE`, `DG_DEDUP_ARGMAX`, and
  `DG_VLLM_TRACE` default off; K defaults to 48. Record all four settings plus Gumbel mode.
- For vLLM, split queue time, pure prefill, denoise, commit, trace capture, and replay. API-visible
  tokens/s is not the block rate.
- Pure prefill and serving prefill are different measurements. The current 64K-build pure-prefill
  artifact is `context_window_prefill_only_chunkedlong_20260713_msl65536.json`.
- `DG_PREFILL_RAGGED_LONG` defaults on: every multi-token prefill uses ragged top-8 experts and
  sequences above 4096 are processed in 4096-token slices. The 4K→16K dense-MoE cliff in the
  similarly named artifact without `chunkedlong` is a superseded pre-fix control. Set the flag to
  `0` only when intentionally reproducing that fallback.
- The July-15 quality decision supersedes blanket “expected garbage” language. Compare any
  serving garbage against prompt-correct fp32/HF and current TT controls.

This skill assumes you have runnable TTNN code with passing correctness tests. If not, first use the appropriate bringup or debugging skill. This guide is written for autoregressive LLMs with prefill and decode phases. If the target model differs, map each requirement to the nearest equivalent path and record that mapping; do not drop correctness or performance evidence.

This guide does not choose a completely new model-level parallelization strategy from scratch, such as TP vs DP vs EP for the whole model. But if you are optimizing an existing multi-device TTNN path, changing activation layout, residual layout, collective placement, fused CCL+matmul use, and helper-module decomposition is in scope. Do not treat the inherited multichip implementation as fixed when the perf report shows material collectives, resharding, or layout movement.

Read the advice in `tech_reports/LLMs/llms.md`, particularly section 4 "Best practices and optimizations". In this skill we will strive to optimize *on-device* performance. For decode it is required to always measure the performance of a traced execution run; untraced/eager decode performance is not acceptable optimized evidence. Teacher-forcing decode must also use the traced path. For complete model or serving paths, avoidable host gaps are part of the optimization target and must be removed rather than merely noted. Always perform optimization using real tensor shapes, sequence shapes, batch size, sharding, and dtypes. Do not shrink hidden sizes, head counts, sequence lengths, or weight shapes just to make evidence easier to collect.

Optimize primarily for batch-1 single-user latency, but preserve larger-batch correctness. Do not hard-code batch 1 into the optimized path, trace inputs, cache/page-table handling, sampling, or output formatting. For complete model and serving optimization, keep or add evidence that larger batches/concurrency still work, normally up to 32 when the target hardware, memory, and harness allow it.

Optimization must preserve the model's capability and context contract. If a change affects KV-cache dtype, cache layout, trace buffers, activation memory, CCL buffers, or any other persistent allocation, update `models/experimental/diffusion_gemma/doc/context_contract.json`. Do not improve performance by lowering `max_model_len`, benchmark context, eval context, or any other advertised capability. A reduction is acceptable only when a hard physical device limit prevents the advertised capability from fitting or running, such as device DRAM capacity for weights + KV/cache/state + required persistent buffers. If reduced, record the byte calculation or failed capacity probe, the largest feasible supported value, and the exact construction/serving setting that uses it.

Optimization must also preserve valid non-aligned logical sequence lengths. Faster code that only works when prompt or prefill length is divisible by a chunk, tile, block, page, or trace size is not complete; keep padding, masking, cache fill, position handling, and output slicing inside the model path.

When direct traced generator decode is already fast but vLLM/serving decode is slower, treat the gap as orchestration overhead before retuning decoder math. First fix the adapter/generator path: async decode split, nonblocking trace replay, on-device traced sampling, host readbacks, page-table/input refreshes, and fallback sampling. Keep same-harness primary single-user and CI serving-burst before/after metrics.

A note on the term "sharding" - tt-metal uses this to mean two things. On-device sharding means sharding across the cores or DRAM banks of one device, such as L1-sharded activations or DRAM-sharded weights. Multi-chip sharding means distributing tensors across devices in a mesh. On-device sharding is in scope for this skill. When `tt-perf-report` mentions sharding, it usually means on-device sharding.

Profile warmed prefill and decode separately. Use `tt-perf-report` to find bottlenecks and suggestions for decoder, module, and non-serving full-model optimization. Try applicable advice. Keep changes that improve the target without unacceptable correctness or complexity cost. Record why rejected advice was rejected. If advice seems wrong, incomplete, or misleading, call that out as a candidate improvement to `tt-perf-report`.

Before local knob tuning, do an operation-topology audit of the measured path. Read the code and perf report together, then write a small table with the current operation sequence, material repeated matmuls, material collectives, reshard/layout conversions, candidate fused or lower-movement replacements, dtype/fidelity constraints for each candidate, and the action taken. This is not a comparison to any one existing model. Derive it from the target model's dataflow: if multiple projections consume the same activation, if a collective feeds a matmul, if a matmul is immediately followed by a collective, or if tensors gather/reshard only to satisfy a local helper, that is optimization work.

For multi-device decode, audit topology as coherent families, not isolated knobs. For every material row/column-parallel boundary, compare the feasible families:

- local matmul followed by all-reduce or all-gather to a replicated output;
- matmul followed by reduce-scatter, with a later hidden all-gather only if the next op or residual contract needs it;
- all-gather followed by matmul fused into one op when a matmul consumes gathered input;
- matmul output fused with reduce-scatter or all-reduce when supported;
- residual kept sharded or fractured with distributed norm/residual ops instead of gathered to a replicated boundary.

For each candidate, record residual layout before and after, collective axis and dtype, expected bytes moved, ops removed or added, persistent-buffer use, and next-layer compatibility. If a fused or lower-movement candidate is slower, prove it was tried under a compatible residual/layout contract. Otherwise redesign the contract and remeasure before rejecting the family.

Do not measure a lower-movement collective family only after restoring the old replicated residual layout. That is a useful compatibility candidate, but it does not reject the family whose purpose is to carry a sharded or fractured residual forward. If a reduce-scatter, fused matmul+reduce-scatter, or fused all-gather-matmul candidate should win by changing the residual contract, adapt the next residual, norm, attention, or MLP boundary to consume that layout and measure the stack-compatible path. For single-layer correctness comparison, gather or convert only in the test harness outside the measured layer path, and label that boundary cost separately. Reject the lower-movement family only after this adapted path is measured slower or a minimal repro proves the required next op cannot consume the layout.

Optimize against the best correct measured path you have, not only against the original functional path. If this checkout or run already contains an earlier optimized artifact, read its compact perf summary and `tt-perf-report` tables before accepting a new candidate. Also look for same-model, same-stage, same-hardware-family references in the current repository, run root, or provided experiment artifacts. Use them as comparators, not as source code to copy: the final report must say whether such a reference was found, what headline latency and dominant op rows it achieved, and whether the new result is materially slower. If a same-model optimized reference exists and the new result is materially slower, the stage is not complete unless the report proves an intentional workload difference, incompatible contract, correctness failure, or TTNN/runtime blocker with exact evidence. If there is no earlier optimized artifact or reference, keep a candidate table during this stage. A candidate is not accepted just because it has fewer ops or a nicer topology; it must beat the best correct candidate for the target workload, normally traced warmed decode for decoder work. If a topology change forces a worse dtype, fidelity, layout, or program config, compare that full candidate against the dtype-compatible separate path and keep the faster correct path. Do not reject a dtype, fidelity, fusion, or CCL candidate only because it loses in isolation; if the candidate changes communication volume or residual layout, measure it as part of the compatible topology family.

Before finalizing, rerun the selected default path with the same evidence harness used for candidates. The headline optimized numbers are the final default-run numbers, not the best candidate run copied from an earlier environment. If the final default run is materially slower than the candidate it is meant to preserve, fix the default wiring or explain the difference before completing the stage.

When this skill is invoked as part of vLLM integration, do not collect Tracy, `tt-perf-report`, or `TT_METAL_DEVICE_PROFILER` metrics from the live server. This checkout has no generic readiness runner; use the direct server and block-serving harness recorded in `models/experimental/diffusion_gemma/doc/vllm_integration/`. Compare prefill+block-0 TTFT, block latency, blocks/s, and tokens-per-block/s under the same launch and request shape. Keep on-device sampling, qualitative, multi-block, cleanup, and no-host-fallback evidence. Use earlier non-serving profiles for device-op context.

Every optimization stage that can generate text must preserve prompt-correct qualitative behavior. From optimized-full-model onward, use `qualitative-check` to rerun the shared qualitative prompt suite after selecting the optimized path. If text quality changes after an optimization, compare against a prompt-correct HF or previous-stage control before blaming the checkpoint.

Do not run Tracy or device-profiler collection on a full-model stack with every layer present. Full-stack profiling can create multi-GB profiler dumps, overflow device-profiler buffers, and distort the measurement. For full-model profiling, build a reduced profiling variant with one real layer of each layer kind and the real surrounding path: embeddings or input projection, the representative layers, final norm, LM head, sampling or token feedback when relevant, real KV-cache/page-table shapes, and the same trace path. Capture one warmed traced decode replay, or the smallest signposted prefill/decode window that answers the question. Use this reduced-layer profile for `tt-perf-report`; use the complete model only for end-to-end timing and correctness.

Run watcher and profiler evidence as separate hardware runs for non-vLLM optimization. Do not combine `TT_METAL_WATCHER` with device-profiler collection. Do not run profiler evidence in vLLM serving stages at all. Use `tt-device-usage` for general TT command serialization, reset/list retries, hang triage, and ARC/ERISC/remote-Ethernet recovery. On T3K, the dangerous pattern seen in Phi-3.5 Mini experiments was: a vLLM/serving profiler failure or watcher failure, followed by a full in-process 32-layer serving-adapter profile under device-profiler env such as `TT_METAL_DEVICE_PROFILER=1`, `TT_METAL_PROFILER_CPP_POST_PROCESS=1`, `TT_METAL_PROFILER_MID_RUN_DUMP=1`, `TT_METAL_PROFILER_TRACE_TRACKING=1`, and `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000`, then explicit `ttnn.ReadDeviceProfiler(mesh)` readback. With signatures such as `Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, or ARC lock/readback waits, this can leave the T3K undiscoverable: `tt-smi -ls --local` hangs and `tt-smi -r` may hang. If this happens, stop profiler collection, preserve the logs, mark the evidence `hardware-profiler-limited`, and run the T3K reset recovery procedure below before declaring the optimization stage blocked.

## T3K Reset Recovery

ARC, ERISC, remote Ethernet, or `tt-smi` discovery/reset failures during optimization are recoverable infrastructure events until the recovery steps below fail. Do not mark the model implementation blocked just because a board is temporarily undiscoverable after watcher, profiler, serving, or reset trouble.

When you see signatures such as `Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, ARC lock/readback waits, `tt-smi -ls --local` hanging, or a failed `tt-smi` reset:

1. Stop only the risky or stale test/server/profiler processes for this run. Preserve agent config and state, repo state, stage logs, work logs, README files, benchmark JSON, and reduced profiler outputs. Do not delete authenticated config or successful stage evidence.
2. Do not collect more Tracy, watcher, device-profiler, serving-adapter profiler, or `ttnn.ReadDeviceProfiler(mesh)` evidence while the card is unhealthy.
3. Run a bounded list/reset/list sequence from the host:

```bash
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
```

4. If reset returns but some expected devices or Ethernet links are missing, run the bounded reset sequence once more. If all expected devices are visible, verify a minimal source-backed mesh open/close before resuming optimization:

```bash
python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

5. If reset or the mesh smoke fails, ask the monitor/operator for a host reboot and reservation re-acquire. If you have direct experiment-monitor authority, reboot the host, reacquire the same or equivalent T3K reservation if needed, restore the run root or preserved agent config and state, and repeat the device list plus mesh smoke check.
6. After recovery, resume the same optimization stage from the preserved run state. Re-run the stage slash-commands under `commands/` in order (each stage is a `/dg-NN-...` command), picking up at the current stage instead of restarting earlier completed stages. Verify the resumed objective still names the exact target model and expected stage skill.
7. Record the recovery in the stage work log: failure signature, commands run, whether reset or reboot was required, final `tt-smi -ls --local` health, mesh smoke result, and resumed stage/thread. This is infrastructure evidence, not a model correctness or performance result.

Keep large raw Tracy/profiler dumps and generated tensor artifacts out of copied-back artifacts after a recovery. Preserve code/docs/tests plus compact evidence such as `*_perf_report.txt`, `*_perf_report.csv`, reduced summaries, benchmark JSON, logs, and READMEs. Exclude `*.tensorbin`, `*.pt`, `*.refpt`, `ops_perf_results.csv`, `prefill_decode_ops.csv`, `*_decode_ops.csv`, and raw multi-GB Tracy CSVs; they are not worth destabilizing the node or evidence copy unless Mark explicitly asks for a specific raw artifact.

For decoder or module-level optimization, do not use a blunt global dtype policy. Start with a named precision/fidelity policy and tune tensor groups separately: attention weights, MLP/expert weights, KV cache, activations/residuals, CCL communication, norms, logits, and layer exceptions. Use the fallback policy below as the starting point unless an earlier stage has already selected a faster correct policy. Move one tensor group at a time.

Then tune precision and fidelity one group at a time so regressions can be assigned. For precision tuning always use real weights and recorded input activations; synthetic weights and activations are not representative enough to veto a policy. A test named "representative semantics" is still synthetic if it uses random or synthetic tensors. Synthetic/random-weight tests may catch op crashes, shape bugs, trace replay bugs, and numerical explosions, but they cannot by themselves make a slower higher-precision policy "best correct" when the lower-precision policy passes real-weight evidence for the target model. If synthetic coverage fails while real-weight coverage passes, treat that as a discrepancy to debug: inspect the synthetic distribution and threshold, add real-weight non-aligned/paged/trace replay coverage for the same contract, and reject the lower-precision policy only on model-visible correctness loss, trace/runtime failure, unacceptable latency, or an exact op-contract blocker. A common fallback starting point, when no prior policy exists, is BF16 activations and norms, BFP8 attention/MLP weights, BFP8 KV cache if PCC allows it, and selective BFP4 trials for MLP/expert weights.

If a prior-good policy fails in the generated code, debug the mismatch before discarding it. Check loader grouping, tensor layout, KV-cache update math, scale/transpose handling, and whether the validation harness is exercising the same full-model policy. For KV-cache precision, compare cache shape and mapper as well as dtype. Local-head replicated caches, global-head sharded caches, page-table distribution, and `paged_fill_cache`/`paged_update_cache` input dtype restrictions are different contracts. Lower-precision cache fill should cast the prefill K/V fill tensors to the cache dtype before `paged_fill_cache`; decode update tensors should stay BF16/FLOAT32 for `paged_update_cache`.

When optimizing a complete full model in the repo-local autonomous bringup flow, keep the main focus on full-model parallelism, tracing, sharding, data movement, program configs, compute-kernel configs, and removing host boundaries. `datatype-sweep` owns the final accuracy/performance frontier, but this pass must still try targeted precision/fidelity changes when the measured full-model decode is materially below a credible target and the decoder-layer roofline says reduced precision could be the difference. Do not reject such work as "datatype sweep" by default. Try small, evidence-backed policies such as MLP gate/up BFP4, selected layer exceptions, KV/cache/CCL dtype changes, or compute-fidelity changes, then validate on the same traced full-model token-out path. Leave broad Pareto exploration to `datatype-sweep`.

Before finishing non-vLLM optimization, review a current `tt-perf-report` output **when the build supports Tracy**. For the current DiffusionGemma QB2 build (`ENABLE_TRACY=OFF`), use the approved substitute: traced Metal capture/replay for ranking plus synchronized per-op/component device-time tables, and label the evidence `hardware-profiler-limited`. If an applicable optimization remains untried, try it or record an exact blocker. For vLLM serving optimization, review serving benchmark and contract evidence instead; do not create a profiler run to satisfy this paragraph.

Sometimes you will encounter a ttnn limitation or a bug. If, for example, you try an optimization and find that L1 buffers overlap (insufficient L1 space) do not take this as an excuse to give up on that optimization entirely. Instead, dive in to the code of the op and its shapes and configs and understand how you can reduce the L1 requirements in this part of the model. Or perhaps your specific shapes is not supported by the op and you need another one. Or the op does not support padding -> change the model contract so the tensors are manually padded in torch before conversion - all these things are in scope. If the failure crosses several ops, kernels, layouts, or planner/runtime boundaries and you are not making progress, use `autofix`; it will run `autodebug` if needed, then verify or refute each proposed bug before keeping any fix. Solve problems. Be curious. Be tenacious. Be creative. Be brilliant!

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
  "workload": {"profile": "single_user_decode", "prompt_len": 128, "gen_len": 128, "batch": 1},
  "ttft_ms": 0.0,
  "decode_ms_per_token_e2e": 0.0,
  "decode_ms_per_token_device": 0.0,
  "roofline_ms_per_token_estimate": 0.0,
  "named_limitations": ["..."]
}
```

The template above is the per-token `single_user_decode` shape. **DiffusionGemma does NOT use it.** The optimization unit is the denoise step over the 256-token canvas, so the summary uses `profile: "block_diffusion_denoise_step"` with per-step / per-block fields. Map every per-token field (ms/token, t/s/u, per-token roofline) onto per-step / per-block; report tokens-per-block and blocks-per-second, never `1000/mean_tpot_ms`. The diffusion FIELD shape below is illustrative and deliberately leaves changing measurements null. Populate them from the newest `doc/optimize_perf/perf_campaign_worklog.md` entries plus `l1_residency.md`, `l1_residency_summary.json`, `norm_fullcanvas_flip_gate.md`, and `early_halt.md`. Use `path_to_100tps.md` only for roadmap arithmetic and lever provenance, never as the current-performance authority; do not use the dense `perf_summary.json` numbers.

```json
{
  "workload": {"profile": "block_diffusion_denoise_step", "prompt_len": 32, "canvas_length": 256, "max_denoise_steps": 48, "batch": 1, "mesh": [1, 4], "tensor_parallel": 4},
  "ms_per_denoise_step": null,
  "steps_per_block": 48,
  "ms_per_block": null,
  "tokens_per_block_per_s": null,
  "roofline_ms_per_step_estimate": null,
  "decode_ms_per_token_device": null,
  "named_limitations": ["model-faithful @48 ceiling (early-halt blocked by #48291); hardware-profiler-limited: ENABLE_TRACY=OFF, tt-perf-report op-CSV unavailable; per-step/per-block/full-generation numbers come from traced Metal capture/replay plus synchronized per-op device-time tables"]
}
```

Keep the FIELD shape above; take the NUMBERS from the current docs, not from the shipped `doc/optimize_perf/perf_summary.json` (that file is the earlier dense-MoE snapshot: ≈4176 ms/step, 137.55 ms/layer — superseded by true-sparse+tuned MoE). Set `decode_ms_per_token_device` to `null` with the profiler reason — the device-time op table cannot be collected on this build (see `Profiling without Tracy` in the playbook).

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
- Use a common sampling implementation for token-out decode. Compare `models/common/sampling/` and `models/common/modules/sampling/sampling_1d.py`, choose the one that fits this model's state, seed, topology, trace, and logprob requirements, and record the choice. Trace the chosen sampling path or a correct generator-owned wrapper around it. Pass `tt_out_tok=<persistent decode token input tensor>` so the sampled token becomes the next token input on device.
- Keep sampling trace keys distinct for greedy, penalties, and log-prob modes. Warm and capture the active mode before measuring.
- For vocab-sharded greedy decode, keep the split-sampling tensors tile-shaped. A good default is local `topk(..., k=max_top_k)`, usually `max_top_k=32`, on each vocab shard; all-gather those candidates; then pass sampling params that are semantically greedy (`k=1`, `p=0`, `temp=1`). Do not build a physical `top_k=1` per-shard path if it creates a gathered width smaller than a tile or forces a fallback. This keeps greedy behavior while avoiding full-vocab all-gather plus global argmax.
- For greedy decode, use the vocab-sharded split-sampling path above. Do not replace it with full-vocab all-gather plus global argmax because an unpadded split path is slow; fix the split path first.
- The split-sampling greedy benchmark must be semantically greedy. Do not use a generic sampled `top_k=32` or top-p-capable path as the only comparison against force-argmax. If `top_k=1` or equivalent greedy split sampling fails because of sampler shape, layout, or tiling requirements, fix that contract or keep a minimal repro and leave the stage incomplete.
- If `ArgMaxDeviceOperation`, full-vocab all-gather, generic `TopKDeviceOperation`, or sampling trace replay dominates token-out decode, fix the LM-head/sampling contract before retuning decoder dtypes or CCLs. Do not mark the optimization complete with this bottleneck still in the measured path.
- Make vLLM reuse the same optimized terminal path. Do not add adapter-side host argmax, full-logits readback, or a separate fallback sampler for serving.

For vLLM serving performance, do not profile the live server or serving adapter to split these terminal costs. Reuse the full-model or reduced non-serving terminal evidence above, then prove the serving adapter uses that path with same-harness benchmark JSON and contract checks.

The same measured path must be used for before/after comparisons. A teacher-forcing or device-logit replay number is useful, but it does not prove a token-out generator or vLLM path is fast unless it includes the same sampling and token-feedback work. Record both when they differ.

If a decoder optimization was disabled in the full model because the stacked model hit L1, semaphore, trace, or CCL limits, do not accept the fallback as final until you have tried to reduce or pool that resource. Examples include persistent CCL buffers, output buffers, ring buffers, semaphores, trace input tensors, and page-table buffers. If it still cannot fit, record the exact allocation or runtime failure and the measured cost of the fallback.

Preserve the multichip decoder's data-layout contract across the stack. If the decoder was optimized around a sharded/fractured residual stream, do not insert a layer-to-layer all-gather merely to simplify the full-model wrapper. Try fused collective/matmul or sharded-output patterns first and find a way to make the performant solution work. `autofix` can help you if you are running into bugs here.

## DiffusionGemma denoise-step optimization playbook

Load `diffusion-gemma` first, then read this before touching a knob. It grounds where the denoise-step headroom is (and is not), so you neither re-grind exhausted levers nor reach for the shared gemma4 backbone.

**Read the live perf docs first — the numbers here churn.** Use the tail of `models/experimental/diffusion_gemma/doc/optimize_perf/perf_campaign_worklog.md` plus `l1_residency.md`, `early_halt.md`, `norm_fullcanvas_flip_gate.md`, and `l1_residency_summary.json` as the current record. Use `path_to_100tps.md` only for roadmap arithmetic and lever provenance: its “current landed state” predates OPT-004, batched commit, traced denoise, and the L1 pass. The older `perf_summary.json` + `work_log.md` are the **dense-MoE snapshot** and must not supply current performance numbers.

### The current landed state (not the dense snapshot)

The denoise path has moved through three states; anchor on the latest:

1. **Dense-128 MoE** (the old `perf_summary.json`/`work_log.md` snapshot): ~137.6 ms/layer, ~4176 ms/step, dominated by an ~87 ms expert-major `Permute` + dense sparse_matmul over all 128 experts.
2. **True-sparse token-gather MoE** (`tt/sparse_moe.py`, landed): 137.6 → **10.54 ms/layer (13×)**; per step ~379 ms. The dense-128 all-ones sparse_matmul + its Permute are retired (`path_to_100tps.md` "Current landed state").
3. **Sparse MoE + OPT-004 geometry tuning** (`DG_SPARSE_MOE_TUNED=1`, **default ON**, commit `9c5c999`): tuned MoE ≈ **2.90 ms/layer** (3.47× over untuned, PCC 0.99967). At this point the step is ~258 ms and the **MoE is only ~34% of it — ~66% is now NON-MoE** (attention/SDPA/o_proj, norms, RoPE, TP all-reduce/CCL ~40 ms/step, terminal argmax+entropy over the 262144 vocab ~43 ms/step, dispatch).
4. **L1-residency pass** (`fbabe620f21` plus fidelity follow-ups): `DG_NORM_FULLCANVAS=1` removes 8× slice/norm/DRAM-concat glue and improves traced @48 **17.855 → 20.676 t/s (+15.8%)**, but is opt-in/default OFF because the BF16 reduction-order change is not bit-identical. Absolute HF comparison on one prompt/seed was fidelity-neutral (committed agreement 0.160 vs 0.168 for the default), not a population proof. `DG_MOE_L1` is bit-identical but an end-to-end wash (−0.6% @48, +0.4% @12).

**The single most important reframing: the expert matmul is ALREADY ~roofline-optimal (weight-bound at M=1 tile, ~92% of the 256 GB/s roofline when tuned) — so more MoE/dtype tuning does not help. To beat the current model-faithful ~18 t/s @48, target the ~66% NON-MoE surface, not the experts.** Two durable architectural facts behind this: (a) at S=256 with top-8 routing the canvas activates ~all 128 experts (coupon-collector `E[distinct] ≈ 128`), so the weight floor is the **all-128** bank (12.58 GB/chip/step) and top-8 sparsity buys compute/data-movement, never weight bytes; (b) bf16 expert precision is not the lever — bf8 was tried and rejected (below).

The device-backed L1 pass corrected one earlier conclusion: **de-chunking the 256-canvas RMSNorm is fresh headroom** after MoE tuning. It saves ~41 ms/step and is the largest remaining DG-local measured win, although it stays opt-in for output-identity policy. The same pass closed the other placement ideas: MoE activation L1 is overlap-hidden, residual-stream L1 requires a whole-layer contract rewrite, attention L1 is blocked by the flash-SDPA CB clash, and mask placement is immaterial.

### Already landed — do NOT re-grind

These are done or measured to closure; redoing them without new evidence wastes cycles:

- **True-sparse token-gather MoE** (`tt/sparse_moe.py`): retired the dense-128 all-ones `sparse_matmul` + its ~87 ms expert-major `Permute`; 137.6 → 10.54 ms/layer (13×). The dense-path "compute fewer experts" question is answered — this IS the fewer-experts path.
- **OPT-004 matmul-geometry tuning** (`DG_SPARSE_MOE_TUNED=1`, default ON, `9c5c999`): tuned MoE ≈ 2.90 ms/layer (3.47×, PCC 0.99967); the expert matmul now runs ~92% of the 256 GB/s roofline (weight-bound at M=1 tile — no utilization headroom left).
- **Batched commit** (default since 2026-07-04, `3d71dee`): the old 256-sequential single-token decode-append commit (~31 s/block) is NOT the live path; `DG_COMMIT_BATCHED=0` only to disable.
- **Traced denoise loop** (`d25626f` + 2026-07-13 production-Gumbel/growing-prefix increment): the historical RUN-first argmax @48 anchor was **17.92 t/s**, but that capture-once multi-block benchmark held the denoise prefix at the initial prompt and is performance provenance only. Single-step traces support materialized and bounded-memory chunked Gumbel through refreshable full-noise/device-seed inputs. Correct full-depth K=48/max_seq_len=1024 growing-prefix evidence releases/recaptures after commit and is **1.42 output t/s** (`doc/vllm_integration/traced_chunked_gumbel_20260713.json`). Dynamic Gumbel does not support grouped trace windows; eliminating prefix recapture needs paged/fixed-shape inputs; traced 256K remains a separate allocator gate.
- **Terminal trim / `DG_DEDUP_ARGMAX`**: the RUN-first path dedups the 2nd full-vocab argmax over 262144 (argmax is scale-invariant when `gumbel_noise=None`). ROW_MAJOR multi-core argmax (`tt/sampling.py:43`, 86× vs single-core TILE, bit-identical) is already wired.
- **Multi-step trace batching** (opt-in `DG_DENOISE_TRACED_MULTISTEP`, `8ce1904`): measured `+0.3%` @48 — a no-op because the step is compute-bound there. Not default; do not expect a win from it @48.
- **bf8 experts — TESTED and REJECTED** (`5df3175`): the DG-local `DG_EXPERTS_BFP8` knob fails the diffusion-decision gate (argmax agreement 0.60, committed match 0.23 → repetition) and is only ~9% faster (denoise is not weight-bound). Do NOT re-try expert precision as a speed lever.
- **Self-conditioning embedding prechunk + logits/denominator L1 — LANDED DEFAULT ON** (`DG_SELFCOND_PRECHUNK_EMBED`, diagnostic opt-out `0`; `DG_SELFCOND_LOGITS_L1`, diagnostic control `off`): prechunking removes 32 repeated embedding slices/step; the L1 chain retains each dynamic logits slice, subtract/exp, denominator reduction, and ordered denominator accumulator in L1 while numerator matmuls/accumulation stay DRAM. The final reviewed unset-default reproduction is **18.844 t/s @48**, **257.575 ms/warmed traced step**, with exact 48-step RUN-first argmax and production chunked-Gumbel decisions. Current evidence: `doc/optimize_perf/selfcond_logits_l1_e2e.json`.
- **Full-canvas RMSNorm — LANDED OPT-IN BUT INELIGIBLE AS DEFAULT** (`DG_NORM_FULLCANVAS=1`): +15.8% @48 / +23.3% @12, but its flip gate failed badly (commit agreement 0.145). Do not enable it for a precision/decision-preserving campaign.
- **MoE activation L1 — MEASURED WASH** (`DG_MOE_L1`): isolated MoE improves 3.2%, but traced end-to-end is noise. Keep default OFF.
- **Traced early-halt mechanism — LANDED OPT-IN** (`DG_DENOISE_EARLY_HALT`): correct one-scalar/window control flow, but 0/5 prompts halt under #48291 and no-halt overhead is ~2%; keep default OFF until the entropy floor changes.

### Where the remaining headroom is (and why in-repo is largely exhausted)

At the tuned default the step is **~66% NON-MoE**. The L1 pass recovered one material part of that surface through full-canvas RMSNorm. Beyond that opt-in win, the remaining concrete non-MoE lever is blocked in-repo:

- **Sharded terminal — VALIDATED in concept, BLOCKED in-repo.** The replicated full-vocab terminal does 4× redundant argmax/entropy work. A per-chip `[256,65536]` terminal measured an estimated ~7% block win, but the traced on-device cross-shard combine needs an exact 18-bit token index; Blackhole fp32 TILE reduction loses low index bits. The fix is an int32 reduction or custom terminal kernel. See `nonmoe_roofline/README.md` and the sharded-terminal entries in `perf_campaign_worklog.md`.
- **DRAM-sharded expert weights are no longer a current Python-level recommendation.** That roadmap item predates OPT-004. The tuned expert matmul already reaches ~235 GB/s/chip (~92% of practical DRAM roofline), while a second sharded expert copy would conflict with the 256K budget (only ~2.16 GiB/chip free). Revisit only with a loader-native non-duplicated layout and a batched-matmul contract that can consume it.
- **Current verdict:** the historical selected-default @48 result is **18.844 t/s** on a traced RUN-first argmax benchmark (`selfcond_logits_l1_e2e.json`) that did not grow committed-prefix visibility across blocks. The 20.7 t/s full-canvas-norm row is rejected by decision fidelity. Production chunked-Gumbel decisions and eager full-budget 256K capacity pass; correct production chunked-Gumbel tracing is validated full-depth at `max_seq_len=1024` (K=48, **1.42 output t/s including block-1 recapture**). Traced 256K remains allocator-blocked. Recovering capture-once throughput needs paged/fixed-shape growing-prefix inputs; reaching 60–100 t/s additionally requires a faithful ≤16–20-step regime, blocked by #48291.

### When the in-repo levers are exhausted: the ceiling

These are genuinely not fixable DiffusionGemma-locally or are hard device/kernel limits. Do not re-investigate them as DG-local Python knobs; use the in-repo evidence named above.

1. **At 48 faithful steps, the shipping default is ~18.8 t/s and the measured opt-in ceiling is ~20.7 t/s.** 100 t/s at 48 steps remains arithmetically impossible. The landed early-halt controller currently fires for 0/5 prompts because #48291 keeps the entropy/stability condition from converging, so the favorable ≤16–20-step regime is not yet model-faithful.
2. **#48291 decision fidelity (correctness, and it gates the step count).** The bf16 / MoE / TP=4 backbone argmax-agrees with HF only ~50% and diffusion commits the clean argmax with no cushion. Decomposition (2026-07-07): the gap is the backbone hidden, and **attention is the #1 lever** — full-fp32 attention alone lifts logits PCC to ≥0.92. Config precision knobs are DEAD (`sparse_matmul` + flash SDPA ignore `fp32_dest_acc_en`; HiFi4 is worse). The clean fix is a **C++ flash-SDPA kernel change** (fp32 softmax/PV accumulation) + fp32 qkv/o projections — scoped shared/upstream kernel work, NOT a DG-local Python knob. `ttnn.topk` is bf16-only (TT_FATAL on FLOAT32); fp32 experts exceed QB2 DRAM.
3. **The MoE is already ~roofline-optimal; the remaining MoE gap is a fused kernel (upstream).** At the tuned state the expert matmul reads the weight bank at ~92% of the 256 GB/s roofline (weight-bound at M=1 tile). The ~3.6 ms/layer of dispatch + gather/combine + all-reduce overhead in `tt/sparse_moe.py` (dense gather/combine matmuls because TTNN has no gather-experts primitive) is the residual — removing it needs a **fused gather-experts-combine kernel (upstream ttnn)** or a per-token/down-layout `sparse_matmul` variant (upstream). At S=256, top-8 activates ~all 128 experts, so the weight floor is all-128 (12.58 GB/chip/step) — sparsity never buys weight bytes.
4. **The commit still rides the ungated shared-gemma4 decode footprint.** Batched commit is the live default (landed, above), but a fully DG-local **sparse causal 256-token commit** (`path_to_100tps.md` lever 7) or reverting the ungated decode-footprint edits (RoPE-per-user, SDPA 1×1 grid k=32) touches shared gemma4 → needs a gate/rebaseline or copy-into-DG (plan.md R0.4 / R-new / line 149).
5. **Hard device limits (mitigated, not on the critical path).** Full-vocab on-device MATERIALIZED Gumbel `[1,256,262144]` fp32 does not fit DRAM — mitigated by chunked Gumbel (`sampling.py:165`) + RUN-first argmax (`gumbel_noise=None`). The sharded-terminal ~7% lever is blocked by the on-device 18-bit-index fp32-reduction wall (above) — needs ttnn int32 reduction or a custom kernel.

Any of the above requires a SCOPED, separately-owned shared-backbone/upstream change — never an in-place `models/demos/gemma4/` edit (hard rule #1) — or a product decision on #48291. Do not present them as DG-local Python work still to be done.

### Profiling without Tracy

The build has `ENABLE_TRACY=OFF` (`build_Release/CMakeCache.txt: ENABLE_TRACY:BOOL=OFF`, confirmed). So `TT_METAL_DEVICE_PROFILER=1`, `python -m tracy`, and `tt-perf-report` op-level CSV all raise `TT_FATAL: ... requires a Tracy-enabled build`. Enabling requires a full ttnn rebuild that would replace the shared venv's `_ttnn` bindings on a shared QB2 — an environment limit, not in-repo fixable. **Do NOT attempt that rebuild to satisfy the generic `tt-perf-report` paragraph.** For this stage the op-CSV requirement is *substituted*, not skipped, and the substitute is a legitimate measured path:

- Metal TRACE capture/replay itself WORKS — so the per-step, per-block, and full-generation numbers ARE from a traced measured path (not eager). Only the Tracy op-attribution CSV is unavailable.
- Substitute the op-CSV with: **synchronized per-op device-time tables** (`time.perf_counter` + `ttnn.synchronize_device` around each op; work_log 2a/2b/2e), the **reduced-layer per-layer sweep** (work_log 3a/3c), and the **traced e2e terminal microbench** (work_log 2d). Together these give the same "which ops dominate + before/after" signal a `tt-perf-report` table would.
- The prof/bench scripts still `import from tracy import signpost` and document a `python -m tracy -r -p` invocation, but that op-table path cannot run with `ENABLE_TRACY=OFF`; the recorded numbers come from the `time.perf_counter` + `ttnn.synchronize_device` path.
- Mark the evidence `hardware-profiler-limited` and note that trace capture/replay works. Set `decode_ms_per_token_device` (or the per-step device-time field) to `null` with this reason in `perf_summary.json`.
- `artifacts/*.log` are NOT in the committed tree (only the `.py` harnesses + `work_log.md` + `perf_summary.json` + `README.md`), so the raw run logs backing the tables cannot be re-read from this checkout; the numbers are as recorded in `work_log.md` / `perf_summary.json`.

### Denoise-step evidence artifacts

Read before acting; reproduce before trusting. All under `models/experimental/diffusion_gemma/doc/optimize_perf/`. **Read the current-state docs first; the dense-stage ones are superseded:**

Current authoritative record:
- `perf_campaign_worklog.md` — read newest entries first; it records the dense→sparse→tuned evolution, early-halt, and the L1 pass.
- `l1_residency.md` / `l1_residency_summary.json` — default ~18 t/s, `DG_NORM_FULLCANVAS` 20.68 t/s @48, and `DG_MOE_L1` wash.
- `norm_fullcanvas_flip_gate.md` — why the norm win remains opt-in despite one-prompt absolute HF neutrality.
- `early_halt.md` — landed controller, ~2% no-halt overhead, and 0/5 halts under #48291.
- `path_to_100tps.md` — roadmap arithmetic and historical lever inventory only; its starting-line state is stale.
- `perf_progress.md` — earlier sparse-MoE evolution and supporting measurements.
- `path_to_30tps.md`, `multistep_trace_batching.md`, `landed_levers_47465_comment.md` — the stacked-lever roofline, the multi-step no-op finding, and the landed-lever summary.

Earlier dg-08 **dense-MoE snapshot (SUPERSEDED — do not quote as current):**
- `perf_summary.json` / `work_log.md` — the dense ≈4176 ms/step, 137.55 ms/layer state before true-sparse MoE. Useful only for the diffusion-shaped `perf_summary.json` FIELD shape (`profile: block_diffusion_denoise_step`), not for its numbers.
- `prof_denoise_step.py`, `bench_sampling_step.py`, `diag_sampling_ops.py` / `diag_argmax_alt.py` / `verify_trace_safe_loop.py` — the reduced-layer fit + terminal/op-diagnosis + trace-safety harnesses.

## Evidence To Leave

Final optimized evidence checklist - these items MUST be completed:

- Functional checks still pass against the optimized path.
- Prefill and decode PCC remain at the functional acceptance bar, with any material delta explained.
- Paged KV-cache and warmed trace replay still behave correctly.
- Runtime fallback audit remains clean.
- Stress or repeated-run coverage appropriate to the risk of the changes.
- Warmed prefill and decode latency before/after optimization.
- `tt-perf-report` output when Tracy is available; otherwise the documented DiffusionGemma substitute (traced Metal ranking plus synchronized per-op/component tables, labeled `hardware-profiler-limited`). This requirement does not apply to vLLM serving stages.
- Watcher still clean. Watcher should be run by setting `TT_METAL_WATCHER=10`, don't skip asserts or anything. Keep watcher runs separate from device-profiler runs. If watcher/profiler collection produces remote Ethernet, ARC, or ERISC errors and `tt-smi` starts hanging, do not retry more profiler collection; preserve compact evidence, run T3K reset recovery, and resume the same stage if the node returns healthy.
- For vLLM decode-serving optimization: same-harness primary single-user and CI serving-burst vLLM before/after metrics, and proof the measured path used on-device sampling without host greedy argmax or full-logits readback.
- For vLLM decode-serving optimization: no Tracy, `tt-perf-report`, live-server device profiler, or serving-adapter profiler collection was attempted; if profiler evidence is absent, record that this is intentional.
- For optimized-full-model and vLLM-serving optimization: `qualitative-check` evidence for the shared qualitative prompt suite after the selected optimization, with HF or previous-stage controls.
- Optimization checklist:
-[ ] Decoder path fully traced with no host fallbacks
-[ ] Decode activations generally width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
-[ ] Prefill activations generally DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
-[ ] Operation-topology audit completed: current op sequence, repeated same-input matmuls, collectives, reshard/layout conversions, candidate fused/lower-movement replacements, dtype/fidelity constraints, and action taken are recorded.
-[ ] Multi-device topology candidates were measured as coherent families when applicable: residual layout, collective placement, fused CCL+matmul use, projection packing or separation, activation/CCL dtype, and persistent-buffer use. A rejection measured only under an incompatible residual/layout contract does not complete this item.
-[ ] Lower-movement residual candidates were measured without an immediate old-contract restore when applicable. If a reduce-scatter or fused CCL+matmul path only lost after an immediate all-gather or full replication, a stack-compatible sharded/fractured residual path was also measured or a minimal repro proves the next op cannot consume that layout.
-[ ] Best-candidate comparison completed: the final path is compared against the strongest available correct baseline, earlier optimized artifact, same-model same-stage reference when available, and material candidates from this stage. The final choice wins traced warmed decode or has an explicit target-specific reason for prioritizing another workload. A synthetic-only precision veto does not count as a correctness reason when real-weight evidence passes. A geometry sweep measured only under a different dtype/fidelity does not reject the final dtype/fidelity policy.
-[ ] Final default performance reproduced the selected best candidate under the final code path. If the final default is slower, the report uses the final number and explains why the candidate was not preserved.
-[ ] Final dtype/fidelity policy is verified in the measured runtime rows, not only in policy JSON or constructor defaults. For each dominant matmul, the `tt-perf-report` row or an equivalent profiler artifact must show the expected input/weight dtype and math fidelity. If the row shows BF16 or BFP8 where the selected policy claims BFP4, the policy did not reach the measured op and the stage is incomplete.
-[ ] Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives where the target model fits their contracts.
-[ ] Fused or packed repeated same-input projections where legal and beneficial, such as Q/K/V-style projections, paired gate/up projections in 3-matmul MLPs, or other model-specific projection groups. If kept separate, there is measured evidence or a specific unresolved TTNN/runtime blocker after adapting layout, rank, padding, weight packing, and output splitting. If kept packed, it wins against a well-tuned legal separate candidate after counting split, activation, binary elementwise, and layout overhead, or the evidence explains why the separate candidate is invalid.
-[ ] Explicitly configured `memory_config`, `program_config`, and `compute_kernel_config` for important ops.
-[ ] For any matmul or repeated matmul group that is one of the largest decode-time consumers: swept legal program configs separately for each dominant role, including core grid, larger legal `in0_block_w` values, output subblocks, output blocks, memory configs, and compute kernel config where applicable. The stage is incomplete without a before/after evidence table or an exact TTNN/runtime blocker.
-[ ] Decode compute fidelity was swept as a real performance knob for each dominant projection group. Do not assume BFP8 implies HiFi2 is fastest; try legal LoFi and HiFi2 candidates with the same dtype and real traced decode evidence, then keep the fastest policy that passes correctness.
-[ ] Attention projection weight dtype/fidelity was swept separately from MLP weight dtype/fidelity when QKV, Q/K/V, output projection, or fused attention matmul rows are material. If attention projections remain BFP8 or BF16, the report names the BFP4 attention candidate tried on real weights or recorded real activations, plus the precise correctness, latency, or op-contract blocker.
-[ ] If dense MLP or expert matmuls are among the largest decode-time consumers: BFP4/LoFi trials for FF1/FF3 or equivalent gate/up projections were run before lower-priority prefill-only advice was pursued to completion. FF2/down BFP4 was also tried or rejected with PCC/runtime evidence.
-[ ] Shard specs and core grids that divide tensor dimensions cleanly into tiles where possible, code grids as large as this and the model/hardware allows.
-[ ] DRAM-sharded decode matmuls.
-[ ] Collective topology minimized. Avoidable gather, reshard, all-reduce, reduce-scatter, and all-gather operations have been removed, moved to cheaper boundaries, or justified with before/after evidence.
-[ ] Fused matmul-CCL ops used where possible, including fused all-gather-matmul or matmul-reduce-scatter patterns when a collective and matmul are adjacent or can be made adjacent. If rejected, the rejection includes an adapted attempt, not only the first API error.
-[ ] Repeated decode CCLs use persistent or preallocated intermediate/output buffers where the API supports it. If unavailable or slower, the reason and measurement are recorded.
-[ ] DiffusionGemma uses the tuned true-sparse token-gather `tt/sparse_moe.py` path; no retired dense-128 runtime or avoidable activation round trip was reintroduced.
-[ ] Final norm, vocab projection, entropy/Gumbel/accept terminal, canvas feedback, and commit are included in the measured block path; no host argmax or full-logits readback was introduced.
-[ ] Precision/fidelity candidates are gated on injected-noise diffusion decisions; the rejected BF8-expert result is not silently re-enabled.
-[ ] Performance accounting reconciles per-step roofline, traced step time, commit, and end-to-end block time. vLLM evidence uses direct per-block serving metrics with profiler fields marked intentionally unavailable.
-[ ] Batch/concurrency claims match current implementation: batch-1 is measured; larger values are claimed only after paged-cache ownership and batched canvas decode are tested.

If this checklist is not completed, go back and perform those optimization steps. For decoder/module-level work the main focus is on-device performance. For complete model and serving work, host orchestration, synchronizations, readbacks, and input-refresh overhead are also in scope and must be driven out of the measured path where the runtime contract allows it.

# Additional reference

For generic TTNN optimization patterns, program-config advice, and reusable model examples, read [REFERENCE.md](REFERENCE.md) only when relevant. The DiffusionGemma playbook and completion gates above remain binding.
