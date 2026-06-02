---
name: optimize
description: Optimize per-device performance of runnable TTNN code, preserving correctness while improving layout, precision, sharding, program configs, data movement, and warmed latency with tt-perf-report evidence.
---

# Optimize TTNN code

This skill assumes you have some runnable TTNN code already with passing correctness tests. If you do not, this is the wrong skill to use. Assuming you do, let's continue. This guide is written for autogenerative LLMs with prefill and decode phases. If your model doesn't look like this, you'll have to try to adapt it to your situation as makes sense.

This guide does not explain how to make more efficient multi-device mesh layout decisions (e.g. mixtures of TP/DP/EP) but if you have multi-device TTNN code it will make every device run as fast as it can given the existing multi-device weight layout choices.

Read the advice in `tech_reports/LLMs/llms.md`, particularly section 4 "Best practices and optimizations". In this skill we will strive to optimize *on-device* performance. For decode it is important to always measure the performance of a traced execution run. If there are significant op/host gaps you can note them but you should still follow the steps below to optimize on-device performance. Always perform optimization using real model shapes, do not use reduced shapes!

A note on the term "sharding" - tt-metal uses this to mean two things. On-device sharding (e.g. L1-sharded activations, DRAM-sharded weights) are sharded across the cores/dram banks of a single device (which is a grid of cores). You should absolutely consider these as in-scope for this stage! Multi-chip sharding (e.g. with a mesh mapper) is about distributing tensors across multiple devices in a mesh. Any time tt-perf-report mentions sharding it is probably talking about on-device sharding and is in-scope for you.

Profile warmed prefill and decode separately. Use `tt-perf-report` as a conversation with the hardware, not as an oracle: classify bottlenecks, try applicable advice, keep changes that improve the target without unacceptable correctness or complexity cost, and record why rejected advice was rejected. We'd like to improve tt-perf-report and its advice to be more useful so please call out potential improvements in your final report.

Tune precision and fidelity one group at a time so regressions can be assigned. A common starting point is BF16 activations and norms, BFP8 attention/MLP weights, BFP8 KV cache if PCC allows it, and selective BFP4 trials for MLP/expert weights. After that follow tt-perf-report, read the kernels, explore and be methodically creative until you are satisfied we've got everything out of the hardware that we can without rewriting the ttnn ops themselves!

Before you finish, take another look over a current tt-perf-report output. Is everything optimized that can be optimized, or were some things left deferred? If so, now is the time to take a breath and then systematically address them. After all, there *is* no "deferred". We are the optimization pass. If we defer something, it will forever be left unfinished. Now is the time to reach for our goal of a decoder that comes as close to full hardware performance as we can within the bounds of ttnn's capabilities! If there are specific ttnn op limitations preventing performance optimizations call these out in your report, we want to continue to improve it.

## Evidence To Leave

Final optimized evidence should show:

- Functional checks still pass against the optimized path.
- Prefill and decode PCC remain at the functional acceptance bar, with any material delta explained.
- Paged KV-cache and warmed trace replay still behave correctly.
- Runtime fallback audit remains clean.
- Stress or repeated-run coverage appropriate to the risk of the changes.
- Warmed prefill and decode latency before/after optimization.
- `tt-perf-report` output with advice enabled and the main performance conclusions.
- Watcher still clean. Watcher should be run by setting TT_METAL_WATCHER=10, don't skip asserts or anything.
- Optimization checklist:
-[ ] Decoder path fully traced with no host fallbacks
-[ ] Decode activations generally width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
-[ ] Prefill activations generally DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
-[ ] Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives where the target model fits their contracts.
-[ ] Explicitly configured `memory_config`, `program_config`, and `compute_kernel_config` for important ops.
-[ ] Shard specs and core grids that divide tensor dimensions cleanly into tiles where possible, code grids as large as this and the model/hardware allows.
-[ ] DRAM-sharded decode matmuls.
-[ ] Fused matmul-CCL ops used where possible (or profiled and discarded with evidence).
-[ ] For MoE models: optimized the routed active-expert path using `all_to_all_dispatch_metadata` + `moe_compute` where the model/hardware fits, including packed W0/W1/W2 weights, expert mapping, combine/reduce, and no dense all-expert runtime path.

If this checklist is not completed, take this as a sign that you should go back and perform those optimization steps to improve on-device performance. For this stage that is what we are most interested in optimizing; op/host gap will be reduced by tracing.

# Useful Optimization Knowledge

Use this reference while optimizing functional TTNN code. It captures repo-local optimization patterns and the strongest current LLM guidance. If you are not optimizing an LLM, use your best judgement about what applies in your case.

## Code Paths Worth Reading

- `models/tt_transformers/tt/model_config.py`: precision/fidelity settings, sharded activation configs, DRAM-sharded matmul helpers, prefill and decode program configs.
- `models/tt_transformers/PERF.md`: empirical precision/performance tradeoffs for Llama/Qwen/Mistral/Phi/Mixtral families.
- `tech_reports/LLMs/llms.md`: LLM memory configs, matmul variants, DRAM-sharded matmul guidance, and perf-report interpretation.
- `models/common/modules/attention/attention_1d.py`: reusable attention configs with BFP8 attention weights, BFP8 KV cache, DRAM-sharded decode matmuls, SDPA configs, and L1-sharded decode residual paths.
- `models/common/modules/mlp/mlp_1d.py`: decode/prefill MLP split, DRAM-sharded decode matmuls, sharded outputs, and precision knobs.
- `models/common/tensor_utils.py`: helpers to serialize program and compute-kernel configs for artifact reporting.
- `ttnn/ttnn/_experimental/moe_compute_utils.py`: packed W0/W1/W2 layouts, DRAM-sharded weight memory configs, shared-expert helpers, and bias variants for `ttnn.experimental.moe_compute`.
- `models/demos/deepseek_v3/tt/moe_optimized.py`, `tests/nightly/tg/ccl/moe/test_all_to_all_dispatch_metadata_6U.py`, and `tests/nightly/tg/ccl/moe/test_moe_compute_6U.py`: current routed MoE decode examples using `all_to_all_dispatch_metadata` and `moe_compute`.
- `models/common/modules/moe/tt_moe_decode.py`: preferred common MoE decode wrapper when present in the checkout.
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
- Keep the optimization target single-user prefill/decode. For MoE decoders, preserve gate-selected active-expert execution and prefer the `all_to_all_dispatch_metadata` -> `moe_compute` -> model-appropriate combine/reduce pipeline. Dense all-expert execution is a debug baseline, not the optimized target.

## Matmul Choices

- Decode matmuls with small activations and large weights are usually DRAM-bound. Use `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
- Prefill matmuls with large M and N are usually compute-bound. Use `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` over a large 2D grid.
- `in0_block_w` should be at least 2 when possible and must divide the tiled K dimension. Higher is usually better until L1 pressure or correctness fails. There can be a trade-off between the number of cores for shard spec and the value of `in0_block_w` - if you end up in the case where the only valid `in0_block_w` is `1` then definitely consider using a different number of cores to allow an `in0_block_w` of `2`, even if this means fewer cores. Note that for dram-sharded matmuls the number of compute cores is always fixed (12 on wormhole for example) regardless of the input/output shard spec core counts, which can give you more flexibility than you would otherwise assume. This setting is so important for matmul performance that it can even be worth padding the weights (although changing the shard spec is probably prefereable to that).
- Output subblock size should usually be at least `2x1` or `1x2` when legal.
- If any `in0_block_w` or output subblock sizes are <2 for a matmul that is a non-trivial percentage of the runtime, call them out explicity in your final output summary and list the exhaustive set of things you tried to enable a value >=2 and why they failed.
- If an op runs out of L1, first try to increase the core count. If that's not possible, reduce `in0_block_w`, `out_subblock_h`, or `out_subblock_w` and see which combination preserves the most performance whilst avoiding the L1 OOM issue.
- If `tt-perf-report` says a matmul is DRAM-bound and it is not DRAM-sharded, trying DRAM-sharded matmul is mandatory. You can usually figure out a way to make it work with resharding if necessary and it's also usually worth it. If you find it is not a performance win, record that you tried and why it was not in your final output summary.

## Precision And Fidelity

- Start optimized decoder with BF16 activations and BFP8 weights. Keep norms BF16.
- Try BFP8 KV cache. Keep it if PCC remains above threshold and perf/memory improve.
- Try BFP4 for MLP FF1/FF3; these often tolerate BFP4 well.
- Try BFP4 for FF2/down-projection, but expect it to be more sensitive. Fall back based on PCC evidence, not preference.
- For BFP8 weights, HiFi2 is the normal starting point. LoFi may work but needs PCC evidence.
- For BFP4 weights, LoFi is expected.
- For BF16 weights or numerically sensitive operations, use HiFi4 or FP32 accumulation where PCC demands it.
- Evaluate precision changes one group at a time so regressions can be assigned to the right tensor group.
- Activation size matters for CCLs. Try using BFP8 activations and see if PCC (and final top-1/top-5/benchmark eval scores if run) remain high enough.
- Prefer fused CCL + matmuls where possible, models/demos/tt_transformers has some good examples of these.
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

Copy the final CSV into the artifact directory and run:

```bash
export ARTIFACT_DIR="models/autoports/<model>/doc/optimized_decoder"
cp <ops_perf_results_*.csv> "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --csv "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.csv" \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.txt"
```

Use the same pattern for prefill with `PERF_PREFILL` signposts and `prefill_*` filenames. If your installed `tt-perf-report` version uses different flags, run `tt-perf-report --help`, use the equivalent flags, and record the exact command. You'll have to add these signposts to your code, of course.

`tt-perf-report` runs should keep advice enabled. If you also need a compact no-advice report for a table, run that as a secondary command and keep the advice-backed run in your work log and final reports.

Check time units before computing latency. Filtered `tt-perf-report` CSVs may expose `Device Time` in microseconds; raw Tracy ops CSVs often expose `DEVICE KERNEL DURATION [ns]`.

## Advice Policy

For every actionable `tt-perf-report` recommendation:

- try it. If there is a good reason to reject it, call this out in your summary output - we want to improve tt-perf-report's recommendations;
- record before/after latency, PCC, and any watcher or correctness issue;
- keep it if it improves the target metric without unacceptable PCC or complexity;
- reject it only with evidence, then continue optimizing the rest of the decoder.

Avoid suppressing advice in the report used to guide optimization. When applicable advice remains untried, call that out as remaining work rather than implying the optimization pass is complete.

## Final Audit Checks

- No unnecessary `InterleavedToSharded`, `ShardedToInterleaved`, `reshard`, `tilize`, `untilize`, `to_torch`, or `from_torch` in the optimized runtime path.
- Decode trace replay still measures the optimized path, not a fallback path.
- Program configs and compute-kernel configs are described in the final report or a compact structured summary.
- PCC covers prefill and decode for every representative layer kind.
- Optimized stress runs and passes for every representative layer kind and exercised mode; skipped stress is not a passing optimized result.
- Final perf reports cover warmed prefill and warmed decode separately.
- Watcher-clean evidence exists for the optimized correctness run.
