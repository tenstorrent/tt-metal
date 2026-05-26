# Multichip Decoder Parallelization Knowledge

This reference is a compact map of tt-metal model code worth reading before adding multi-chip decoder execution.

## Primary Style Recommendation

Prefer the GPT-OSS structured approach and `models/common` TTTv2 modules:

- `models/demos/gpt_oss/config.py`: `ModeConfig`, `MeshConfig`, semantic mapper helpers such as `column_parallel`, `row_parallel`, `sequence_parallel`, and CCL helper methods.
- `models/demos/gpt_oss/tt/ccl.py`: small generic `CCLManager` with ping-pong semaphores instead of model-specific CCL control flow.
- `models/demos/gpt_oss/tests/test_factory.py`: target mesh setup, fabric config, and hardware-shape parametrization.
- `models/common/modules/attention/attention_1d.py`: 1D attention with local heads, paged KV cache, optional fused all-gather plus WO matmul, and reduce-scatter output.
- `models/common/modules/mlp/mlp_1d.py`: 1D MLP with W1/W3 output sharding, W2 input sharding, and reduce-scatter output.
- `models/common/modules/rmsnorm/rmsnorm_1d.py` and `rmsnorm_2d.py`: reusable RMSNorm paths, including distributed stats.
- `models/common/modules/tt_ccl.py`: common CCL semaphores, topology detection, and link-count helpers.

Use these as the first implementation references. If they do not fit the target model, write model-local code in the same structured style: config dataclasses, explicit mesh helpers, setup-time weight conversion, and straight-line runtime forwards.

## Galaxy As Evidence, Not Template

`models/demos/llama3_70b_galaxy/` is highly optimized and useful for understanding where collectives belong, but it is too model-specific to copy into this skill's default path.

Useful lessons:

- Attention splits Q/K/V work across devices, runs SDPA on local heads, gathers or concatenates heads before WO, then reduces output to restore residual layout.
- MLP performs W1/W3 work on sharded intermediate chunks, gathers when W2 needs the full intermediate, then reduces output.
- The residual layout after each sublayer is a first-class performance contract.

Avoid copying:

- custom `llama_rs_*` ops as a baseline;
- 32-chip asserts;
- prefetcher-specific control flow;
- environment-variable topology switches as the normal user interface;
- model-specific hardcoded memory configs where generic config helpers can express the same decision.

Read these files when targeting Galaxy:

- `models/demos/llama3_70b_galaxy/tt/model_config.py`
- `models/demos/llama3_70b_galaxy/tt/llama_attention.py`
- `models/demos/llama3_70b_galaxy/tt/llama_mlp.py`
- `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`
- `models/demos/llama3_70b_galaxy/tt/distributed_norm.py`

## Dense 1D Tensor Parallel Default

For 1D meshes up to 8 devices, start with the common TP pattern used in LLM implementations:

- WQKV: column/output sharding, so each device owns local Q/K/V heads.
- KV cache: per-device local KV-head cache, paged if the baseline is paged.
- SDPA: local to the local heads.
- WO: row/input sharding over concatenated local heads, then reduce-scatter or all-reduce to restore residual layout.
- W1/W3: column/output sharding over intermediate dimension.
- W2: row/input sharding over intermediate dimension, then reduce-scatter or all-reduce to restore residual layout.

The output of each decoder layer must match the input layout contract. Avoid a design that requires an extra gather or reshard between stacked decoder layers unless evidence shows it is faster overall.

## RMSNorm Correctness

The distributed RMSNorm primitive pair is:

1. `ttnn.rms_norm_pre_all_gather`
2. `ttnn.experimental.all_gather_async` or `ttnn.all_gather` for the stats
3. `ttnn.rms_norm_post_all_gather`

Use this when hidden activations are sharded across the normalized dimension and local RMSNorm would compute incorrect statistics. The reusable implementations to read are:

- `models/common/modules/rmsnorm/rmsnorm_1d.py::_prefill_1d_distributed`
- `models/common/modules/rmsnorm/rmsnorm_2d.py::decode_forward`
- `models/common/modules/rmsnorm/rmsnorm_2d.py::prefill_forward`
- `models/demos/deepseek_v3/tt/rms_norm/distributed_rms_norm.py::DistributedRMSNorm`
- `models/tt_transformers/tt/ccl.py::tt_distributed_rmsnorm`
- `models/tt_transformers/tt/ccl.py::tt_sharded_distributed_rmsnorm`

Do not phrase distributed RMSNorm as always mandatory. Correct RMSNorm is mandatory. A faster replicated-activation stream plus local RMSNorm is acceptable if it preserves the decoder chain layout and improves measured performance.

## MoE And Expert Replication

For TP up to 8 devices, the default is to run each active expert selected by the gate with tensor parallelism. Keep the gate-selected active-expert path from the single-chip baseline; do not run every expert densely as the final path unless there is no practical alternative.

For Galaxy 4x8 MoE, read GPT-OSS throughput experts:

- `models/demos/gpt_oss/tt/experts_throughput/config.py`
- `models/demos/gpt_oss/tt/experts_throughput/weights.py`
- `models/demos/gpt_oss/tt/experts_throughput/decode.py`
- `models/demos/gpt_oss/tt/experts_throughput/fused_decode.py`

The interesting GPT-OSS strategy is to dispatch tokens to expert owners along one axis while using the other axis as replicated expert groups / TP groups for single-user throughput. Only accept this strategy if `memory_capacity_plan.json` proves enough DRAM for all model layers plus full KV cache on the target max sequence length, not just the current decoder layer.

If expert replication does not fit, document the alternative: ordinary TP active experts, 2D TP, expert parallel without replication, or a hybrid. Record the rejected strategy and evidence.

## 2D Mesh Planning

On Galaxy 4x8, do not blindly flatten to TP=8. Make an explicit 2D plan:

- Choose which axis owns TP, expert parallelism, sequence parallelism, or replication.
- Identify which collectives cross rows and which cross columns.
- Estimate activation, weight, expert, and KV-cache memory with all layers loaded.
- Compare communication volume against the 1D alternative.
- Record why the selected strategy should improve single-user latency.

GPT-OSS `MeshConfig` is the preferred shape for representing this plan. Galaxy Llama is useful for validating where reductions and gathers happen in an optimized 2D decoder, not for copying bespoke ops.

## Validation Heuristics

Compare multi-chip output to the single-chip TTNN baseline first. This removes HF-vs-TTNN numerical differences from the bug hunt. If multi-chip PCC is close to the minimum threshold, split the decoder into component comparisons:

- input RMSNorm;
- QKV projection;
- RoPE;
- SDPA output;
- WO projection and reduction;
- post-attention residual;
- post-attention RMSNorm;
- router/top-k for MoE;
- active expert outputs;
- W2/down projection and reduction;
- final residual.

Check layouts and collectives before changing precision. Most multi-chip bringup bugs are wrong sharding, wrong gather/reduce axis, bad padding/slicing, repeated bias application after all-reduce, wrong local head count, wrong local KV-cache shape, or mismatched input/output residual layout.

## Runtime Gotchas

- Bind static mode choices at construction time when possible; keep runtime forward paths straight-line.
- Avoid `ttnn.from_torch`, `ttnn.to_torch`, host reads, host writes, or tensor allocation after trace capture inside measured prefill/decode paths.
- Use source-built watcher separately from profiler runs.
- Make CCL semaphore ownership explicit; reuse generic CCL managers unless the target model already has a better local abstraction.
- Pad CCL-sensitive hidden dimensions deliberately and slice at a documented boundary.
- If a bias is applied before an all-reduce, prove it is not applied once per TP shard unless that is intended.
