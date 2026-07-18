# AutoFix: fused MoE on the fixed 1x4 decoder

## Starting evidence

- Candidate command already run by this stage:
  `RUN_MULTICHIP_FUSED_MOE=1 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_fused_moe_compute_ep_local_candidate`.
- Result: pass on one Blackhole device; retained as
  `fused_moe_ep_local_retry.junit.xml`.  It proves the exact GPT-OSS dimensions
  (`H=I=2880`), fused biases/SwiGLU, eight resident experts, and `top_k=4`
  compile and run.  It does **not** prove that outputs from arbitrary experts
  survive: the imported validator intentionally checks only the last two
  rolling experts.
- No implementation file was edited during this AutoFix investigation.

## Hypothesis experiments

### H1: `compute_only` can expose every active local expert

**Prediction.** If slot 4 retained all eight experts, arbitrary rank-local
top-4 contributions could be combined after the op.

**Source experiment/result.** Refuted.

- `compute_output_specs` allocates exactly two token buffers, shape
  `[all_cores, 2, 32, H]`, and slot 4 is only a row-major alias of that storage
  (`moe_compute_device_operation.cpp:275-313,320-327`).
- DM1 uses a boolean `output_buffer_idx`, writes expert `e` into `e % 2`, and
  toggles after every expert (`device/kernels/dm1.cpp:213-222,381,427`).
  Full mode consumes each buffer before reuse; `compute_only` explicitly has no
  consumer and only flushes writes (`dm1.cpp:302-310,386-424`).
- The shared validator consequently derives exactly the final two experts and
  checks only those.  For eight experts these are local experts 6 and 7
  (`tests/nightly/tg/ccl/moe/test_moe_compute_6U.py:777-806`).

Increasing the buffer depth from 2 to 8 is not a viable buffer-only repair:
the present BF16 buffer is already `2*32*2880*2 = 368,640` bytes per worker
core; eight slots require 1,474,560 bytes per core before the other CBs and
semaphores. Repeated two-expert calls also do not preserve the traced dynamic
routing contract: they require dynamic expert-weight repacking/selection, or
force unselected experts into the indices and execute them.

**Verdict:** refuted. The passing compute-only test is a kernel-math smoke, not
a usable arbitrary-active-expert decoder path.

### H2: full `moe_compute` can represent the fixed 1x4 EP decoder

**Prediction.** Full mode's fused selective-reduce consumer can preserve every
expert before the two-slot buffer is reused.

**Source experiment/result.** Verified for the API/dataflow, with an important
production-contract cost.

- Full mode constructs `SelectiveReduceCombineParams` on `cluster_axis=1` and
  accepts Ring or Linear topology (`moe_compute_device_operation.cpp:451-491`).
- The upstream test explicitly records that the same suite works on a 1x4
  mesh (`test_moe_compute_6U.py:52`) and its sparse-input contract is
  `[devices, tokens_per_device * dispatch_devices, H]`, sharded on device dim
  0 (`test_moe_compute_6U.py:1045-1117,1933-1973`).
- Full mode fixes `batch_size=1, seq_size=total_tokens`; selective combine then
  shards those tokens across the four devices (`moe_compute_device_operation.cpp:478-491`;
  `selective_reduce_combine_device_operation.cpp:80-99`).

For this decoder, the residual is replicated and each rank holds the same one
decode token. Therefore full mode can keep the public `[1,1,1,H]` replicated
residual only by treating the replicas as four source tokens: the internal
input has four identical token rows, and each device receives one identical
combined row. This preserves arbitrary top-4 routing and does not execute all
experts, but it duplicates the selected-route compute four times. A one-source
token variant cannot use the existing output contract: `total_tokens / 4`
would be zero for decode, and there is no broadcast/local-combine mode.

The op is also fixed to BFP4 expert weights: the program factory creates the
weight CB as `Bfp4_b` (`moe_compute_program_factory.cpp:856-875`). The stage's
real decoder sweep already showed all-BFP4 expert weights below the accepted
0.99 PCC floor, while the selected path needs BFP8 gate/up plus BFP4 down.

**Verdict:** full mode preserves all contributions, but it is not an acceptable
drop-in rank-local path. It requires fourfold duplicated routed work and a
precision policy already rejected by the real decoder PCC gate. A focused A/B
may still be run as a negative performance/control test; it cannot become the
default unless both PCC >= 0.99 and end-to-end latency beat the selected path.

Focused shape/topology control (synthetic weights): add a temporary mesh test
that calls `_run_moe_compute_impl` with
`mesh_shape=(1,4), cluster_axis=1, experts_per_device=8,
tokens_per_device=1, selected_experts_k=4, num_layers=1, N=2880,
hidden_size=2880, output_height_shard_dim=4,
output_width_shard_dim=2, dtype=ttnn.bfloat16,
activation_type=SWIGLU, has_bias=True, num_links=1, topology=Ring`.
Then run the same case with 32 tokens/device to check the prefill-shaped path.
The production gate must use real stage weights and compare the weighted sum
of the four combine slots against the existing EP output at >=0.99 PCC.

### H3: another existing fused path fits this decoder

**Source experiment/result.** Refuted.

- `ttnn.experimental.moe_gpt` is the older GPT-OSS path, but it hard-codes four
  experts/device and a 12-core, 90-tile ring
  (`moe_gpt_ring_common.h:11-35`). Its compute kernel always performs 12 ring
  steps (`moe_gpt/device/kernels/compute.cpp:76-116`). The target P300
  Blackhole path uses the generic eight-core ring; this is why the newer
  `moe_compute` candidate was tested instead.
- `unified_routed_expert_moe` is a DeepSeek prefill composite, not an exact
  GPT-OSS replacement: it computes `silu(gate) * up`, requires BFP8 input, has
  a documented ~0.97 PCC target, and launches a separate FFN program per local
  expert (`unified_routed_expert_ffn_nanobind.cpp:20-43,83-98`). It lacks the
  GPT-OSS clamp/alpha/`up+1` and bias contract and is not a decode megakernel.
- The remaining MoE-named ops are routing/remap/reduction pieces; none fuses
  the exact GPT-OSS gate/up/bias/SwiGLU/down sequence while exposing all local
  active-expert results.

## Final status

**Limitation.** No existing in-scope Python/API path exposes arbitrary local
expert 0-7 results from `compute_only`. Full mode can preserve them only by
replacing the local-EP-plus-all-reduce topology with a four-source dispatch and
selective combine, duplicating routed compute and forcing BFP4 weights.

The smallest performant fix is a C++ `moe_compute` **local-combine mode**: keep
the two-slot producer/consumer handshake, consume each expert before overwrite,
apply its recorded routing score, accumulate into one `[tokens,H]` rank-local
BF16/BFP8 output, and skip fabric. To satisfy this decoder's accepted PCC it
also needs a non-BFP4 gate/up weight format. This is new kernel/dataflow work,
not a Python decoder integration fix. Until then, retain the measured
gate-selected sparse-matmul EP path; dense all-expert execution remains
unacceptable.
