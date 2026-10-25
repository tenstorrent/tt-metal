# Runtime precision contract

`tt/precision.py::load_precision` loads an explicit constructor argument, then
`QWEN38_PRECISION_CONFIG`, then the selected artifact, otherwise the inherited
baseline. Both `Model` and `build_generator` use that loader. There is no vLLM
adapter in this stage; a later adapter must construct this same generator/model.
No vLLM integration is added here.

|Field|Runtime meaning|
|---|---|
|weight_groups.attention|Packed full-attention Q/K/V/gate or linear-attention Q/K/V/Z/A/B inputs, plus attention output projection; `MultichipDecoder.matrix`|
|weight_groups.mlp|Gate/up separate prefill weights and packed gate+up decode weights|
|weight_groups.down|MLP down projection|
|weight_groups.head|Actual terminal device weight conversion in `Model`; ordinary terminal matmul uses head fidelity|
|weight_groups.embedding, norm|Required BF16 representations; loader rejects unsupported values before construction|
|layer_exceptions|Per-index attention/MLP/down weight dtype and ordinary projection fidelity overrides, merged before loading weights|
|compute_fidelities.projection|Actual ordinary matmul compute config in `OptimizedDecoder._linear`|
|compute_fidelities.fused_prefill|Actual persistent fused4096-row matmul/RS compute config|
|compute_fidelities.fused_decode|Actual native batched fused matmul/RS compute config|
|compute_fidelities.sdpa|Decode paged SDPA compute config only; existing FP32-destination=false policy retained|
|compute_fidelities.head, norm|Head compute config; norm is the required shared HiFi4/FP32-destination config also used by prefill chunked SDPA, RoPE and recurrent normalization kernels|
|activation_dtype.ordinary_attention, ordinary_mlp|Inputs to ordinary projection matmuls; MLP includes ordinary down|
|activation_dtype.prefill_column|Prefill mix/gate/up/gateup inputs; exact per-role policy applied after inherited geometry|
|activation_dtype.fused_row|Required BF16 inputs to existing fused row paths, not an ordinary-matmul override|
|residual_dtype|Required BF16 residual representation and buffers, validated before construction|
|ccl_dtype.decode_b1|Projection reduction payload for replicated B1–7 decode; workspace owner validates payload dtype|
|ccl_dtype.prefill|Required BFP8 projection CCL, including persistent fused prefill buffers|
|ccl_dtype.native_batched|Required BF16 native batched row reduction; batch8–32 views preserve their existing topology|
|kv_cache_dtype|Actual full-attention paged K/V allocation dtype. Fill casts to cache dtype; updates remain BF16|
|recurrent_state_dtype|Required FP32 recurrent state; convolution history remains BF16|
|logits_sampling|Required BF16 logits/sampling inputs and uint32 feedback/output IDs|
|max_context|Must match checkpoint262144; runtime may allocate a smaller requested pool but cannot advertise a larger model context|

Fixed entries are supported-representation checks, not tunable fields silently
ignored by hard-coded kernels. Unsupported changes raise immediately. Norm and
convolution weights, RoPE, residual boundaries and embedding/terminal gathers stay
BF16; recurrence A/dt/normalization constants and recurrence math stay FP32/HiFi4.
Projection CCL fields do not describe BF16 embedding/terminal or sampler transfers.
Ordinary/fused projection accumulation retains FP32-destination=false and
packer-L1=true; head/norm retain FP32-destination=true. No MoE routing exists.

Weight packing, residual topology, trace lifecycle, non-aligned prompt handling
and sampling contracts are inherited. BFP8 weights and BF16 decode CCL require
validated dtype/workspace-dependent K-block and reader-count adaptations to fit
L1; AUTOFIX.md records these exact changes. Baseline BFP4/BFP8-CCL geometry stays
unchanged. Fused-row geometry is unchanged. Full-model runtime records include actual weight/cache dtypes
and resolved layer policies. After the first two runs, untimed preparation also
records actual linear/fused kernel dtype/fidelity arguments, with instrumentation
removed before the measured repeats. Final selected evidence must include that
concrete kernel inventory.

BFP4 KV reset uses same-dtype in-place device fill because public zeros_like
otherwise packs/uploads this dtype on the host. This preserves existing cache
addresses and CQ ordering; normal preparation warms its exact TensorSpec before
trace capture. BFP8/BF16/FP32 reset paths are unchanged. Initial BFP4 allocation
may still use host construction, but repeated request reset no longer does.
See kv_reset_autofix.md for the exact primitive and replay validation status.

BF16 full-attention KV also selects decode SDPA K128 (cap32), because K256 plus
the native page-table staging buffer exceeds live L1 in the full generator.
The existing mapped-capacity divisor guard still applies. BFP8/BFP4 KV retain
K256. This is a dtype-dependent admission adaptation, not a smaller context or
an altered page-table ownership contract. AUTODEBUG_kv_bf16.md and
kv_bf16_autofix.md contain the exact budget and validation evidence.
