# Router checkpoint storage: lossless BF16 candidate

Read-only source/CPU investigation, 2026-11-13. No F edit, hardware run, or measured speed claim.

**Both tested layer kinds have native BF16 checkpoint router weights.** CPU `safetensors.safe_open` read only `model.language_model.layers.{0,3}.mlp.gate.weight` at pinned revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`:

| Layer | Checkpoint shard | Shape/dtype | Finite; FP32→BF16→FP32 exact | Raw BF16 tensor SHA256 |
|---|---|---|---|---|
|0|model-00002-of-00026.safetensors|[256,2048], torch.bfloat16|yes;yes|7e79efc3ff4f395be4667108f668061a4f40d32db4d0889431ffdbc70aafff40|
|3|model-00003-of-00026.safetensors|[256,2048], torch.bfloat16|yes;yes|86a1ba236777d0207ee3661e316b294b3e684c642ed0b27e9063fb5117383bbe|

Current setup uploads `value.T` as FP32 specifically for `mlp.gate.weight` (`FusedDecoder.from_state_dict`, line84). That widens a1,048,576-byte BF16 matrix to2,097,152 bytes without adding information. Native row geometry is2048×256, already tile-aligned. Removing this one upload exception is a lossless storage/interface change; it is not quantization of FP32 checkpoint values.

## Prior evidence and boundaries

Searched stage work log, endurance memos, artifact names, JSON/logs, and snapshots for router BF16-weight/storage trials. No same-router BF16-weight trial/rejection was found. `router_pack_dtype_{0,3}.json` and `router_pack_dtype.py.snapshot` test **BF16 output packing**, while retaining FP32 weight storage. `work_log.md:128–135` and `README_before_endurance_repair.md.snapshot:24` explicitly distinguish those policies. Generic attention projection format probes are different workloads and do not establish this router candidate's correctness/performance.

Canonical HF reference constructs the layer on meta and loads the checkpoint with `load_state_dict(...,assign=True)`, preserving BF16 router weight dtype (`tests/reference.py:46–50`). Installed `Qwen3_5MoeTopKRouter.forward`, `modeling_qwen3_5_moe.py:784–791`, computes `F.linear(hidden_states,self.weight)` with BF16 inputs/weights and BF16 logits, then FP32 softmax, top8, selected-probability renormalization, and BF16 scores.

Current F deliberately has different prefill output precision: `_moe` passes `dtype=ttnn.bfloat16` for singleton/decode and `dtype=ttnn.float32` for multirow prefill. The storage candidate should preserve both choices, the current native router/topk interface, and explicit `self.compute` HiFi4 / FP32 destination / nonapproximate / packer_l1_acc=False. BF16 operands and FP32 output are accepted by native matmul; storage dtype does not set the explicitly requested output dtype or disable explicit FP32 destination accumulation (`matmul_device_operation.cpp` input-format validation and compute-config construction).

## Minimal candidate

Change only the setup upload selection for `mlp.gate.weight` to BF16 (the ordinary projection upload policy); leave `linear()`, `_moe`, compute config, output dtypes, and normalization untouched. Weight values are exactly preserved. The same generic native matmul consumes either format without a runtime host operation or cast fallback. Setup already transposes/uploads the weight once.

**Value preservation is not a proof of identical native outputs.** Changing operand data format can alter native scheduling/config selection or arithmetic implementation. Previous seed154 failures demonstrate why a BF16 router-logit boundary is sensitive to small changes. Compare actual same-input router logits, top8 identities/scores, and final decoder contribution for both layer kinds, including singleton and multirow prefill; retain the existing seeded endurance/continuation gates. A changed BF16 routing set must be assessed by those unchanged gates, not dismissed as a tie or accepted from weight equality alone.

Logical streamed weight savings are1,048,576 bytes per router call, or2.048 us at the report's optimistic512GB/s ceiling. This alone does not predict the~24.9us kernel's latency reduction: dispatch, native configuration, and compute remain. The resident storage saving is also1MiB per layer on this single-device implementation. If retained, update roofline `router_fp32` to `router_bf16` and reduce its useful-byte numerator by1,048,576; do not present the same speedup against an inflated FP32 weight-byte lower bound. No accounting change is made while this candidate is untested.
