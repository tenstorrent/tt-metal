# Runtime Fallback Audit

Scope: one measured `prefill_forward` or `decode_forward` pass in `tt/functional_decoder.py`.

The runtime source audit covers:

- `FunctionalDecoder.prefill_forward`
- `FunctionalDecoder.decode_forward`
- `_QwenFullAttention._project_qkgv`
- `_QwenFullAttention._reshape_prefill_heads`
- `_QwenFullAttention._reshape_decode_heads`
- `_QwenFullAttention._norm_and_rope`
- `_QwenFullAttention._decode_update_mem_config`
- `_QwenFullAttention._cache_update_tensor`
- `_QwenFullAttention.prefill_forward`
- `_QwenFullAttention.decode_forward`
- `_QwenLinearAttention._conv_step`
- `_QwenLinearAttention._step`
- `_QwenLinearAttention._conv_prefill`
- `_QwenLinearAttention._reshape_prefill_heads`
- `_QwenLinearAttention._fold_prefill_heads`
- `_QwenLinearAttention._pad_linear_chunk`
- `_QwenLinearAttention._solve_chunk_attn`
- `_QwenLinearAttention._chunk_gated_delta_rule`
- `_QwenLinearAttention._finish_prefill_chunk`
- `_QwenLinearAttention.prefill_forward`
- `_QwenLinearAttention.decode_forward`
- `_QwenMoe._router_dense`
- `_QwenMoe._shared`
- `_QwenMoe._routed_decode`
- `_QwenMoe._routed_prefill_chunk`
- `_QwenMoe._routed_chunk`
- `_QwenMoe._forward_chunk`
- `_QwenMoe.forward`
- shared runtime helpers `_slice`, `_slice_last`, `_concat_dim2_bounded`, `_silu_mul`, `_rms_norm`, `_l2_norm_last_dim`, `_rotate_half`, `_apply_partial_rope`, and `_sparse_matmul_program_config`

Forbidden runtime calls:

- `torch`
- `ttnn.from_torch`
- `ttnn.to_torch`
- `get_fallback_function`

Evidence:

- Source audit test: `test_runtime_fallback_audit_source`
- Runtime fallback guard command: `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest ... -k 'test_synthetic_functional_decoder_prefill_decode_against_hf or runtime_fallback_audit_source'`
- Artifact: `logs/runtime_fallback_audit.log`
- Result: `3 passed, 19 deselected`

Allowed boundaries:

- `from_state_dict`, `__init__`, and state allocation helpers may use Torch to load, shape, and convert HF-format weights or static masks before the measured pass.
- Tests may use `ttnn.from_torch`/`to_torch` for input construction and PCC comparison.
- The audited runtime decoder pass itself stays in TTNN.
