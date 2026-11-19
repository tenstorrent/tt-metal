Applied checkpoint: `c873e26136c8140fe419e523525265ee7d3939dc` (2026-11-16). The analysis below records the pre-change gap and original proposal; v5/v6 measurements validate the applied code.

The current runtime has an evidence gap for activation/residual boundaries. `precision.py` rejects unsupported values, but `model.precision_summary().fixed_boundaries` merely echoes those policy fields. Actual BF16 transport enforcement exists, yet its assertions are not connected to activation/residual policy values. Logits/sampling have partial consumption: generator construction asserts the real sampler buffers against policy, but the selected model head output and sampling allocations remain hard-coded and their observed dtypes are not included in the model summary. Do not present the echoed fields as measured observations.

The adjacent patch originally proposed four bounded changes, without changing any permitted dtype or tensor math:

- Resolve the four validated fields into device dtypes once. Use activation_dtype at the existing terminal norm cast, residual_dtype for prefill workspace allocations, logits_dtype for both head paths, and sampling_dtype for existing sampler buffers.
- Assert actual embedding, bounded-prefill residual transport, decoder-layer residual, terminal logits, and sampler-input tensor dtypes against those fields. No new device operations/readbacks/casts are introduced; existing BF16 operation arguments resolve to the same values.
- Record actual tensor metadata as observed_boundaries in precision_summary, keeping the existing fixed_boundaries field for compatibility. Observations are Python dispatch/warmup/trace-capture evidence, not trace-replay execution counts. Report fields are snapshots, not live mutable references.
- Keep all four fields fixed to BF16 under the existing strict resolver. This is explicit consumption of a fixed model contract, not permission to claim variable BF16/BFP8/FP32 activation support. Internal FP32 recurrence, normalization accumulation and native grouped-MoE BFP8 intermediates are outside these boundary-field meanings. PrefillTransport independently requires BF16 ROW_MAJOR storage; common sampling support should not be characterized as universally BF16-only without separate native evidence. Sampled token IDs remain integer tensors; sampling_dtype means the model-to-sampler logits interface, not token IDs or every internal sampler tensor.

Acceptance evidence must use runtime_policy_after observed_boundaries following real evaluation/warmup, and require embedding.output, embedding.residual, bounded_prefill.decoder_input, bounded_prefill.residual_write_source, terminal.normalized_activation, the active terminal logits branch, sampler.logits_buffer, sampler.model_logits_compatibility, and sampler.zero_logits_row. Post-selection token-out evidence additionally requires sampler.forward_input. Python dtype guards execute during trace capture and check immutable dtype metadata of the captured buffers; they do not run on each device replay.

At proposal creation, this patch was unapplied. All four transformed Python files passed AST parsing; the patch passes git apply --check. No production source or hardware was touched during that CPU-only proposal task. Applying it changes source identity and requires fresh comparable baseline/candidate/P1 evidence. Python observation/guard overhead must not be assumed performance-free even though the TTNN operation sequence and legal BF16 values are unchanged.

Source SHA256 at proposal generation:

{
  "models/autoports/qwen_qwen3_6_35b_a3b/tt/model.py": "b314e1972dc8c3ed8131dc6a46e45ed453aa4d83712ca19ca4f2325e87e9a93c",
  "models/autoports/qwen_qwen3_6_35b_a3b/tt/lm_head.py": "584ee756c38d094b2de6ae11c42a4bbba1a696abd305d460f3a3978b11a86d82",
  "models/autoports/qwen_qwen3_6_35b_a3b/tt/prefill.py": "eda33d5d9023ca28d785a871ae58032685aff37558fc4f119cf439e42f28356c",
  "models/autoports/qwen_qwen3_6_35b_a3b/tt/generator.py": "5f9d84f0d9b9878260159600458b54512c3f06d77d67bebe6c4a986fc4043dbd"
}
