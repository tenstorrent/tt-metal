# Candidate107: opt-in convolution and recurrence integration

Immutable assembly over `candidates/candidate103.py`; no hardware execution, main edits, input-snapshot edits or new numerical research. Cumulative hardware signoff is pending.

Only two existing class methods change:

- `_batched_conv`: inside the existing validated `sfpu_decode_conv` branch, `conv_untilize_pipeline:true` selects the exact helper from `conv_decode_sfpu_untilize_pipeline2.py`. Otherwise it calls candidate103's original `conv_decode_sfpu`. Arguments, carry ownership, outputs and surrounding guards are unchanged.
- `_gdn_compact`: `decay_predict_fusion:true` selects the exact helper from `gdn_sfpu_decay_predict.py`. Otherwise candidate103's existing `grouped_recurrent_init` selector is unchanged. The new decay helper includes the grouped initialization settings from its independently probed source; it takes precedence over the grouped selector when enabled.

Both switches default to **False**. `DEFAULT_POLICIES` is unchanged. Exact norm, MLP packer, MLP input sharing, correction chunk-add, role-specific prefill config and phase98 `decode_weight_correction` behavior are inherited unchanged from candidate103. No mutable phase state is introduced. Prefill/S1 correction selection is unchanged; as with the previous helpers, a compact S1 prefill can use a selected convolution/recurrence kernel, so cumulative boundary checks remain required.

All top-level helper names/constants are namespaced separately (`_conv107_...`, `_decay107...`). They do not shadow candidate103's original or `_grouped96...` definitions. Python identifiers are the only changes inside imported helpers: function ASTs match after reversing namespacing, and embedded C++ kernel strings remain byte-identical.

Explicit policy additions for parent comparison:

```json
{
  "conv_untilize_pipeline": true,
  "decay_predict_fusion": true
}
```

Keep `sfpu_decode_conv:true`, the compatible packed-RM convolution policy and the previously selected cumulative policies. Test each new switch independently and both together. The convolution helper performs one native untilize per invocation; its measured cost must remain included in any end-to-end timing claim. The source integration does not claim that independently passing helpers compose correctly or improve the final latency.

Artifacts:

- `candidates/candidate107.py`
- `candidates/make_candidate107.py`
- `candidates/candidate107.generation.json` — input/output hashes and complete rename maps
- `tests/test_candidate107_source.py`
- `candidate107_cpu_checks.json`

CPU/source validation passed: deterministic regeneration, frozen-source hashes, exact imported helpers, no identifier collisions, eight selector combinations, unchanged defaults/top-level helpers, and candidate103-identical math AST when the two selectors are disabled. Only `_batched_conv` and `_gdn_compact` differ. `_linear_decode`, `_linear`, `_linear_raw`, `_finish`, `_norm`, `from_state_dict`, `decode_forward` and `prefill_forward` remain AST-identical.

Commands executed without importing TTNN:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_candidate107.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_candidate107_source.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/candidate107.py
```

The generator refuses to overwrite an existing output. Long traced correctness, state/boundary checks, watcher and final before/after latency remain parent-run gates.
