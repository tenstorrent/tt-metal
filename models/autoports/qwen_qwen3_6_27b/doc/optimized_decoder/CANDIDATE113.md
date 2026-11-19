# Candidate113: batch-one full-attention decode controls

Immutable source-only assembly over `candidates/candidate107.py`. No hardware run, main-source edit, cache-dtype change or default-policy selection. This prepares the adaptive branch for cumulative signoff.

Optional policies:

```json
{
  "single_user_classic_attention": true,
  "single_user_sdpa_grid": [11, 10]
}
```

Both are independent. Classic attention defaults to False; the grid override defaults to absent/None. Existing `DEFAULT_POLICIES`, weight precision/fidelity policies and KV-cache dtype are unchanged. No batch-specific cache dtype was implemented.

`single_user_classic_attention:true` applies only to full-attention **public decode with logical batch one**. It bypasses both the `compact_residual` shortcut in `decode_forward` and the `compact_attention` shortcut in `_attention_decode`, reaching the existing classical attention implementation. Logical batch counts account for ordinary `[B,1,W]` and compact `[1,B,W]` input shapes. Other batches and GDN routing retain their existing choices.

`single_user_sdpa_grid` applies only to full-attention public decode with batch one. A shared `_decode_sdpa_grid` helper supplies the grid in both classical and compact SDPA paths. All other calls retain `sdpa_grid` (or its existing `[8,4]` fallback). Selecting a grid does not itself establish its memory feasibility or latency; parent hardware evidence decides whether it is retained.

## Prefill phase isolation

Unaligned full-attention prefill also calls `_attention_decode`, so shape-only selection would accidentally alter prefill. Candidate113 explicitly propagates `decode=True` from `decode_forward` to full-attention decode helpers. Their default is False; prefill callers are unchanged. Thus unaligned prefill, one-token prefill and S1 tails retain candidate107 routing and grid even when both new policies are enabled.

No persistent mutable phase flag is used. Phase98 GDN correction selection and all candidate103/107 switches remain unchanged.

## Exact source changes

Only these existing methods differ:

- `decode_forward`: adds the B1 classic bypass condition and passes the explicit decode phase to full-attention helpers.
- `_attention_decode`: adds a default-False decode phase, the B1 classic bypass and the shared decode-grid lookup.
- `_compact_attention_decode`: adds a default-False decode phase and uses the shared decode-grid lookup.

Two methods are added: `_single_user_classic_decode` and `_decode_sdpa_grid`.

`prefill_forward`, `_attention_prefill`, `_gdn`, `_gdn_compact`, `_linear_decode`, `_linear`, `_linear_raw`, `_finish`, `_norm`, `from_state_dict`, `create_paged_state` and `_batched_conv` remain AST-identical. Every original module-level helper, kernel string, constant and default is also AST-identical.

## Validation completed

`tests/test_candidate113_source.py` passed:

- Source/output hashes and deterministic generation.
- Exact unchanged prefill/GDN/default/cache/projection-phase AST checks.
- Candidate107-identical tensor math after removing only the optional selector/phase plumbing.
- 80 routing cases across both layer kinds, B1/B2/B3/B31/B32, ordinary/compact inputs, enabled/disabled compact residual, and enabled/disabled classic policy.
- 80 grid cases across both layer kinds, B1/B2/B3/B31/B32, decode/prefill phase, present/absent base grid and present/absent override.
- Enabled B1 policies never change unaligned prefill routing.

These checks execute metadata/stub routing and pure policy helpers, without importing TTNN or opening a device. They do not establish PCC, paged-cache correctness, L1 feasibility, watcher status or performance.

Artifacts: `candidates/candidate113.py`, `candidates/make_candidate113.py`, `candidates/candidate113.generation.json`, `tests/test_candidate113_source.py`, and `candidate113_cpu_checks.json`. The generator refuses to replace an existing output.

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_candidate113.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_candidate113_source.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/candidate113.py
```

Parent signoff should compare classic/compact B1 with the selected BF8 KV policy and any grid override, retain B>1 controls, then run paged continuation/mixed-position correctness and traced latency on the same candidate. Existing gates remain unchanged.
