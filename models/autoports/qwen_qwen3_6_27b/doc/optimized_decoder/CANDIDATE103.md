# Candidate103 cumulative integration

Source-only assembly over frozen `candidates/phase98.py`. No hardware execution, main-source edit, default-policy selection, or modification to any input snapshot. This is a cumulative signoff candidate, not a final optimized-decoder delivery.

## Exact changes

Only four existing class methods differ from phase98:

1. `_finish`: copied exactly from `shared90.py`. `share_mlp_input:true` shares the compatible gate/up input reshard in compact decode. Its explicit `not prefill` guard remains intact. Default is False.
2. `_gdn_compact`: selects the namespaced final grouped96 recurrence helper when `grouped_recurrent_init:true`; otherwise calls the original phase98 helper. All recurrence arguments, post-GDN math and phase98 projection selection remain unchanged.
3. `_linear`: an opt-in `chunk_add_prefill_correction:true` guard calls the exact chunk94 helper for corrected projections with more than 32 logical flattened rows. Existing dispatch follows unchanged when disabled or at compact decode sizes. The original flat helper remains available and unchanged, so chunk-add versus ordinary correction is a direct switch.
4. `_linear_raw`: adds the existing `make_prefill_role_configs.py` transformation. `prefill_projection_configs` overrides grid, K block, subblock and input placement by original projection key, with residual keys inheriting their base projection settings. Empty overrides reproduce the original calls. Decode DRAM-sharded matmul configs remain unchanged.

The final effective grouped96 definitions were copied after its last `_GDN_FUSE_ACCUMULATOR` assignment, including the grouped initialization flags, compute-config builder, reader/writer/compute strings, partition helper and recurrence function. All Python identifiers in this suffix were namespaced (`_GROUPED96...` / `_grouped96...`). C++ source strings are byte-identical. They cannot overwrite phase98's effective globals. The old and new kernels can coexist for direct controlled comparison.

The chunk94 function is copied as `corrected_projection_chunk_add_prefill`; its AST is identical after reversing this single function rename. No packed correction-pair candidate is included.

## Defaults and phase behavior preserved

`DEFAULT_POLICIES`, `from_state_dict`, `_linear_decode`, `_norm`, `_gdn`, `decode_forward`, and `prefill_forward` are AST-identical to phase98. Every original effective top-level helper and constant remains AST-identical. Thus:

- `decode_weight_correction` retains phase98 semantics: excluding the output projection removes its correction only in explicitly marked decode.
- Prefill, including one-token prefill and final S1 tails, continues to use both correction terms.
- No mutable phase flag or public logical-length restriction was added.
- Original weight/correction storage and state ownership are unchanged.
- No exact-norm, MLP-packer, grouped recurrence, or correction-selection policy is silently promoted to a default.

The grouped recurrence is the one intentional opt-in kernel substitution that also applies to compact S1 prefill when selected, exactly as in grouped96. Its arithmetic/state semantics require the parent’s cumulative hardware check; projection phase routing itself remains unchanged.

## Suggested explicit cumulative policy

Root selects settings and retains measured comparisons. The following is the requested combination, not a new default:

```json
{
  "exact_tree_hidden_norm": true,
  "mlp_packer_l1_acc": true,
  "share_mlp_input": true,
  "decode_weight_correction": ["packed_gdn"],
  "grouped_recurrent_init": true,
  "chunk_add_prefill_correction": true,
  "prefill_projection_configs": {}
}
```

Set `chunk_add_prefill_correction:false` for ordinary corrected prefill. Set `grouped_recurrent_init:false` for the original phase98 retained recurrence. Role settings can be compared independently, for example:

```json
{
  "prefill_projection_configs": {
    "linear_attn.out_proj.weight": {"grid": [10,8], "in0_block_w": 16},
    "mlp.down_proj.weight": {"grid": [10,8], "in0_block_w": 16}
  }
}
```

That example is not a performance recommendation over the parent's current measured winner. Subblock-height overrides are upper bounds, reduced to legal divisors for arbitrary short tails. Full output blocks are retained.

## Source validation

`tests/test_candidate103_source.py` passed:

- Input/output SHA verification and deterministic regeneration.
- Exact default, prefill/S1 routing, decode correction-selection and norm preservation.
- Exact shared90 `_finish`, exact namespaced final grouped96 definitions, exact renamed chunk94 helper.
- Only the four class methods listed above changed.
- No original effective top-level helper or constant changed.
- Direct decode-selection checks: both corrected by default; input-only selection still corrects input and emits raw-output→BF16 for decode output.
- 243 source/config cases spanning B1/B3/B32, S1/17/33/65/128/511/512/513/1025, three projection roles and three grids. Default matmul calls matched phase98; all enabled configurations retained valid full output blocks and tail subblocks.

Commands executed (CPU only):

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_candidate103.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_candidate103_source.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/candidate103.py
```

Artifacts: `candidates/candidate103.py`, `candidates/make_candidate103.py`, `candidates/candidate103.generation.json`, `candidate103_cpu_checks.json`, and the source test above. The generator refuses to replace an existing output; source hashes include phase98, grouped96, shared90, chunk94 and the role-config generator.

Cumulative B1/B32 correctness, nonaligned prefill/state checks, 512-step traced stress, watcher and latency comparisons remain pending. Source checks do not claim that individually passing changes compose numerically or improve latency.
