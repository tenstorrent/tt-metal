# Reviewed compact lookup / direct coefficient audit extensions

The new auditor explicitly admits the following frozen references, with per-method setup allowlists and unchanged runtime host-fallback restrictions:

| Candidate | SHA256 |
| --- | --- |
| Compact norm/SiLU | `951391e44ed08d1646f857091d74c011b3c93e0c8735f7a0d941890291fd5579` |
| Direct coefficients | `c823e6f6f8480a9550276856a7aa660b9308ff598c6565a38c2bc54460440ece` |
| Combined | `5ff93fd27348d9e49b6c09da6fd77a362243c7c575b4b92431232638247de242` |

Only four new setup scopes gain exemptions: `WeightedNormTail.__init__`, `LocalSiluMultiply.__init__`, `local_silu_math_table`, and `compact_silu_math_table`. Their complete method/function AST must match a pinned reference. The direct-coefficient constructor does not require an exemption; it and every new `__call__` remain checked as runtime. The new helper classes/functions, compact range constants, and kernel source assignments are additionally pinned to the reviewed ASTs. The auditor does not accept arbitrary helper classes merely because their constructors upload constants.

## Mathematical setup review

`local_silu_math_table` enumerates all 65536 BF16 bit patterns and computes their mathematical SiLU in FP32, independently of model weights, activations or reference outputs. `compact_silu_math_table` compares integer FP32 encodings, derives the finite active ranges `[13249,16773]` and `[46144,49841]`, stores those ranges plus all positive/negative nonfinite encodings, and reconstructs every original integer encoding before asserting exact equality. Outside the stored ranges it uses the exactly representable BF16-input FP32 half-value, large-positive identity, or large-negative signed zero. Packing contains 7479 used entries padded to 7488, or 29952 bytes per worker. The local runtime reader uses the same pinned boundaries and index offsets. The pre-upload integer assertion includes signed zero, subnormal values and nonfinite payloads. The setup proof is preserved, not disabled or replaced with a weaker numerical tolerance.

This source audit does not rerun Torch table generation or open hardware. Prior component evidence is separately retained in `v6_silu_compact_lookup_cpu.json` and `v6_silu_compact_lookup_device.json`; no numerical/device acceptance is inferred merely from static matching. Runtime lookup and weighted-tail helpers contain only device tensor operations, descriptor construction and metadata scheduling. Direct coefficients are read by the device reader from original BF16 tensor buffers; no new host tensor conversion is allowed.

Eighteen mutation controls now reject the original 11 defects plus compact readback, setup-table rebuilding, an aliased setup function, aliased `exec`, weighted-tail upload, direct-coefficient `.cpu()`, and removal of the compact setup reconstruction assertions. Runtime references to setup functions/classes/methods are rejected even when assigned to aliases instead of immediately called.

## Production formatting equivalence

Parent selected the combined donor and Black formatted production to SHA `c71ea01a4b38458f0f53808d288d9a39ca5563a13cb5e97f71e701391670c3f2`. The first audit correctly rejected the changed setup method hash. Source inspection confirmed only trailing whitespace changed in two legacy bare-string descriptions at the start of full-attention branches in `prepare_prefill` and `prefill_forward`.

The final comparator normalizes **only trailing whitespace** in those two method-leading descriptions and their explicit full-attention guarded copies. It neither removes general string expressions nor normalizes assigned constants or kernel source. Negative controls confirm assigned kernel strings and ordinary executable constants still compare differently after edits. `v6_snapshot_static_audit_production_c71ea.json` records both exact before/after strings and whole-module normalized AST SHA `2f1a11e469917c14146b8e0198b90e8c6e8a0b1d16168fb1b02905ae06a9ab35`, identical to the frozen 5ff93 donor. Parent's separate selection proof is `v6_qualification_selection.json`.

## Production entrypoint migration

With explicit parent authorization, `tests/audit_fused_source.py` now delegates to the stronger host-only auditor. The old entrypoint is preserved as `v6_previous_audit_fused_source.py.snapshot` (SHA `43df3a27d974f54067b4b243617b6fa32c50a54021e79ac3e37657b9a02a55cc`) and its preceding evidence as `v6_previous_fallback_audit.json`. The original new-auditor source (`2de52ebc...`) remains `v6_snapshot_static_audit.py.snapshot`; intermediate extension versions are retained separately.

The replacement preserves the `fallback_audit.json` fields used by `check_evidence`/`check_fused_evidence`, including `python_audit_pass` and decoder identity. It records `unchanged_planning_body=false` for V6, rather than claiming the implementation matches functional. `default_prefill_chunk=4096` is explicitly a public hint; actual V6 linear chunks are 64 relative to the public call, and full attention retains at-most 256 aligned interiors plus singleton boundaries. Public single-token prefill preserves recurrence at every valid start. Logical lengths remain unrestricted by alignment, and context remains 262144. Detailed planner, runtime, guard, mutation and mock-dispatch evidence is embedded in the fallback report.

No PCC threshold, context contract, numerical/performance gate, consumer checker, production decoder or frozen candidate was changed by this audit work. Device-only runtime instrumentation remains mandatory in the delivered runners. Static pass is explicitly not stage acceptance or hardware-capacity proof.

Final sources are frozen before qualification provenance:

- `tests/audit_fused_snapshot_v6.py`: `0248adae72df6d5a16ef51fae1ede107c8f12dba84bc918016a0292a4ed923c4`.
- `tests/audit_fused_source.py`: `f823c1e5d91e322548dc757497f8b683c0d546b0b938368f04015cbaff8437ce`.
- Exact copies: `v6_final_audit_fused_snapshot_v6.py.snapshot` and `v6_final_audit_fused_source.py.snapshot`.

Reproduction (repository root):

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_source
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_snapshot_v6 --snapshot <frozen-snapshot> --output <new-report.json>
```

The snapshot auditor refuses to overwrite evidence. The production entrypoint retains its existing convention of updating `doc/fused_decoder/fallback_audit.json` only after the complete audit passes.

Final validation: all four frozen candidates pass in `v6_snapshot_static_audit_{951391,5ff93,c823,8cda}_final.json`; the production entrypoint passes and writes the source-bound fallback report. `v6_snapshot_static_audit_extension_summary.json` records report/source hashes and per-candidate helper counts. Each candidate passes 80,784 planner cases, 168 mocked dispatch cases and 18 mutation controls. No device was opened.
