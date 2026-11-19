# Candidate178: residual prefill K10/SW3 on candidate173

Source-only cumulative candidate: `candidates/integrated178.py`. Frozen base is `integrated173.py`; generator is `make_integrated178.py`, with SHA256 inputs/output and selected defaults recorded in `integrated178.generation.json`.

Only `_linear_raw` and GDN default-policy entries differ. The method is transplanted exactly from measured `prefill_split_role_configs.py`. This permits exact residual-key overrides before legacy primary-key fallback and legal divisor-clamped output-height ceilings for arbitrary short tails. Output width remains full per-core width, avoiding the previously diagnosed DRAM-width-sharded partial-width reader problem.

Selected explicit defaults:

```python
{
    "prefill_projection_configs": {
        "packed_gdn_residual": {"in0_block_w": 10, "subblock_w": 3}
    },
    "gdn_fused_affine_gate": True,
    "gdn_affine_gate_cores": 64,
}
```

Core64 and fused affine/gate match the parent-selected173 run configuration; the original173 source did not contain explicit entries for those defaults. Primary projection, out projection and MLP configs are unchanged. Recurrence, affine helper, kernel strings, numerical precision, phase routing, state ownership, decode and full-attention policy are unchanged. There is no new logical sequence-length restriction.

Parent-reported isolated171 measurements motivating the selection (milliseconds; identical PCC in each comparison):

| Shape | Control | Residual K10/SW3 |
|---|---:|---:|
| B32 S33 | 24.574096 | 24.379934 |
| B32 S512 | 176.515634 | 174.322562 |
| B1 S512 | 6.314413 | 6.270639 |

These are isolated role-config results, not measurements of cumulative178. Primary K10 results were pending at generation. Candidate178 is immutable; a subsequent winner requires another frozen output.

CPU validation executed:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_integrated178_source.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_integrated178.py models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/integrated178.py models/autoports/qwen_qwen3_6_27b/tests/test_integrated178_source.py
```

`integrated178_cpu_checks.json` records success: generator reproducibility, source hashes, full AST equality after excluding the exact allowed method/default changes, all 3,840 existing geometry cases, and 640 selected residual tail cases. No TTNN import or device execution occurred in this child task. Parent must perform cumulative correctness, long stress, performance and final gates.
