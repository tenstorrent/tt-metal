# Combine measured persistence and partition improvements

The parent-run JSON artifacts show both independent changes improve B32:

| Variant | B32 short median ms | B32 512 median ms | B1 median ms |
|---|---:|---:|---:|
| packed150 | 1.765976 | 1.782593 | 1.298467 |
| persist160 | 1.734883 | 1.747842 | 1.295988 |
| balance161 | 1.752951 | 1.766422 | 1.298017 |

Sources are the corresponding `{tag}_b32.json`, `{tag}_sequence.json` and
`{tag}_b1.json` files. Both changes passed all 54 isolated exact output/state
comparisons. All three 512-step runs report identical minimum per-user PCC
0.9953641404219116. Balance is therefore retained for combination, with no
assumption that its gain adds to the persistence gain.

`gdn_sfpu_fused_persistence_balance.py` combines the two already-tested changes.
Its host functions exactly match the partial-group balance helper. Its complete
compute string exactly matches the persistence helper after reverting only the
global start argument, group-entry/group-exit conditions and their comment.
Reader, writer, all other constants, CB sizes, rounding arithmetic and state
packing retain their tested implementations.

Frozen artifacts:

- `candidates/gdn_sfpu_fused_persistence_balance.py`
- `candidates/integrated_fused_persistence_balance134.py`
- `candidates/make_gdn_fused_persistence_balance_candidate.py`
- `candidates/check_gdn_fused_persistence_balance_cpu.py`
- `candidates/gdn_fused_persistence_balance_cpu.json`

The composition checker passed and reran the 524,288-pattern native-RNE check
and all 7,040 partition/CB-lifetime cases. Python compilation passed. The combined
helper's device accuracy and latency are unverified; no device was accessed and
no main/shared source was edited.

Parent-serialized exact probe:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_recurrence_exact_comparison \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_fused_persistence_balance.py \
  --reference models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_packed_prepared_qk.py \
  --state-dtypes bf16 fp32 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/persistence_balance_probe.json
```

Require 54 exact cases, then B32 short/512 and B1 validation with
`integrated_fused_persistence_balance134.py`. This whole candidate isolates the
recurrence change over integrated134. Other independent decoder gains can use
the helper through the existing namespaced-append integration pattern.
