# Compute and broadcast only the live delta row

Profiler163 reports recurrence call 1264640 at 409.222 us: BRISC 409.222,
NCRISC 370.054, TRISC0/1/2 403.280/405.583/405.813 us. Explicit CB-wait counters
are absent, and these kernel durations include waiting. They do not establish a
NoC bottleneck. The reader already finishes about 39 us before the operation;
increasing reader look-ahead is therefore not justified by these totals alone.

Source exposes a concrete unnecessary dependency chain after prediction:
`sub_binary_tile(4,6,6)` subtracts V and prediction across 1024 values, then
`gdn_mul_scalar(6,7)` multiplies all 1024 by beta, then `gdn_copy_row_to(6,4)`
keeps only row zero and replicates it. The 31 discarded rows cannot influence
the subsequent update. Both intermediate full-tile stores also serialize the
next SFPU traversal through DST.

The new `_gdn_delta_row_` loads the first four-row band of V and prediction and
uses the native SFPTRANSP mapping to place each useful 32-element row zero into
one LREG. It computes a separate FP32 subtraction, invokes the unchanged native
beta scalar broadcast, then computes a separate FP32 multiply. It uses the
incumbent inverse SFPTRANSP row replication and writes the full delta tile once.
No fused multiply-add replaces the two rounding steps. The unused rows never
participate in arithmetic or reductions.

The rounded subtraction survives in LREG1 while the native scalar helper uses
only LREG0, LREG2 and programmable LREG14. Three register copies replicate the
result into LREG0–3 before inverse transpose. V's output slot is safely reused
only after its input row has been loaded. Prediction DST6 is no longer rewritten
by delta formation, but its next consumer overwrites it with the first readout
term, exactly as before. Q/K, state arithmetic, persistence rounding, reductions,
CBs, reader/writer, partitioning and host interfaces are unchanged.

Frozen files, based on the combined persistence/balance helper:

- `candidates/gdn_sfpu_fused_delta_row.py`
- `candidates/integrated_fused_delta_row134.py`
- `candidates/make_gdn_fused_delta_row_candidate.py`
- `candidates/check_gdn_fused_delta_row_cpu.py`
- `candidates/gdn_fused_delta_row_cpu.json`

CPU/source checks passed. The full compute string reverts exactly to its parent
when the new helper/wrapper and three-call replacement are removed. Twenty-eight
scale/beta cases verify bit-identical full output tiles through the transpose
model, including signed zeros and subnormals. Python compilation passed. These
checks do not replace the SFPU device comparison: explicit negate-plus-add must
match the compiled native subtraction, and instruction hazards need hardware
verification. No timing improvement is claimed and no hardware/main source was
touched.

Parent-serialized probe:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_recurrence_exact_comparison \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_fused_delta_row.py \
  --reference models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_fused_persistence_balance.py \
  --state-dtypes bf16 fp32 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/fused_delta_row_probe.json
```

Require all 54 cases exact, then compare B32 short/512 and B1 latency and accuracy
with the whole candidate. Profile the recurrence if whole-run timings improve.
