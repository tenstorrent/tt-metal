# Decode output projection precision discriminator

Prepared two isolated snapshots on `v5_combined_accurate_residual_explicit_gate_norm.py.snapshot` (de82dc83…). Only public decode's linear-attention output projection uses the new path; the existing native output projection remains for prefill, including internal singleton tails. Setup adds an odd/even-permuted BF164096×2048 weight copy and UINT16 ROW_MAJOR pair indices `[1,128,32]`. No production/hardware changes occurred.

- `v5_ordered_outproj_k512.py.snapshot`: eight K512 sequential scans followed by original ordered block reduction.
- `v5_ordered_outproj_k1024.py.snapshot`: four K1024 scans followed by ordered block reduction.

The adapter is the existing qualified compact projection geometry specialized to input4096/output2048: RM input conversion, adjacent-pair gather, BF16 coefficient tiles and BF16 ordered weights, accurate FP32 SFPU multiply, uncompensated H-axis cumsum, aligned terminal-band extraction, last-row extraction, explicit logicalH1 FastReduce output, BF16 boundary and DRAM return. Each FP32 product/prefix occupies32 MiB, overlapping64 MiB before small CBs/other temporaries; the additional BF16 weight is16 MiB. No allocation or performance result is inferred from this accounting.

A new `ordered_out=False` keyword preserves all previous linear behavior. Only the decode gated tail passes `ordered_out=cache_boundary`; public prefill explicitly uses cache_boundaryFalse even for internal singleton tails, so those retain the old native producer. Existing packed QZ/KVAB branches and other linear roles remain untouched. The CPU source proof removes the inserted setup assignments, guard, keyword and decode-call argument, then verifies **full-module AST equality** to de82. This is a complete scope proof, not only a method-name comparison.

Probe `tests/v5_ordered_outproj_probe.py` loads only the actual layer34 output weight and captured gated BF16 tensor from explicit-gate181. It executes the actual frozen `linear` methods and actual setup AST, including native DRAM-sharded output matmul geometry. Both BF16 and FP32 outputs are compared on **all2048 columns** to canonical CPU F.linear; selected columns additionally require exact agreement with explicitly ordered FP32 sums. Each candidate preserves final DRAM placement. No timing claim is made; all adapter operations are executed in the device comparison.

CPU-only proof completed without TTNN import: all10 selected columns match canonical BF16 output for both K policies, and gather/weight permutations are exact. This does not establish all-column equality or a full decoder pass. The upcoming device probe supplies all-column evidence. Artifacts are not overwritten; try/finally closes physical chip0 and records errors. Existing accuracy thresholds remain unchanged.

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_ordered_outproj_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_outproj_device.json
```

Full evidence hashes are in each snapshot's `.py.provenance.json` and `v5_ordered_outproj_cpu.json`. The CPU report distinguishes the selected ordered sum proof from the full canonical projection comparison. No router/expert or upstream-state behavior is changed by these candidates.

## Completed device/model results

Both K schedules produce exact BF16 output on all2048 actual181 columns; native differs335 codes. Both full8192+33 controls repair181 but fail201 other stream positions, with minimum contributionPCC.9125192963759651 at7944. No schedule is promoted or claimed faster. The completed JSON/logs are v5_ordered_outproj_k512_layer34_seed154 and v5_ordered_outproj_k1024_layer34_seed154; adjacent provenance stores their hashes. Formatted probe source was rerun successfully in v5_ordered_outproj_device_formatted.json; original measured source and exact AST-equality proof are preserved in probe_formatting_provenance.json.
