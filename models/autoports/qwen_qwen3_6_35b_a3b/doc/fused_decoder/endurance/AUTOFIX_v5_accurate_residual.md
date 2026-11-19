# Selected-V5 first-residual-only control

[Candidate](v5_accurate_residual.py.snapshot), SHA256 `7cb0aecda4c52bcf579aef631c26e3dd22174f0027c94505c607ac0a6bd9efe9`, applies one keyword to selected V5 `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f`:

```python
x = ttnn.add(hidden_states, x, fast_and_approximate_mode=False)
```

[Full-module AST proof](v5_accurate_residual.py.provenance.json) establishes that removing this one keyword restores selected V5 exactly. Native Q/Z, native post-attention RMSNorm, native BF16 decode gated norm, KVAB projection, recurrence and final MoE residual addition are unchanged. The shared first add changes for both layer kinds. There are no added persistent allocations, setup copies or host operations.

This isolates the minimum proposed repair from the [combined-precision control](v5_combined_accurate_residual.py.snapshot). Existing source evidence and the CPU-versus-device residual diagnostic limitation are documented in [the combined-control memo](AUTOFIX_v5_combined_accurate_residual.md). Neither the combined implementation nor its precision changes are assumed necessary. This snapshot is unselected; CPU syntax/scope checks pass, and the subsequent parent-owned model test fails as recorded below. Performance is untested. No production or harness edits were made.

Parent-only same-stream diagnostic:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_accurate_residual.py.snapshot \
  --layer 34 --seed 154 --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_accurate_residual_layer34_seed154.json
```

## Completed model discriminator

Selected V5 with native Q/Z, native postnorm and native BF16 gated norm retained. The parent-owned [layer34 seed154 result](v5_accurate_residual_layer34_seed154.json) completed but **fails** the unchanged model gate: **287 failing rows**, minimum contribution PCC **0.9125780044745395** at position 7944. This is an insufficient repair and remains unselected. The original selected-V5 run had 296 unique failing positions; changing that count does not satisfy the acceptance threshold or establish that all affected rows share a root cause.

The first-residual-only intervention does not establish exact postnorm on the actual device residual. The pending actual-norm-input capture will separate residual rounding from normalization on equal operands. Final residual arithmetic was not changed. There is no performance or full qualification claim.

| Position | Output PCC | Contribution PCC |
| --- | ---: | ---: |
| 137 | 0.999973363867 | 0.999964933446 |
| 165 | 0.991531798650 | 0.989694480267 |
| 181 | 0.993937430015 | 0.992876336388 |

Raw model JSON SHA256: `d7bf4a58761ea83143a5ab63a131dd2fbb7ae30ad2c749ac45621a33300563f6`. [Updated provenance](v5_accurate_residual.py.provenance.json) records the raw evidence hashes and diagnostic scope.
