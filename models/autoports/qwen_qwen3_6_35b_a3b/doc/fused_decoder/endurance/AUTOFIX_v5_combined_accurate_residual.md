# Combined-source accurate attention-residual add control

[Candidate](v5_combined_accurate_residual.py.snapshot), SHA256 `e3aea1c50aaf54b0ffa69949f8fdcc6bbe7232a64fac3a9fed8d3eaf052f2e6c`, is based on the captured combined source `16fdf3f632f685af1b8e279f933fb28137308d8dc1f3965dafbb50c797401870`. It changes exactly one call in `_forward`:

```python
x = ttnn.add(hidden_states, x, fast_and_approximate_mode=False)
```

[Full-module AST provenance](v5_combined_accurate_residual.py.provenance.json) proves that removing this keyword restores the complete combined-source AST. It leaves norm arithmetic, projections, gated norm, recurrence and final MoE residual add unchanged. The shared `_forward` applies this first-add change to both layer kinds; no kind guard or other behavior is introduced. No new persistent weights/state are allocated. CPU syntax/scope checks pass. The parent-owned model test subsequently failed as recorded below; this agent performed no device work.

## Source-backed hypothesis and diagnostic limitation

[Binary dispatch](../../../../../../ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp), `resolve_fast_and_approximate_mode` at922, defaults ADD/SUB/RSUB's unset flag to true, keeping the FPU route. [BinaryNg validation/routing](../../../../../../ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp),21–64, explicitly documents the accurate BF16 nearest-even SFPU path when false. It accepts this option for BF16 output and selects SFPU for ADD. The actual first residual operands are BF16, so the source contract is applicable. Installed-device behavior remains for the parent to verify.

The existing [combined capture](v5_combined_layer34_stream_probe.json) labels `post_norm_local_arithmetic` by constructing a **CPU** BF16 residual, `x + tt_values["attention"]`, then applying HF postnorm. In `tests/v5_layer_stream_probe.py` around477–486 it does not yet compare norm on the actual TT residual input. Thus that field includes differences from the preceding device residual add: it is not proof that the ordered norm itself disagrees on identical inputs. This explains why exact inputnorm can coexist with hundreds of apparent postnorm-local BF16 differences. The independently owned capture extension will record actual norm input to separate residual rounding from same-input normalization. No causal conclusion is made before that comparison.

A final residual-add precision change would alter output arithmetic but cannot repair the current token's upstream router input. It is deliberately excluded from this first-add discriminator. Accuracy gates remain unchanged; full model and capacity evidence are still pending.

Parent-only same-stream diagnostic:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_combined_accurate_residual.py.snapshot \
  --layer 34 --seed 154 --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_combined_accurate_residual_layer34_seed154.json
```

## Completed model discriminator

Combined ordered Q/Z, ordered postnorm and BF16 gated-norm control. The parent-owned [layer34 seed154 result](v5_combined_accurate_residual_layer34_seed154.json) completed but **fails** the unchanged model gate: **247 failing rows**, minimum contribution PCC **0.9127081311510953** at position 7944. This is an insufficient repair and remains unselected. The original selected-V5 run had 296 unique failing positions; changing that count does not satisfy the acceptance threshold or establish that all affected rows share a root cause.

The first-residual-only intervention does not establish exact postnorm on the actual device residual. The pending actual-norm-input capture will separate residual rounding from normalization on equal operands. Final residual arithmetic was not changed. There is no performance or full qualification claim.

| Position | Output PCC | Contribution PCC |
| --- | ---: | ---: |
| 137 | 0.999984699048 | 0.999980112215 |
| 165 | 0.999989508576 | 0.999986019703 |
| 181 | 0.993941115902 | 0.992883149128 |

Raw model JSON SHA256: `f98ef02228e8e33149b24ed0f667cb063ecd0e19f221946713e8cef2af56b348`. [Updated provenance](v5_combined_accurate_residual.py.provenance.json) records the raw evidence hashes and diagnostic scope.
