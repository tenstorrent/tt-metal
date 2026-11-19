# Isolated V5 ordered post-attention norm control

[Candidate](v5_ordered_postnorm.py.snapshot), SHA256 `ff8227f036acd9ecdacfb83824cac07b1576e6b68594e459327b1f783316d83c`, changes only the first guard in `FusedDecoder.norm` against selected V5 `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f`:

```python
if key in ("input_layernorm.weight", "post_attention_layernorm.weight") and self.kind == "linear_attention":
```

It applies the existing ordered 2048-wide input RMSNorm implementation to the post-attention norm of linear-attention layers, for both prefill and decode. Input norm remains unchanged, and full-attention layers retain their original path. The existing checkpoint loader already uploads both effective gamma vectors as FP32 `weight.float() + 1`; the ordered path reads `self.w[key]`. No new weights, constants or persistent buffers are introduced. The path retains the original `x.memory_config()`, uses its existing interleaved DRAM arithmetic, and restores that original memory configuration before returning. The decode caller's subsequent transfer to DRAM is unchanged.

[AST provenance](v5_ordered_postnorm.py.provenance.json) proves that restoring the single guard makes the entire candidate module AST identical to V5. Arithmetic, normalization epsilon, BF16 output boundary, projections, recurrence, routing and all quality thresholds are unchanged. CPU syntax and scope checks pass. Production and test harnesses were not edited; no device execution was performed by this agent.

This is a discriminator for the shared layer34 failure described in [AutoDebug](AUTODEBUG_v5_layer34.md) and the later [stream capture](v5_layer34_stream_probe.json), which contains positions137,165,181. Applying an exact same-input norm does not remove attention/residual differences presented to that norm, nor establish that those differences or native norm alone cause expert-membership changes. Actual unchanged model gates decide whether the control helps; no speed or quality claim follows from its source.

Parent-only same-stream diagnostic:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_postnorm.py.snapshot \
  --layer 34 --seed 154 --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_postnorm_layer34_seed154.json
```

The cached-oracle, one-repeat run is diagnostic evidence only. Source-bound full validation, capacity, repeated-run and performance evidence would still be needed for selection.

## Device result, 2026-11-13 07:15 UTC

The bounded same-oracle8192+33 diagnostic completed with clean device closure and failed the unchanged .995 gate at279 positions. Minimum output/contribution PCC .927830430452/.912473714217. Position137 is repaired, position181 remains below threshold. This rejects ordered postnorm as a sufficient standalone repair; it remains a useful controlled component for combined hypotheses. Exact command/output/source hashes are in `v5_ordered_postnorm_layer34_seed154.log`, `.json`, and `.json.provenance.json`; comparison is `v5_layer34_repair_comparison.json`. No speed or repeatability claim is made for this single-repeat diagnostic.

## Completed model discriminator

The parent-owned [same-stream8192 result](v5_ordered_postnorm_layer34_seed154.json) fails the unchanged gate: 279 failing rows. Position137 is repaired (output PCC 0.9999692535454229, contribution PCC 0.9999610004005614), while181 still fails (output PCC 0.993931909667549, contribution PCC 0.9928757729326335). This agrees with the bounded CPU counterfactual prediction and shows that this change alone is insufficient. It does not assign the remaining failures to one producer. The candidate remains unselected; the separate ordered Q/Z control is prepared next.
