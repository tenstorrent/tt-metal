# AutoFix Report

## Starting Evidence

- Source report: `AUTODEBUG.md`.
- Original command: `timeout 300 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_full_attention_smoke.py`.
- Original failure: `TT_FATAL Expect input_tensor to be sharded` at TP4 `paged_update_cache`.

## Hypothesis Experiments

- Hypothesis: the TP4 loader replaced the optimized baseline's required batch-height-sharded decode-attention layout with interleaved DRAM.
  Experiment: compare loader/head-creation memory configs and paged-update validation; restore only the workload-derived one-core-per-user HEIGHT layout and rerun the original command.
  Result: both K/V paged updates passed and execution advanced through paged SDPA.
  Verdict: verified.
  Fix: derive the batch grid exactly as the optimized baseline and pass `self.decode_attention_memory_config` to local Q/K/V head creation.
  Verification: unchanged original command advanced beyond the original fatal.

- Hypothesis: the later `nlp_concat_heads_decode` fatal was the same omitted layout restoration after SDPA, whose output must remain DRAM for the SDPA op itself.
  Experiment: restore the optimized baseline's single `to_memory_config(..., self.decode_attention_memory_config)` after paged SDPA and rerun unchanged.
  Result: command exited zero; output PCC against serialized optimized single-chip TTNN baseline was 1.0, all four replicas agreed, local key-cache shapes were four copies of `(1, 1, 64, 256)`, and fallback hard-failure mode remained enabled.
  Verdict: verified.
  Fix: retained the SDPA DRAM output contract and restored the height-sharded concat input boundary.
  Verification: `MULTICHIP_FULL_DECODE 1.0 replicas_equal=True cache_shapes=[(1, 1, 64, 256), ...] fallback_audit=True`.

## Final Status

- Fixed for the focused full-attention B1 decode smoke.
- Remaining stage work: target-batch, prefill/non-aligned, permuted pages/positions, linear-attention state, trace, watcher, and performance gates.
