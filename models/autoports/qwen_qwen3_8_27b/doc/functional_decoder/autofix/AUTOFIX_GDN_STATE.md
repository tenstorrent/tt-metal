# AutoFix: GDN persistent-state L1 ownership

## Starting Evidence

- Sources: `AUTODEBUG.md` and the three prior norm/residual refutation reports.
- Original command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`.
- Original failure: post-attention RMSNorm program 240 needed static CB space through byte 1,520,512 while a live L1 allocation began at byte 1,447,616.

## Hypothesis Experiments

- Hypothesis: a returned persistent GDN state, rather than the residual or norm output, owns the earlier live L1 allocation.
- Probe: an autoport-test-only `ffn_norm` wrapper logged tensor memory configs and addresses immediately before the failing norm. The wrapper was removed after the probe.
- Probe command: the original real-weight command above; log `/tmp/autofix_gdn_state_probe.log`.
- Result: prefill `recurrent_state` was DRAM at 58,784,384 and `fused_conv_state` was DRAM at 58,210,944. After the first decode, `recurrent_state` was L1 at exactly 1,447,616 and `fused_conv_state` was L1 at 1,515,520. `ffn_norm_input` remained DRAM at 57,660,032. `split_conv_state` was absent on this fused-state path. Device-resident projection, norm, and precomputed `A_neg` tensors were DRAM; the three raw convolution-weight tensors reported host storage and therefore had no device buffer address.
- Verdict: **verified** for `recurrent_state` as the exact original owner. Moving only it to DRAM exposed the adjacent `fused_conv_state` allocation at 1,515,520, still below the same 1,520,512 CB end; this verified the second returned state as a co-owner.
- Fix: the autoport wraps only the inherited GDN forward boundary. When returned `recurrent_state` or `fused_conv_state` is L1, it copies that persistent state to DRAM, updates the owning attribute, and deallocates the obsolete L1 tensor before inherited post-attention RMSNorm. No GDN math, dtype, weight, residual, norm, MLP, or demo/common source changed.
- Placement A/B log: `/tmp/autofix_gdn_state_ab.log` moved the collision from 1,447,616 to 1,515,520. `/tmp/autofix_gdn_state_ab_both.log` passed the original RMSNorm/MLP decode and reached trace setup.
- Trace setup finding: the test changed from non-inplace eager decode directly to capture, so capture tried to compile new in-place state-update programs. A single in-place warmup was added before capture, and the HF reference step count was updated from three to four.
- Trace verification command: the original command above; log `/tmp/autofix_gdn_state_trace_warm_retry.log`.
- Trace verification result: eager decode, in-place warmup, trace capture, and trace replay all completed. Replay output PCC was **0.9740499002269837**, below the required 0.995.
- Device handling: hardware commands were serialized on device 0. One failed-capture pytest remained alive holding the device lock; only the two experiment pytest processes were terminated before retry. The successful retry fixture closed device 0 cleanly.

## Final Status

- **L1/CB clash fixed with verified autoport-only state placement.**
- **Stage-critical traced decode correctness still failing:** PCC 0.9740499002269837 < 0.995 after successful replay.
- Retained changes are the smallest proven state-placement fix plus the required in-place warmup/reference-step correction. The next AutoFix pass should localize eager-inplace versus captured/replayed output/state drift; it should not revisit the refuted norm/residual placement hypotheses.
