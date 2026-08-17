# AutoFix: GDN decode post-attention norm placement

## Starting Evidence

- Source: `AUTODEBUG.md`, ranked hypothesis 1.
- Original command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`
- Original failure: program 240 RMSNorm CB region ends at 1,520,512 while a live L1 buffer starts at 1,447,616.

## Hypothesis Experiments

- Hypothesis: the GDN decode post-attention `ffn_norm` L1 output destination causes the L1/CB clash.
- Experiment A: autoport-only wrapper changed the inherited decode norm's post-op `output_mem_config` from L1 to DRAM, leaving attention norm, GDN, MLP, weights, shapes, and demo sources unchanged.
- Result: failed at the same `ttnn.rms_norm`, program 240, and identical addresses before the post-op conversion ran.
- Verdict: refuted for the post-op conversion.
- Experiment B: the wrapper invoked the same `ttnn.rms_norm` directly with `memory_config=ttnn.DRAM_MEMORY_CONFIG`, changing only the norm kernel's output allocation.
- Result: failed at the same `ttnn.rms_norm`, program 240, and identical addresses (L1 allocation 1,447,616; static CB end 1,520,512).
- Verdict: refuted. The conflicting allocation is already live when RMSNorm is admitted; it is not this norm's requested output allocation.
- Verification command (both experiments): `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`
- Device handling: each pytest run opened device 0 and the fixture closed it cleanly. `timeout 60 tt-smi -ls --local` could not run because `tt-smi` is absent from PATH.
- Fix: none retained. Both speculative wrapper variants were removed.

## Final Status

- First-ranked norm-output-placement hypothesis refuted; original test still fails.
- Next focused work should identify the live L1 allocation at 1,447,616 or isolate attention-norm lifetime, as ranked in `AUTODEBUG.md` experiments 2 and 3.
