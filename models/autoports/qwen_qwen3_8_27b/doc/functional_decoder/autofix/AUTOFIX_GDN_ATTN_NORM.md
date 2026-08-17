# AutoFix: GDN decode attention-norm placement

## Starting Evidence

- Sources: `AUTODEBUG.md` experiment 2 and the refuted post-attention norm experiments in `AUTOFIX_GDN_NORM.md`.
- Original command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`
- Original failure: the first eager decode reaches post-attention `ffn_norm`; program 240 requires static CB space through byte 1,520,512 while a live L1 buffer begins at byte 1,447,616.

## Hypothesis Experiment

- Hypothesis: the earlier GDN decode `attention_norm` L1 result remains live and causes the later `ffn_norm` RMSNorm CB clash.
- Prediction: forcing only `attention_norm`'s decode post-conversion to DRAM, while leaving `ffn_norm`, GDN, MLP, weights, and shapes unchanged, should let the original command advance through the later RMSNorm.
- Experiment: an autoport-local wrapper intercepted only the linear-attention layer's decode `attention_norm` call and replaced `output_mem_config=L1` with `output_mem_config=DRAM`. The first attempted wrapper compared the `Mode` enum directly with a string and was therefore an invalid/no-op setup check; the corrected experiment compared `Mode.value` and is the result used below.
- Command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`
- Result: failed at the same post-attention `ttnn.rms_norm`, program 240. The live L1 allocation moved only 4,096 bytes, from 1,447,616 to 1,451,712, and still overlapped the unchanged static CB end at 1,520,512. Trace capture and decode PCC were therefore not reached.
- Verdict: **refuted**. Attention-norm L1 placement affects allocator position but is not the allocation whose removal makes the later RMSNorm admissible.
- Fix: none retained. The autoport wrapper was removed after the refutation; demo/common sources were not modified.
- Device handling: the pytest fixture opened device 0 only and closed it during teardown. Hardware-facing runs were serialized.

## Final Status

- The original real-weight GDN decode remains failing before trace capture.
- Both decode norm-placement hypotheses are now refuted. The next focused AutoFix experiment should identify the live allocation around byte 1,451,712 using tensor memory configs/buffer addresses around GDN output, residual add, and state buffers, then test only the identified owner or the next ranked MLP-L1 hypothesis.
