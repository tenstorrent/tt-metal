# AutoFix: GDN decode residual/output L1 ownership

## Starting Evidence

- Sources: `AUTODEBUG.md`, `AUTOFIX_GDN_NORM.md`, and `AUTOFIX_GDN_ATTN_NORM.md`.
- Original command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`.
- Original failure: the first eager real-weight GDN decode reaches post-attention `ffn_norm`; program 240 requires static CB space through byte 1,520,512 while a live L1 allocation begins at byte 1,447,616.

## Hypothesis Experiments

- Hypothesis: the live GDN attention output or residual `h` is L1-resident and blocks admission of the post-attention RMSNorm CBs.
- Prediction: an autoport-local reproduction of the GDN decode composition will identify either tensor in the overlapping L1 range; moving only the verified owner to DRAM and deallocating its L1 buffer before `ffn_norm` will let the original test advance through traced PCC.
- Address probe command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`.
- Address probe result (successful retry): input `hidden_states` was interleaved DRAM at 57,602,688; `attn_output` was interleaved L1 at 1,503,232; residual `h` was already interleaved DRAM at 57,660,032. A DRAM conversion of `h` was therefore an identity (same address 57,660,032). The run still failed at `ffn_norm`. An immediately preceding run ended in a native segfault at the same RMSNorm boundary before buffered probe output was emitted, so it was treated as invalid infrastructure evidence and retried.
- Residual verdict: **refuted**. `h` is not L1-resident and cannot own the reported L1 allocation.
- GDN-output A/B: copied only `attn_output` from L1 to DRAM, deallocated its original L1 buffer, then performed the unchanged residual add, RMSNorm, MLP, and residual math. Precision, weights, recurrent state, shapes, and norm policy were unchanged.
- GDN-output command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`.
- GDN-output result: failed at the same `ffn_norm` program 240 with the original lower live allocation at byte 1,447,616 and unchanged static CB end at byte 1,520,512. Trace capture and decode PCC were not reached.
- GDN-output verdict: **refuted as the blocking owner**. Although the output occupies L1 at byte 1,503,232, removing it exposes the earlier allocation at byte 1,447,616; another GDN live allocation remains the actual blocker.
- Fix: none retained. All instrumentation and both speculative placement variants were removed; no demo/common source was modified.
- Evidence logs: `/tmp/autofix_gdn_residual_retry.log` (address probe) and `/tmp/autofix_gdn_output_dram.log` (output A/B). These are machine-local transient logs; the exact durable results are recorded above.
- Device handling: hardware-facing runs were serialized. The pytest fixture opened device 0 only and the completed A/B run closed it during teardown. `tt-smi` is unavailable on `PATH`, consistent with the prior reports.

## Final Status

- The requested residual/GDN-output ownership hypothesis is refuted; the original real-weight GDN decode remains failing before trace capture.
- The next focused experiment should inspect persistent and transient GDN state allocations, especially the allocation beginning at byte 1,447,616, before changing any additional placement policy.
