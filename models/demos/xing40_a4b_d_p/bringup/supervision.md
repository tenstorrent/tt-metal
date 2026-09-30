# Xing4.0-29B-A4B bring-up: supervision log

Overseer log: time, task, trigger, classification, action, resulting commit.

## 2026-09-30 intake
- Owner prompt: xing_bringup_prompt.md (repo root). Box: 8x Blackhole p150b (LoudBox), mesh 2x4, FABRIC_2D opens and
  all_gather passes on both axes (probe 02:56).
- XingChen-AGI/Xing4.0-29B-A4B @ baae3c3e813cad5f888f1f485cfff659c89076c5, not gated, apache-2.0, 62 GB bf16,
  downloaded to /localdev/dnijemcevic/bringup/xing40_a4b_d_p/hf; every spec.checkpoint.expect shape verified.
  Custom modeling code (trust_remote_code) builds on python_env's transformers 5.12.1; nothing vendored.
- Fit: all 40 layers, <= ~11 GiB per chip (experts bfp8 3.3 + rest bf16 replicated 5.1 + MLA cache 2.4).
- 03:05 owner approved the spec ("yes"); intake approved, ledger --early 8 tasks, run1.
