# Hy4 Preview (layers 0-5) bring-up: supervision log

Overseer log: time, task, trigger, classification, action, resulting commit.

## 2026-09-28 intake
- Owner prompt: hy4_bringup_prompt.md (repo root). Study: hy4_preview_findings.md (committed with the scaffold).
- Checks: tencent/Hy4-preview @ 705d81ee51566a186d645b74c974d642ef2828fe, not gated, 1.56 TB BF16, instruct model
  (model card; chat template with reasoning_effort, no_think used). Config and tensor shapes match the study; HF eager
  applies the top-k mask and sinks, so hf.parity_seq 4096 (> index_topk 2048). Tokenizer + template work on
  transformers 5.12.1; hy_v4 5.17 code vendored in /localdev/dnijemcevic/bringup/hy4_preview_d_p/hf_code.
- Correction to the study: layers 0-5 + non-layer tensors are 35 shards, 127 GB (not 415 GB).
- The R.1/R.2 HF sanity needs the whole model. Owner chose the full BF16 download (2026-09-28 22:35); asked for the
  framework to drop the unneeded weights after the sanity -> F47 (task R.4, commit 8351a01c23c), spec
  checkpoint.trim_drop [model.mtp_layers.*]. Owner also had the MiMo checkpoint trimmed to layers 0-5 (023405b50b0).
- Owner approved the spec ("yes, full BF16 download, go ahead"); intake approved 22:36; ledger --early 9 tasks; run1.
- Launch waits for the full download (R.1's checkpoint counts and R.2's sanity need every shard).
