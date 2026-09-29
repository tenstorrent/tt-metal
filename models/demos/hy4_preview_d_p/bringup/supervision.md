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
- 2026-09-29: owner: keep the MTP layer out of the trim ("we'll see what next"; a possible follow-up would attach the
  MTP block after the layer subset). Removed checkpoint.trim_drop from the spec; R.4 now keeps layers 0-5, every
  non-layer tensor and model.mtp_layers.* (~20 GB). The spec edit voided the intake approval; re-approved on the
  owner's word.
- 2026-09-29 01:45: full BF16 download complete (131/131 shards, 1.5 TB; 588 GB free). run1 launched (orchestrator in
  the background, log /localdev/dnijemcevic/bringup/hy4_preview_d_p/runs/run1/orchestrator.log). R.1 PASS 1ba084cc58f.
- 2026-09-29 02:30-02:57 R.2 attempt 1: HF sanity first gave smoke "Paris" but text_top1_acc 0.237 (MiMo 0.955). Owner:
  if no bug is found, lower the floor to 0.2. The agent found one: the transformers 5.17 hy_v4 port uses rotate-half
  RoPE, but SGLang serves Hy4 with interleaved RoPE (configs/hy_v4.py hardcodes rope_interleave = True and
  indexer_rope_interleave = True; checked by the overseer). With the fix in its HF copy, accuracy 0.965. Floor left
  at 0.4; no spec change.
- 2026-09-29 03:40 owner delegated approvals and board resets: "can you just take over resetting the board, approving
  the spec etc". Overseer now approves the plan, the perf picks and op requests, and resets the board (only with no
  other device job running). Still asked: pushes and launching op-gen.
- 03:51 R.2 PASS 0f409282718 (attempt 1): acc 0.965, smoke ok, parity PCC 1.0 on L0-5 and logits. Reviewed: the
  reference (hy4_ref.py) is standalone (weights read directly, no HF import); the RoPE fix is in both the vendored
  oracle (marked) and the reference, with a finding (R2-hf-rope-interleaved). Accepted.
- R.4 PASS 704c7d08b3e: checkpoint trimmed 1.5 TB -> 116 GB (1436 GB freed, 188 tensors kept incl. 27 MTP, 0 verify
  errors). R.3 PASS 6eb263e888b on the precheck (chunked == one-shot, graph replay exact). Dashboards republished.
- 04:40 goldens s4096/s16384/s56320, B.1, PL.0 (104 tasks) PASS. PL.1 plan written (attempt 1): SP2 (rows) x TP2
  (cols), 32 heads per chip for sparse_sdpa, block-cyclic MLA latent + index-key caches, EP=4 with the DeepSeek 2D
  dispatch on axis 0 (MiMo 2x2 precedent), dense MLP / shared expert TP2, fp32 iHC streams, bf16 attention / indexer
  weights and index keys, bfp8 experts, HiFi4 + fp32 accumulation everywhere; sink passed as sink x 16 with scale 1/16;
  indexer dims permuted on the host so the op's RoPE half matches; no host RoPE permutation in the MLA (R.2 finding).
  Gate numbers checked by the overseer: 20.60 of 27.20 GiB per chip, 0 unplaced tensors, 0 plan / component / ledger
  errors, no CPU or OPGEN step, tasks.yaml unchanged. Approved by the overseer (owner delegated 03:40); resumed.
