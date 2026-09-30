# XiaomiMiMo/MiMo-V2.6-Flash-RL bring-up on mesh 1x4: breadcrumbs

Prior bring-up: mimo_v2_6_d_p (mesh 1x4); goldens and CPU reference shared. Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## PL.1 plan (attempt 1, 2026-09-30)

- Wrote `plan.yaml`, `plan.md` and `components.yaml` for the owner's CP=4 scheme on 1x4. The residual and the KV cache
  are split by sequence (chip c holds slice c of each chunk, in a chunk-major ring cache). Attention is TP=1 (whole
  qkv/o_proj, all heads, no all_reduce) through `ttnn.transformer.ring_joint_scaled_dot_product_attention`: causal
  ring for the full layers, window 128 + sink + compact halo for the sliding layers. The dense MLP and the router run
  locally. The experts are EP=4 with one fabric dispatch group of 4 on axis 1 (`ttnn.bringup.dispatch/combine/offset_cumsum`,
  then `post_combine_reduce`, no reduce after). The LM head is vocab-sharded after a [32, 4096] last-row all_gather.
- The gate prints 15.95 GiB per chip of the 27.20 budget. plan_fits 1, 0 unplaced, 0 plan / component / ledger errors.
  plan_approved 0 until a person approves (`approvals.yaml`).
- Forced by the ring op's validation (`ring_joint_sdpa_device_operation.cpp`): V is padded 128 -> 192 (tensor-V mode
  needs VDH == DH), the sliding K/V cache is bfp8 (the sliding path requires BFP8_B), and the sink needs streaming
  compute (fp32 dest off, as preset S). Full-layer caches stay bf16.
- Risks for the implement steps: MiMo's 64Q:8KV at D192 on the ring sliding path is unexercised (documented for the
  GPT-OSS 8Q:1KV D64 shape), so probe it at s4096 first. A bfp8 sliding cache may cost sink-layer accuracy. Both have
  a fallback: a `ttnn.bringup` ring_joint fork behind a default-off option. Axis-1 dispatch with DGS 4 on the forks is
  new in this repo.
- Gotcha: this checkpoint map lists `model.mtp.*` (48 tensors), so the plan skips them explicitly. In the plan.yaml
  flow mappings, a `group:` with a comma must be quoted, or YAML splits it into another key.
- tasks.yaml is unchanged (same block graphs and components).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`
