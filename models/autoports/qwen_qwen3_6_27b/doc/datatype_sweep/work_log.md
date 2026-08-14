# Qwen3.6-27B Datatype Sweep Work Log

Stage target: Qwen/Qwen3.6-27B on 4x Blackhole p300c, TP4 `MeshShape(1,4)`.
Accuracy gates: top-1 >= 90%, top-5 >= 98%, top-100 = 100% on the refreshed
AIME24 chat-template reference with 100 generated tokens. Candidate ranking uses
only trace-verified teacher-forcing decode tokens/s/user.

## 2026-08-14 discovery and hardware health

- Preserved pre-existing dirty/untracked files shown by `git status`; stage
  changes are isolated to datatype-sweep artifacts and precision-policy runtime
  plumbing.
- `timeout 60 tt-smi -ls --local`: four Blackhole p300c devices visible.
- `timeout 180 tt-smi -r`: exit 0, devices 0-3 reset.
- post-reset `timeout 60 tt-smi -ls --local`: all four devices visible/resettable.
- `ttnn.open_mesh_device(ttnn.MeshShape(1,4), trace_region_size=0)` then close:
  `MESH_SMOKE_OK`.
- Firmware 19.8.0 emitted a newer-than-tested warning; the mesh opened and
  closed cleanly. This is tracked as environment provenance, not a model result.

## Precision propagation prerequisite

The prior generator always constructed decoder candidate `default` and exposed
only `lm_head_dtype`; CCL dtype and a complete dtype/fidelity artifact could not
flow into the measured path. Added a strict `precision_config.py` loader, default
selected-artifact/env resolution, per-layer policy construction, CCL and LM-head
propagation, and a constructed-runtime precision summary. Unsupported or ignored
schema fields fail validation instead of being silently documented.

Host verification:

```text
python -m compileall -q models/autoports/qwen_qwen3_6_27b/tt/{precision_config.py,model.py,multichip_decoder.py}
pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_full_model_public_contract.py
6 passed
```

## Reference and full-model sweep

- Refreshed the exact pinned HF revision with `--prompt-source aime24
  --chat-template --gen-len 100 --top-k 100`; S161 + 100 continuation tokens.
- Baseline prefill: 92/100 top-1, 100/100 top-5/top-100.
- Baseline traced teacher forcing: 97/100 top-1, 100/100 top-5/top-100,
  TTFT 5153.68 ms, 6.91 t/s/u.
- Evaluated nine full-model policies. Exact rows and commands are in
  `sweep_results.{json,csv}`; every ranking metric comes from the traced
  teacher-forcing runner.
- Selected `full_attention_bfp4_lofi`: 93/100 top-1, 100/100 top-5/top-100,
  TTFT 5628.30 ms, 7.00 t/s/u. It is the fastest passing evaluated config.

## AutoFix: BFP8 MLP-down policy plumbing

All-projection BFP8 HiFi2 and LoFi initially failed identically with a
14,656-byte L1/static-CB overlap. Fresh `AUTODEBUG.md` isolated the first-layer
MLP-down `[32,4352] x [4352,5120]` op and refuted fidelity as cause. Both TP4
MLP-down call sites hard-coded the BFP4-tuned width 17 instead of consuming
`OptimizationPolicy.mlp_down_in0_block_w`. The smallest fix consumes the policy
field and uses the only smaller legal divisor, width 1, for BFP8 candidates.
Original commands then passed: HiFi2 97/100 at 6.26 t/s/u; LoFi 98/100 at 6.64
t/s/u. Pre-fix logs are preserved as `*_pre_autofix.log`; see `AUTOFIX.md`.

## Post-selection gates

- Default-path B1/S128/G128 warmed token-out: TTFT 5244.60 ms, 17.8968 t/s/u,
  128 no-readback replays, device token feedback, semantic greedy overwrite.
- Non-aligned mixed S65/S63: pass; inactive KV exact; reset/reuse pass.
- Shared six-prompt qualitative suite: exact Qwen2Tokenizer chat template,
  exact HF revision, 50 greedy tokens, matched HF controls; coherent pass.
- Selected capacity recomputation: exact BFP4 tile ratio changes full-attention
  projection residency from 2,264,924,160 to 637,009,920 bytes/device; selected
  full-model weights are 11,832,539,648 bytes/device. Isolated physical probe:
  B1 C262144 pass; B32 C78016 pass / C78080 fail. No capability reduction.

Independent stage review subsequently returned `clean-pass`; the local checkpoint commit is recorded below.

## Stage-review remediation

First independent verdict: `more-work-needed` (`STAGE_REVIEW.md`). Findings and
resolutions:

- Missing BFP4+HiFi2 family controls: added dtype-matched full-model rows.
  The first generated controls were confounded by restoring full-attention
  BF16; their logs are retained as `*_confounded.log` and are not sweep rows.
  Corrected all-BFP4 controls differ from selected only in the named family:
  MLP HiFi2 passes at 6.72 t/s/u and linear-attention HiFi2 at 6.90 t/s/u;
  selected LoFi remains faster at 7.00.
- Missing activation/residual candidate: plumbed activation dtype through the
  real embedding, projection, residual, CCL, and runtime-summary boundaries.
  The first BFP8 run proved `nlp_create_qkv_heads_decode` requires BF16/FP32;
  a local BF16 adapter fixed that exact boundary. The rerun passes accuracy at
  5.97 t/s/u, so selected BF16 residual is faster.
- Pareto frontier bug: replaced order-dependent accumulation with pairwise
  maximize-accuracy/maximize-throughput dominance, including equal-accuracy
  handling. Independently derived frontiers are top-1 = BFP8-attention HiFi2,
  selected BFP4+LoFi, BFP8-CCL; top-5 = selected BFP4+LoFi only.

Fresh independent rereview returned `clean-pass` after verifying the corrected same-dtype fidelity controls, activation/CCL candidate, Pareto construction, runtime policy consumption, capacity probes, non-aligned coverage, qualitative evidence, and post-selection token-out result. See `STAGE_REVIEW.md`.

## Local checkpoint

- Branch: `mvasiljevic/fmf/qwen-qwen3-6-27b`
- Stage-owned changes are committed locally without pushing. The implementation/evidence commit SHA is recorded in the final handoff because a commit cannot contain its own SHA.
