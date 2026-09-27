# Supervision log: MiMo-V2.6-Flash-RL (layers 0-5), run1

Overseer: the /bringup session. Owner delegated every decision for this run on 2026-09-26 ("Do it yourself from
beginning to the end"; "Reset the board if you think you need it, you are in autonomous mode now"): the overseer
approves the intake, the plan and the perf picks, and may reset the board when a stop is a box fault and nothing else
holds the device. No push.

Dashboards: standard https://claude.ai/artifact/U84wcPFKnAWw5bwk1zbPRD, teletext
https://claude.ai/artifact/DiE4z2vUUEXV2snyj4sWAf (republish both after every gate).

| Time | Task | Trigger | Classification | Action | Commit |
|---|---|---|---|---|---|
| 15:35 | intake | MiMo chat template thinks by default | framework | F38: spec `text.template_kwargs` reaches the prompt wrap and smoke | 54c3eb51e00 |
| 15:40 | intake | spec written, validates | approval (delegated) | approved intake: layers 0-5, bfp8 experts, 56320/5120, both dashboards, enable_thinking false | - |
| 15:50 | R.1 | stock HF loader cannot run the checkpoint on this host | framework | F39: hf.custom_loader moves HF sanity into R.2; spec edited, intake re-approved (delegated) | cbbbedc2824 |
| 16:59 | R.2 | gate PASS after 1 attempt | review | accepted: reference independent of HF modeling (shares dequant weights.py with the oracle; full-model smoke 'Paris' and top-1 0.955 validate the dequant) | f6f26163b79 |
| 17:04 | R.3 | gate PASS, no code change | routine | accepted | 028a9233b76 |
| 19:30 | G.s56320 | 6-layer subset golden ran all 48 layers on CPU (~2 h) | framework | owner: fix it; F40: subsets stop at their last layer (goldens, check_hf parity), CPU gates use physical cores; R.2 agent's 3 known-issue proposals taken | c755b88b557 |
| 19:50 | PL.1 | plan ready (14.65/27.2 GiB per chip; TP4 attention by the checkpoint's TP ranks, EP4 bfp8 experts, V padded to 192, sinks via SDPA attention_sink) | approval (delegated) | reviewed plan.md, approved; V pad noted as a perf candidate | - |
| 20:00-23:30 | C.*, S.*, M.1, L.* | 41 gates passed, each on its first agent attempt | review | spot-checked attention, router, experts, M.1 diffs: forwards free of torch/host round trips, experts bfp8 (assert against bfp4), host_transfers_per_layer 0 | 54b087ec1ca..785d65d0a4d |
| 00:05 | K.1 | STOPPED after 3 attempts: acks 12 vs 96 | framework | contract test counted all 48 layers, not the 0-5 subset (agent found it); F41 served_layers; agent's contract code kept in the tree; rerun K.1 | 012ad529c80 |
| 00:25 | X.2 | perf picks (delegated) | decision | X.1: device 1122 ms, experts 959 (85%), attention 142 (13%); host transfers already 0, so "remove host work" has nothing to remove. P.1: fused experts (unified_routed_expert_moe honours HiFi2 + fp32 dest for Silu, as ce75b1f0d7f did for GeluTanh; loop -> unified), gate: frozen experts tests + L.last + profile, experts < 350 ms, wall < 700 ms. P.2: SDPA config A (as Gemma P.2), attention < 120 ms. X.3 deps -> P.2 | - |
| 00:42 | P.1 | gate PASS, attempt 1 | review | accepted: opt-in high_precision flag in unified_routed_expert_ffn (default off, program-cache key, 143 existing op tests pass); device 1122 -> 257 ms, experts 959 -> 94 ms, wall 1280 -> 258 ms, PCC unchanged | e3150c04460 |
| 01:05 | P.2 | gate PASS, attempt 1 | review | accepted: full layers config A (q512/k128), sliding layers streaming HiFi4 preset S (sink logit ~28 above the rest makes HiFi2 lose 3-6% per head); attention 142 -> 111 ms, wall 258 -> 226 ms, chunk PCC 0.9984 | 3c999ad8afa |
| 01:15 | X.3 | gate FAIL: prefill_ms_full MISSING, every other metric in range | framework | full_prefill skipped any layer subset (fix agent found it); stopped the fix agent and the orchestrator after the gate's device tests; F42; rerun X.3 | bfb77b8ce2e |
| 01:25 | X.3 | gate PASS on the pre-agent check after F42 | routine | run complete: 60/60 | 2d4df7dd229 |

## Final state (2026-09-27 01:25)

Run1 complete, 60/60 tasks PASS, nothing pushed. Layers 0-5 of 48 (a subset result, not a full-model result).
- Accuracy, full 56320-token ladder (11 chunks of 5120, all on device, tuned model): min layer PCC 0.9984, state min 0.9963,
  host transfers per layer 0. HF sanity of the checkpoint on the full 48-layer CPU model: smoke "Paris", top-1 0.955.
- Warm full prefill 0 -> 55k (6 layers, no readback, no LM head): 2.21 s, 25.5k tok/s; per chunk 175 -> 228 ms.
- Chunk 50k -> 55k: 226 ms (X.1 baseline 1280 ms): experts 93 ms, attention 111 ms (SDPA 54 ms full + 2.5 ms sliding,
  qkv 24, o_proj 21), router 7, dense MLP 6.
- Chunk time by position (5k chunk at 0 / 50k / 100k / 150k / 200k): 175 / 223 / 274 / 324 / 375 ms.
- Perf picks: P.1 fused experts via opt-in `high_precision` in unified_routed_expert_ffn (1122 -> 257 ms device),
  P.2 SDPA config A on full layers + streaming HiFi4 on sliding-with-sink layers (257 -> 225 ms).
- Framework fixes this run: F38 template kwargs (thinking models), F39 custom HF loader, F40 subsets stop at their last
  layer (goldens, parity) + physical-core threads, F41 contract serves the subset, F42 full prefill of a subset.
- Open: V padded 128 -> 192 in the KV cache and SDPA (bandwidth); LM head on host; qkv / dense MLP in bf16 (bfp8 is a
  candidate); the full 48-layer model does not fit the box.
