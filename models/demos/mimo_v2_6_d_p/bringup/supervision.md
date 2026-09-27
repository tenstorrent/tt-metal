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
