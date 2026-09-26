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
