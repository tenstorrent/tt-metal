
## Session 17 — MVBench accuracy vs. published (Docker image, single-instance DP=4)

Model card only publishes the 15-benchmark average (63.1); per-benchmark numbers are in the tech report
(arXiv 2601.10611, Table 2: MVBench 75.9 for Molmo2-8B). Ran MVBench through the production
`molmo2-ti-dev:local` container (greedy, `max_tokens=16`, MCQ letter prompt, `file://` videos, 4 concurrent).
Script: `verification/eval_mvbench.py`; per-question results: `verification/mvbench_results.jsonl`.

| Result | Value |
|---|---|
| Overall (18 of 20 tasks, 3600 questions) | **74.81%** (2693/3600), 0 server errors |
| Published (all 20 tasks) | 75.9 |

Not run: `episodic_reasoning` (TVQA frame directories, not video files) and `fine_grained_pose` (NTU videos not in
the HF dataset), 400 questions. Per task: action_antonym 99.5, moving_attribute 98.0, object_existence 97.0,
scene_transition 94.5, unexpected_action 89.5, object_interaction 85.0, character_order 84.0, moving_count 83.0,
counterfactual_inference 81.5, action_sequence 81.0, moving_direction 78.5, state_change 78.0, action_count 74.5,
action_prediction 61.0, fine_grained_action 49.5, action_localization 42.5, object_shuffle 37.5, egocentric_navigation 32.0.
Prompt/frame-sampling details of the paper's eval harness are not public, so a ~1 pt gap is within expected variation.
