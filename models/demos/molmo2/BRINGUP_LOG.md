
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

## Session 18 — Model-card benchmark accuracy (host DP=4 server, same code as the Docker image)

Model card publishes only the 15-benchmark average (63.1). Per-benchmark references are Table 2 of the Molmo2 tech report
(arXiv 2601.10611). All runs: greedy, `max_tokens=16`, letter-only MCQ prompt, `file://` videos, 4 concurrent requests, server
frame sampling = Molmo2 default (uniform_last_frame, <=2 fps, <=384 frames). Runner: `verification/evals/run_eval.py`
(MVBench: `verification/evals/eval_mvbench.py`); per-question MVBench results: `verification/evals/mvbench_results.jsonl`.

| Benchmark | Ours | Published (Molmo2-8B) | Notes |
|---|---|---|---|
| MVBench | 74.81% (2693/3600) | 75.9 | 18/20 tasks (episodic_reasoning needs frame dirs, fine_grained_pose needs NTU videos) |
| TempCompass (multi-choice) | 74.43% (1176/1580) | 73.4 | |
| LongVideoBench (val, no subtitles) | 65.37% (874/1337) | 67.5 | <=15 s 78.8, 60 s 75.6, 600 s 64.6, 3600 s 58.3 |
| Video-MME (no subtitles) | 69.00% (1863/2700) | 69.9 | short 77.3, medium 70.4, long ~60 |
| NExT-QA (MC, 8564) | 84.77% | 86.2 | |
| MotionBench (dev, 4018) | 62.29% | 62.2 | test split answers are hidden |
| LVBench | 53.15% (785/1477) | 52.8 | 72 questions skipped (see bug below) |
| EgoSchema (public 500-question subset) | 59.20% (296/500) | 62.0 (full test) | full set has hidden answers |
| PerceptionTest (val, every 16th question) | 81.37% (974/1197) | 82.1 (test) | subsample for time (full val = 19140); semantics 91.1, memory 82.0, physics 80.2, abstraction 71.9 |

Not run: MLVU, VideoEval-Pro (150-216 GB datasets), Tomato (no open copy found), Molmo2 Caption (needs an LLM judge, not named),
Molmo2 Count (data not located). Server errors across all runs: 0. No comparison against the GPU reference was done.

### Server bugs found while running the evals
1. **vLLM decoded every video frame into RAM before sampling** (8.6 min of 720p30 = 148 GB resident, a 1.4M-token input and HTTP 400).
   Fixed in the Molmo2 specs with `vllm_args: media-io-kwargs: {video: {frame_sample_mode: uniform_last_frame, num_frames: 384,
   max_fps: 2}}` (tt-inference-server commit 3b1e5e50): the same clip takes 29 s and 3.4 GB; 105-suite still 98/100 (135 s, was 203 s).
   The Docker image was rebuilt with it (see the decoder-crash fix below).
2. **Five LVBench videos segfault the OpenCV decoder inside the API server** (Aiem1w_TvaA, KLIa2UaE2KE, T1yhBv1ytzw, rSE2YPcv89U,
   oZEVgDXJwCc; exit -11 in `Molmo2VideoBackend.load_bytes`). One request kills the API server and leaves `VLLM::EngineCore`
   orphaned holding the device. **Fixed** (plugin commits `6e4fd11`, `49a986f`, `video_guard.py`): decoding runs in persistent helper processes, so a crash returns HTTP 400 in ~8 s and the server stays up. The helper sets RLIMIT_CORE=1 (0 still runs the piped apport handler, which stalled a 3.4 GB crash for >10 min). Verified in the rebuilt `molmo2-ti-dev:local`: 2 crashing videos -> 400, normal videos before/after OK, health 200, 105-suite 98/100 (same misses idx 22, 84), 0 errors. The LVBench run excludes
   these videos (`verification/evals/lvbench_bad_videos.json`, found with `screen_videos.py`).
3. Prefill is serialised across the 4 DP replicas (engine admits one prefill at a time; ~3.4k prompt tokens/s), so hour-long-video
   benchmarks run at ~3 questions/min. Candidate for the chunked/concurrent-prefill optimization step.
