# K3.1 stage tool (local only, not part of PR #64830)

Copied from tt-metal branch mvasilijevic/k31/main @ 70d4fe79170 by tt_metal_tracker-8v1.2 before
removal from the PR. Files are verbatim copies at that commit:

- test_kimi_k3_1_stage.py: capacity sweep (stub/real weights, STAGE_MAX_USERS), 4-process handoff
  chain, kvpe probe, 10K/128K prompt perf with realtime-profiler dumps and the 1M deep probe.
- stage_handoff.py: save/load of the D2D payload as a raw bf16 file.
- realtime_profiler_utils.py: the version whose records carry start_ns/end_ns (needed by
  _profiled_replay); the PR reverts that file to its base.
- test_transformer_depth.at-70d4fe79170.py: depth ladder with the stage1-3 rungs that start from
  $K31_HANDOFF_DIR handoffs.
- test_transformer_pipeline_split.at-70d4fe79170.py: split test that hands off through a file.

To use: copy into a tt-metal checkout at the same paths (k31_stage support module of the PR provides
the CI constants; the tool expects its own definitions, which it still carries).
