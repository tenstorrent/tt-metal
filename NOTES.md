# t48 notes: all LTX-2.5 wins on one branch

Branch ttp/t48-ltx25-integrated (= ttp/t48-integrate-all-ltx-2-5-wins-on-one-branch), base t36 16ba9a383dc.
Merged: t20+t40 (9e336c44b71, includes 0533827a419), t13 (eee3baf7c0d), t18 (63902277007),
t44 tip (1968790b040 + its A/B harness), t8 ltx_eval harness. Python-only diff against t36.

Conflicts:
- pipeline_ltx_distilled.py: t13 and t40 both capture the Gemma encode trace after gen #0. Kept t40's
  open_trace_gate() + capture_trace() (guarded by _trace_captured). t13's open_trace_gate(capture_prompt=) was removed in t55 (no caller).
- utils/video.py: t18's YuvVideoExport (worker-thread video encode) + t13's zero-copy frame wrap and start_encoding;
  the AAC encode runs in finish() before joining the worker, so it overlaps the video encode as in t13.
  test_yuv_export_encodes_audio_alongside_video now gates the video worker on the audio encode starting
  (fails if finish() encodes audio after the join; checked).
- test_ltx_export_latency.py: gemma -> gemma3 import path.

CPU tests (python_env, PYTHONPATH=worktree): export/trace/eval/cache/ltx set (13 files) 78 passed, 8 skipped;
13 pre-existing failures in test_ltx_euler_tail.py and test_ltx_embedding_cache_identity.py (they read
models/tt_dit/encoders/gemma/, renamed to gemma3); same 13 fail on the t36 base tree.
Fold CPU reference (--noconftest): 5 passed. The 78 include the ltx_eval harness (8) and the 13 export/trace tests.

Device: not run (blx03 paused; full-mesh barred by the 22:10 rule). Ready job: tmp/READY_48.md, tmp/blx03/run48.sh.
Next: when the user allows full-mesh runs on blx03, follow tmp/READY_48.md (setup, one job, timings, ltx_eval vs t20).

t113: folded t100 47aecb9bdd7 (halo sweep harness + CPU test) and 118ed6de1f4 (conv3d _BLOCKINGS (4,8):
s4_res (128,64,6,4,8), s1_up (128,64,5,2,16); bit-identical, traced decode 519.7 -> 506.2 ms, blx03 job 469, 544x960/145f).
