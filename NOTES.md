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

# t98 notes: fused neighbor_pad+conv3d (t93 PLAN task 3)

Before porting origin/kevinmi/neighbor-pad-conv3d-fused (5f3d20c0fac..a7cf0f9164a, base cab8c46af3e), I measured the
most it could save. models/tt_dit/tests/models/ltx/test_vae_ltx_np_ceiling_ab.py times the halo-only decode in 4
interleaved arms (eager; t96 found eager = traced, device-bound): base; skip_routed (no halo exchange on the convs the
router would fuse, i.e. keys in _HALO_LAST_KEYS/_FORCE_SPATIAL_KEYS; conv3d reads a stale halo buffer); skip_all;
grid_routed (routed convs on one compute-grid column fewer, a stand-in for the 8 cores the fused op reserves for fabric).
Scripts: tmp/blx03/t98/{stage98,run98,driver98}.sh. No C++ change; ran on the blx03 t48 build (83c11ee2b3).

## Result (blx03 job 455, 2026-10-03 07:00 UTC, exit 0, 49 s, no chip drop or broker ERROR)
- Routed: 31 of 42 convs (s1_up 1, s2_res+s2_up 9, s3_res 12, s3_chg 1, s4_res 8).
- base        decode_s 0.5366 0.5370 0.5293  min 529.3 mean 534.3 ms  md5 18e86950... (= t97 pad arm)
- skip_routed decode_s 0.5171 0.5290 0.5183  min 517.1 mean 521.5 ms  delta -12.3 (min) / -12.8 (mean)
- skip_all    decode_s 0.5242 0.5183 0.5220  min 518.3 mean 521.5 ms  delta -11.0 / -12.8
- grid_routed decode_s 0.5405 0.5356 0.5387  min 535.6 mean 538.3 ms  delta +6.3 / +4.0, bit-identical to base
- Run-to-run spread within an arm is 5-12 ms. AICLK clamped at 1150 MHz, as in t97; the A/B is interleaved, so deltas stand.

## What it means
- Hiding every routed halo exchange perfectly and for free saves at most ~12.5 ms (2.4% of the decode, ~0.2% of a
  ~6 s e2e). The other 11 exchanges cost ~0.
- The fused op gives up 8 conv cores (~3-5 ms here, from grid_routed) and, per its README
  (ttnn/.../neighbor_pad_conv3d/README.md on a7cf0f9164a, sec. 6 and 7.3), pays ~262 us fixed co-residence overhead
  per call (~8 ms over 31 calls). Its own rule: fusion wins only if min(NP, conv) > ~262 us per call. Our routed
  exchanges average ~0.4 ms each, so the expected net is about 0 to -7 ms: not measurable with this noise.
- The port also needs features the fused op lacks (logical H/W masks with pad_offset, replicate T pad), so it means
  re-forking t48's conv3d reader/compute/writer. Not worth it for <= 12 ms. Decision: no port; task 3 should be dropped.
- Side note: one column fewer (-8% cores) slows the routed convs (~270 ms) by only ~1.5-2%, so their work does not
  scale with core count. That points at work-unit imbalance or NoC limits, which a blocking re-sweep (PLAN task 5) or
  exact-shard (t97) is better placed to fix.
- Cleaned /var/tmp/fasth3/t98/src on blx03 (logs kept).
