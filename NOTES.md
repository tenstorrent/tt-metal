# t427 notes (cut the FastH3 VAE decode's off-device blend)

## Where the time was
DECODE_SPLIT blend=0.428 s of 1.48 s (job 852, 5 s yuv420) is the temporal assembly in `_decode`
(`assemble_clip_parts`), run after the last wave. It scales with frames (~3.4 ms/frame; 1.24 s at 15 s).
It is slow on blx01 because tt-metal pins the main thread to the few cores left free
(device_manager.cpp `bind_current_thread_to_free_cores`) and Python threads inherit it: the 8-thread
pool contends on them. Emulated with `taskset -c 5` on g15blx02: ~220 ms, seam blends ~25 ms each.

## Change (b0f45278d72, code only)
- Cross-fade a frame at a time, writing straight into the output (`_crossfade_planar_into`):
  bit-identical (all 65536 uint8 pairs, extents 1-8), ~2.3x faster on one core.
- `StreamedClipAssembly`: each clip is assembled on the yuv readback worker as it lands, under the
  remaining waves; only the last 5-frame tail is left after the last wave. Flag `stream_blend=True`
  (default), strip-stitched yuv420 only; other paths keep the batch assembly. Profile: `blend` is now
  the post-wave remainder, `blend_streamed` the hidden worker time.
- CPU tests: test_stitch_device_minimax_h3.py -k 'crossfade or streamed' (8 pass).
- Device A/B: test_performance_vae_minimax_h3.py::test_decode_stream_blend_ab.

## A/B job
blx01 tree /var/tmp/fasth3/t427/b (setup427-blx01.sh, python-only on the t284 build, marker rc=0).
blx01 broker job 858 (submitted 2026-10-11 23:23 PDT, -t 450, unmeasured est. ~150 s, inner 400 s):
`bash /var/tmp/fasth3/t427/run427.sh` = stream-blend A/B, 5 s, seeds 0-4, 2 repeats per arm.
Log: /var/tmp/fasth3/t427/out_ab/run.log. Read STREAM_AB_SUMMARY (off_min/on_min/saved) and
STREAM_AB_QUALITY (identical expected on every seed) and the affinity line.
Judgment: all 5 seeds in one job (one A/B pair, one config); separate jobs would each make the live
LTX worker yield and reload (~87 s).

## Next step
Check `ssh g15blx01 tt-device-mcp status -j 858`, read the log. If the gate passes: write results here,
`ttp checks`, `ttp push --own --detach` onto ttp/t423-trace-the-fasth3-video-vae-decode-waves-.
