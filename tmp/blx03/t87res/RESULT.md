# t87: traced vs eager conv VAE decode, blx03 job 063 (full mesh open + create_submesh(2,4))
Build: blx03 t48 build (a613d669eef) + t87 Python overlay (branch ttp/t87-vae-trace-2x4-check).
544x960/145f saved latents, LTX_TIME_STAGES=1, trace_region_size=500 MB. No chip drop; post-job health gate OK.

| arm | runs (s, test wall) | min | VAE_DECODE_SPLIT decode (ms) |
|---|---|---|---|
| eager (1 warmup + 3) | 2.0579 2.0638 2.0533 | 2.0533 | 1979.1 1978.5 1978.5 |
| traced (capture + 3 replays) | 2.3162 2.0547 2.0550 | 2.0547 | 2238.8 1979.7 1979.1 |

- Trace size 2,490,368 B (2.5 MB); fits the 500 MB region with large margin. Capture took 4.33 s.
- Output YUV (145x816x960 uint8) bit-identical: capture vs eager and replay vs eager, max_abs_diff 0.
- Gain: none. Warm replay decode 1979 ms vs eager 1979 ms (+0.0-0.6 ms). The first replay
  after capture is 260 ms slower. The decode is device-bound; with fast runtime mode, eager host
  dispatch already overlaps device execution.
- Recommendation: do not make LTX_VIDEO_VAE_TRACE the default. Keep it opt-in (default 0).
