
## Step 1 result (job 852, exit 0, 108.8 s, blx01; relative numbers only, box clock not checked)
Warm yuv420 5 s decode (7 waves, 28 units/wave), MINIMAX_H3_VAE_PROFILE=1 (syncs after each wave):
- dispatch per wave: median 49 ms [44 48 49 49 50 52 58]
- device still busy after dispatch returns (wait): median 82 ms [87 84 83 82 81 79 74]
- so device time per wave is >= 82 ms and about 131 ms (dispatch overlaps device work); total 0.96 s device of 1.48 s
- 15 s (19 waves): dispatch median 59 ms, wait median 74 ms.
Host dispatch is NOT the bottleneck: the device outruns it. Without the profile sync the host queues the
next wave while the device works, so dispatch is hidden after wave 1. Step 2's condition (device time
well under dispatch) fails, so the A/B was not run and nothing was landed. The traced_decode code
(5a8c59583a9, opt-in, default off) stays on this branch only, unvalidated on device.
Bigger leftover: ~0.4 s of the 1.48 s is outside the device (blend=0.428 in DECODE_SPLIT).
Correction (#430, job 862): that blend is a profile-mode artifact; with the profile off the old code
decodes 5 s in 1.01 s with blend 0.04 s, and #427 brings it to 0.98 s (~2.7%).
