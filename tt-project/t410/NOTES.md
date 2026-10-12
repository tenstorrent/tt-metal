# t410: shorten the FastH3 VAE decode output path. Result: already in the base, no code change

Base: ttp/t400-adaln-seeds a15eb37ec6a. Branch ttp/t410-shorten-the-fasth3-vae-decode-output-pat. No device jobs run.

## The PLAN.md (t407) items and the #414 survey idea 3, checked against the code
1. Port main's H3 decode work (rgb_to_yuv_wide, device stitch, unpatchify gather, convtranspose polyphase):
   already ancestors of the base. 8e1dc77c837 (#56922: strips stitch, readback path), c792f807557 (#59060),
   e24a2b93d79 (yuv readback without mesh events on multi-host). unpatchify_tiled is in
   models/tt_dit/models/vae/minimax_h3/unpatchify_minimax_h3.py:63, used at vae_minimax_h3.py:954.
   Cherry-picking ad2792e70b0 again only conflicts, because its content is already here.
2. Read back uint8/YUV and overlap it with compute: already the default yuv420 strip path.
   vae_minimax_h3.py readback() (~1236): `_read_canvas_yuv(rows_partitioned=True)` clamps,
   typecasts and converts to YUV 4:2:0 on device (fast_device_to_host_yuv, defer=True). The finish
   runs in `_yuv_finish_pool` and is joined one wave later (`len(pending) > 1`), so wave i's readback
   overlaps wave i+1's device work. read_bytes = T*H*W*3/2, i.e. uint8.
   The "0.30 GB FP32" claim in PLAN.md and the #414 survey is a misreading: 196 frames x 1344x768 x 1.5 B
   = 0.303 GB, exactly what job 643 printed. FP32 would be 2.43 GB. "FLOAT32" in the profile is
   the dtype label of the canvas on device, not of the readback.
3. conv3d blocking-table fallbacks: the "[fallback]" log lines come from MiniMaxH3CausalConv3d in the
   **encoder** (conv_minimax_h3.py), which calls get_conv3d_config without T/H/W. So the lookup
   lands on `_H3_ENCODER_BLOCKINGS`, which already holds swept per-channel values. The
   decoder is a ViT and has no conv3d. Filling these in buys nothing for decode, and encode is 0.24 s.
4. (survey) Rebalance 196 units over 32 chips: 196/32 = 6.1, so it still takes 7 waves unless the unit
   size changes. There is no gain from rebalancing alone.

## Where the decode time actually goes (job 643, 900 MHz clamp, relative only)
VAE decode 1.0133 s: host dispatch ("device") 0.50 s (~72 ms per wave over 7 waves), readback issue 0.03 s,
readback_join 0.40 s (waiting for the device to finish plus the YUV finish), residual 0.41 s (includes the join).
The video VAE decode is **not traced**: only the audio decoder uses traced=True. That makes the open lever
trace capture/replay of the decode waves (fixed unit shape, so likely bit-identical). First measure
whether the device is starved by dispatch: run test_decode_stage[yuv420-5s] with profile=True
(it syncs per wave). If device time per wave is well under the 72 ms dispatch time, tracing pays.
