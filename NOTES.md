# t83 — LTX audio decode on the critical path

Branch ttp/t83-profile-and-cut-ltx-audio-decode-on-the- (based on t48 a613d669eef).

## Off-device findings
Critical path after VAE decode: max(libx264 video encode on worker thread, audio decode + AAC ~0.15 s) + mux
(YuvVideoExport). So audio decode counts only where it exceeds the (now ultrafast) video encode.

Default traced audio path (LTX_AUDIO_DEVICE_CHAIN=0), per decode — 5 uploads, 5 downloads, 2 eager device stages:
1. host unpatchify + denormalize -> upload bf16 (mel)        2. mel-VAE trace -> download, host crop/permute
3. host transpose/pad -> upload fp32 (vocoder)               4. vocoder trace -> download, host crop
5. host pad -> upload -> EAGER mel STFT -> download          6. upload -> BWE trace -> download
7. upload -> EAGER resampler -> download                     8. host add + clamp + trim
LTX_AUDIO_DEVICE_CHAIN=1 (exists, default 0) runs 3-8 in one trace: 1 upload + 1 download.
CPU (torch, 64 threads) reference: mel-VAE 0.32 s, vocoder+BWE ~10 s -> host CPU audio is not viable.
Device overlap with VAE decode: same chips, one CQ -> no real overlap; audio is already overlapped with the
video encode thread.

## Device jobs
blx03 job 041 (both arms, one job): chain=0 replay 550.4 ms/decode min (first replay 1080.6), eager_vs_replay 0.
Failed on the split probe's own assert (split differs from decode_audio ~1e-4), so chain=1 never ran. No drop during 041;
the 10:17 PDT blx03 drop was during ltx-host job 051 (not ours). chain0 wave kept at blx03:/var/tmp/fasth3/t83/wave_chain0.pt.
Fix 54771db51fb: split prints drift instead of asserting. run83.sh now takes ARMS=0|1|01 (hook caps -t at 600 s).
blx03 job 062: `ARMS=1 bash ~/fasth3/t83/run83.sh` (-w /home/smarton -t 590), chain=1 arm + T83_CMP vs saved chain0 wave.
Log: g14blx03:~/fasth3/t83/run83.log. Copies: tt-project/t83/.

Next: read run83.log for `AUDIO chain=1 replay_ms`, `T83_CMP`, `T83_EXIT`; check for drops during 062 (stop ALL device
work if one started during it). Then submit `ARMS=0` (job ~2 min, warm cache) for the split timings.
If chain=1 is bit-identical and faster: report the saving (flag exists, default 0). If not identical: STFT device_framing
(set only for the chain) is the likely source; compare device vs host framing.
After: copy logs to tt-project/t83/, `rm -rf ~/fasth3/t83 /var/tmp/fasth3/t83` on blx03.
