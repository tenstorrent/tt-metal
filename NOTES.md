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

## Device job
blx03 job 041 (queued 14:47 behind t81 job 040): `bash ~/fasth3/t83/run83.sh` (copy in tt-project/t83/run83.sh).
Log: g14blx03:~/fasth3/t83/run83.log. Status: `ssh g14blx03 tt-device-mcp status -j 041`.
Pass lines: `AUDIO chain=0 replay_ms=...`, `AUDIO split ...`, `AUDIO chain=1 replay_ms=...`,
`T83_CMP chain0_vs_chain1 identical=...`, `T83_EXIT=0`.

Next: read the log, check broker log for drops during 041 (stop ALL device work if a drop started during it),
copy log to tt-project/t83/, `rm -rf ~/fasth3/t83 /var/tmp/fasth3/t83` on blx03. If chain is bit-identical and
faster: report the saving (flag already exists, default 0). If not identical: look at device add/clamp vs host.
