# t84 NOTES (conv3d precision A/B: LoFi / bf8 weights)

Branch ttp/t84-vae-conv3d-precision-a-b-lofi-and-bf8-we @f45ce450901 (pushed), on t48 tip 29a0e8dfdc8 (+369f4c3c773 cherry-pick;
8ff9ba539e4 skipped: it patches matmul_blocks_split, which t48 lacks).

Code: LTX_VAE_CONV_FIDELITY (LoFi|HiFi2|HiFi3|HiFi4), LTX_VAE_CONV_WEIGHT_DTYPE (bf16|bf8), up-block convs only, default off.
conv3d C++ now accepts bf8 weights with bf16 input (weight CB format/tile size, writer bias tile size fixed).
NOTE: production default conv fidelity is HiFi4 (not HiFi2), so the reference arm "base" = HiFi4/bf16.

## Attempt 2 (2026-10-02 18:45 UTC)
Attempt 1 never ran: its build died during the CPM download on 10-01 19:48 (session ended) and left truncated
cpmcache entries (capnproto, protobuf, cadical, blake3 dirs dated 10-01 19:48); removed. blx03 rebooted 10-02 18:23
(chip 9 off PCIe, not during our job). Relaunched with setsid. Build dir now /var/tmp/fasth3/t84/build_Release
(blx03 /home has 65 GB free, <150 GB rule); ~/fasth3/t84/build is a symlink to it. Driver submit retry raised to 300 min.
Earlier logs: *.attempt{1,2,3}.log next to the current ones.

## Running on blx03 (launched 2026-10-01 19:48 UTC, relaunched 2026-10-02 18:45)
- setup/build: ~/fasth3/t84-setup.log (marker SETUP84_DONE rc=N), worktree ~/fasth3/t84 (~3.3 GB with build)
- driver: ~/fasth3/t84drv/driver.sh, log /var/tmp/fasth3/t84/driver.log, final marker "T84_DRIVER_DONE <stage> <rc>"
  stages: setup!=0 -> build failed; ab 8 -> broker never healthy in 2h (just relaunch driver); drop 9 -> drop/reboot
  during OUR job -> STOP ALL device work, report; ab 0 -> success.
- one job, arms base,hifi2,lofi,bf8,lofi_bf8: log /var/tmp/fasth3/t84/run84.log (lines "AB arm=... decode_s=... min=")
- outputs: /var/tmp/fasth3/t84/ab/yuv_<arm>.pt, still_/crop_/diff16_<arm>.png, summary.json; scorer log compare.log

## Next step
Read driver.log, run84.log (decode min per arm), ab/summary.json (psnr_min >= 40 dB vs base = pass), look at stills.
Copy stills to g15blx02 under tt-project/state/runs/<run>/ for the report. Then write result.json.
Cleanup on blx03 when done: git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/t84; rm -rf /var/tmp/fasth3/t84/build_Release; rm -rf ~/fasth3/t84drv
~/fasth3/t84-setup.* (keep /var/tmp/fasth3/t84/ab stills; yuv .pt files can go).
