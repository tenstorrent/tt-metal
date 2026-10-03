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

## Attempt 3 (2026-10-03 00:48 UTC)
Build finished OK on 10-02 18:47 (SETUP84_DONE rc=0, _ttnn.so links to /var/tmp build). The driver then waited for
health and died with blx03's host power-cycle (up again 10-03 00:34). None of our jobs ran. Relaunched the driver
(setup step is skipped since the marker exists); wait_health raised to 240 min. Old log: driver.attempt4.log.
If the driver is gone without a T84_DRIVER_DONE marker (another reboot), just relaunch it:
  cd /var/tmp/fasth3/t84 && setsid nohup bash ~/fasth3/t84drv/driver.sh > driver.out 2>&1 < /dev/null &
(run it via `ssh g14blx03 '...' < /dev/null` with a timeout; ssh otherwise hangs on the backgrounded chain).

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

## RESULT (blx03 job 354, 2x4 submesh from full mesh, 2026-10-03 00:55-00:58 UTC, no drop)
544x960/145f real latent, 40 up-block convs, min of 3 warm decodes, PSNR/SSIM vs base (YUV, per frame):
| arm      | decode s | delta    | PSNR min/mean | SSIM-Y min | max abs |
| base     | 1.9672   | -        | inf           | 1.0        | 0       |
| hifi2    | 1.9739   | +0.3%    | inf (bit-id)  | 1.0        | 0       |
| lofi     | 1.7899   | -177 ms, -9.0% | 45.00 / 45.92 | 0.9951 | 32 |
| bf8      | 1.9446   | -1.1%    | 10.0 / 10.4   | 0.022      | 221 (garbage) |
| lofi_bf8 | 1.7920   | -8.9%    | 10.0 / 10.4   | 0.023      | 221 (garbage) |
- base == hifi2: the production conv default for bf16 is already HiFi2 (HiFi4 only for fp32). The earlier
  "base = HiFi4" note was wrong.
- LoFi passes (min 45 dB >= 40); still/crop look the same as base, x16 diff is faint and edge-only.
- bf8 output is noise: a bug in the bf8-weight conv3d path (not precision loss). Not worth debugging: at best
  -23 ms, and nothing on top of LoFi. Removed the bf8 knob and reverted the conv3d C++ changes (commit after
  f45ce450901). LTX_VAE_CONV_FIDELITY stays opt-in, default unchanged.
- Stills: g15blx02 tt-project/state/runs/t84-ab/ (still_/crop_ base, lofi, diff16_lofi, still_bf8, summary.json,
  run84.log). Full set on blx03 /var/tmp/fasth3/t84/ab.
- Next: LoFi on by default needs a 4x8 E2E 5-seed check (VBench + visual), once 4x8 runs are allowed.
