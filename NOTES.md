# t99 NOTES: fused RMSNorm + residual add (LTX_FUSE_NORM_ADD=1), blx03 2x4 A/B

## RESULT (job 457, 2026-10-03 07:09-07:12 UTC, blx03 full mesh -> create_submesh(2,4), LTX_CONV3D_BLOCKING_MESH=4,8) — DONE
- Unit RMSNorm+add (fused vs reference): all 7 shapes RS_OK, incl. (1,19,17,15,1024) W=15 and s2/s3/s4 (W=30/32/60/64/120/128).
- Decode 544x960/145f, min of 3:   fuse0 eager 513.2 / traced 518.1 ms;  fuse1 eager 515.2 / traced 518.2 ms.
  => no gain (delta within noise). Trace size 2.69 -> 2.42 MB (fewer ops), so the fused path ran.
- Quality fused vs unfused: PSNR 51.58 dB, PCC 1.000000, max_abs 4/255 (not bit-identical).
- AICLK clamped at 1150 MHz (same as #96/#97); A/B is same-job so the comparison holds. No drops/reboots.
- Verdict: keep LTX_FUSE_NORM_ADD opt-in, default off. Not worth adding to the eval pack.
- Cleanup done: blx03 ~/fasth3/t99 worktree+build removed; /var/tmp/fasth3/t99 kept logs only (472K).
  Summary log: tmp/blx03/t99/results/job457_summary.log.

Code: c6073fcc69e on ttp/t99-t93-4-fold-resnet-residual-add-into-next (Kevin's dual-output RMSNorm from f9cc61e24ad
+ row-major residual fix for W=15 + flat RM tile-row count + LTXUNetMidBlock3D wiring). Not compiled locally.
Job scripts: tmp/blx03/t99/ (6afe1031681). Stage branch ttp/t99-stage = t99 + t96 trace harness (057e841c056,
489e09bcd36), staged to blx03:/var/tmp/fasth3/t99/src.

## Run 2 (2026-10-03 07:07 UTC): build of a6e02770e0e OK, broker job 457 running at 07:09
Run 1 (c6073fcc69e) failed to compile: tt::tt_metal::create_device_tensor not visible (fixed in a6e02770e0e).
Old logs kept as driver.log.1 / build.log.1. Relaunch from ssh must not background a chain holding the ssh channel.

## Running on blx03 (launched 2026-10-03 06:59 UTC)
- driver: /var/tmp/fasth3/t99/driver.log, marker "T99_DRIVER_DONE <stage> <rc>"; build log /var/tmp/fasth3/t99/build.log
  (worktree ~/fasth3/t99 at c6073fcc69e, own build). Job log /var/tmp/fasth3/t99/run99.log.
- stages: build !=0 -> compile error, read build.log, fix, push, relaunch driver (T99_REV=<new>).
  ab 9 = drop/ERROR/reboot during OUR job -> STOP ALL device work on every galaxy, report. ab 8 = broker never healthy.
  ab 0 = done. ab other = test failure: grep T99_PART1_EXIT / T99_DECODE*_EXIT / RS_OK / AB / CMP99 in run99.log.
- results: "AB arm=eager/traced ... min=" for fuse0 and fuse1 (t96 baseline traced 445 ms decode),
  "CMP99 ... psnr_db pcc" fused vs unfused.
- relaunch: ssh g14blx03 'cd /var/tmp/fasth3/t99 && T99_REV=<rev> setsid nohup bash src/tmp/blx03/t99/driver99.sh > driver.out 2>&1 < /dev/null &'
  (re-stage first: bash tmp/blx03/t99/stage99.sh ttp/t99-stage)
- cleanup when done: ~/fasth3/t99 worktree+build on blx03 (git -C ~/fasth3/tt-metal worktree remove --force), /var/tmp/fasth3/t99/jit, fuse*/ .pt
