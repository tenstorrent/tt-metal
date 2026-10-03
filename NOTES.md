# t99 NOTES: fused RMSNorm + residual add (LTX_FUSE_NORM_ADD=1), blx03 2x4 A/B

Code: c6073fcc69e on ttp/t99-t93-4-fold-resnet-residual-add-into-next (Kevin's dual-output RMSNorm from f9cc61e24ad
+ row-major residual fix for W=15 + flat RM tile-row count + LTXUNetMidBlock3D wiring). Not compiled locally.
Job scripts: tmp/blx03/t99/ (6afe1031681). Stage branch ttp/t99-stage = t99 + t96 trace harness (057e841c056,
489e09bcd36), staged to blx03:/var/tmp/fasth3/t99/src.

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
