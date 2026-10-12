# t430 notes (real old-vs-new A/B for the #427 cross-fade rewrite)

Why: #427's -33% / -43% compared job 852 (MINIMAX_H3_VAE_PROFILE=1, device sync after every wave)
with job 858 (profile off, new code). Not a fair A/B.

## Step 1: device A/B (blx01 broker job 862, submitted 2026-10-11 23:44 PDT, -t 560, unmeasured)
`bash /var/tmp/fasth3/t430/run430.sh` (copy here): test_decode_stage yuv420 5 s, profile unset, seed 0,
four pytest processes in order old, new, old, new. old = /var/tmp/fasth3/t423/b @ 5a8c59583a9 (VAE
code identical to b0f45278d72^), new = /var/tmp/fasth3/t427/b @ b0f45278d72. -k
'decode_stage and yuv420 and 5s and not 15s and not neighbor'. Per-arm logs out_ab/run_<n>_<tree>.log,
summary out_ab/summary.log (DECODE_STAGE total, DECODE_SPLIT blend, thread affinity per arm).
Lint needs the script's own setsid process-group re-exec (same as jobs 852/858); pytest itself runs
under `timeout --foreground -k 10 <=400`, not under setsid.

## Step 2: CPU assemble_clip_parts old vs new (blx01, no device)
Driver drv430.sh (detached, bench430drv) waits for job 862 to end, then bench430.sh: 5 s and 15 s
synthetic planar clips (124/362 x 1152 x 1344 uint8), unpinned / taskset 32-63 / taskset 40, old/new
alternated, after the 1-min load drops below 20. Output /var/tmp/fasth3/t430/bench.log.
tt-metal pins the main thread only with TT_METAL_NUMA_BASED_AFFINITY=1 (device_manager.cpp); by default
its 32 dispatch threads go round-robin on cores 0-31 and the main thread is unpinned (job 858 saw 64 cpus).

## Next step
`ttp detach --check --host g15blx01 /var/tmp/fasth3/t430/bench430drv` exits 0 -> read summary.log and
bench.log, fix the numbers in NOTES.md (t427 section), commit, `ttp push --own --detach`.
