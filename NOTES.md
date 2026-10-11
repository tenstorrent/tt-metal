# t376 notes (conv VAE blocking sweep of the layers #363 skipped)

- Base: origin/ttp/t48-ltx25-integrated 90ed8257bac. Code commit 2af2bbe4a06: exact-shard halo sweep entries
  s4res_x (res128), s4out_x (conv_out), s1up_x (up_all_x1), s0res_x (res1024), s0up_x (up_all/2); CPU test relaxed.
- Keys: tt-project/t376/keys376.py (42 convs, 10 keys). up_time = s2_res key, up_space = s3_chg key (#363 tuned both);
  conv_in skipped (0.4 ms total).
- Candidates: tt-project/t376/cands376.py -> cands376.txt (40 per layer, table C_in_block, prefetch shard fits).
- Driver on blx01: /var/tmp/fasth3/t376/drv376.sh (log drv376.log, jobs.txt, res/, marker drv376.done).
  Started with `ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t376/detach t376drv -- bash /var/tmp/fasth3/t376/drv376.sh`.
- Next on wake: read drv376.done, res/winners.txt, res/run_ab*_job*.log ("AB arm=..." medians, identical/pcc/psnr).
  Winners that the A/B confirms (identical) go into _BLOCKINGS (conv3d.py ~486) + test_ltx25_halo_winners_in_table;
  then ttp checks, cherry-pick code commits onto a -land branch from origin t48, `ttp push --detach`.

## Run 2 (2026-10-10 17:10 PDT): hangs found and fixed
- All 8 sweep hangs on blx01 (broker jobs 529 539 546 553 567 574 582 588, smarton/t376) stopped on blocking
  (128,64,5,2,8): T*H*W=80, unaligned and >64. conv3d's vol2col_rm CB is min(n,64) pages and the reader pushes
  min(left,32)-row chunks, so for unaligned n>64 a reservation straddles the ring end; cb_push_back only wraps
  on wr_ptr == fifo_limit (dataflow_api.h:217) -> device hang. Table entries are all safe (unaligned ones <=56).
- Fix (code f141ee96e70): vol2col_ring_safe() drops such blockings in build_all_blockings and run_sweep;
  per-combo watchdog (SWEEP_COMBO_WATCHDOG_S=90, os._exit(3)); CPU tests (47 pass).
- Device held after each reap: plain `timeout` calls setpgid, so pytest left the setsid group and the group
  kill missed it (job 588's pytest 1937092 still held blx01 at 00:08 UTC; killed by pid at ~00:10 UTC).
  run376.sh now uses `timeout --foreground -k 10` plus a by-pid reap; tested locally: TERM to the outer pid
  clears script, timeout and a TERM-ignoring python. device-lint passes.
- Old driver t376drv (pgid 693788) killed ~00:10 UTC. No queued jobs of ours were left.
- drv376b.sh: s4out_x, s4res_x, s0up_x, 16 ring-safe candidates each, no retry, any hang stops everything
  (marker drv376b.done = HUNG ...). s0res_x and s1up_x skipped (coordinator: hung more than twice).
