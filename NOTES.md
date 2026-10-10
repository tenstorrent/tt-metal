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
