# t78 notes
- Branch ttp/t78-conv3d-halo-only-input-for-ltx-vae-decod @f5b8175bc7b (pushed), based on t48 a613d669eef.
- LTX_VAE_HALO_ONLY=1 (default off): LTXCausalConv3d calls ccl_manager.neighbor_pad_halo_only (Linear topology)
  and conv3d(halo_buffer=..., padding=(pT,1,1), logical_h_mask/logical_w_mask, pad_offset_tensor). No neighbor_pad
  interior copy, no W mask multiply. Only when H and W are both sharded.
- conv3d: halo mode now accepts padding_mode="replicate" and clamps T (FOLD_TIME_PAD); H/W boundary still comes
  from the halo buffer. neighbor_pad_halo does NOT mask; conv3d's mask check runs before the halo read, by global
  coordinate, so halo sticks past logical_h/logical_w are zeroed.
- CPU: models/tt_dit/tests/models/ltx/test_vae_ltx_halo_only_ref.py (13 pass, --noconftest): gather emulation equals
  the default path's padded input exactly for 2x4/4x8 shapes, zeros/replicate/causal T pad, masked/unmasked; flag
  default off; forward wiring. All ltx *_ref.py: 37 pass. Kernel not compiled (JIT on device).
- Removed test_conv3d.py's rejects_replicate case (now supported); not run (device).
- Next (device, not submitted): on blx03 build off-device:
    setsid nohup bash tmp/blx03/setup78.sh > ~/fasth3/t78-setup.log 2>&1 &   # marker "SETUP78_DONE rc=0"
  then one broker job (full 4x8 open + create_submesh(2,4)):
    cd ~/fasth3/tt-metal && tmp/blx03/submit.sh 1500 bash /home/smarton/fasth3/t78/tmp/blx03/run78.sh
  Read /var/tmp/fasth3/t78/run78.log: T78_CMP identical=True and AB min decode_s h0 vs h1 (baseline ~2.05 s).
- Cleanup after the A/B: git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/t78; rm -rf /var/tmp/fasth3/t78/jit.

## t79 run (2026-10-01 14:35 UTC)
- blx03 /home 194 GB free (>=150). Build started 14:31: ~/fasth3/t78-setup.log (setup78.sh extracted to ~/fasth3/setup78.sh).
- Detached driver on blx03: ~/fasth3/t78drv/driver.sh (copy in tmp/blx03/t78drv/), log /var/tmp/fasth3/t78/driver.log.
  build -> health -> job ab (run78.sh) -> if identical: health -> job halo (run78b.sh: test_conv3d.py -k halo on a
  2x4 submesh of the full mesh via an untracked conv/conftest.py override of `device`). Stops on any broker ERROR,
  non-OK HEALTH-GATE or reboot since our submit. Final marker "T78_DRIVER_DONE <stage> <rc>".
- Next: read driver.log, /var/tmp/fasth3/t78/run78.log (T78_CMP, AB decode_s per arm), run78b.log (T78B_EXIT).
  Then clean up: git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/t78; rm -rf /var/tmp/fasth3/t78/jit
  ~/fasth3/t78drv ~/fasth3/setup78.sh.

## t79 result (2026-10-01 16:55 UTC)
- blx03 job 043 (run78.sh, full mesh + create_submesh(2,4), 544x960/145f conv decode): exit 0, no drop
  (post-job gate 32/32 OK, no reboot since 08:09). h0 decode_s 2.0580/2.0577/2.0542 (min 2.0542);
  h1 1.9917/1.9963/1.9937 (min 1.9917), 42/42 convs halo-only. -62.5 ms (-3.0%).
  T78_CMP h1_vs_h0 identical=True max_abs_diff=0 (145,816,960) uint8.
- Driver stopped after 043 on a false alarm: gate_fail_since() treats "device healthy; no reset needed" and
  "heartbeat: HEALTHY" as failed gates (it only accepts ": OK"). Same bug made wait_health idle ~16 min.
- Halo unit tests submitted by hand as blx03 job 049 (run78b.sh, log /var/tmp/fasth3/t78/run78b.log, T78B_EXIT).
- Job 049 (test_conv3d.py -k halo on a 2x4 submesh of the full mesh): exit 0, 2 passed / 33 deselected
  (rejects_dilation, rejects_undersized_halo). Only the reject cases match -k halo; the functional halo path
  is covered by the bit-identical A/B. Post-job gate OK, no reboot.
- Cleaned up on blx03: ~/fasth3/t78 worktree, /var/tmp/fasth3/t78 (jit 972M, yuv 2x109M, logs), ~/fasth3/t78drv,
  ~/fasth3/setup78.sh, ~/fasth3/t78-setup.log. Trimmed logs in tmp/blx03/t78res/.
