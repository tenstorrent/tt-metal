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
