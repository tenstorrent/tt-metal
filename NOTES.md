# t76 notes
- Branch ttp/t76-cut-vae-conv-decode-neighbor-pad-and-ln- @428e0104180 (pushed), based on t48 a613d669eef.
- 8a2f5548851: TT_NEIGHBOR_PAD_LOCAL_BATCH=<n> (default off) batches neighbor_pad_async's local copy
  (one NOC barrier per row-batch instead of per stick). Host C++ syntax-checked (clang-20); kernels not compiled yet
  (JIT on device). CPU test models/tt_dit/tests/models/ltx/test_neighbor_pad_local_batch_ref.py: 17 pass.
- blx03 build started 2026-10-01 11:30 (off-device): ~/fasth3/t76, log ~/fasth3/t76-setup.log, marker "SETUP76_DONE rc=0".
- Next: once built, on blx03: cd ~/fasth3/tt-metal && tmp/blx03/submit.sh 1500 bash /home/smarton/fasth3/t76/tmp/blx03/run76.sh
  Read /var/tmp/fasth3/t76/run76.log: T76_CMP identical=True and the AB timing lines (b0 vs b128).
- After the A/B: git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/t76; rm -rf /var/tmp/fasth3/t76/jit.
- Result (2026-10-01, blx03 job 039, full 4x8 open + create_submesh(2,4), 544x960/145f, 1 warmup + 3 decodes):
  T76_CMP b128_vs_b0 identical=True max_abs_diff=0 shape=(145, 816, 960) uint8.
  b0 decode_s 2.0510 2.0549 2.0552 (min 2.0510); b128 decode_s 2.0747 2.0716 2.0760 (min 2.0716): +20.6 ms (+1.0%), slower.
  b128 built 6 new local_copy kernel variants in the JIT cache, so the batched path ran. Not made default; not on t48.
  Arm order was fixed (b0 first); a reversed-order run or a per-op profile would show whether neighbor_pad itself got slower.
