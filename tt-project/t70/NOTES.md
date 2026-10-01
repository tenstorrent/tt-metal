# #70 — device check of LTX_FUSE_GATE_ON_DEVICE (t51 @80882b291c2) + LTX_FUSE_NORM_ADALN A/B

blx03 broker job **035** (submitted 2026-10-01 10:50, timeout 1500 s): `bash ~/fasth3/t70/run70.sh`.
Log: `g14blx03:~/fasth3/t70/run70.log` (also `/var/log/tt-device-broker/2026-10-01_105004_035.log`).
Status: `ssh g14blx03 tt-device-mcp status -j 035`.

Setup: full (4,8) mesh with FABRIC_1D, then create_submesh(2,4). models/tt_dit comes from 80882b291c2
(`~/fasth3/t70/src`, 6.5 MB, ahead of blx03's tree on PYTHONPATH); build and models/common come from blx03's tree.

- Phase 1 (fold): AV block 0 with real weights at TP=4 (sp_axis 0, tp_axis 1, Ring CCL manager, no forward).
  Folds on device and compares every chip's fused weight/bias with an LTX_FUSE_GATE=1 load. Log line `T70_FOLD`;
  any `T70_FOLD_BAD` line is a mismatch. The fused forward can't run on a 2x4 (Ring needs wraparound; Linear's
  minimal_matmul_split ignores chunk_sizes), so a bit-identical result means its speed and quality are those of LTX_FUSE_GATE.
- Phase 2 (norm+adaln): AV block 0 with Linear sp1/tp0, F,H,W = 10,34,60 (5100 video tokens per chip), traced,
  5 replays per arm. Log lines `T70_ADALN` and `T70_ADALN_AB` (ms off/on, PCC between arms).

Next step: grep `T70_` and `T70_EXIT` in the log, check the broker log for drops, write the result, then delete `~/fasth3/t70` on blx03.
