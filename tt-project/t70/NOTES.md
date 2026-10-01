# #70 — device check of LTX_FUSE_GATE_ON_DEVICE (t51 @80882b291c2) + LTX_FUSE_NORM_ADALN A/B

Job 035 failed in 22 s: `python -m pytest` from $M put blx03's tree ahead of src/, so LTXAttention had no
`fold_gate_on_device` (harness bug, no device fault; broker fabric check 036 passed after it). Fixed: run from $D and
assert at collection that attention_ltx comes from src/.

blx03 broker job **037** (submitted 2026-10-01 10:51, timeout 1500 s): `bash ~/fasth3/t70/run70.sh`.
Log: `g14blx03:~/fasth3/t70/run70.log` (also `/var/log/tt-device-broker/2026-10-01_105149_037.log`).
Status: `ssh g14blx03 tt-device-mcp status -j 037`.

Setup: full (4,8) mesh with FABRIC_1D, then create_submesh(2,4). models/tt_dit comes from 80882b291c2
(`~/fasth3/t70/src`, 6.5 MB, ahead of blx03's tree on PYTHONPATH); build and models/common come from blx03's tree.

- Phase 1 (fold): AV block 0 with real weights at TP=4 (sp_axis 0, tp_axis 1, Ring CCL manager, no forward).
  Folds on device and compares every chip's fused weight/bias with an LTX_FUSE_GATE=1 load. Log line `T70_FOLD`;
  any `T70_FOLD_BAD` line is a mismatch. The fused forward can't run on a 2x4 (Ring needs wraparound; Linear's
  minimal_matmul_split ignores chunk_sizes), so a bit-identical result means its speed and quality are those of LTX_FUSE_GATE.
- Phase 2 (norm+adaln): AV block 0 with Linear sp1/tp0, F,H,W = 10,34,60 (5100 video tokens per chip), traced,
  5 replays per arm. Log lines `T70_ADALN` and `T70_ADALN_AB` (ms off/on, PCC between arms).

Next step: grep `T70_` and `T70_EXIT` in the log, check the broker log for drops, write the result, then delete `~/fasth3/t70` on blx03.

## Result (job 037, 2026-10-01 10:51-10:53, 88.7 s, exit 0; broker post-job health gate clean, no drop)
- `T70_FOLD folded=6/6 fold_ms=1320.6 mismatches=0`: on-device fold of all 6 block-0 attentions is bit-identical,
  on every chip, to the LTX_FUSE_GATE=1 fused-cache load. So the fold's forward speed and quality are LTX_FUSE_GATE's
  (t51: block -3.7% S1 / -4.9% S2, ~-0.20 s e2e est.; block PCC >= 0.99999 vs unfused, not bit-exact).
  Not measured here: fused forward on device (needs Ring on 4x8), fold time over all 48 blocks (1.32 s for block 0
  includes first-time JIT of pad/concat), DRAM peak over a full load.
- `T70_ADALN_AB off_ms=30.05 on_ms=29.49 delta=-0.56 ms (-1.9%) pcc_video=0.999958 pcc_audio=0.999982 maxabs_video=28`
  (2x4 Linear, 5100 video tokens/chip, traced, 5 replays/arm). Norm work is per-token local, so the absolute
  -0.56 ms/block should carry to the 4x8 S2 block (4845 tokens/chip): ~144 S2 block calls -> ~-0.08 s, plus S1
  -> ~-0.1 s total, matching job 582's -0.10 s e2e.
- Log: run70.log.gz. blx03 ~/fasth3/t70 removed.

Recommendation: cherry-pick 80882b291c2 onto t48 as opt-in (LTX_FUSE_GATE_ON_DEVICE default 0); it removes the
37 GB fused-cache blocker. Turn the fold and LTX_FUSE_NORM_ADALN on by default only after one shared 5-seed
VBench + visual eval of both together (~-0.3 s est.), which needs a 4x8 e2e run.
