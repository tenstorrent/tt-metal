# READY (#60): fold the LTX VAE W pad-mask MUL into neighbor_pad_async (2x4 A/B on blx03, not yet run)

## What changed
- `neighbor_pad_async(..., logical_w=N)`: in a fused 2D [H, W] pad, every kernel that reads an input stick
  (local copy writer, H fabric reader, W reader phase 1) reads zeros for W columns >= N. Output == pad of the
  masked input, so the per-conv `ttnn.mul(x, w_mask)` can go.
- `LTX_VAE_FOLD_W_MASK=1` (opt-in): `LTXCausalConv3d` skips the mul and passes `logical_w` when the halo is the
  fused 2D one (H and W sharded). Otherwise it keeps the mul.
- The temporal concat is not folded into neighbor_pad: #44's `LTX_VAE_FOLD_TIME_PAD=1` already removes it via
  conv3d replicate padding, which reads the clamped edge frames and writes nothing extra. Doing it in
  neighbor_pad would write T+2 frames (plus their halos) and needs a replicate-both-ends T pad the op lacks.

## Off-device evidence (g15blx02)
- Host C++ builds (Release, clang-20, warnings as errors): `bash build_metal.sh --release`.
- Kernels compile with the exact BH JIT commands of job 994: `bash tmp/t60/kernel_compile.sh .` (6/6 OK; a
  deliberately broken kernel fails it).
- CPU emulation of the five kernels' index math vs "mask, then pad", 11 configs incl. pad columns crossing a
  W halo, logical_h, t_front_pad, replicate, 4x8 1080p grid: 16 passed
  (`python -m pytest --noconftest models/tt_dit/tests/models/ltx/test_vae_ltx_fold_w_mask_ref.py`).
  Dropping any one of the four mask points in the emulation fails 2-10 cases.

## Expected saving
42 MULs = 67 ms/chip in the 544x960/145f decode on 2x4 (job 994, same per-chip shard as 1080p on 4x8). The
masked sticks become zero-writes inside neighbor_pad's existing copy (same bytes written, fewer DRAM reads), so
expect ~60-67 ms per decode, bit-identical output. With #44 (concat, 38 ms/chip): ~100 ms per decode.

## Run (blx03; one project device job at a time; check the broker CLI after its 1.0.0 upgrade)
1. Setup, CPU only (~5 min, ~2.5 GB in ~/fasth3/t60):
   `ssh g14blx03 'bash -s -- <t60 commit>' < tmp/t60/blx03_setup60.sh`
   Ready when `ssh g14blx03 tail -1 ~/fasth3/t60-setup.log` prints `SETUP60_DONE rc=0`.
2. Optional device unit checks (op level + conv level, fold on vs off bit-identical; short):
   `ttp lock g14blx03-device -- ssh g14blx03 '~/fasth3/tt-metal/tmp/blx03/submit.sh 900 bash /home/smarton/fasth3/t60/tmp/t60/check60.sh'`
   Pass: `/var/tmp/fasth3/t60/check60.log` ends `T60_CHECK_EXIT=0`.
3. Decode A/B, one arm per job, each after the previous job ended:
   `ttp lock g14blx03-device -- ssh g14blx03 '~/fasth3/tt-metal/tmp/blx03/submit.sh 900 bash /home/smarton/fasth3/t60/tmp/t60/run60.sh t1w0'`
   then the same with `t1w1`. Add `t0w0` for a same-tree baseline (#46 already ran #44's t0/t1 on the t36 tree).
4. Compare: `ssh g14blx03 '~/fasth3/tt-metal/python_env/bin/python ~/fasth3/t60/tmp/t60/compare60.py /var/tmp/fasth3/t60'`
   Pass: every `identical=True` and `T60 PASS=True`; saving = the `T60 saving t1w0->t1w1` line.
   Append `/var/tmp/fasth3/t44/yuv_fold1.pt` to also check against #44's output.
5. Cleanup: `ssh g14blx03 'git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/t60; rm -rf /var/tmp/fasth3/t60 ~/fasth3/t60-setup.*'`

Logs: `/var/tmp/fasth3/t60/run60_<arm>.log` (AB lines, T60_EXIT[<arm>]). Stop all device work at the first drop
while one of our jobs runs.
