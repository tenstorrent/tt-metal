# t44 notes: conv VAE decode op cuts (off-device)

Branch ttp/t44-off-device-prototype-conv-vae-decode-op- (pushed): 1968790b040 (fold), 33beb4d89ca (A/B harness).

## (1) T frame-repeat pad fold: done, opt-in LTX_VAE_FOLD_TIME_PAD=1
- neighbor_pad_async's t_front_pad writes ZERO frames (local_copy_writer Phase A, phase2_w_reader), and it is
  front-only. LTX decode is non-causal (causal_decoder=False) and repeats the first AND last frame, so the
  #43 idea of reusing t_front_pad would have given wrong edge frames.
- What I did instead: conv3d already has padding_mode="replicate" (reader clamps indices). When H and W are both
  sharded (4x8 / 2x4), conv3d's internal H/W pad is 0, so replicate only acts on T, which matches LTX exactly. The
  decoder passes padding=(1,0,0), mode replicate, and drops the slice+concat. No C++ change, no rebuild.
  Guarded: only non-causal calls, even time_pad, internal H/W pad == 0. Encoder (causal) keeps the concat.
- Proof (CPU): models/tt_dit/tests/models/ltx/test_vae_ltx_fold_time_pad_ref.py emulates the reader index math
  against the diffusers LTX-2 conv (5 pass; also shows replicate with an internal H/W pad would be wrong).
  Run: python -m pytest --noconftest <file>.
- Device unit test (my branch only): test_vae_ltx.py::test_ltx_conv3d_fold_time_pad (4x8 open, 2x4 submesh,
  fold on vs off must be torch.equal).
- Expected saving (estimate, unmeasured): 42 k=3 convs per decode; the concat copies ~3.5 GB/chip of activations
  (read+write ~7 GB) at 1080p/145f on 4x8, dominated by the 145-frame stages (res 256ch x12, 128ch x8, conv_out).
  ~25-70 ms of the ~0.70 s decode window, depending on RM concat bandwidth. Expect bit-identical output.

## (2) "2 untilizes per resnet block": already gone, nothing to cut
dit_rms_norm_unary_fused -> prim::layer_norm accepts ROW_MAJOR interleaved input, tilizes/untilizes in-kernel,
and returns ROW_MAJOR (compute_output_specs: output layout = input layout). Resnet inputs are ROW_MAJOR (conv3d
and RM+RM add outputs), so ttnn.to_layout(h, ROW_MAJOR) returns early (to_layout_op.cpp:62). Fixed the stale
comment that said the norm outputs TILE. A profile will still show the norm's in-kernel tilize cost.

## Ready-to-run A/B on blx03 (do NOT run while device work is paused)
Files here: run44.sh, test_vae_ltx_fold_time_pad_ab.py, vae_ltx_t36_fold.py (blx03 tree's vae_ltx.py =
origin/ttp/t36-blx03-ltx25 + the fold patch, loaded as an overlay; the shared blx03 tree is untouched), compare44.py.
Opens (4,8) and runs on create_submesh(2,4); 544x960/145f, real 2.5 latent from t37 job 931, fused YUV on,
AICLK cap 1150, 1 warmup + 3 timed decodes. One arm per job.
  ssh g14blx03 'mkdir -p ~/fasth3/t44' && scp tt-project/t44/{run44.sh,test_vae_ltx_fold_time_pad_ab.py,vae_ltx_t36_fold.py,compare44.py} g14blx03:fasth3/t44/
  ttp lock g14blx03-device -- ssh g14blx03 '~/fasth3/tt-metal/tmp/blx03/submit.sh 900 bash /home/smarton/fasth3/t44/run44.sh 0'
  (after that job ends) same with '... run44.sh 1'
  ssh g14blx03 'cd ~/fasth3/tt-metal && python_env/bin/python ~/fasth3/t44/compare44.py'
Pass: identical=True; saving = min decode_s fold0 - fold1 (AB44 lines). Cleanup: blx03 ~/fasth3/t44, /var/tmp/fasth3/t44.
To rebuild the overlay: git show origin/ttp/t36-blx03-ltx25:models/tt_dit/models/vae/vae_ltx.py > f;
  git diff b9f8587ce6c 1968790b040 -- models/tt_dit/models/vae/vae_ltx.py | patch f
