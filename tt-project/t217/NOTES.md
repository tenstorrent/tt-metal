# t217: e2e time on 2.3 VAE with 8-bit DiT (user #148)

Meaning: LTX 2.5 pipeline, 2.3 conv VAE (LTX25_DIFFVAE=0), 4x8 1088x1920 145f, DEFAULT prompt, traced.
8bit = LTX_QUANT=all_bf8_lofi (bf8 DiT linear weights, bf8 activations into ColParallel linears
[LTX_QUANT_ACTIVATIONS=1 default], LoFi; attn-out weights stay bf16; SDPA stays bf16).
The DiT is not 8-bit by default, so #208 (bf16, mean 4.800 s, 3 seeds) is the comparison, not the answer.
Needs LTX_FUSE_GATE_ON_DEVICE=0: fold_gate_on_device concats bf8 QKV with the bf16 gate and
ttnn.concat TT_FATALs on mixed dtypes. So the 8-bit arm also runs without the gate fold.

Box: blx01, same tree as #208 (t208/tree = t48 5e4e0cd643a py, /var/tmp/fasth3/t48 build).
Driver: blx01:/var/tmp/fasth3/t217/driver.sh (pid 3599339, started 2026-10-07 19:43 UTC), waits for the
t209 and t211 drivers, then: fill job (seed 0, writes ...transformer-bf16.q-all_bf8_lofi cache, ~25 GB on
/var/tmp), then the time job (seeds 0-4 in one process). Log: driver.log; marker: driver.marker.
Outputs: blx01:/var/tmp/fasth3/t217/res/{fill,time}/run.log + ltx_av_fast_*.mp4 (gen N = seed N-1, gen 0 = capture).

Next step on wake:
1. cat driver.marker / driver.log; grep E2E_WALL_S and stage timings in res/time/run.log.
2. Score res/time mp4s gen1..5 vs tt-project/baselines/ltx25_1080p_6s/ref_t48_s2x2/seed{0..4}.mp4 (ltx_eval batch, see t185/post.sh).
3. Keep one mp4 + t3s still under ~/fasth3/tt-project-artifacts/t217; delete res/ on blx01 except that clip.
   The bf8 DiT cache under /var/tmp/fasth3/cache/dit-ltx25 is project-created; keep only if 8-bit work follows.
4. result.json summary also lists other 8-bit knobs: LTX_QUANT_ACTIVATIONS=0 (weight-only), LTX_QUANT_SDPA_BF8=1
   (not quality-checked), uint8 yuv420p output (already default), conv-VAE bf8 weights (removed #84, noise),
   LTX_VAE_CONV_FIDELITY=LoFi (not 8-bit).
