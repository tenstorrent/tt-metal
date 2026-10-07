# t211 notes: same LTX-2.5 video, 2.3 conv VAE vs 2.5 DiffVAE

Search (2026-10-07 18:30 UTC): no existing pair decoded the same latents with both VAEs.
- ref_dv145 (DiffVAE) is an older DiT config (3c13c445c79, S1 8 / S2 3); ref_t48_f6b8, ref_t48_s2x2 and t208 are 2.3-VAE only.
- t20 jobs 874/879 used different commits/prompts. No latents were saved by #208 or #210 (#210 was CPU-only).

Plan (running): driver on blx01 `/var/tmp/fasth3/t211/driver.sh` (pid 3200465, started 18:32 UTC, setsid).
- Waits for #209's driver (`/var/tmp/fasth3/t209/driver.marker`), then blx01 health (two passes 60 s apart).
- Job conv: t48 HEAD python 5e4e0cd643a (t208 overlay tree, read only), C++ bf7db12a149, 4x8 ring 1088x1920/145f,
  traced, warmup, gen#0 capture + gen#1 seed 0, DEFAULT prompt, LTX25_DIFFVAE=0 (2.3 VAE). -t 220.
- Job diffvae: same, LTX25_DIFFVAE=1 (DiffVAE production options, slab 78). -t 600 (unmeasured; cold DiffVAE
  weight cache + JIT on blx01; one retry on plain failure). Drops: rerun, skip after 2.
- Both dump S2 latents (`res/<arm>/s2lat.gen{0,1}.pt`). post.sh checks they are identical
  (`/var/tmp/fasth3/t211/latent_check.txt`) and grabs frame 72 (3.0 s) stills `*_f072.png`.
Scripts: `blx01/` here (copies). Log: blx01 `/var/tmp/fasth3/t211/driver.log`.

Check: `ssh g15blx01 cat /var/tmp/fasth3/t211/driver.marker`

Next step after the marker:
1. scp `res/{conv,diffvae}/ltx_av_fast_1920x1088_1.{mp4,json}`, `*_1_f072.png`, run.log, latent_check.txt to
   `tt-project/share/vae_compare/` (names: ltx25_seed0_vae23.mp4 / ltx25_seed0_diffvae.mp4 + pngs).
   Keep `s2lat.gen1.pt` (latents, ~80 MB) on blx01 `/var/tmp/fasth3/t211/res/conv/` as the DiffVAE reference input for #210.
2. PCC/PSNR between the two decodes: ltx_eval batch (t170 worktree), cand=diffvae, ref=conv, no VBench.
3. VAE decode times: grep 'VAE decode' run.log (gen#1).
4. Answer with `scp g15blx02:<abs path> .` lines.
