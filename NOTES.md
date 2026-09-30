# t22 notes
- Plan: tt-project/research/denoise_plan.md
- Win: seeded-noise prefetch, commit cc51d4a21bc (CPU test test_ltx_seeded_noise.py passes, bit-exact draws).
- E2E job 601 (submit log tmp/submit_noisepf.log): bash tmp/e2e.sh noisepf, env tmp/e2e_env.yaml
  (TT_METAL_HOME=t14 worktree for kernel-cache hits; python from t22). Outputs tmp/e2e/noisepf/.
- Next: when 601 is done: grep "denoise init\|Stage [12] denoise" from `tt-device-mcp logs -n 100000 601`;
  compare latents tmp/e2e/noisepf/latents.gen2.pt vs ../t14/tmp/e2e/safe/latents.gen2.pt (expect bit-identical);
  baseline init S1 ~85 ms, S2 ~135 ms, S1 2.18 s, S2 2.46 s (job 574).
