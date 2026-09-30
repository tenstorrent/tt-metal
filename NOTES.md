# t22 notes
- Plan: tt-project/research/denoise_plan.md
- Win: seeded-noise prefetch, commit cc51d4a21bc (CPU test test_ltx_seeded_noise.py passes, bit-exact draws).
- E2E job 601 (submit log tmp/submit_noisepf.log): bash tmp/e2e.sh noisepf, env tmp/e2e_env.yaml
  (TT_METAL_HOME=t14 worktree for kernel-cache hits; python from t22). Outputs tmp/e2e/noisepf/.
- Next: when 601 is done: grep "denoise init\|Stage [12] denoise" from `tt-device-mcp logs -n 100000 601`;
  compare latents tmp/e2e/noisepf/latents.gen2.pt vs ../t14/tmp/e2e/safe/latents.gen2.pt (expect bit-identical);
  baseline init S1 ~85 ms, S2 ~135 ms, S1 2.18 s, S2 2.46 s (job 574).
- E2E job 602: same + LTX_DEVICE_PROMPT_HANDOFF=1 (plan item #2), outputs tmp/e2e/noisepf_ph/. Compare its latents to safe too
  (handoff may change nothing numerically — the prompt comes from the encoder straight into device buffers).
- 2026-09-30 14:10 UTC (attempt 2): 601/602 still queued (first in line); device HELD by broker for recovery
  (chip 20 left PCIe bus). Nothing run. On resume: `bash tmp/cmp.sh noisepf 601` and `bash tmp/cmp.sh noisepf_ph 602`.
- 2026-09-30 14:15 UTC (attempt 3): 601/602 FAILED in 3 s — t22 had no ttnn/ttnn/_ttnn.so (ImportError
  get_all_unsafe_tracked_ids). Fixed: symlink ttnn/ttnn/_ttnn.so -> t14's build (t22 has no ttnn diff vs t14; import checked OK).
  Resubmitted directly: job 630 (noisepf), 631 (noisepf_ph). On resume: `bash tmp/cmp.sh noisepf 630`, `bash tmp/cmp.sh noisepf_ph 631`.
