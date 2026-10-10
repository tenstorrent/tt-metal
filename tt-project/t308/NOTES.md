# t308: open the #194 draft PR

State (2026-10-10 04:50 UTC):
- #305 confirms the gain: main 6.41/6.19 s (jobs 210/214) vs PR 6.24 s (job 213), ~-0.2 s, all in S1+S2. Quality: t301 cmp PCC 0.9983-0.9988, PSNR 38-39.6 dB, stills identical.
- Revert of 05401286709 (mel-VAE trace) committed on local ttp/ltx23-main-pr as ae3469d0a26 (worktree tt-project/worktrees/t308-mainpr). NOT pushed yet. Revert (not drop) so the push stays a fast-forward.
- PR body ready: tt-project/t308/PR_BODY.md (this branch).
- Blocker: project push_checks (harness/project.json) run ltx-rt-only test files (test_vae_ltx_*_ref.py, test_conv3d_sweep_halo_cpu.py, test_denoise_trims.py). None exist on a main-based branch, so `ttp checks` fails on ae3469d0a26 and gh/ttp push --own refuse. Needs a harness fix (skip missing files).
- Branch conflicts with current origin/main (13 hunks: dit_fused_distributed_rmsnorm op, normalization.py, rotary_embedding_llama factory; main #59195, #58676). Noted in the PR body.

Next, once checks are fixed:
1. cd tt-project/worktrees/t308-mainpr; ttp checks -- (CPU tests, see PR_BODY Checks); ttp push --own --detach (fast-forward df9e5ecaac6 -> ae3469d0a26).
2. gh pr create --draft --repo tenstorrent/tt-metal --base main --head ttp/ltx23-main-pr --title "LTX-2.3 distilled: turn on bit-identical speed gains and a faster mp4 export by default" --body-file <PR_BODY.md>
