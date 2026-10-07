# Skill / guide / kit defects found during the MiMo bring-up (coordinator list)
1. tt-lab-ops sec.1: worktree recipe is prose only; symlinked build artifacts appear untracked (gitignore entries end in "/", symlinks are files) so launch.py / log_tool.py refuse the "dirty" tree. Fixed in bringup/mimo_v26/tools/mk_worktree.sh (writes .git/info/exclude); skill text still to update.
2. Pre-commit end-of-file hook rewrites json written by scorecard.py template, first commit fails; re-add and commit again.
3. ONBOARDING says mimo_sdpa_probe.md "may be in progress": the file does not exist; attention track must run the probe itself.
4. COMMON.md / study: `bringup_study/mimo/sgl.py` is a 14-byte "404: Not Found" stub (the sglang qkv-loading code the brief points to was never fetched); no local MiMo checkpoint exists anywhere on /mnt/tt-data (ONBOARDING implies one). Track 0 cannot do real-token tests until a checkpoint location is given. (found by ref agent)
5. tt-model-reference sec.1-2 has no step for "no local checkpoint": add "check for local weights first; ask before download". Not yet edited in the skill.
