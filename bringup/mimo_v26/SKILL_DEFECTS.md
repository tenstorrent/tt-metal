# Skill / guide / kit defects found during the MiMo bring-up (coordinator list)
1. tt-lab-ops sec.1: worktree recipe is prose only; symlinked build artifacts appear untracked (gitignore entries end in "/", symlinks are files) so launch.py / log_tool.py refuse the "dirty" tree. Fixed in bringup/mimo_v26/tools/mk_worktree.sh (writes .git/info/exclude); skill text still to update.
2. Pre-commit end-of-file hook rewrites json written by scorecard.py template, first commit fails; re-add and commit again.
3. ONBOARDING says mimo_sdpa_probe.md "may be in progress": the file does not exist; attention track must run the probe itself.
