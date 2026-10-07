# Skill / guide / kit defects found during the MiMo bring-up (coordinator list)
1. tt-lab-ops sec.1: worktree recipe is prose only; symlinked build artifacts appear untracked (gitignore entries end in "/", symlinks are files) so launch.py / log_tool.py refuse the "dirty" tree. Fixed in bringup/mimo_v26/tools/mk_worktree.sh (writes .git/info/exclude); skill text still to update.
2. Pre-commit end-of-file hook rewrites json written by scorecard.py template, first commit fails; re-add and commit again.
3. ONBOARDING says mimo_sdpa_probe.md "may be in progress": the file does not exist; attention track must run the probe itself.
6. COMMON.md / study: `bringup_study/mimo/sgl.py` is a 14-byte "404: Not Found" stub (the sglang qkv-loading code the brief points to was never fetched); no local MiMo checkpoint exists anywhere on /mnt/tt-data (ONBOARDING implies one). Track 0 cannot do real-token tests until a checkpoint location is given. (found by ref agent)
7. tt-model-reference sec.1-2 has no step for "no local checkpoint": add "check for local weights first; ask before download". Not yet edited in the skill.
4. Kit: models/common/bringup_kit/__pycache__/*.pyc were tracked; running any kit tool dirtied the tree so log_tool.py/launch.py refused. Found by the moe agent (b93346bd3bc), fixed on ssinghal/mimo by untracking + .gitignore.
5. Host list in briefs/ONBOARDING goes stale within minutes: DSV4 agents grab "free" hosts (.33 and .34 went busy within ~30 min of an OK health check). Always run host_health.py immediately before launch; keep a spare host per agent.
