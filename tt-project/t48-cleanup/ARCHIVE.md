# t48 notes archive (#150, cleanup steps 1-2)

Archive branch: `ttp/t48-notes-archive` @ `b81bb403d86c5fa6261beb89ae2581f3393d9562`
(origin/ttp/t48-ltx25-integrated tip on 2026-10-06, 03:47:56 UTC commit time). Created by a plain
`git push origin <sha>:refs/heads/ttp/t48-notes-archive` (new branch, no force). t48 itself was
not changed. Every note and driver below stays reachable there with today's hashes.

## Step 2: shared blx03 scripts rehomed

Harness commit `dd2ebdc` (tt-project/harness, local repo): `templates/blx03/`
- `submit.sh` <- ttp/t36-blx03-ltx25 @ 86076afc66 `tmp/blx03/submit.sh` (the live blx03 copy; t48's
  copy is older: no duplicate-submit or bare-2x4 guard)
- `env.yaml`, `run25.sh`, `run48.sh` <- t48 `tmp/blx03/`
- `driver.sh` <- t48 `tmp/blx03/drv_template/driver.sh`
- `test/test_health.sh` <- t48 `tmp/blx03/drv_template/test_health.sh` (path to driver.sh fixed; 16/16 pass)
- `README.md` (new): file map, sources, overlap with `templates/blx03-runner/`
- `templates/blx03-launch.sh` comment now points at `templates/blx03/driver.sh`

Not copied (task notes, stay in the archive): `tmp/blx03/NOTES.md`, `tmp/blx03/READY_blx03.md`.

## Step 3 (NOT applied): what the removal commit would delete

`git ls-tree` of t48 @ b81bb403d86 under `tmp/`, `tt-project/` and top-level `NOTES.md`:
39 files (PLAN.md says 38; its own list adds up to 39).

- `NOTES.md`
- `tmp/READY_22.md`
- `tmp/READY_26.md`
- `tmp/READY_48.md`
- `tmp/block_env.yaml`
- `tmp/blx03/NOTES.md`
- `tmp/blx03/READY_blx03.md`
- `tmp/blx03/drv_template/driver.sh`
- `tmp/blx03/drv_template/test_health.sh`
- `tmp/blx03/env.yaml`
- `tmp/blx03/run25.sh`
- `tmp/blx03/run48.sh`
- `tmp/blx03/submit.sh`
- `tmp/chunk_ab.sh`
- `tmp/cmp.sh`
- `tmp/cmp_blk.py`
- `tmp/done.sh`
- `tmp/drive6.sh`
- `tmp/e2e.sh`
- `tmp/na_kernel_cmds.txt`
- `tmp/na_kernel_compile.sh`
- `tmp/t138/NOTES.md`
- `tmp/t138/driver.sh`
- `tmp/t138/run_e2e.sh`
- `tmp/t140/configs.txt`
- `tmp/t140/driver.sh`
- `tmp/t140/post.py`
- `tmp/t140/run_cfg.sh`
- `tmp/t141/NOTES.md`
- `tmp/t141/driver.sh`
- `tmp/t141/run_e2e.sh`
- `tmp/t60/READY_60.md`
- `tmp/t60/blx03_setup60.sh`
- `tmp/t60/check60.sh`
- `tmp/t60/compare60.py`
- `tmp/t60/kernel_compile.sh`
- `tmp/t60/np_kernel_cmds.txt`
- `tmp/t60/run60.sh`
- `tt-project/t93/PLAN.md`
