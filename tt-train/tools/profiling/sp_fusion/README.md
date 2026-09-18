# SP fusion campaign: notes and harness (paused 2026-09-18)

Start with `PAUSE.md` (state, open questions in order, commands), then `DESIGN.md` (design and every measured
number), `HANDOFF_S.md` (two-queue backward overlap) and `HANDOFF_F.md` (fused ops round 2).

The scripts here are the harness used for the measurements. They run from a git-ignored working copy: copy this
directory to `<repo>/generated/spfuse/` (that is the `SPFUSE` path `env.sh` exports; logs and locks are created
there), then `source env.sh`. `build.sh` serialises builds and installs the ttnn libraries; `devrun.sh` serialises
device runs behind a lock with an inactivity watchdog that resets the chips on a hang; `bench_sp_train.sh` runs the
Llama-8B tp4 SP training step (`BATCH=`, `MEMEFF=1` for recompute, impl `composed|fused|nocomm` where `nocomm` is the
zero-communication ideal); `matrix.sh` and `final_regress.sh` drive the sweeps; `bench_sp_collectives.py` times the
standalone collectives and matmuls (run from the repo root with `-p conftest -s`); `kit/` holds the tracy ops-CSV
analyzer (device profiling of the 32-layer step stalls; profile a 2-4 layer variant instead).
