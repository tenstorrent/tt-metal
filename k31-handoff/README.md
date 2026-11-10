# K3.1 P->D handoff

Everything needed to rebuild the K3.1 prefill run if /data/rmiller is lost.

- Code: this branch is rmiller's K3.1 prefill work at 7370b8c2685, plus this directory.
- jobs/k31/*.sh: the Slurm job and build scripts (hostprep, build, engine build, sc1 job, venv clone, cpm relink).
- jobs/k31/patches/: patches that the build applies.
- PLAN.md: run plan and environment notes.

Not included: build_Release (2.4 GB, rebuild with jobs/k31/k31_build.sh), python_env (3.7 GB, rebuild with
create_venv.sh or venv_clone.sh), the weight cache overlay (symlinks to the Weka K3.1 cache), ssh keys, run logs.
