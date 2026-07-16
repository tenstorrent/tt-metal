# Task: TT software stack, end-to-end, proven on dual T3K

## What (from ../task.md)

Study how the Tenstorrent software layers work together end-to-end
(tt-topology → tt-fabric → TTFM → tt-run → ttnn), document which layer does what and how
they hand off, and **prove it** with one minimal-but-complete Big-Mesh TTNN workload:
open a single 1×16 MeshDevice across both hosts, create sharded/replicated tensors, run
compute (add, matmul) and collectives (**all-gather + all-reduce + reduce-scatter**), each
verified against a torch golden. Output is a team-presentable layer-by-layer doc + the script.

## Deliverables (this folder)

- `STACK.md` — **headline**: layer-by-layer walkthrough (5 layers below) with file-path
  citations + a runtime init trace + a hand-off diagram; embeds the captured PASS output and the
  fabric-manager demo transcript. Cross-links `../dual_t3k_ops/GUIDE.md` (how-to-run) and
  `../dual_t3k_ops/topology.html` (wiring) instead of duplicating them.
- `scripts/stack_workload.py` — Big-Mesh 1×16 SPMD: add, matmul, all_gather, all_reduce,
  reduce_scatter; torch golden per op; per-local-shard verification.
- `progress.md`, `findings.md` — journal + stable facts (delta over `../dual_t3k_ops/findings.md`).
- Optional: render `STACK.md` as an Artifact for the team.

## Decisions already made

- New job folder (this one), separate from `dual_t3k_ops`.
- TTFM: **document + live demo** the in-repo `run_fabric_manager` split lifecycle.
- Big-Mesh only (the Multi-Mesh socket pipeline stays in `dual_t3k_ops`).

See `PLAN.md` in this folder for the full plan (layer map with verified file paths, CCL
signatures, workload verification design, execution steps, cautions, definition of done).

## Environment (mounted framework — from dual_t3k_ops/findings.md)

```bash
cd /home/namvu/dual-t3k/tt-metal && source python_env/bin/activate
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/namvu/dual-t3k/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/namvu/dual-t3k/tt-metal
export TT_METAL_CACHE=/home/namvu/.cache/tt_metal_local   # LOCAL per host
```
Hosts: launcher `t3k-node-b` .243, remote `t3k-node-a` .247, NIC `ens18`.
