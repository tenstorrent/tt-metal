# Task: Run ttnn ops (add, matmul, CCL) across two cabled T3Ks

## What

Produce a **reproducible guide** for running real `ttnn` tensor operations
(**add + matmul + one CCL collective**) across the two cabled T3K boxes, covering
**both** distributed models:

- **Big-Mesh (scale-up)** — one logical `1×16` MeshDevice spanning both hosts; a single
  op runs SPMD across all 16 chips.
- **Multi-Mesh + sockets (scale-out)** — each host owns a `2×4` mesh; op runs per-host and
  tensors cross the QSFP fabric via `MeshSocket` send/recv.

## Why

`tests/scripts/multihost/run_dual_t3k_tests.sh` already validates the fabric, but it only
runs **C++** fabric/mesh-socket binaries. There is no worked example of a real `ttnn`
Python op running across the two hosts. This job fills that gap with runnable scripts and
a guide the user can reproduce.

## Hardware

- Launcher: `t3k-node-b` = 192.168.1.243, NIC `ens18` (this host).
- Remote:   `t3k-node-a` = 192.168.1.247, NIC `ens18`.
- Each box: 8× Wormhole_b0 (2×4 mesh). Cabled by QSFP fabric.
- Checkout `/home/namvu/tenstorrent/tt-metal`, `build → build_Release`, identical on both.
- Launcher: `mpirun-ulfm` (OpenMPI 5.0.7 ULFM), `tt-run` from `python_env`.

## Definition of done

Each model's script prints a rank-0 PASS with PCC vs a torch golden and exits 0 under
`tt-run` across both hosts. Working commands + gotchas captured in `GUIDE.md` /
`findings.md`; real run output captured in `progress.md`.

See `../WORKFLOW.md` for the folder convention.
