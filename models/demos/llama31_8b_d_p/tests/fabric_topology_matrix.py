# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-FABRIC-MATRIX` — which (mesh, topology, num_links, axis) combinations can actually run a
collective on this Blackhole Galaxy. The evidence behind `DEC-080` and `DEC-081`.

Two findings this script exists to reproduce, both measured rather than argued:

1. **A top-level partial mesh cannot bring the fabric up.** `ttnn.open_mesh_device(MeshShape(1, 8))`
   opens, and then fabric init dies with
   ``Fabric Router Sync: Timeout after 10000 ms … furthest-behind stage: STARTED``
   (`tt_metal/fabric/fabric_firmware_initializer.cpp:206`) — the routers on the opened devices wait
   for an ethernet handshake with partners outside the mesh, which have no kernel running. So every
   P8 sub-shape is carved from the full 32-device mesh with `create_submesh`
   (`tt_metal/api/tt-metalium/mesh_device.hpp:307`) instead.
2. **Two OVERLAPPING submeshes, live at once, hang the machine** unless `parent.quiesce_devices()`
   separates them. `tt_metal/api/tt-metalium/mesh_device.hpp:296` documents exactly this ("insert a
   barrier between phases that use overlapping submeshes on the same physical devices"), and the
   failure mode is the worst kind: the second collective never returns, and **every** later
   collective on the box — including ones that had just passed — then hangs too, until `tt-smi -r`.
   The `overlap-nobarrier` / `overlap-quiesce` pair below is the controlled measurement.

   This is also a correction. The first P8 draft hit the hang while running a `(1,2)` submesh
   collective and then a `(1,8)` one in the same process, and blamed `Topology.Linear` on the 8-wide
   logical row (the system mesh is `MeshShape([8, 4])`, so a logical `(4,8)` row of 8 is linear index
   `r*8 + c` -> physical `(idx // 4, idx % 4)` = two physical rows, which made a missing linear route
   a plausible story). The matrix refutes it: `1x8:linear:1:1:submesh` passes in isolation. Topology
   and link count are innocent; the overlap is the cause. `DEC-081` records the correction.

Each case runs in its own subprocess with a timeout, so a hang is recorded as `HANG` instead of
taking the harness down with it.

Run (the descriptor matters — see `DEC-020`)::

    export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
    python3 models/demos/llama31_8b_d_p/tests/fabric_topology_matrix.py

Exit status is 0 when every case matched its expectation (`ok` cases completed, the `hang` case hung),
1 otherwise. **After a run that includes the hanging case, reset the box: `tt-smi -r`.**

Measured on 2026-09-03 (`bringup_log/raw/G-FABRIC-MATRIX_*.log`): 12/12 single-mesh cases as listed,
`overlap-quiesce` ok, `overlap-nobarrier` hang.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# (rows, cols, topology, num_links, cluster_axis, expectation). `submesh` is implied for anything
# smaller than the full galaxy; `toplevel` cases open the shape directly.
FULL = (4, 8)
CASES = [
    # shape,  topology, links, axis, mode,       expectation
    ((4, 8), "ring", 2, 1, "toplevel", "ok"),
    ((4, 8), "ring", 2, 0, "toplevel", "ok"),
    ((1, 2), "ring", 2, 1, "submesh", "ok"),
    ((1, 4), "ring", 2, 1, "submesh", "ok"),
    ((1, 8), "ring", 2, 1, "submesh", "ok"),
    ((1, 8), "ring", 1, 1, "submesh", "ok"),
    ((2, 8), "ring", 2, 1, "submesh", "ok"),
    ((2, 8), "ring", 2, 0, "submesh", "ok"),
    ((1, 2), "linear", 1, 1, "submesh", "ok"),
    # Fabric init refuses these outright (fast failure, no poisoning).
    ((1, 8), "linear", 1, 1, "toplevel", "fabric-init-fail"),
    ((2, 8), "ring", 2, 1, "toplevel", "fabric-init-fail"),
    ((1, 8), "linear", 1, 1, "submesh", "ok"),
]

# Cases that are not a single (shape, topology, links, axis) tuple. `overlap-nobarrier` is the one
# that hangs the machine, so it runs LAST: everything after it would report a false HANG.
SPECIAL_CASES = [
    ("overlap-quiesce", "ok"),
    ("overlap-nobarrier", "hang"),
]

CASE_TIMEOUT_S = 240


def _run_overlap(barrier: bool) -> int:
    """Two OVERLAPPING submeshes, live at the same time, each running a collective.

    `mesh_device.hpp:296-305` says a barrier is required "between phases that use overlapping
    submeshes on the same physical devices" and names `quiesce_devices()` as that barrier. This case
    measures what happens without it, because the first P8 draft hit exactly this and misread it as a
    topology problem: a `(1,2)` submesh collective followed by a `(1,8)` one on the same devices hung
    the machine, and `(1,8)` alone then looked guilty.
    """
    import torch

    import ttnn
    from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
    from models.demos.llama31_8b_d_p.tt.config import MeshConfig

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*FULL))
    try:
        subs = []
        for phase, (rows, cols) in enumerate(((1, 2), (1, 8))):
            sub = parent.create_submesh(ttnn.MeshShape(rows, cols), ttnn.MeshCoordinate(0, 0))
            subs.append(sub)  # deliberately kept live: that is the configuration under test
            ccl = CCLManager(sub, num_links=1, topology=ttnn.Topology.Linear)
            mesh_config = MeshConfig((rows, cols), tp=cols)
            host = torch.randn(1, 1, 128, 4096)
            tt_in = ttnn.from_torch(
                host,
                device=sub,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
            )
            print(f"[fabric] START overlap phase {phase} {rows}x{cols} barrier={barrier}", flush=True)
            out = mesh_config.allreduce(tt_in, ccl, axis=1)
            ttnn.synchronize_device(sub)
            got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
            rel = (got - host * cols).abs().max().item() / (host * cols).abs().max().item()
            print(f"[fabric] DONE overlap phase {phase} {rows}x{cols} rel={rel:.2e}", flush=True)
            out.deallocate(True)
            if barrier:
                parent.quiesce_devices()
                print("[fabric] quiesce_devices() barrier taken", flush=True)
    finally:
        for submesh in parent.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


def _run_one(rows, cols, topology, num_links, axis, mode) -> int:
    """Open the mesh (or carve the submesh), run one all-reduce, check it summed `ring` copies."""
    import torch

    import ttnn
    from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
    from models.demos.llama31_8b_d_p.tt.config import MeshConfig

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING if topology == "ring" else ttnn.FabricConfig.FABRIC_1D,
    )
    if mode == "toplevel":
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
        mesh = parent
    else:
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*FULL))
        mesh = (
            parent
            if (rows, cols) == FULL
            else parent.create_submesh(ttnn.MeshShape(rows, cols), ttnn.MeshCoordinate(0, 0))
        )
    try:
        ccl = CCLManager(
            mesh,
            num_links=num_links,
            topology=ttnn.Topology.Ring if topology == "ring" else ttnn.Topology.Linear,
        )
        mesh_config = MeshConfig((rows, cols), tp=cols)
        host = torch.randn(1, 1, 128, 4096)
        # Replicated input: an all-reduce over `ring` devices must return exactly `ring x` the input,
        # so the check is a value check and not just a shape check.
        tt_in = ttnn.from_torch(
            host,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"[fabric] START {rows}x{cols} {mode} topology={topology} num_links={num_links} axis={axis}", flush=True)
        out = mesh_config.allreduce(tt_in, ccl, axis=axis)
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
        ring = (rows, cols)[axis]
        max_err = (got - host * ring).abs().max().item()
        rel = max_err / (host * ring).abs().max().item()
        print(
            f"[fabric] DONE {rows}x{cols} {mode} topology={topology} num_links={num_links} axis={axis} "
            f"out={tuple(got.shape)} ring={ring} max|err vs {ring}x input|={max_err:.4f} rel={rel:.2e}",
            flush=True,
        )
        # bf16 rounding of a `ring`-way sum only; anything larger means the reduction is wrong.
        assert rel < 1e-2, f"all-reduce did not sum {ring} copies: rel err {rel}"
    finally:
        for submesh in parent.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="ROWSxCOLS:topology:num_links:axis:mode — run one case in-process")
    args = parser.parse_args(argv)

    if args.case in ("overlap-nobarrier", "overlap-quiesce"):
        return _run_overlap(barrier=args.case == "overlap-quiesce")
    if args.case:
        shape, topology, num_links, axis, mode = args.case.split(":")
        rows, cols = (int(x) for x in shape.split("x"))
        return _run_one(rows, cols, topology, int(num_links), int(axis), mode)

    rows_out = []
    failures = 0
    cases = [
        (f"{rows}x{cols}:{topology}:{num_links}:{axis}:{mode}", expectation)
        for (rows, cols), topology, num_links, axis, mode, expectation in CASES
    ] + SPECIAL_CASES
    for case, expectation in cases:
        print(f"\n=== case {case} (expect {expectation}) ===", flush=True)
        try:
            done = subprocess.run(
                [sys.executable, "-u", __file__, "--case", case],
                timeout=CASE_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
            tail = [ln for ln in done.stdout.splitlines() if "[fabric]" in ln]
            for line in tail:
                print(line, flush=True)
            if done.returncode == 0:
                got = "ok"
            else:
                got = "fabric-init-fail" if "Fabric Router Sync" in (done.stdout + done.stderr) else "error"
                for line in (done.stdout + done.stderr).splitlines():
                    if "Fabric Router Sync" in line or "TT_THROW" in line or "TT_FATAL" in line:
                        print(f"    {line.strip()[:220]}", flush=True)
                        break
        except subprocess.TimeoutExpired:
            got = "hang"
            print(f"    no completion in {CASE_TIMEOUT_S}s", flush=True)
        ok = got == expectation
        failures += 0 if ok else 1
        rows_out.append((case, expectation, got, "MATCH" if ok else "MISMATCH"))

    print("\n| case | expected | measured | |", flush=True)
    print("|---|---|---|---|", flush=True)
    for case, expectation, got, verdict in rows_out:
        print(f"| `{case}` | {expectation} | **{got}** | {verdict} |", flush=True)
    print(
        f"\n[fabric] {len(rows_out) - failures}/{len(rows_out)} cases matched expectation. "
        f"DEC-080: top-level partial meshes cannot init the fabric, so every sub-shape is a submesh "
        f"of the open galaxy. DEC-081: two OVERLAPPING submeshes live at once hang the machine "
        f"unless quiesce_devices() separates them; topology and link count are innocent.",
        flush=True,
    )
    print("[fabric] a HANG case was run — reset the machine now: tt-smi -r", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
