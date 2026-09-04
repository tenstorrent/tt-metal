# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The (mesh, topology, links, axis) sweep for one Blackhole Galaxy. Gate: `G-FABRIC-MATRIX`.

**HF anchor:** none — this file holds no model math. It maps which mesh/fabric/topology
combinations can run a collective **at all** on this box, and it must run **before every other P8
gate** (`BRINGUP_RECIPE.md:1755-1757`), because two of the combinations below do not fail: they
**hang the machine**, and a hang is not contained. After one, every later collective on the box
hangs too — including one that passed forty seconds earlier — until `tt-smi -r`
(`BRINGUP_RECIPE.md:1706-1712`, `models/demos/common/bringup/LANDMINES.md` "Two overlapping live
submeshes").

**So each case runs in its own subprocess with a timeout**, which is what turns a hang into a
recorded measurement instead of a lost session, and the parent runs `tt-smi -r` after any case that
timed out (`DEC-070`). `tt-smi -r` was verified on this box before the first hazardous case ran:
exit 0 in 41.8 s.

**Why a *submesh* sweep and not the template's `mesh_device` parametrisation.** The obvious form —
`@pytest.mark.parametrize("mesh_device", [(1,2), (1,4), (1,8), (2,8)], indirect=True)`, which is
what `models/demos/minimax_m3/tests/test_factory.py:89` does — opens a **top-level partial mesh**,
and on this galaxy that dies in fabric bring-up: the routers on the opened devices wait for an
ethernet handshake with partners outside the mesh, which have no kernel running
(`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:200`). The two `toplevel_*` cases
below are that claim, stated as an expectation and measured. Every other case opens the full
`(4, 8)` once and carves a submesh (`tt_metal/api/tt-metalium/mesh_device.hpp:307`).

**Every expectation in `CASES` was written before the sweep ran**, from the recipe's own measured
claims, and the gate is that each case matches it — *including* the ones expected to fail or hang.
Cases whose expectation is `ok` additionally check the collective's **result bit-exactly**
(`torch.equal`), not just that it completed: a collective that returns garbage without hanging is
the failure this file would otherwise wave through.

Run (parent; ~4 min, resets the box twice):

    python models/demos/llama31_8b_d_p/tests/fabric_topology_matrix.py

Run one case (child):

    python models/demos/llama31_8b_d_p/tests/fabric_topology_matrix.py --case submesh_1x8_ring_l1_ax1

The harness's own negative control — proving a wrong expectation is reported rather than absorbed:

    python models/demos/llama31_8b_d_p/tests/fabric_topology_matrix.py --control
"""

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------------------------
# The single-galaxy torus descriptor.
#
# `BRINGUP_RECIPE.md:80-83` names `bh_galaxy_sp4_torus_xy_graph_descriptor.textproto` and
# `32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto` as "BH Galaxy mesh descriptors ... The
# Ring topology P8 needs the torus descriptor". **Neither is usable on one galaxy** and neither is
# what this package uses (`DEC-071`):
#
#   * `bh_galaxy_sp4_...` declares FOUR meshes of `[32, 4]` with `host_topology [4, 1]` — a
#     super-pod of 4 galaxies, 512 devices, 16 hosts;
#   * `32x4_quad_bh_galaxy_...` declares one `[32, 4]` mesh with `host_topology [4, 1]` — a quad
#     galaxy, 128 devices, 4 hosts.
#
# The single-galaxy torus descriptor is `single_bh_galaxy_torus_xy_graph_descriptor.textproto`
# (`[8, 4]`, `dim_types: [RING, RING]`, `host_topology [1, 1]`), which is also what the in-repo
# galaxy harness this file's Ring cases follow uses
# (`models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:26`).
TORUS_DESCRIPTOR = "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_DESCRIPTOR = "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

GALAXY_SHAPE = (4, 8)
GALAXY_NUM_DEVICES = 32

# Per-case subprocess timeout. A case that has not finished by then is recorded `hang`. 240 s is
# ~6x the slowest healthy case measured here (the (4,8) two-axis case, 38 s including mesh open and
# kernel JIT), so a `hang` verdict is a hang and not a slow machine (`DEC-070`).
CASE_TIMEOUT_S = 240

# ---------------------------------------------------------------------------------------------
# The sweep. `expect` is the *stated* expectation and was written before any case ran.
# ---------------------------------------------------------------------------------------------
CASES = [
    # --- the top-level partial mesh, which is the trap the template's parametrisation walks into --
    dict(
        id="toplevel_1x8_fabric1d",
        open="toplevel",
        shape=(1, 8),
        fabric="1d",
        topology="linear",
        links=1,
        axis=1,
        expect="error",
        why="a top-level partial mesh: the opened devices' fabric routers wait on partners outside "
        "the mesh (fabric_firmware_initializer.cpp:200). BRINGUP_RECIPE.md:1672-1693.",
    ),
    dict(
        id="toplevel_2x8_fabric1d",
        open="toplevel",
        shape=(2, 8),
        fabric="1d",
        topology="linear",
        links=1,
        axis=1,
        expect="error",
        why="same as above at the shape P8 step 3 needs for 2-link Ring. BRINGUP_RECIPE.md:1672-1693.",
    ),
    # --- submeshes of the full (4,8): the form this package actually uses ------------------------
    dict(
        id="submesh_1x2_linear_l1_ax1",
        open="submesh",
        shape=(1, 2),
        fabric="1d",
        topology="linear",
        links=1,
        axis=1,
        expect="ok",
        why="G-TP-PARITY's smallest TP shape.",
    ),
    dict(
        id="submesh_1x4_linear_l1_ax1",
        open="submesh",
        shape=(1, 4),
        fabric="1d",
        topology="linear",
        links=1,
        axis=1,
        expect="ok",
        why="G-TP-PARITY.",
    ),
    dict(
        id="submesh_1x8_linear_l1_ax1",
        open="submesh",
        shape=(1, 8),
        fabric="1d",
        topology="linear",
        links=1,
        axis=1,
        expect="ok",
        why="the shape whose apparent failure was misdiagnosed as a routing problem when the real "
        "variable was submesh overlap; running it ALONE is what falsified that story. "
        "BRINGUP_RECIPE.md:1713-1723.",
    ),
    dict(
        id="submesh_1x8_ring_l1_ax1",
        open="submesh",
        shape=(1, 8),
        fabric="1d_ring",
        topology="ring",
        links=1,
        axis=1,
        expect="ok",
        why="G-KV-TP8's mesh with the deployment topology. BRINGUP_RECIPE.md:1720-1721 measured "
        "(1,8)+Ring passing at 1 and 2 links.",
    ),
    dict(
        id="submesh_2x8_ring_l2_ax1",
        open="submesh",
        shape=(2, 8),
        fabric="1d_ring",
        topology="ring",
        links=2,
        axis=1,
        expect="ok",
        why="`get_default_num_links` returns 1 for ANY single-row mesh "
        "(models/demos/gpt_oss_d_p/utils/general_utils.py:33), so every (1,N) case above runs "
        "num_links=1 and never touches the deployment fabric. (2,8) is the cheapest shape that "
        "exercises 2-link Ring. BRINGUP_RECIPE.md:1724-1732.",
    ),
    dict(
        id="full_4x8_ring_l2_ax1",
        open="full",
        shape=GALAXY_SHAPE,
        fabric="1d_ring",
        topology="ring",
        links=2,
        axis=1,
        expect="ok",
        why="the deployment mesh, TP axis. G-MESH-KV / G-RACE run here.",
    ),
    dict(
        id="full_4x8_ring_l2_ax0",
        open="full",
        shape=GALAXY_SHAPE,
        fabric="1d_ring",
        topology="ring",
        links=2,
        axis=0,
        expect="ok",
        why="the deployment mesh, SP axis — the axis the ring SDPA's halo exchange runs on.",
    ),
    # --- the two that do not fail, they hang ----------------------------------------------------
    dict(
        id="submesh_1x8_ring_on_fabric1d",
        open="submesh",
        shape=(1, 8),
        fabric="1d",
        topology="ring",
        links=1,
        axis=1,
        expect="hang",
        why="a Ring topology on a plain FABRIC_1D fabric HANGS rather than erroring "
        "(BRINGUP_RECIPE.md:82-84, LANDMINES.md 'Ring collectives hang rather than error'). This "
        "is why DEC-027 couples the fabric config and the topology behind one variable.",
    ),
    dict(
        id="overlap_1x2_then_1x8_no_quiesce",
        open="overlap",
        shape=(1, 8),
        first_shape=(1, 2),
        quiesce=False,
        fabric="1d",
        topology="linear",
        links=1,
        axis=1,
        expect="hang",
        why="two overlapping submeshes live at once with no barrier between their phases. "
        "mesh_device.hpp:296-305 requires one and names quiesce_devices(); nothing enforces it. "
        "THE WORST LANDMINE IN THE SET: the hang is not contained and poisons the box until "
        "tt-smi -r. BRINGUP_RECIPE.md:1694-1712.",
    ),
    # --- ADDENDUM, added after the first sweep -------------------------------------------------
    # The first sweep found that **every** `FABRIC_1D_RING` case fails on this machine: the only
    # single-galaxy RING/RING mesh-graph descriptor
    # (`single_bh_galaxy_torus_xy_graph_descriptor.textproto`) cannot be mapped to the discovered
    # physical topology (`topology_mapper.cpp:544`, "32 target node(s) are not mapped to any global
    # node"), and it is not the channel policy — a RELAXED copy fails identically. Restricting
    # instead (`FABRIC_1D_RING` on a LINE/LINE or LINE/RING descriptor) is refused a step earlier,
    # at `mesh_graph.cpp:447-454`: "FabricConfig can only restrict topology (e.g., torus->mesh), not
    # create new connections". So **there is no ring fabric on this galaxy** (`DEC-079`).
    #
    # It also found that `ttnn.Topology.Ring` **collectives** run correctly on the plain `FABRIC_1D`
    # fabric — bit-exact at `(1,8)` — which is the opposite of `BRINGUP_RECIPE.md:82-84`'s claim
    # that they hang. These cases extend that finding to the shapes P8 actually uses, so the
    # topology the rest of the phase runs on is measured rather than assumed. Expectations written
    # before the addendum ran, from the `(1,8)` result.
    dict(
        id="submesh_2x8_ring_on_fabric1d_l2_ax1",
        open="submesh",
        shape=(2, 8),
        fabric="1d",
        topology="ring",
        links=2,
        axis=1,
        expect="ok",
        why="2-link Ring on the fabric this galaxy actually has. The (1,8) arm ran 1 link.",
    ),
    dict(
        id="full_4x8_ring_on_fabric1d_l2_ax1",
        open="full",
        shape=GALAXY_SHAPE,
        fabric="1d",
        topology="ring",
        links=2,
        axis=1,
        expect="ok",
        why="the deployment mesh and TP axis, on the fabric this galaxy has.",
    ),
    dict(
        id="full_4x8_ring_on_fabric1d_l2_ax0",
        open="full",
        shape=GALAXY_SHAPE,
        fabric="1d",
        topology="ring",
        links=2,
        axis=0,
        expect="ok",
        why="the SP axis — the one the ring SDPA's halo exchange runs on. Only 4 devices long.",
    ),
    dict(
        id="full_4x8_linear_l2_ax1",
        open="full",
        shape=GALAXY_SHAPE,
        fabric="1d",
        topology="linear",
        links=2,
        axis=1,
        expect="ok",
        why="the Linear fallback on the TP axis, so PREFILL_TOPOLOGY=linear is a measured option " "and not a guess.",
    ),
    dict(
        id="full_4x8_linear_l2_ax0",
        open="full",
        shape=GALAXY_SHAPE,
        fabric="1d",
        topology="linear",
        links=2,
        axis=0,
        expect="ok",
        why="the Linear fallback on the SP axis.",
    ),
    dict(
        id="overlap_1x2_then_1x8_quiesce",
        open="overlap",
        shape=(1, 8),
        first_shape=(1, 2),
        quiesce=True,
        fabric="1d",
        topology="linear",
        links=1,
        axis=1,
        expect="ok",
        why="the same two phases with parent.quiesce_devices() between them. This is the case that "
        "makes the one above a statement about the barrier rather than about the shapes.",
    ),
]

CASES_BY_ID = {c["id"]: c for c in CASES}


# =============================================================================================
# child: run exactly one case
# =============================================================================================
def _fabric_config(name):
    import ttnn

    return {"1d": ttnn.FabricConfig.FABRIC_1D, "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING}[name]


def _topology(name):
    import ttnn

    return {"linear": ttnn.Topology.Linear, "ring": ttnn.Topology.Ring}[name]


def _run_collective(mesh, *, topology, links, axis):
    """All-gather a rank-labelled tensor along `axis` and check the result **bit-exactly**.

    Goes through the package's own `MeshConfig.allgather` / `CCLManager`, not a raw
    `ttnn.experimental.*` call, so what this matrix clears is the code path the modules take
    (`bringup_log/04_CCL_PLAN.md` §2).

    Device `(r, c)` holds a `[1,1,32,32]` tile filled with its index **along `axis`**; the gather
    concatenates on dim 3, so every device must come back holding `[0]*32 | [1]*32 | ...`. Integer
    payloads stay <= 31, well inside bf16's exact-integer ceiling of 256
    (`BRINGUP_RECIPE.md:663-670`), so a mismatch cannot be the probe's own numerics.
    """
    import torch

    import ttnn
    from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
    from models.demos.llama31_8b_d_p.tt.config import MeshConfig

    rows, cols = tuple(mesh.shape)
    n = (rows, cols)[axis]
    mesh_config = MeshConfig((rows, cols), tp=cols)
    ccl = CCLManager(mesh, num_links=links, topology=_topology(topology))

    # Per-device payload = the device's coordinate along `axis`.
    host = torch.zeros(rows, cols, 32, 32)
    for r in range(rows):
        for c in range(cols):
            host[r, c] = float((r, c)[axis])
    tt_in = ttnn.from_torch(
        host,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    gathered = mesh_config.allgather(tt_in, ccl, axis=axis, dim=3)
    ttnn.synchronize_device(mesh)

    expected = torch.cat([torch.full((1, 1, 32, 32), float(i)) for i in range(n)], dim=3)
    per_device = ttnn.get_device_tensors(gathered)
    bad = []
    for i, dev_t in enumerate(per_device):
        got = ttnn.to_torch(dev_t).float()
        if tuple(got.shape) != tuple(expected.shape) or not torch.equal(got, expected):
            bad.append(
                (i, tuple(got.shape), float((got - expected).abs().max()) if got.shape == expected.shape else -1)
            )
    gathered.deallocate(True)
    if bad:
        raise AssertionError(
            f"all_gather on axis {axis} of a {(rows, cols)} mesh returned wrong values on "
            f"{len(bad)}/{len(per_device)} devices (device, shape, max|delta|): {bad[:4]}"
        )
    return f"all_gather axis={axis} n={n} bit-exact on all {len(per_device)} devices"


def run_case(case):
    """Run one case in this process. Raises on failure; returns a one-line detail string."""
    import ttnn

    ttnn.set_fabric_config(_fabric_config(case["fabric"]))
    kind = case["open"]

    if kind == "toplevel":
        # The trap: open the partial shape directly, with no full-mesh parent.
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*case["shape"]))
        try:
            detail = _run_collective(mesh, topology=case["topology"], links=case["links"], axis=case["axis"])
        finally:
            ttnn.close_mesh_device(mesh)
        return detail

    parent = ttnn.open_mesh_device(ttnn.MeshShape(*GALAXY_SHAPE))
    try:
        if kind == "full":
            return _run_collective(parent, topology=case["topology"], links=case["links"], axis=case["axis"])

        if kind == "submesh":
            sub = parent.create_submesh(ttnn.MeshShape(*case["shape"]))
            detail = _run_collective(sub, topology=case["topology"], links=case["links"], axis=case["axis"])
            parent.quiesce_devices()
            return detail

        assert kind == "overlap", f"unknown open kind {kind!r}"
        first = parent.create_submesh(ttnn.MeshShape(*case["first_shape"]))
        d1 = _run_collective(first, topology=case["topology"], links=case["links"], axis=case["axis"])
        if case["quiesce"]:
            parent.quiesce_devices()
        # `first` is deliberately still LIVE here: the landmine is two overlapping submeshes with
        # no barrier between their phases, not two submeshes existing.
        second = parent.create_submesh(ttnn.MeshShape(*case["shape"]))
        d2 = _run_collective(second, topology=case["topology"], links=case["links"], axis=case["axis"])
        parent.quiesce_devices()
        return f"phase1: {d1} | phase2: {d2}"
    finally:
        parent.quiesce_devices()
        ttnn.close_mesh_device(parent)


def child_main(case_id):
    case = CASES_BY_ID[case_id]
    try:
        detail = run_case(case)
    except Exception as e:  # noqa: BLE001 — the child's job is to report, the parent's to judge
        first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        print(f"RESULT {case_id} error {type(e).__name__}: {first_line[:300]}", flush=True)
        return 1
    print(f"RESULT {case_id} ok {detail}", flush=True)
    return 0


# =============================================================================================
# parent: sweep, judge, and clean up after a hang
# =============================================================================================
def _reset_box(reason):
    print(f"[fabric-matrix] {reason} -> tt-smi -r (a hang poisons every later collective)", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(["tt-smi", "-r"], capture_output=True, text=True, timeout=600)
    print(
        f"[fabric-matrix] tt-smi -r exit={proc.returncode} in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    if proc.returncode != 0:
        print(f"[fabric-matrix] tt-smi -r FAILED; the box is poisoned:\n{proc.stdout[-2000:]}", flush=True)
    return proc.returncode == 0


def _repo_root():
    """The tt-metal checkout root — five levels up from `<root>/models/demos/<pkg>/tests/<this>`."""
    root = os.path.abspath(__file__)
    for _ in range(5):
        root = os.path.dirname(root)
    assert os.path.isdir(os.path.join(root, "tt_metal", "fabric", "mesh_graph_descriptors")), (
        f"{root} does not look like a tt-metal checkout; the mesh-descriptor directory is missing. "
        f"A wrong root silently makes TT_MESH_GRAPH_DESC_PATH nonexistent and every case reports "
        f"`std::filesystem::exists(mesh_graph_desc_path)` instead of what it was measuring."
    )
    return root


def _child_env(case):
    env = dict(os.environ)
    root = _repo_root()
    env["TT_METAL_HOME"] = root
    env["PYTHONPATH"] = root
    # The mesh-descriptor path and the fabric config are set TOGETHER, always: a Ring topology on a
    # non-torus descriptor is one of the two hang cases below, so leaving either to a default is
    # the bug the matrix is here to map (`DEC-027`, `DEC-071`).
    env["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(
        root, TORUS_DESCRIPTOR if case["fabric"] == "1d_ring" else MESH_DESCRIPTOR
    )
    return env


def run_one_in_subprocess(case):
    """-> (outcome, detail, seconds). `outcome` is one of `ok`, `error`, `hang`."""
    cmd = [sys.executable, os.path.abspath(__file__), "--case", case["id"]]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=CASE_TIMEOUT_S, env=_child_env(case), cwd=_repo_root()
        )
    except subprocess.TimeoutExpired:
        return "hang", f"no result in {CASE_TIMEOUT_S}s", time.perf_counter() - t0
    dt = time.perf_counter() - t0

    # Keep the child's diagnosable tail on EVERY case, not only the ones with no RESULT line: the
    # first sweep reported five cases as bare `TT_FATAL @ topology_mapper.cpp:544` and the sentence
    # that says *why* — "32 target node(s) are not mapped to any global node" — was in the part
    # discarded. A harness that throws away the reason costs a diagnosis run (`DEC-070`).
    tail_lines = (proc.stderr or "").splitlines() + proc.stdout.splitlines()
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(f"RESULT {case['id']} ")), None)
    if line is None:
        # No RESULT line: the child died before it could report (a TT_FATAL in mesh open, a signal).
        # That is still an `error` outcome, and the reason is what makes the case readable.
        reason = _first_interesting(tail_lines)
        return "error", f"child exit={proc.returncode} with no RESULT line: {reason}", dt
    parts = line.split(None, 3)
    outcome = parts[2]
    detail = parts[3] if len(parts) > 3 else ""
    if outcome != "ok":
        detail = f"{detail} || why: {_reason_sentence(tail_lines)}"
    return outcome, detail, dt


_NOISE = ("info     |", "warning  |", "DEBUG", "Config{", "| UMD |", "AICLK")

# The sentences a fabric/mapping failure actually explains itself with. Matched on substrings so
# the harness records the cause and not only the assert's file and line.
_REASON_MARKERS = (
    "could not fit in the discovered physical topology",
    "not mapped to any global node",
    "can only restrict topology",
    "Fabric Router Sync",
    "LOCAL_HANDSHAKE_COMPLETE",
    "furthest-behind stage",
)


def _reason_sentence(lines, limit=420):
    """The first line carrying one of `_REASON_MARKERS`, trimmed — else the last non-noise line."""
    for ln in lines:
        if any(marker in ln for marker in _REASON_MARKERS):
            return ln.strip()[:limit]
    return _first_interesting(lines)


def _first_interesting(lines):
    for ln in reversed(lines):
        if ln.strip() and not any(tok in ln for tok in _NOISE):
            return ln.strip()[:300]
    return "(no diagnosable output)"


def parent_main(control=False, only=None):
    cases = [c for c in CASES if c["id"] in only] if only else list(CASES)
    if only:
        missing = sorted(set(only) - {c["id"] for c in cases})
        assert not missing, f"unknown case id(s): {missing}"
    if control:
        # The harness's own negative control: take one KNOWN-ok case, state its expectation as
        # `hang`, and require the sweep to report a mismatch. Without it, "every case matched"
        # could mean the comparison never ran (recipe §1.4: a test that only ever sees the correct
        # input cannot tell you it is measuring anything). The `id` is unchanged so the child still
        # resolves it; only `expect` — which lives in the parent — is wrong.
        cases = [dict(CASES_BY_ID["submesh_1x8_linear_l1_ax1"], expect="hang")]
        print("[fabric-matrix] CONTROL: one ok case is stated as `hang`; the sweep must report MISMATCH", flush=True)

    print(
        f"[fabric-matrix] {len(cases)} cases, {CASE_TIMEOUT_S}s timeout each, one subprocess per case; "
        f"tt-smi -r after any hang",
        flush=True,
    )
    rows, mismatches = [], []
    for case in cases:
        print(f"\n[fabric-matrix] === {case['id']}  (expect {case['expect']})", flush=True)
        outcome, detail, dt = run_one_in_subprocess(case)
        matched = outcome == case["expect"]
        rows.append((case, outcome, detail, dt, matched))
        print(
            f"[fabric-matrix] {case['id']}: expect={case['expect']} got={outcome} "
            f"{'MATCH' if matched else 'MISMATCH'} in {dt:.1f}s :: {detail}",
            flush=True,
        )
        if not matched:
            mismatches.append(case["id"])
        if outcome == "hang":
            _reset_box(f"{case['id']} hung as {'expected' if matched else 'NOT expected'}")

    print("\n" + "=" * 110, flush=True)
    print(f"{'case':<36} {'expect':<7} {'got':<7} {'match':<6} {'secs':>7}  detail", flush=True)
    print("-" * 110, flush=True)
    for case, outcome, detail, dt, matched in rows:
        print(
            f"{case['id']:<36} {case['expect']:<7} {outcome:<7} {'yes' if matched else 'NO':<6} "
            f"{dt:>7.1f}  {detail[:120]}",
            flush=True,
        )
    print("=" * 110, flush=True)
    print(
        f"[fabric-matrix] {len(rows) - len(mismatches)}/{len(rows)} cases matched their stated expectation"
        + (f"; MISMATCHED: {mismatches}" if mismatches else ""),
        flush=True,
    )
    if control:
        ok = bool(mismatches)
        print(
            f"[fabric-matrix] CONTROL {'PASSED' if ok else 'FAILED'}: a wrong expectation was "
            f"{'reported' if ok else 'ABSORBED — the comparison is not running'}",
            flush=True,
        )
        return 0 if ok else 1
    return 1 if mismatches else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="run exactly this case in this process (child mode)")
    parser.add_argument("--control", action="store_true", help="the harness's own negative control")
    parser.add_argument("--cases", help="comma-separated subset of case ids to run (an addendum run)")
    parser.add_argument("--list", action="store_true", help="print the case ids and their expectations")
    args = parser.parse_args()

    if args.list:
        for case in CASES:
            print(f"{case['id']:<36} expect={case['expect']:<7} {case['why']}")
        return 0
    if args.case:
        return child_main(args.case)
    only = set(args.cases.split(",")) if args.cases else None
    return parent_main(control=args.control, only=only)


if __name__ == "__main__":
    sys.exit(main())
