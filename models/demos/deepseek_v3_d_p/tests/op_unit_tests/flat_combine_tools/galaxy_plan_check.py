# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side check of the flat expert's plans for test_flat_combine_overlap.py, one chip, seconds (no fabric).

For each model at the test's sizes (8 x 1 scaling: experts per chip, capacity) and each layout the overlap runs:
  full     the serial baseline's plan (whole grid)
  rows     MIMO_FL_ROWS=1,<last> (combine's senders on row 0)
  xdown    rows + MIMO_FL_XDOWN=<the row-0 cells combine leaves> (argv[1], "x,y;x,y;..", default: none)
prints the grid, the gate/up split (np, g), gate/up / down / reader / relay counts, the most down tile columns per
core, and fails if a rows / xdown plan puts a core on row 0 other than the XDOWN cells. On a Galaxy chip (12 x 10)
the rows plans should keep np 1 (64 gate/up cores) for GLM 5.3 and Kimi K2.7.
  python flat_combine_tools/galaxy_plan_check.py ["1,0;4,0;9,0;10,0;11,0"]
Without hardware: TT_METAL_MOCK_CLUSTER_DESC_PATH=<umd>/tests/cluster_descriptor_examples/blackhole_galaxy.yaml."""

import os
import subprocess
import sys

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests import test_hybrid_routed_expert_combine as hyb
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants

MODELS = {"glm-53": ("glm-53", None), "kimi-k27": ("kimi-k27", None), "kimi-k3": ("kimi-k27", (3584, 3072))}


def plan(m, H, I, E, NG, M):
    p = dict(ttnn._ttnn.operations.bringup.flat_routed_expert_plan(m, H, I, E, NG, M))
    cv = lambda v: [cv(x) for x in v] if isinstance(v, (list, tuple)) else ([v.x, v.y] if hasattr(v, "x") else v)
    return {k: cv(v) for k, v in p.items()}


def check(lay, xdown):
    """One layout, in its own process: the op caches plans by shape, and reads MIMO_FL_* once per shape."""
    xcells = {tuple(int(v) for v in c.split(",")) for c in xdown.split(";") if c}
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    g = m.compute_with_storage_grid_size()
    if lay == "full":
        print(f"PLAN_CHECK grid {g.x} x {g.y}")
    bad = 0
    try:
        for name, (base, hi) in MODELS.items():
            model = hyb._scaled_model((8, 1), base)
            H, I = hi or (model.EMB_SIZE, model.MOE_INTERMEDIATE_SIZE)
            experts_per_chip, _, _, max_per_expert = compute_constants(
                hyb._SEQ_LEN_PER_CHIP, model.NUM_ROUTED_EXPERTS, model.NUM_EXPERTS_PER_TOKEN, 8, 8, hyb._CAPACITY_FACTOR
            )
            try:
                p = plan(m, H, I, experts_per_chip, model.NUM_ROUTED_EXPERTS, max_per_expert)
            except Exception as e:  # noqa: BLE001
                print(f"PLAN_CHECK {name:8s} {lay:5s} FAILED: {str(e).splitlines()[0][:200]}")
                bad += 1
                continue
            cores = [tuple(c) for k in ("gu", "down", "readers", "relays") for c in p[k]]
            row0 = sorted(c for c in cores if c[1] == 0)
            ok = lay == "full" or set(row0) <= xcells
            bad += not ok
            print(
                f"PLAN_CHECK {name:8s} {lay:5s} H {H} I {I} E {experts_per_chip} M {max_per_expert}: np {p['np']} "
                f"g {p['g']} gate/up {len(p['gu'])} down {p['nd']} (max {max(p['pcds'])} cols) readers "
                f"{len(p['readers'])} relays {len(p['relays'])} rdown {p['rdown']} row-0 cells {row0 or '-'}"
                + ("" if ok else "  <-- uses combine's row")
            )
    finally:
        ttnn.close_mesh_device(m)
    return bad


def main():
    if os.environ.get("_PLAN_CHECK_LAYOUT"):  # a child: one layout
        sys.exit(check(os.environ["_PLAN_CHECK_LAYOUT"], os.environ.get("MIMO_FL_XDOWN", "")))
    xdown = sys.argv[1] if len(sys.argv) > 1 else ""
    last = int(os.environ.get("FLAT_CMB_LAST_ROW", "9"))
    layouts = {"full": {}, "rows": {"MIMO_FL_ROWS": f"1,{last}"}}
    if xdown:
        layouts["xdown"] = {"MIMO_FL_ROWS": f"1,{last}", "MIMO_FL_XDOWN": xdown}
    bad = 0
    for lay, env in layouts.items():
        child = {k: v for k, v in os.environ.items() if k not in ("MIMO_FL_ROWS", "MIMO_FL_XDOWN")}
        bad += subprocess.run([sys.executable, __file__], env={**child, **env, "_PLAN_CHECK_LAYOUT": lay}).returncode != 0
    print(f"PLAN_CHECK {'OK' if not bad else f'problems in {bad} layout(s)'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
