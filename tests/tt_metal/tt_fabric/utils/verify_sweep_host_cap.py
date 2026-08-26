#!/usr/bin/env python3
"""Enumerate topology-mapping solutions via ``generate_rank_bindings --all-solutions`` and assert that EVERY
solution stays within the minimal host-group cap.

This guards the multi-solution host-cap enforcement in topology_mapper_utils
(map_multi_mesh_to_physical_n / MultiMeshSolutionEnumerator): the enumeration solver does not honor
set_max_same_rank_groups_used itself, so the utils layer must drop / skip over-cap placements. Without that
enforcement most enumerated solutions occupy more host-ranks than k_min.

Exit 0 if all enumerated solutions are within the cap, 1 otherwise. Intended for the fabric CPU-only suite.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-graph-descriptor", required=True)
    ap.add_argument("--mock-cluster-rank-binding", required=True)
    ap.add_argument("--max-solutions", type=int, default=4)
    ap.add_argument("--max-ranks", type=int, default=16, help="Per-solution host-rank cap to assert (k_min).")
    ap.add_argument("--timeout", type=int, default=400)
    args = ap.parse_args()

    tt_home = os.environ.get("TT_METAL_HOME", os.getcwd())
    sys.path.insert(0, str(Path(tt_home) / "ttnn"))
    from ttnn.distributed.ttrun import (
        build_generate_rank_bindings_mpi_cmd,
        find_generate_rank_bindings_executable,
        load_mock_rank_to_descriptors,
    )
    sys.path.insert(0, str(Path(tt_home)))
    from tools.scaleout.sweep_rank_binding_solutions import _inject_solution_flags

    outdir = Path(tempfile.mkdtemp(prefix="sweep_cap_"))
    exe = find_generate_rank_bindings_executable()
    mock_rank_to_desc = load_mock_rank_to_descriptors(Path(args.mock_cluster_rank_binding).resolve())
    cmd = build_generate_rank_bindings_mpi_cmd(
        executable=exe,
        mgd_path=Path(args.mesh_graph_descriptor),
        hosts=None,
        output_dir=outdir,
        mock_rank_to_desc=mock_rank_to_desc,
        mpi_args=["--allow-run-as-root", "--oversubscribe"],
    )
    cmd = _inject_solution_flags(cmd, ["--all-solutions", "--max-solutions", str(args.max_solutions)])

    env = dict(os.environ, TT_METAL_SLOW_DISPATCH_MODE="1")
    subprocess.run(cmd, cwd=tt_home, env=env, timeout=args.timeout, check=False)

    idxs = glob.glob(str(outdir / "**" / "solutions_index.yaml"), recursive=True)
    if not idxs:
        print("[sweep-cap] FAIL: no solutions_index.yaml produced", flush=True)
        return 1
    sols = yaml.safe_load(open(idxs[0])).get("solutions", []) or []
    if not sols:
        print("[sweep-cap] FAIL: enumeration produced 0 solutions", flush=True)
        return 1

    over = [s for s in sols if int(s.get("num_ranks", 1 << 30)) > args.max_ranks]
    for s in sols:
        print(f"[sweep-cap]   {str(s.get('id',''))[:10]}  num_hosts={s.get('num_hosts')}  num_ranks={s.get('num_ranks')}")
    if over:
        print(f"[sweep-cap] FAIL: {len(over)}/{len(sols)} enumerated solution(s) exceed the {args.max_ranks}-rank cap")
        return 1
    print(f"[sweep-cap] PASS: all {len(sols)} enumerated solution(s) within the {args.max_ranks}-rank host cap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
