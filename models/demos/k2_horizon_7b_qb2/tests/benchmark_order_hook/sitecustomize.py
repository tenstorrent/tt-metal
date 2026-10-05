"""Activate only for an explicitly opted-in benchmark_stage evaluate child."""

import os
import sys
from pathlib import Path

plan = os.environ.get("K2_BENCHMARK_REQUEST_ORDER")
arguments = getattr(sys, "orig_argv", [])
evaluation_child = any(
    arguments[index : index + 3] == ["-m", "benchmark_stage", "evaluate"] for index in range(len(arguments) - 2)
)
if plan and evaluation_child:
    try:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        from models.demos.k2_horizon_7b_qb2.tests.benchmark_request_order import install

        install(
            plan,
            output / "request-order-observed.json",
            activation={"orig_argv": arguments, "opt_in_plan": str(Path(plan).resolve()), "evaluate_child": True},
        )
    except BaseException as error:
        # Python normally ignores sitecustomize errors; opt-in scheduling must
        # fail closed instead of silently launching the original request order.
        sys.stderr.write(f"K2 request-order hook failed: {error!r}\n")
        raise SystemExit(1) from error
