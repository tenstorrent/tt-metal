"""t232 host-only planner tests (no mesh is opened).

test_choose_sharded_brick_regression asks for a (1, 1) mesh_device fixture it never uses; opening a bare
1x1 mesh on the galaxy is not allowed, so its cases run here with mesh_device=None. The host-only tests
(GNA stride + S5_2D guard, key-phase pin, H-shard divide) run under pytest in run.sh.
"""

import sys

from models.tt_dit.tests.unit import test_neighborhood_sdpa as t

fn = t.test_choose_sharded_brick_regression
cases = next(m.args[1] for m in fn.pytestmark if m.name == "parametrize" and m.args[0].startswith("volume"))
failed = 0
for case in cases:
    try:
        fn(None, *case)
        print("PASS", case, flush=True)
    except Exception as e:  # report every case, not just the first failure
        failed += 1
        print("FAIL", case, repr(e), flush=True)
print(f"REGRESSION {len(cases) - failed}/{len(cases)} passed", flush=True)
sys.exit(1 if failed else 0)
