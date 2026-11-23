# Reuse the golden suite's fabric-mesh fixtures (perf ablation harness only).
from eval.golden_tests.all_gather_rms_norm.conftest import *  # noqa: F401,F403
from eval.golden_tests.all_gather_rms_norm.conftest import _fabric_mesh_cache  # noqa: F401
