import os
from eval.golden_tests.all_gather_rms_norm.conftest import _fabric_mesh_cache
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.topology import system_mesh_shape
from eval import ccl_perf
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm
import ttnn

c = _loose_cases()[1]
inputs, axes, extras = c.values
cache = _fabric_mesh_cache.__wrapped__() if hasattr(_fabric_mesh_cache, "__wrapped__") else None
