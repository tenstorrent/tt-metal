# Runs a script or pytest with main's prefill-precision changes (#55948 / #56466) switched off, everything else as
# checked out: the main RMSNorm drops its prefill fp32 compute config and TILE weight copy, and the prefill per-head
# Q/K/V norms run without fp32_accumulate. This is the code's prefill behaviour from before those PRs.
# Usage: prefill_fp32_off.py <script.py> [args]      or      prefill_fp32_off.py pytest [pytest args]
import runpy
import sys

from models.demos.gemma4.tt import rms_norm
from models.demos.gemma4.tt.attention import prefill

_init = rms_norm.RMSNorm.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    self.prefill_compute_kernel_config = None  # op default, as before #55948
    self.tt_weight_tile = None  # prefill uses the row-major weight, as before #55948


rms_norm.RMSNorm.__init__ = init

_per_head = prefill.apply_per_head_norm


def per_head(*a, **k):
    k["fp32_accumulate"] = False  # as before #55948 / #56466
    return _per_head(*a, **k)


prefill.apply_per_head_norm = per_head
print("PREFILL_FP32_OFF patched RMSNorm and prefill.apply_per_head_norm", flush=True)

if sys.argv[1] == "pytest":
    import pytest

    sys.exit(pytest.main(sys.argv[2:]))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
