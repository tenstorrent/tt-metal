# The four LoFi 2x1 cases of test_lf21_prof.py (one file, so the device profiler run needs no -k expression).
import pytest
from test_lf21_prof import CASES, device, test_lf21_mm as _run  # noqa: F401

SEL = [c for c in CASES if c[0] in ("lf_512_4k_256_lofi", "lf_512_4k_256_fp32", "lf_512_4k_256_b8in0_default", "lf_512_4k_256_b8in0_fp32")]


@pytest.mark.parametrize("case", SEL, ids=[c[0] for c in SEL])
def test_lf21s_mm(device, case):
    _run(device, case)
