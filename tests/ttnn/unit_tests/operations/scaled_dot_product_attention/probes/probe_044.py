import os, sys

repo = os.getcwd()
sys.path.insert(0, repo)
import pytest

rc = pytest.main(
    [
        "-s",
        "-p",
        "no:cacheprovider",
        "--no-header",
        "-q",
        "-k",
        "h8_s4096_d64 or h8_s8192_d128",
        "tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_sdpa_perf_juice.py",
    ]
)
print("JUICE_RC", rc)
