import sys, os

sys.path.insert(0, os.getcwd())
from tests.ttnn.unit_tests.operations.rms_norm.perf_zone_harness import main

main(
    [
        "decode7168",
        "decode2304",
        "decode5120",
        "decode1024",
        "prefill1024",
        "prefill7168",
        "wshard1024",
        "wshard7168",
        "bshard1024",
        "rm_interleaved",
        "hshard512",
        "wtail",
    ]
)
