import sys, runpy

sys.argv = [
    "verify_supported",
    "/tmp/g6_verify/golden",
    "ttnn.operations.g6_interleaved_to_sharded",
    "--output",
    "/tmp/g6_verify/golden/verifier_report.json",
]
try:
    runpy.run_module("eval.verify_supported", run_name="__main__")
except SystemExit as e:
    print("exit", e.code)
