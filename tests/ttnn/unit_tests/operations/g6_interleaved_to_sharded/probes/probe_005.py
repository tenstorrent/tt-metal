import sys, runpy

sys.argv = [
    "verify_supported",
    "/localdev/pmilojevic/campaign/bh-lb-10/slots/chip4/ttnn/ttnn/operations/g6_interleaved_to_sharded/verify_results",
    "ttnn.operations.g6_interleaved_to_sharded",
    "--output",
    "/localdev/pmilojevic/campaign/bh-lb-10/slots/chip4/ttnn/ttnn/operations/g6_interleaved_to_sharded/verify_results/verifier_report.json",
]
try:
    runpy.run_module("eval.verify_supported", run_name="__main__")
except SystemExit as e:
    print("exit", e.code)
