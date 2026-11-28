import sys, json
from eval import verify_supported

res = "ttnn/ttnn/operations/g6_zeros/verifier_results"
sys.argv = ["verify_supported", res, "ttnn.operations.g6_zeros", "--output", f"{res}/verifier_report.json"]
try:
    verify_supported.main()
except SystemExit as e:
    print("EXIT", e.code)
print("SUMMARY", json.dumps(json.load(open(f"{res}/verifier_report.json"))["summary"]))
