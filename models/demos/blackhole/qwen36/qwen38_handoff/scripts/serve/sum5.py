import json
import sys

C = sys.argv[1]
for r in sys.argv[2].split(","):
    d = json.load(open(f"{C}/{r}.json"))
    print(
        r,
        "compl",
        d["completed"],
        "fail",
        d.get("failed"),
        "TTFT mean/med/p99 %.1f/%.1f/%.1f" % (d["mean_ttft_ms"], d["median_ttft_ms"], d["p99_ttft_ms"]),
        "TPOT mean/med %.2f/%.2f p99 %.2f" % (d["mean_tpot_ms"], d["median_tpot_ms"], d["p99_tpot_ms"]),
        "ITL mean %.2f" % d["mean_itl_ms"],
        "outtok/s %.2f" % d["output_throughput"],
        "req/s %.3f" % d["request_throughput"],
        "user t/s %.2f" % (1000 / d["mean_tpot_ms"]),
    )
