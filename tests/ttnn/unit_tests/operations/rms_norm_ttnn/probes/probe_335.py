import os
os.environ["RMS_TRACE_BLOCKING"] = "1"
os.environ["RMS_PC_TRACE"] = "1"
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import sys
sys.path.insert(0, "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/passb_op_count")
import ttnn, bench
cases = bench.perf_cases()
dev = ttnn.open_device(device_id=0)
try:
    for i in (0, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 18):
        print(f"=== case {bench.label(i, cases[i])}", flush=True)
        run, exp, live = bench.build(dev, cases[i])
        o = run()
        ttnn.deallocate(o)
        for t in live:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
finally:
    ttnn.close_device(dev)
