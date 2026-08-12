import ttnn
from ttnn.operations.rms_norm.perf_experiments.sumsq_reduce_merge.harness import check

device = ttnn.open_device(device_id=0)
try:
    for v in ("baseline", "merged"):
        try:
            pcc, ratio, got, ref = check(device, "focus_r1_c3", v)
            print(f"SMOKE {v}: pcc={pcc:.6f} ratio_median={ratio:.6f}")
            print("  got[:4] =", [round(x, 5) for x in got[:4].tolist()])
            print("  ref[:4] =", [round(x, 5) for x in ref[:4].tolist()])
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"SMOKE {v}: FAILED {type(exc).__name__}: {exc}")
finally:
    ttnn.close_device(device)
