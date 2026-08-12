import ttnn
from ttnn.operations.rms_norm.perf_experiments.sumsq_reduce_merge.harness import check, measure

device = ttnn.open_device(device_id=0)
try:
    for name in ("focus_r1_c3", "tail_r1_c3_v17", "bshard_r16_c4"):
        for v in ("baseline", "merged_cvalid"):
            try:
                pcc, ratio, _, _ = check(device, name, v, fp32_dest_acc_en=True)
                ns1, _ = measure(device, name, v, iters=1, fp32_dest_acc_en=True)
                nsN, n = measure(device, name, v, fp32_dest_acc_en=True)
                print(
                    f"FP32DST {name:18s} {v:14s} pcc={pcc:.6f} ratio_median={ratio:.6f} one_ns={ns1} per_block_ns={nsN/n:.1f}"
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                print(f"FP32DST {name:18s} {v:14s} FAILED {type(exc).__name__}: {exc}")
finally:
    ttnn.close_device(device)
