# SPDX-License-Identifier: Apache-2.0
# Blackhole (P150a) batch-size sweep for YOLOv11n e2e perf (Trace + 2CQ).
# Run inside the seamless container; see top-level run command in chat/memory.
import sys
import time

import torch
from loguru import logger

import ttnn
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.runner.performant_runner import YOLOv11PerformantRunner

RESOLUTION = (640, 640)
ITERS = 100
TRACE_REGION = 23887872  # generous, same as DP test


def bench(device, bs):
    runner = YOLOv11PerformantRunner(
        device,
        bs,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        resolution=RESOLUTION,
        model_location_generator=None,
    )
    x = torch.randn((bs, 3, *RESOLUTION), dtype=torch.float32)
    # warmup + PCC check
    try:
        runner.run(torch_input_tensor=x, check_pcc=True)
        logger.info(f"[PCC] BS={bs} {runner.runner_infra.pcc_message}")
    except Exception as e:
        logger.warning(f"[PCC] BS={bs} check skipped: {repr(e)[:120]}")
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(ITERS):
        runner.run(torch_input_tensor=x)
    ttnn.synchronize_device(device)
    dt = (time.time() - t0) / ITERS
    fps = bs / dt
    logger.info(f"[RESULT] BS={bs}  iter={dt*1e3:.3f} ms  FPS={fps:.1f}")

    # Cheaper host-prep: feed a bf16 input so from_torch skips the fp32->bf16 convert.
    x16 = x.to(torch.bfloat16)
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(ITERS):
        runner.run(torch_input_tensor=x16)
    ttnn.synchronize_device(device)
    dtb = (time.time() - t0) / ITERS
    logger.info(f"[E2E-BF16] BS={bs}  iter={dtb*1e3:.3f} ms  FPS={bs/dtb:.1f}")

    # Device-only path: reuse the precomputed host shards (skip per-iter from_torch),
    # so this isolates on-device + fixed H2D from the host tilize/convert overhead.
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(ITERS):
        runner._execute_yolov11_trace_2cqs_inference()
    ttnn.synchronize_device(device)
    dt2 = (time.time() - t0) / ITERS
    logger.info(f"[DEVONLY] BS={bs}  iter={dt2*1e3:.3f} ms  FPS={bs/dt2:.1f}  (host-prep overhead={ (dt-dt2)*1e3:.3f} ms)")

    # Pipelined path: prepare the NEXT frame's host shards in a background thread
    # while the device runs the current (non-blocking) trace, so the from_torch
    # tilize/convert overlaps device compute instead of sitting on the critical path.
    from concurrent.futures import ThreadPoolExecutor

    def make_host():
        h, _ = runner.runner_infra._setup_l1_sharded_input(device, x)
        return h

    ex = ThreadPoolExecutor(max_workers=1)
    ttnn.synchronize_device(device)
    fut = ex.submit(make_host)
    t0 = time.time()
    for _ in range(ITERS):
        host = fut.result()
        fut = ex.submit(make_host)  # prefetch next while device runs this one
        runner._execute_yolov11_trace_2cqs_inference(host)
    ttnn.synchronize_device(device)
    dt3 = (time.time() - t0) / ITERS
    fut.result()
    ex.shutdown()
    logger.info(f"[PIPELINED] BS={bs}  iter={dt3*1e3:.3f} ms  FPS={bs/dt3:.1f}")
    runner.release()
    return fps


def main():
    batch_sizes = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "4"])]
    results = {}
    for bs in batch_sizes:
        device = ttnn.CreateDevice(
            0,
            l1_small_size=YOLOV11_L1_SMALL_SIZE,
            trace_region_size=TRACE_REGION,
            num_command_queues=2,
        )
        try:
            results[bs] = bench(device, bs)
        except Exception as e:
            logger.error(f"BS={bs} FAILED: {repr(e)[:300]}")
            results[bs] = None
        finally:
            ttnn.synchronize_device(device)
            ttnn.close_device(device)
    print("\n==== YOLOv11n P150a batch sweep ====")
    for bs, fps in results.items():
        print(f"  BS={bs:>2}: {'FAIL' if fps is None else f'{fps:7.1f} FPS'}")


if __name__ == "__main__":
    main()
