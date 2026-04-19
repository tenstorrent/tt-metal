#!/usr/bin/env python3
"""Quick 640x640 single-device YOLOv8L perf benchmark."""
import time

import torch

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

N_ITERS = 20

l1_small = yolov8l_l1_small_size_for_res(640, 640)
trace_region = 6434816

print(f"Opening single device (l1_small={l1_small}, trace_region={trace_region})...")
device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    l1_small_size=l1_small,
    trace_region_size=trace_region,
    num_command_queues=2,
)
device.enable_program_cache()

inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

print("Building YOLOv8lPerformantRunner (640x640, batch=1)...")
runner = YOLOv8lPerformantRunner(
    device,
    device_batch_size=1,
    inp_h=640,
    inp_w=640,
    mesh_mapper=inputs_mesh_mapper,
    mesh_composer=output_mesh_composer,
    weights_mesh_mapper=weights_mesh_mapper,
)
print("Runner ready.")

torch_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)

# Warmup (2 runs: JIT compile + trace capture)
print("Warmup...")
for i in range(2):
    _ = runner.run(torch_input)
    ttnn.synchronize_device(device)
    t = runner.last_timing
    print(f"  warmup {i}: host_prep={t['host_prep_ms']:.1f}ms  h2d+trace={t['h2d_and_trace_ms']:.1f}ms")

# Benchmark
print(f"Benchmarking {N_ITERS} iterations...")
t0 = time.perf_counter()
for i in range(N_ITERS):
    _ = runner.run(torch_input)
ttnn.synchronize_device(device)
t1 = time.perf_counter()

avg_s = (t1 - t0) / N_ITERS
fps = 1.0 / avg_s
last_t = runner.last_timing

print(f"\n{'='*60}")
print(f"YOLOv8L 640x640  batch=1  single device")
print(f"  {N_ITERS} iterations: total={t1-t0:.3f}s")
print(f"  Avg per iteration: {avg_s*1000:.1f}ms")
print(f"  FPS: {fps:.1f}")
print(f"  Last timing: host_prep={last_t['host_prep_ms']:.1f}ms  h2d+trace={last_t['h2d_and_trace_ms']:.1f}ms")
print(f"{'='*60}")

runner.release()
ttnn.synchronize_device(device)
ttnn.close_mesh_device(device)
print("Done.")
