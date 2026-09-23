"""Sustained (power-throttled) time of the traced BGE-M3 forward, nomask.

Same setup as bench_nomask.py. Replays the trace for SOAK seconds so AICLK
settles under the chip TDC/TDP limits, then times N replays back to back.
Prints both the burst mean (first 20 replays) and the sustained mean.

Usage: python bench_sustained.py <batch> [n_timed] [soak_s]
"""

import statistics
import sys
import time

import ttnn
from models.demos.wormhole.bge_m3.tests.perf import perf
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

batch = int(sys.argv[1])
n_timed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
soak_s = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
seq_len = 512

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000, num_command_queues=1)
args, model, _ = create_tt_model(
    mesh_device=mesh_device, max_batch_size=batch, max_seq_len=seq_len, dtype=ttnn.bfloat8_b
)
inputs = perf._n300_dp_inputs(args.pad_token_id, batch, seq_len, seq_len)
device_tensors = perf._n300_dp_batchshard(inputs, mesh_device, on_device=True)

out = model.forward(**device_tensors, no_padding=True)
ttnn.synchronize_device(mesh_device)
ttnn.deallocate(out)
model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0, no_padding=True)
# Idle so the clock recovers, then take the burst sample.
time.sleep(3.0)
burst = []
for _ in range(20):
    t0 = time.perf_counter()
    model.execute_trace(blocking=True)
    burst.append((time.perf_counter() - t0) * 1000.0)
end = time.perf_counter() + soak_s
while time.perf_counter() < end:
    model.execute_trace(blocking=True)
times = []
for _ in range(n_timed):
    t0 = time.perf_counter()
    model.execute_trace(blocking=True)
    times.append((time.perf_counter() - t0) * 1000.0)
model.release_trace()
ttnn.close_mesh_device(mesh_device)

mean = statistics.mean(times)
print(
    "SUSTAINED batch=%d n=%d mean=%.3f std=%.3f burst_mean=%.3f tok_s=%.0f"
    % (batch, n_timed, mean, statistics.stdev(times), statistics.mean(burst), batch * seq_len / (mean / 1000.0))
)
