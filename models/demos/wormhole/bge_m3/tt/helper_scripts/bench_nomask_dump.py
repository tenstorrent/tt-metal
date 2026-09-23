"""Mean +- std of the traced BGE-M3 forward, nomask, one batch per process.

Follows tests/perf/perf.py::test_perf exactly: the same inputs, the same
warmup, the same trace capture with no_padding=True, 3 warm replays, then
timed execute_trace(blocking=True) calls. perf.py prints only best and mean;
this prints the standard deviation as well, to match the reference table.

Usage: python bench_nomask.py <batch> [iterations]
"""

import statistics
import sys
import time

import ttnn
from models.demos.wormhole.bge_m3.tests.perf import perf
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

batch = int(sys.argv[1])
iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 30
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
for _ in range(3):
    model.execute_trace(blocking=True)

times = []
for _ in range(iterations):
    start = time.perf_counter()
    model.execute_trace(blocking=True)
    times.append((time.perf_counter() - start) * 1000.0)
model.release_trace()
ttnn.close_mesh_device(mesh_device)

print("TIMES " + " ".join("%.1f" % t for t in times))
mean = statistics.mean(times)
std = statistics.stdev(times)
print(
    "BENCH batch=%d n=%d mean=%.3f std=%.3f best=%.3f tok_s_mean=%.0f"
    % (batch, iterations, mean, std, min(times), batch * seq_len / (mean / 1000.0))
)
