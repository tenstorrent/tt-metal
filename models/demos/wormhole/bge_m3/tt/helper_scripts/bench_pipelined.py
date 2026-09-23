"""Pipelined throughput of the traced BGE-M3 forward, nomask.

Same setup and soak as bench_sustained.py, but enqueues N replays without a host
sync in between and syncs once, so the host round trip per forward is hidden (as a
server that enqueues the next batch while the device runs). Prints the per-forward
time for both modes back to back, so the round-trip cost is their difference.

Usage: python bench_pipelined.py <batch> [n] [soak_s]
"""

import sys
import time

import ttnn
from models.demos.wormhole.bge_m3.tests.perf import perf
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

batch = int(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
soak_s = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
seq_len = 512

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000, num_command_queues=1)
args, model, _ = create_tt_model(
    mesh_device=mesh_device, max_batch_size=batch, max_seq_len=seq_len, dtype=ttnn.bfloat8_b
)
inputs = perf._n300_dp_inputs(args.pad_token_id, batch, seq_len, seq_len)
device_tensors = perf._n300_dp_batchshard(inputs, mesh_device, on_device=True)
ttnn.deallocate(model.forward(**device_tensors, no_padding=True))
ttnn.synchronize_device(mesh_device)
model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0, no_padding=True)

end = time.perf_counter() + soak_s
while time.perf_counter() < end:
    model.execute_trace(blocking=True)

t0 = time.perf_counter()
for _ in range(n):
    model.execute_trace(blocking=True)
blocking_ms = (time.perf_counter() - t0) * 1000.0 / n

t0 = time.perf_counter()
for _ in range(n):
    model.execute_trace(blocking=False, synchronize=False)
ttnn.synchronize_device(mesh_device)
pipelined_ms = (time.perf_counter() - t0) * 1000.0 / n

model.release_trace()
ttnn.close_mesh_device(mesh_device)
print(
    "PIPELINED batch=%d n=%d blocking=%.3f pipelined=%.3f round_trip=%.3f ms tok_s_pipelined=%.0f"
    % (batch, n, blocking_ms, pipelined_ms, blocking_ms - pipelined_ms, batch * seq_len / (pipelined_ms / 1000.0))
)
