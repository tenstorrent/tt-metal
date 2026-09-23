"""Op-to-op gaps inside the traced BGE-M3 forward (nomask), from device timestamps.

Captures the trace as bench_nomask.py does, replays it under the device profiler,
then reads generated/profiler/.logs/profile_log_device.csv: each op's span is
[min FW start, max FW end] over its cores. Prints the kernel sum, the gap sum and
the largest gaps by the op pair around them.

Usage: TT_METAL_DEVICE_PROFILER=1 python trace_gaps.py <batch> [replays]
"""

import collections
import csv
import sys

import ttnn
from models.demos.wormhole.bge_m3.tests.perf import perf
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

batch = int(sys.argv[1])
replays = int(sys.argv[2]) if len(sys.argv) > 2 else 1

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000, num_command_queues=1)
args, model, _ = create_tt_model(mesh_device=mesh_device, max_batch_size=batch, max_seq_len=512, dtype=ttnn.bfloat8_b)
inputs = perf._n300_dp_inputs(args.pad_token_id, batch, 512, 512)
device_tensors = perf._n300_dp_batchshard(inputs, mesh_device, on_device=True)
ttnn.deallocate(model.forward(**device_tensors, no_padding=True))
ttnn.synchronize_device(mesh_device)
ttnn.ReadDeviceProfiler(mesh_device)  # drop the eager warm-up records
model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0, no_padding=True)
for _ in range(replays):
    model.execute_trace(blocking=True)
ttnn.ReadDeviceProfiler(mesh_device)
model.release_trace()
ttnn.close_mesh_device(mesh_device)

rows = list(csv.reader(open("generated/profiler/.logs/profile_log_device.csv")))[2:]
freq = 1350.0
span = collections.defaultdict(lambda: [None, None])
for r in rows:
    name, typ = r[10].strip(), r[11].strip()
    if not name.endswith("-FW"):
        continue
    key = (r[9].strip(), int(r[7]))  # (trace counter, op id)
    t = int(r[5])
    s = span[key]
    if typ == "ZONE_START":
        s[0] = t if s[0] is None else min(s[0], t)
    elif typ == "ZONE_END":
        s[1] = t if s[1] is None else max(s[1], t)
ops = sorted((s[0], s[1], k) for k, s in span.items() if s[0] is not None and s[1] is not None)
# Keep the last replay only.
last = ops[-1][2][0]
ops = [o for o in ops if o[2][0] == last]
busy = sum(e - s for s, e, _ in ops) / freq
gaps = [(ops[i + 1][0] - ops[i][1]) / freq for i in range(len(ops) - 1)]
wall = (ops[-1][1] - ops[0][0]) / freq
print(
    "TRACE_GAPS batch=%d ops=%d wall=%.1f us busy=%.1f us gaps=%.1f us mean_gap=%.2f us"
    % (batch, len(ops), wall, busy, sum(gaps), sum(gaps) / max(1, len(gaps)))
)
by_len = collections.defaultdict(list)
for i, g in enumerate(gaps):
    by_len[round((ops[i + 1][0] - ops[i + 1][0]) or 0)].append(g)
hist = collections.Counter(round(g) for g in gaps)
print("gap histogram (us: count):", sorted(hist.items())[:20])
