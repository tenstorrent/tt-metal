import sys, torch, ttnn

dev = ttnn.open_device(device_id=0, trace_region_size=10000000)
x = ttnn.allocate_tensor_on_device(
    ttnn.Shape((1, 1, 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
)


def p(s):
    print("STEP", s, flush=True)


ttnn.graph.up_front_begin_collect(clear=False, real_alloc=True)
try:
    p("eager")
    y = ttnn.relu(x)
    p("eager done")
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    p("begin")
    y2 = ttnn.relu(x)
    p("op in trace")
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    p("end")
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    p("executed")
    ttnn.release_trace(dev, tid)
    p("released")
finally:
    ttnn.graph.up_front_end_collect()
p("collect ended")
ttnn.synchronize_device(dev)
p("synced")
ttnn.close_device(dev)
p("closed")
