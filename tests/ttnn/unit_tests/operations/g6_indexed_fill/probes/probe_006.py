import torch, ttnn
from ttnn.operations.g6_indexed_fill import g6_indexed_fill

device = ttnn.open_device(device_id=0, trace_region_size=1 << 20)


def dev(t):
    return ttnn.from_torch(
        t.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def try_trace(name, fn):
    fn()
    tid = ttnn.begin_trace_capture(device)
    try:
        fn()
        print(f"RESULT {name}: trace capture OK")
    except Exception as e:
        print(f"RESULT {name}: trace capture FAILS:", str(e).splitlines()[0][:110])
    finally:
        ttnn.end_trace_capture(device, tid)
        ttnn.release_trace(device, tid)


try:
    a = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ids = dev(torch.tensor([2]))
    x = dev(torch.zeros(7, 1, 1, 32, dtype=torch.int64))
    s = dev(torch.ones(1, 1, 1, 32, dtype=torch.int64))
    device.disable_and_clear_program_cache()
    try_trace("stock ttnn.add, program cache disabled", lambda: ttnn.add(a, a))
    try_trace("g6_indexed_fill, program cache disabled", lambda: g6_indexed_fill(ids, x, s))
    device.enable_program_cache()
    try_trace("stock ttnn.add, program cache enabled", lambda: ttnn.add(a, a))
    try_trace("g6_indexed_fill, program cache enabled", lambda: g6_indexed_fill(ids, x, s))
    # refusal-only call under graph capture: what the harness's L1 capture sees
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        g6_indexed_fill(ids, x, s, dim=1)
    except Exception as e:
        pass
    trace = ttnn.graph._cpp_end_graph_capture()
    print(
        "RESULT refusal-only call graph peak_total:", int(ttnn.graph.extract_resource_usage_per_core(trace).peak_total)
    )
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = g6_indexed_fill(ids, x, s)
    trace = ttnn.graph._cpp_end_graph_capture()
    print("RESULT graded call graph peak_total:", int(ttnn.graph.extract_resource_usage_per_core(trace).peak_total))
finally:
    ttnn.close_device(device)
