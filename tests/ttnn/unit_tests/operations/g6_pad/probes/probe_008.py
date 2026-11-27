import torch, ttnn

device = ttnn.open_device(device_id=0)
for shape, out in [
    ((1, 3844, 1536), (1, 3872, 1536)),
    ((1, 1, 2, 62080), (1, 1, 32, 62080)),
    ((16, 1, 1536), (16, 32, 1536)),
    ((1, 3844, 12), (1, 3872, 12)),
]:
    x = ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    y = ttnn.reshape(x, ttnn.Shape(out), x.padded_shape)
    g = ttnn.graph.end_graph_capture()
    progs = [
        n["params"].get("name", "?")
        for n in g
        if n.get("node_type") == "function_start" and "program_factory_type" in n.get("params", {})
    ]
    names = [n["params"].get("name") for n in g if n.get("node_type") == "function_start"]
    print(shape, progs, y.buffer_address() == x.buffer_address(), list(y.shape), list(y.padded_shape), names[:6])
ttnn.close_device(device)
