import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for shape, out_shape in (
        ([1, 100, 1536], [1, 128, 1536]),
        ([1, 1, 8, 62080], [1, 1, 16, 62080]),
        ([16, 1, 1536], [16, 32, 1536]),
    ):
        x = ttnn.from_torch(
            torch.randn(shape).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            out = ttnn.reshape(x, ttnn.Shape(out_shape), x.padded_shape)
        finally:
            graph = ttnn.graph.end_graph_capture()
        progs = [
            n["params"].get("name", "?")
            for n in graph
            if n.get("node_type") == "function_start" and "program_factory_type" in n.get("params", {})
        ]
        print(
            "PROBE",
            shape,
            "->",
            list(out.shape),
            "padded",
            list(out.padded_shape),
            "x padded",
            list(x.padded_shape),
            "programs",
            progs,
            "same_addr",
            out.buffer_address() == x.buffer_address(),
            "x shape after",
            list(x.shape),
        )
finally:
    ttnn.close_device(device)
