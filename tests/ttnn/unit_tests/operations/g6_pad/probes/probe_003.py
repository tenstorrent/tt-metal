import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    tiny = torch.finfo(torch.bfloat16).tiny
    specials = torch.tensor([-0.0, float("inf"), float("-inf"), tiny, tiny / 4, -tiny / 8]).to(torch.bfloat16)
    for shape in ((1, 1, 8, 62080), (1, 8, 62080), (1, 8, 64), (1, 1, 196, 196), (1, 196, 196), (1, 100, 1536)):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        flat = x.reshape(-1)
        flat[:6] = specials
        x = flat.reshape(shape)
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        t2 = ttnn.fill_implicit_tile_padding(t, 1000.0)
        g = ttnn.graph.end_graph_capture()
        progs = [
            n["params"].get("name", "?")
            for n in g
            if n.get("node_type") == "function_start" and "program_factory_type" in n.get("params", {})
        ]
        back = ttnn.to_torch(t2)
        d = (back.view(torch.int16) != x.view(torch.int16)).nonzero()
        print("PROBE", shape, "programs", progs, "mismatches", d.tolist()[:6])
finally:
    ttnn.close_device(device)
