import torch, ttnn, importlib

op = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for kind in ("randn", "bits"):
        if kind == "randn":
            x = torch.randn(1, 128, 256, 32, dtype=torch.bfloat16)
        else:
            x = torch.randint(-32768, 32768, (1, 128, 256, 32), dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        xt = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        back = ttnn.to_torch(xt)
        print(kind, "roundtrip mism", (back.view(torch.int16) != x.view(torch.int16)).sum().item())
        y = op.g6_permute(xt, (0, 3, 1, 2))
        got = ttnn.to_torch(y)
        ref = x.permute(0, 3, 1, 2).contiguous()
        print(
            kind,
            got.shape,
            ref.shape,
            "mism",
            (got.view(torch.int16) != ref.view(torch.int16)).sum().item(),
            torch.equal(got, ref),
        )
finally:
    ttnn.close_device(device)
