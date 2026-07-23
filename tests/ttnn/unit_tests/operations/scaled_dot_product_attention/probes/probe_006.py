import torch, ttnn, re
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa

dev = ttnn.open_device(device_id=0)
try:
    for D in [256, 512, 1024]:
        shape = (1, 1, 128, D)
        try:
            to = lambda t: ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = sdpa(to(torch.randn(*shape)), to(torch.randn(*shape)), to(torch.randn(*shape)))
            print(f"D={D} bf16 OK shape={tuple(out.shape)}")
        except Exception as e:
            msg = str(e)
            m = re.search(r"grow to (\d+) B", msg)
            print(f"D={D} bf16 FAIL grow_to={m.group(1) if m else '?'} :: {msg[:160]}")
finally:
    ttnn.close_device(dev)
