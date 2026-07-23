import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa

for D in [256, 512, 1024]:
    for dt in [ttnn.bfloat16]:
        shape = (1, 1, 128, D)
        try:
            q = ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            k = ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            v = ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            out = sdpa(q, k, v)
            print(f"D={D} dt={dt} OK shape={tuple(out.shape)}")
        except Exception as e:
            msg = str(e)
            # extract "grow to N"
            import re

            m = re.search(r"grow to (\d+) B", msg)
            print(f"D={D} dt={dt} FAIL grow_to={m.group(1) if m else '?'} :: {msg[:200]}")
