import os, torch, ttnn
from eval.golden_tests.nomic_sdpa.helpers import make_qkv
from ttnn.operations.nomic_sdpa import nomic_sdpa

tag = os.environ["BX_TAG"]
device = ttnn.open_device(device_id=0)
try:
    outs = {}
    for shape in [(8, 512), (32, 128), (16, 256), (2, 512)]:
        for prof in ["real", "peaky"]:
            q, k, v = make_qkv((shape[0], 12, shape[1], 64), prof, seed=1)
            t = [
                ttnn.from_torch(
                    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                for x in (q, k, v)
            ]
            outs[f"{shape}-{prof}"] = ttnn.to_torch(nomic_sdpa(*t))
    torch.save(outs, f"/tmp/bx_{tag}.pt")
    print("saved", tag, len(outs))
finally:
    ttnn.close_device(device)
