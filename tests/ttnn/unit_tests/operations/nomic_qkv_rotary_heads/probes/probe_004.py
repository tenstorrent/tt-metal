import torch, ttnn
from eval.golden_tests.nomic_qkv_rotary_heads.helpers import model_layout, ln_layout
from ttnn.operations.nomic_qkv_rotary_heads.nomic_qkv_rotary_heads import model_layout_memory_config

device = ttnn.open_device(device_id=0)
for shape in [(2, 1, 512, 768), (3, 1, 400, 768), (8, 1, 512, 768), (1, 1, 1024, 768)]:
    mc = model_layout(shape, device)
    mine = model_layout_memory_config(shape, device)
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    tmc = x.memory_config()
    print(shape, mc == mine, tmc == mc, tmc.memory_layout, tmc.shard_spec, tmc.nd_shard_spec)
    lmc = ln_layout(shape, device)
    print("  ln:", lmc.memory_layout, lmc.shard_spec, lmc.nd_shard_spec is not None)
ttnn.close_device(device)
