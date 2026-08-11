import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
try:
    for shape, ml in [((1, 1, 256, 512), ML.WIDTH_SHARDED), ((1, 1, 256, 512), ML.BLOCK_SHARDED)]:
        W = shape[-1]
        x = torch.ones(shape, dtype=torch.bfloat16)
        g = (torch.arange(W).float() % 64).to(torch.bfloat16)  # repeating ramp, exact in bf16
        mc = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, memory_config=tx.memory_config())).float()
        print(f"== {shape} {str(ml).split('.')[-1]} shard={list(mc.shard_spec.shape)}")
        print("  exp row0[:24] ", [round(v, 2) for v in g.float()[:24].tolist()])
        print("  got row0[:24] ", [round(v, 2) for v in out[0, 0, 0, :24].tolist()])
        print("  got row0[40:64]", [round(v, 2) for v in out[0, 0, 0, 40:64].tolist()])
        print("  exp row0[40:64]", [round(v, 2) for v in g.float()[40:64].tolist()])
        # no-gamma variant: every element should be ~1.0; deviation = sqrt(W/N_used)
        out2 = ttnn.to_torch(rms_norm(tx, memory_config=tx.memory_config())).float()
        print("  no-gamma row0[:16]", [round(v, 4) for v in out2[0, 0, 0, :16].tolist()])
        print("  no-gamma row0 min/max", round(float(out2[0, 0, 0].min()), 4), round(float(out2[0, 0, 0].max()), 4))
finally:
    ttnn.close_device(device)
