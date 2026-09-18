# Generated vs reference GroupNorm on the model's own paths vs torch fp32, for input distributions with growing
# per-channel mean offsets (cancellation stress) — to locate the precision loss seen in the 50-step UNet loop.
import torch, ttnn
import torch.nn.functional as F
from models.demos.stable_diffusion_xl_base.tt.model_configs.model_configs_1024x1024BH import (
    ModelOptimisations1024x1024BH,
)
from models.demos.stable_diffusion_xl_base.tt.sdxl_utility import run_group_norm

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    cfg_gen = ModelOptimisations1024x1024BH(use_generated_groupnorm=True)
    cfg_ref = ModelOptimisations1024x1024BH(use_generated_groupnorm=False)
    G, eps = 32, 1e-5

    def pcc(a, b):
        a = a.flatten().double()
        b = b.flatten().double()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

    def run(path, HW, C, mu_scale, tag):
        torch.manual_seed(0)
        mu = (torch.rand(C) * 2 - 1) * mu_scale
        sig = 0.5 + 1.5 * torch.rand(C)
        x = (torch.randn(1, 1, HW, C) * sig + mu).bfloat16()
        w = 1 + 0.3 * torch.randn(C)
        b = 0.3 * torch.randn(C)
        # torch fp32 golden on the same bf16-rounded input
        xt = x.float().reshape(1, HW, C).permute(0, 2, 1).reshape(1, C, HW, 1)
        gold = F.group_norm(xt, G, w, b, eps).reshape(1, C, HW).permute(0, 2, 1).reshape(1, 1, HW, C)
        out = {}
        for name, cfg in (("gen", cfg_gen), ("ref", cfg_ref)):
            gn_cfg, mem_cfg, mask, negmask, gamma, beta = cfg.get_groupnorm_params(path, w, b, G, dev)
            t = ttnn.from_torch(
                x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            if name == "ref":
                if C in (320, 960):
                    t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
                t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                shard = ttnn.create_sharded_memory_config(
                    shape=t.shape,
                    core_grid=gn_cfg["core_grid"],
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                )
                t = ttnn.to_memory_config(t, shard)
            y = run_group_norm(t, gn_cfg, mem_cfg, mask, negmask, gamma, beta, G, eps, activation="silu")
            y = ttnn.to_torch(y).float().reshape(1, 1, HW, C)
            out[name] = y
        err = {k: (v - gold) for k, v in out.items()}
        print(
            f"PROBE {tag:26s} HW={HW:5d} C={C:4d} mu={mu_scale:4.1f} | gen pcc={pcc(out['gen'],gold):.6f} rms={err['gen'].pow(2).mean().sqrt():.4f} max={err['gen'].abs().max():.3f} | ref pcc={pcc(out['ref'],gold):.6f} rms={err['ref'].pow(2).mean().sqrt():.4f} max={err['ref'].abs().max():.3f}"
        )

    for path, HW, C in (
        ("down_blocks.1.resnets.1.norm1", 4096, 640),
        ("mid_block.resnets.0.norm1", 1024, 1280),
        ("down_blocks.0.resnets.0.norm1", 16384, 320),
    ):
        for mu_scale in (0.0, 3.0, 10.0, 30.0):
            run(path, HW, C, mu_scale, path.split(".norm")[0])
finally:
    ttnn.close_device(dev)
