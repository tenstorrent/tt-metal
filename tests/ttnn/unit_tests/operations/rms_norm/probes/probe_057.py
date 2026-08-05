import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd
from eval.sharding import shard_config

_ML = ttnn.TensorMemoryLayout
CASES = [
    ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "prefill_w1024"),
    ((1, 1, 8192, 2304), None, _ML.INTERLEAVED, "prefill_w2304"),
    ((1, 1, 8192, 5120), None, _ML.INTERLEAVED, "prefill_w5120"),
    ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "prefill_w7168"),
    ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "wshard_w1024_8c"),
    ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "wshard_w2304_9c"),
    ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "wshard_w5120_32c"),
    ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "wshard_w7168_28c"),
    ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "bshard_64c"),
]
device = ttnn.open_device(device_id=0)
try:
    print("avail", ttnn.get_max_worker_l1_unreserved_size())
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )
    for shape, shard, ml, name in CASES:
        W = shape[-1]
        mc = None
        if shard is not None:
            mc = shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        torch.manual_seed(42)
        x = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        g = ttnn.from_torch(
            torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        pd = rpd.create_program_descriptor(x, out, gamma=g, epsilon=1e-6, compute_kernel_config=cfg)
        ct = list(pd.kernels[2].compile_time_args)
        rct = list(pd.kernels[0].compile_time_args)
        rows_per_core = None
        print(
            f"BR {name:20s} BLOCK_ROWS={ct[3]:3d} GROUP_SIZE={ct[13]:3d} COMBINE={ct[12]} "
            f"WT_CHUNK={ct[1]:4d} NUM_W_CHUNKS={ct[2]:3d} NATIVE_IN={ct[16]} BANK_PAGES={rct[20]:3d} "
            f"ncbs={len(pd.cbs)}"
        )
        ttnn.deallocate(x)
        ttnn.deallocate(g)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
