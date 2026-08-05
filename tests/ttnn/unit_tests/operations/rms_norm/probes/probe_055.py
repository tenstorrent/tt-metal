import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd
from ttnn.operations.rms_norm.rms_norm import rms_norm
from eval.sharding import shard_config

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 8192, 1024)
    W = 1024
    mc = shard_config(
        [1024, 128],
        (8, 8),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
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
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    pd = rpd.create_program_descriptor(x, out, gamma=g, epsilon=1e-6, compute_kernel_config=cfg)
    ct = list(pd.kernels[2].compile_time_args)
    print(
        "BENCH-STYLE  BLOCK_ROWS",
        ct[3],
        "GROUP_SIZE",
        ct[13],
        "COMBINE",
        ct[12],
        "WT_CHUNK",
        ct[1],
        "NUM_W_CHUNKS",
        ct[2],
        "NATIVE_IN",
        ct[16],
    )
    print("avail", ttnn.get_max_worker_l1_unreserved_size(), "safety", rpd.L1_SAFETY_FRACTION)
    print("GATHER_FACES writer ct15", list(pd.kernels[1].compile_time_args)[15])
    # now via the op's own entry point
    import ttnn.operations.rms_norm.rms_norm as rn

    y = rms_norm(x, gamma=g, compute_kernel_config=cfg, memory_config=mc)
    print("op ran ok", tuple(y.shape))
finally:
    ttnn.close_device(device)
