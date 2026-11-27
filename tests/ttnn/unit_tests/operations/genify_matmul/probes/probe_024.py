import math, torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul

dev = ttnn.open_device(device_id=0)
try:
    for m, k, n, cores, ep in [
        (1, 4352, 5120, 8, "none"),
        (32, 5120, 9216, 10, "swiglu"),
        (1, 5120, 16384, 10, "none"),
    ]:
        torch.manual_seed(0)
        A = torch.randn(1, 1, m, k)
        B = torch.randn(k, n) / math.sqrt(k)
        grid = ttnn.num_cores_to_corerangeset(cores, dev.compute_with_storage_grid_size(), True)
        a_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, k // cores], ttnn.ShardOrientation.ROW_MAJOR),
        )
        banks = dev.dram_grid_size().x
        g = 64 if ep == "swiglu" else 32
        width = -(-n // (g * banks)) * g
        b_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                [k, width],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        ta = create_ttnn_input_tensor(A, dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
        tb = create_ttnn_input_tensor(B, dev, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, memory_config=b_mc)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        )
        kw = {"activation": "swiglu"} if ep == "swiglu" else {}
        out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, **kw)
        y = A.float() @ ttnn.to_torch(tb).float()
        if ep == "swiglu":
            t = y.reshape(1, 1, m, n // 64, 2, 32)
            y = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(1, 1, m, n // 2)
        got = ttnn.to_torch(out).float().reshape(y.shape)
        pcc = torch.corrcoef(torch.stack([got.flatten(), y.flatten()]))[0, 1].item()
        rms = ((got - y).pow(2).mean().sqrt() / y.pow(2).mean().sqrt()).item()
        print(f"RES {m}x{k}x{n} {ep}: pcc={pcc:.6f} rel_rms={rms:.4f}")
finally:
    ttnn.close_device(dev)
