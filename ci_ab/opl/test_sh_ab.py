# Scratch, not for merge: sharded softmax numeric_stable (softmax_sharded.cpp: ROW MAX over block_w resident tiles), 8x4 cores.
import pytest
import torch
import ttnn


@pytest.mark.parametrize("dt", ["bf16", "bfp8"])
@pytest.mark.parametrize("fp32", [0, 1])
@pytest.mark.parametrize("mask", [0, 1])
@pytest.mark.parametrize("w", [384, 1536])
def test_sh(device, w, mask, fp32, dt):
    torch.manual_seed(61433)
    b, nh, h = 8, 4, 384
    grid = (b, nh)
    t = ((torch.rand((b, nh, h, w)) * 2 - 1) * 1000).to(torch.bfloat16)
    m = ttnn.from_torch(torch.zeros(b, 1, 1, w), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mc = ttnn.create_sharded_memory_config(t.shape, core_grid=ttnn.CoreGrid(y=nh, x=b), strategy=ttnn.ShardStrategy.HEIGHT,
                                           orientation=ttnn.ShardOrientation.ROW_MAJOR)
    pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=grid, subblock_w=3 if fp32 else 6,
                                                   block_h=h // 32, block_w=w // 32)
    cc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                fp32_dest_acc_en=bool(fp32), packer_l1_acc=False)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16 if dt == "bf16" else ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=mc)
    if mask:
        out = ttnn.scale_mask_softmax_in_place(x, 1.0, m, program_config=pc, compute_kernel_config=cc, numeric_stable=True)
    else:
        out = ttnn.scale_mask_softmax_in_place(x, program_config=pc, compute_kernel_config=cc, numeric_stable=True)
    ttnn.to_torch(out)
