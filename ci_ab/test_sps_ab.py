# Scratch, not for merge. Single-device sharded rms_norm / layer_norm post_all_gather (layernorm_sharded_post_allgather.cpp) at
# decode shapes; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

LF, H2, H4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
# name: (rms, tiles per core, cores, fidelity, fp32 dest, gamma, beta)
CASES = {
    "rms_t4_hf4": (True, 4, 16, H4, False, True, False),
    "rms_t4_hf2": (True, 4, 16, H2, False, True, False),
    "rms_t4_lofi": (True, 4, 16, LF, False, True, False),
    "rms_t8_hf4": (True, 8, 16, H4, False, True, False),
    "rms_t8_lofi": (True, 8, 16, LF, False, True, False),
    "ln_t4_hf4": (False, 4, 16, H4, False, True, True),
    "ln_t4_hf2": (False, 4, 16, H2, False, True, True),
    "ln_t4_lofi": (False, 4, 16, LF, False, True, True),
    "ln_t8_hf4": (False, 8, 16, H4, False, True, True),
    "ln_t8_lofi": (False, 8, 16, LF, False, True, True),
    "ln_t4_hf2_f32": (False, 4, 16, H2, True, True, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_sps(device, case, request):
    rms, tpc, cores, fid, f32, gamma, beta = CASES[case]
    torch.manual_seed(0)
    W = tpc * 32 * cores
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=False)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores // 2 - 1, 1))])
    mc = ttnn.create_sharded_memory_config(
        shape=(32, tpc * 32), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(cores // 2, 2), subblock_w=1, block_h=1, block_w=tpc, inplace=False)
    x = torch.randn(1, 1, 32, W).bfloat16()
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    mk = lambda: ttnn.from_torch(torch.rand(1, 1, W // 32, 32).bfloat16(), dtype=ttnn.bfloat16,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = mk() if gamma else None
    b = mk() if beta else None
    pre = ttnn.rms_norm_pre_all_gather if rms else ttnn.layer_norm_pre_all_gather
    post = ttnn.rms_norm_post_all_gather if rms else ttnn.layer_norm_post_all_gather
    stats = pre(tt_x, compute_kernel_config=cfg, program_config=pc)
    stats_w = stats.shape[-1]
    smc = ttnn.create_sharded_memory_config(
        shape=[32, stats_w], core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    stats = ttnn.to_memory_config(stats, smc)
    kw = dict(epsilon=1e-5, weight=w, stats=stats, compute_kernel_config=cfg, program_config=pc, memory_config=mc)
    if not rms:
        kw["bias"] = b
    out = None
    for _ in range(2):
        out = post(tt_x, **kw)
    ttnn.synchronize_device(device)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")
