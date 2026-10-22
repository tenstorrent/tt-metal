# Scratch, not for merge. Block callers: sharded Welford layernorm (layernorm_sharded_welford.cpp) and sharded post all-gather
# norm (layernorm_sharded_post_allgather.cpp); each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import math
import os
import sys

import pytest
import torch
import ttnn

BF16, BFP8 = ttnn.bfloat16, ttnn.bfloat8_b


def record(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(out).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


def cfg(name):
    if name == "def":
        return None
    fid, fp32 = {"lofi": (ttnn.MathFidelity.LoFi, False), "lofi32": (ttnn.MathFidelity.LoFi, True),
                 "hifi2": (ttnn.MathFidelity.HiFi2, True), "hifi4": (ttnn.MathFidelity.HiFi4, False)}[name]
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=fp32,
                                            packer_l1_acc=False)


# name: (cores_h, cores_w, block_ht, block_wt, subblock_wt, gamma, beta, residual)
WLF = {
    "w_s1": (8, 8, 4, 4, 1, True, True, False),
    "w_s2": (8, 8, 4, 4, 2, True, True, False),
    "w_s3": (8, 8, 2, 6, 3, True, True, False),
    "w_s4": (8, 8, 4, 8, 4, True, True, False),
    "w_s8": (8, 8, 2, 16, 8, True, True, False),
    "w_s4_ht8": (8, 8, 8, 8, 4, True, True, False),
    "w_plain": (8, 8, 4, 8, 4, False, False, False),
    "w_res": (8, 8, 4, 8, 4, True, True, True),
    "w_res_s2": (8, 8, 4, 4, 2, True, False, True),
    "w_gamma": (8, 8, 4, 8, 4, True, False, False),
    "w_s4_wide": (4, 8, 4, 16, 4, True, True, False),
}


@pytest.mark.parametrize("config", ["def", "lofi", "hifi2"])
@pytest.mark.parametrize("case", list(WLF))
def test_wlf(device, case, config, request):
    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/work"))
    from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import create_sharded_mem_config

    nch, ncw, bht, bwt, sub, g, b, res = WLF[case]
    if config != "lofi" and sub > 4:
        pytest.skip("fp32 dest limits the subblock to 4 tiles")
    h, w = 32 * bht * nch, 32 * bwt * ncw
    torch.manual_seed(0)
    mem = create_sharded_mem_config(h, w, nch, ncw, False)
    x = ttnn.from_torch(torch.randn((h, w)), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    r = ttnn.from_torch(torch.randn((h, w)), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem) if res else None
    mk = lambda t: ttnn.from_torch(t, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gamma = mk(torch.rand((1, w)) + 0.5) if g else None
    beta = mk(torch.rand((1, w)) - 0.5) if b else None
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        block_h=bht, block_w=bwt, subblock_w=sub, use_welford=True, inplace=False)
    sp = x.memory_config().shard_spec
    recip = ttnn.create_layer_norm_reciprocals(device, sp.grid, sp.shape[1])
    out = None
    for _ in range(2):
        out = ttnn.layer_norm(x, epsilon=1e-5, residual_input_tensor=r, weight=gamma, bias=beta, memory_config=mem,
                              program_config=pc, recip_tensor=recip, compute_kernel_config=cfg(config))
    record(request, out)


# name: (cores_x, cores_y, input width per device, rms, weight, bias)
PAG = {
    "p_n1": (8, 8, 2048, False, True, True),   # block_w 1
    "p_n2": (8, 4, 2048, False, True, True),   # 2
    "p_n3": (8, 2, 1536, False, True, True),   # 3
    "p_n4": (8, 2, 2048, False, True, True),   # 4
    "p_n8": (4, 2, 2048, False, True, True),   # 8
    "p_n4_rms": (8, 2, 2048, True, True, False),
    "p_n8_rms": (4, 2, 2048, True, True, False),
    "p_n4_plain": (8, 2, 2048, False, False, False),
    "p_n2_rms": (8, 4, 2048, True, True, False),
    "p_n6": (4, 2, 1536, False, True, True),
}


@pytest.mark.parametrize("config", ["def", "lofi", "hifi2"])
@pytest.mark.parametrize("case", list(PAG))
def test_pag(device, case, config, request):
    cx, cy, width, rms, g, b = PAG[case]
    ndev = 4
    torch.manual_seed(0)
    full = width * ndev
    xs = torch.normal(0, 1, size=(1, 1, 32, full), dtype=torch.bfloat16)
    chunks = torch.chunk(xs, ndev, dim=-1)
    stats = []
    for c in chunks:
        if rms:
            ex2 = torch.nn.functional.pad(torch.mean(c ** 2, dim=-1, keepdim=True), (0, 31))
            stats.append(ex2)
        else:
            stats.append(torch.nn.functional.pad(torch.mean(c, dim=-1, keepdim=True), (0, 31)))
            stats.append(torch.nn.functional.pad(torch.mean(c ** 2, dim=-1, keepdim=True), (0, 31)))
    tst = ttnn.from_torch(torch.cat(stats, dim=-1), layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=ttnn.L1_MEMORY_CONFIG)
    tst = ttnn.to_memory_config(tst, ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, tst.padded_shape[-1]), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.WIDTH))
    ncores = cx * cy
    sub = (width // ncores) // 32
    if config != "lofi" and sub > 4:
        pytest.skip("fp32 dest limits the subblock to 4 tiles")
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(cx, cy), subblock_w=sub,
                                                     block_h=1, block_w=sub, inplace=False)
    tg = torch.normal(0, 1, size=(1, 1, 1, width), dtype=torch.bfloat16)
    tb = torch.normal(0, 1, size=(1, 1, 1, width), dtype=torch.bfloat16)
    x = ttnn.from_torch(chunks[0], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=BF16)
    rng = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cx - 1, cy - 1))
    x = ttnn.to_memory_config(x, ttnn.create_sharded_memory_config(
        shape=(32, width // ncores), core_grid=ttnn.CoreRangeSet({rng}), strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True))
    wt = ttnn.from_torch(tg, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=BF16) if g else None
    bt = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=BF16) if b else None
    out = None
    for _ in range(2):
        if rms:
            out = ttnn.rms_norm_post_all_gather(x, epsilon=1e-6, weight=wt, bias=bt, program_config=pc, stats=tst,
                                                dtype=BF16, compute_kernel_config=cfg(config))
        else:
            out = ttnn.layer_norm_post_all_gather(x, epsilon=1e-6, weight=wt, bias=bt, program_config=pc, stats=tst,
                                                  dtype=BF16, compute_kernel_config=cfg(config))
    record(request, out)
