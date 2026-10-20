# Scratch, not for merge. Sharded layer norm with Welford (layernorm_sharded_welford.cpp) at model shapes; each output's sha1 goes to
# $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os
import sys

import pytest
import torch
import ttnn

BF16 = ttnn.bfloat16
SHARD = {  # name: (h, w, cores_h, cores_w, block_ht, block_wt, subblock_w, gamma, beta)
    "w1k": (1024, 1024, 8, 8, 4, 4, 2, True, True),
    "w4k": (1024, 4096, 8, 8, 4, 16, 4, True, True),
    "w1k_plain": (1024, 1024, 8, 8, 4, 4, 2, False, False),
}
CFG = {"lofi": (ttnn.MathFidelity.LoFi, False), "hifi2": (ttnn.MathFidelity.HiFi2, False),
       "hifi4": (ttnn.MathFidelity.HiFi4, False), "hifi2f32": (ttnn.MathFidelity.HiFi2, True)}


@pytest.mark.parametrize("config", list(CFG))
@pytest.mark.parametrize("case", list(SHARD))
def test_wsh(device, case, config, request):
    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/work"))
    from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import create_sharded_mem_config

    h, w, nch, ncw, bht, bwt, subw, g, b = SHARD[case]
    torch.manual_seed(0)
    mem = create_sharded_mem_config(h, w, nch, ncw, False)
    dev = lambda t: ttnn.from_torch(t, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.from_torch(torch.randn((h, w)), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    gamma = dev(torch.rand((1, w)) + 0.5) if g else None
    beta = dev(torch.rand((1, w)) - 0.5) if b else None
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        block_h=bht, block_w=bwt, subblock_w=subw, use_welford=True, inplace=False)
    fid, f32d = CFG[config]
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=f32d,
                                          packer_l1_acc=False)
    sp = x.memory_config().shard_spec
    recip = ttnn.create_layer_norm_reciprocals(device, sp.grid, sp.shape[1])
    out = None
    for _ in range(2):
        out = ttnn.layer_norm(x, epsilon=1e-5, weight=gamma, bias=beta, memory_config=mem, program_config=pc,
                              recip_tensor=recip, compute_kernel_config=ck)
    ttnn.synchronize_device(device)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        hh = hashlib.sha1(ttnn.to_torch(out).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {hh}\n")
