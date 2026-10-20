# Scratch, not for merge. Step 4 callers: layer and RMS norm (layernorm.cpp), softmax and scale-mask softmax (attention
# softmax.cpp), SDPA prefill (sdpa.cpp) at model shapes; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

BF16, BFP8 = ttnn.bfloat16, ttnn.bfloat8_b


def cfg(name):
    if name == "def":
        return None
    fid, fp32 = {"lofi": (ttnn.MathFidelity.LoFi, False), "hifi2": (ttnn.MathFidelity.HiFi2, True),
                 "hifi2nf": (ttnn.MathFidelity.HiFi2, False), "hifi4": (ttnn.MathFidelity.HiFi4, False)}[name]
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=fp32,
                                            packer_l1_acc=False)


def dev(t, device, dtype=BF16, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


NORM = {  # name: (rms, shape, dtype, gamma, beta, memory)
    "ln_bert": (False, (1, 1, 384, 1024), BF16, True, True, ttnn.DRAM_MEMORY_CONFIG),
    "ln_bfp8": (False, (1, 1, 1024, 2048), BFP8, True, True, ttnn.DRAM_MEMORY_CONFIG),
    "ln_l1": (False, (1, 1, 256, 768), BF16, True, True, ttnn.L1_MEMORY_CONFIG),
    "ln_plain": (False, (1, 1, 512, 1024), BF16, False, False, ttnn.DRAM_MEMORY_CONFIG),
    "ln_f32": (False, (1, 1, 512, 1024), ttnn.float32, True, True, ttnn.DRAM_MEMORY_CONFIG),
    "ln_f32_plain": (False, (1, 1, 256, 2048), ttnn.float32, False, False, ttnn.DRAM_MEMORY_CONFIG),
    "rms_f32": (True, (1, 1, 512, 2048), ttnn.float32, True, False, ttnn.DRAM_MEMORY_CONFIG),
    "rms_llama": (True, (1, 1, 128, 4096), BF16, True, False, ttnn.DRAM_MEMORY_CONFIG),
    "rms_long": (True, (1, 1, 1024, 4096), BF16, True, False, ttnn.DRAM_MEMORY_CONFIG),
}
SOFTMAX = {  # name: (shape, masked)
    "sm_bert": ((1, 16, 384, 384), False),
    "sm_128": ((1, 32, 128, 128), False),
    "sm_1k": ((1, 8, 1024, 1024), False),
    "smm_bert": ((16, 1, 384, 384), True),
}
SDPA = {  # name: (b, nh, nkv, s, d, q_chunk, k_chunk, dtype)
    "sdpa_1k_d64": (1, 8, 8, 1024, 64, 128, 128, BF16),
    "sdpa_2k_d128": (1, 4, 1, 2048, 128, 128, 256, BFP8),
    "sdpa_4k_d128": (1, 2, 2, 4096, 128, 256, 256, BF16),
    "sdpa_2k_d64_q64": (1, 8, 8, 2048, 64, 64, 128, BF16),
    "sdpa_2k_d64_f32": (1, 4, 4, 2048, 64, 128, 128, ttnn.float32),
    "sdpa_8k_d128": (1, 1, 1, 8192, 128, 256, 512, BFP8),
}


def record(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(out).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


@pytest.mark.parametrize("config", ["def", "lofi", "hifi2"])
@pytest.mark.parametrize("case", list(NORM))
def test_norm(device, case, config, request):
    rms, shape, dtype, g, b, mem = NORM[case]
    torch.manual_seed(0)
    x = dev(torch.randn(shape), device, dtype, mem)
    w = shape[-1]
    gamma = dev(torch.rand((w,)) + 0.5, device) if g else None
    beta = dev(torch.rand((w,)) - 0.5, device) if b else None
    out = None
    for _ in range(2):
        if rms:
            out = ttnn.rms_norm(x, epsilon=1e-5, weight=gamma, memory_config=mem, compute_kernel_config=cfg(config))
        else:
            out = ttnn.layer_norm(x, epsilon=1e-5, weight=gamma, bias=beta, memory_config=mem,
                                  compute_kernel_config=cfg(config))
    record(request, out)


@pytest.mark.parametrize("config", ["def", "lofi", "hifi4"])
@pytest.mark.parametrize("case", list(SOFTMAX))
def test_softmax(device, case, config, request):
    shape, masked = SOFTMAX[case]
    torch.manual_seed(0)
    x = dev(torch.randn(shape) * 4, device)
    out = None
    if masked:
        m = torch.zeros((shape[0], 1, 32, shape[-1]))
        m[torch.rand_like(m) < 0.2] = -1e4
        mask = dev(m, device)
        for _ in range(2):
            out = ttnn.scale_mask_softmax(x, 0.125, mask, compute_kernel_config=cfg(config))
    else:
        for _ in range(2):
            out = ttnn.softmax(x, dim=-1, compute_kernel_config=cfg(config))
    record(request, out)


@pytest.mark.parametrize("config", ["hifi2", "hifi2nf", "hifi4", "lofi"])
@pytest.mark.parametrize("case", list(SDPA))
def test_sdpa(device, case, config, request):
    b, nh, nkv, s, d, qc, kc, dtype = SDPA[case]
    torch.manual_seed(1234)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                                q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=True)
    q, k, v = (dev(torch.randn(b, h, s, d), device, dtype) for h in (nh, nkv, nkv))
    out = None
    for _ in range(2):
        out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=pc,
                                                            compute_kernel_config=cfg(config))
    record(request, out)


SHARD = {  # name: (h, w, cores_h, cores_w, block_ht, block_wt, subblock_w, rms, gamma, beta)
    "lnsh_1k": (1024, 1024, 8, 8, 4, 4, 2, False, True, True),
    "lnsh_4k": (1024, 4096, 8, 8, 4, 16, 4, False, True, True),
    "lnsh_plain": (1024, 1024, 8, 8, 4, 4, 2, False, False, False),
    "rmssh_2k": (1024, 2048, 8, 8, 4, 8, 4, True, True, False),
}


@pytest.mark.parametrize("config", ["def", "lofi", "hifi2"])
@pytest.mark.parametrize("case", list(SHARD))
def test_norm_sharded(device, case, config, request):
    import sys

    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/work"))
    from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import create_sharded_mem_config

    h, w, nch, ncw, bht, bwt, subw, rms, g, b = SHARD[case]
    torch.manual_seed(0)
    mem = create_sharded_mem_config(h, w, nch, ncw, False)
    x = ttnn.from_torch(torch.randn((h, w)), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    gamma = dev(torch.rand((1, w)) + 0.5, device) if g else None
    beta = dev(torch.rand((1, w)) - 0.5, device) if b else None
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        block_h=bht, block_w=bwt, subblock_w=subw, use_welford=False, inplace=False)
    out = None
    for _ in range(2):
        if rms:
            out = ttnn.rms_norm(x, epsilon=1e-5, weight=gamma, memory_config=mem, program_config=pc,
                                compute_kernel_config=cfg(config))
        else:
            out = ttnn.layer_norm(x, epsilon=1e-5, weight=gamma, bias=beta, memory_config=mem, program_config=pc,
                                  compute_kernel_config=cfg(config))
    record(request, out)


SHARD32 = {  # name: (shard case, dtype, fp32 dest, welford)
    "ln1k_bf16_f32d": ("lnsh_1k", BF16, True, False),
    "ln4k_bf16_f32d": ("lnsh_4k", BF16, True, False),
    "rms2k_bf16_f32d": ("rmssh_2k", BF16, True, False),
    "ln1k_f32in": ("lnsh_1k", ttnn.float32, True, False),
    "rms2k_f32in": ("rmssh_2k", ttnn.float32, True, False),
    "ln1k_welford": ("lnsh_1k", BF16, False, True),
}


@pytest.mark.parametrize("case", list(SHARD32))
def test_norm_sharded32(device, case, request):
    import sys

    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/work"))
    from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import create_sharded_mem_config

    base, dt, f32d, welford = SHARD32[case]
    h, w, nch, ncw, bht, bwt, subw, rms, g, b = SHARD[base]
    torch.manual_seed(0)
    mem = create_sharded_mem_config(h, w, nch, ncw, False)
    x = ttnn.from_torch(torch.randn((h, w)), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    gamma = dev(torch.rand((1, w)) + 0.5, device, dt) if g else None
    beta = dev(torch.rand((1, w)) - 0.5, device, dt) if b else None
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        block_h=bht, block_w=bwt, subblock_w=subw, use_welford=welford, inplace=False)
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True,
                                          fp32_dest_acc_en=f32d, packer_l1_acc=False)
    recip = None
    if welford:
        sp = x.memory_config().shard_spec
        recip = ttnn.create_layer_norm_reciprocals(device, sp.grid, sp.shape[1])
    out = None
    for _ in range(2):
        if rms:
            out = ttnn.rms_norm(x, epsilon=1e-5, weight=gamma, memory_config=mem, program_config=pc,
                                compute_kernel_config=ck)
        else:
            out = ttnn.layer_norm(x, epsilon=1e-5, weight=gamma, bias=beta, memory_config=mem, program_config=pc,
                                  recip_tensor=recip, compute_kernel_config=ck)
    record(request, out)
