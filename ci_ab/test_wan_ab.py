# Scratch, not for merge. wan_fused_rmsnorm_post_allgather (fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp), one simulated device
# of 8, HiFi4 with fp32 DEST as the op's tests; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

CASES = {  # name: (seq, hidden, heads per device, weight, rope)
    "w_rope": (1024, 5120, 5, True, True),
    "w_norope": (1024, 5120, 5, True, False),
    "plain": (1024, 5120, 5, False, False),
    "w_rope_2k": (512, 2048, 2, True, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_wan(device, case, request):
    seq, hidden, hpd, use_w, use_rope = CASES[case]
    nd = 8
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)
    torch.manual_seed(1234)
    x = torch.randn((1, 1, seq, hidden)) * 4 - 1
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    chunks = [dev(c) for c in torch.chunk(x, nd, -1)]
    w = dev(torch.rand((1, hidden // nd))) if use_w else None
    cos = sin = tm = None
    if use_rope:
        head_dim = hidden // (hpd * nd)
        c = torch.randn(1, 1, seq, head_dim // 2)
        s = torch.randn(1, 1, seq, head_dim // 2)
        cos = dev(torch.stack([c, c], dim=-1).flatten(-2))
        sin = dev(torch.stack([s, s], dim=-1).flatten(-2))
        m = torch.zeros(1, 1, 32, 32)
        for i in range(0, 32, 2):
            m[0, 0, i + 1, i] = 1.0
            m[0, 0, i, i + 1] = -1.0
        tm = dev(m)
    stats = ttnn.concat([ttnn.experimental.wan_fused_rmsnorm_pre_allgather(c, compute_kernel_config=cfg,
                                                                          dtype=ttnn.bfloat16) for c in chunks], -1)
    out = None
    for _ in range(2):
        out = ttnn.experimental.wan_fused_rmsnorm_post_allgather(
            chunks[0], stats, epsilon=1e-5, num_heads_per_device=hpd, weight=w, transformation_mat=tm,
            rope_cos=cos, rope_sin=sin, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")
