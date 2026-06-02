import torch
from models.experimental.opt_transfer.transforms import apply_transform


def build_fused_qkv(proposal, weights, device, dims):
    """Return a callable x(torch[B,S,embed]) -> (q,k,v) torch[B,H,S,D] that runs the
    concatenated-QKV matmul + nlp_create_qkv_heads on device. Mirrors the verified path."""
    import ttnn

    H = proposal.config["num_heads"]
    embed = dims["embed"]
    folded = apply_transform(proposal.weight_transform, weights, order=proposal.matched_nodes)

    W_tt = ttnn.from_torch(
        folded["weight"].transpose(0, 1).contiguous(), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    B_tt = ttnn.from_torch(folded["bias"].reshape(1, -1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def run(x: torch.Tensor):
        B, S, _ = x.shape
        x_tt = ttnn.from_torch(x.reshape(B, 1, S, embed), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        qkv = ttnn.linear(x_tt, W_tt, bias=B_tt, compute_kernel_config=ckc)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=H,
            num_kv_heads=proposal.config["num_kv_heads"],
            transpose_k_heads=proposal.config.get("transpose_k_heads", False),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.to_torch(q), ttnn.to_torch(k), ttnn.to_torch(v)

    return run
