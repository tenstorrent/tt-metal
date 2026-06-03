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


def placement_to_memory_config(placement):
    """Map a MemoryPlacement to a ttnn memory config. Interleaved only in this plan;
    sharded layouts are a follow-on (raise so a sharded choice can't silently no-op)."""
    import ttnn

    if placement.layout != "interleaved":
        raise NotImplementedError(f"sharded placement not yet supported: {placement.layout}")
    return ttnn.L1_MEMORY_CONFIG if placement.buffer == "L1" else ttnn.DRAM_MEMORY_CONFIG


_EMITTERS = {}


def register_emitter(fused_op):
    def deco(fn):
        _EMITTERS[fused_op] = fn
        return fn

    return deco


def build_fused(proposal, entry, weights, device, dims):
    """Dispatch a resolved FusionProposal to its emitter -> callable(input)->output(s).
    A KB op with no registered emitter raises (the graph routes that to handoff)."""
    if proposal.fused_op not in _EMITTERS:
        raise KeyError(
            f"no codegen emitter for {proposal.fused_op}; KB knows the op but codegen "
            f"can't emit it yet (register one + add a J6 model-shape test)"
        )
    return _EMITTERS[proposal.fused_op](proposal, entry, weights, device, dims)


# the verified QKV emitter becomes the first registered emitter
@register_emitter("ttnn.experimental.nlp_create_qkv_heads")
def _emit_qkv(proposal, entry, weights, device, dims):
    return build_fused_qkv(proposal, weights, device, dims)
