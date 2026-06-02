# tests/test_assemble_device.py
import pytest
import torch
from models.experimental.opt_transfer.references.seamless_m4t_v2 import SeamlessBlock
from models.experimental.opt_transfer.codegen import build_fused
from models.experimental.opt_transfer.assemble import assemble_layers
from models.experimental.opt_transfer.verify import pcc
from models.experimental.opt_transfer.schema import KBEntry, PatternKind, FusionProposal


@pytest.mark.device
def test_two_layer_fused_qkv_assembles_on_device():
    """Multi-block assembly: 2 SeamlessBlocks, fused QKV per layer, stacked + run on device.
    Random-init weights (real HF-checkpoint loading via integration skill is a follow-on)."""
    import ttnn

    H, D, embed = 16, 64, 1024
    entry = KBEntry(
        "nlp_create_qkv_heads",
        "ttnn.experimental.nlp_create_qkv_heads",
        "attention.qkv",
        PatternKind.HORIZONTAL_MERGE,
        ["linear", "linear", "linear"],
        {},
        {},
        "concat_qkv",
        "s",
    )

    def proposal():
        return FusionProposal(
            "nlp_create_qkv_heads",
            "ttnn.experimental.nlp_create_qkv_heads",
            ["q_proj", "k_proj", "v_proj"],
            {"num_heads": H, "num_kv_heads": H, "transpose_k_heads": False},
            "concat_qkv",
            "",
            "s",
        ).resolve({"H": H, "D": D, "embed": embed})

    torch.manual_seed(0)
    blocks = [SeamlessBlock(embed, H).eval() for _ in range(2)]
    device = ttnn.open_device(device_id=0)
    try:
        # Build a per-layer fused-QKV runner; verify each layer's q/k/v vs its golden split.
        worst = 1.0
        for blk in blocks:
            with torch.no_grad():
                x = torch.randn(1, 32, embed)
                h = blk.attn_norm(x)
            weights = {
                n: {"weight": getattr(blk, n).weight.detach(), "bias": getattr(blk, n).bias.detach()}
                for n in ("q_proj", "k_proj", "v_proj")
            }
            run = build_fused(proposal(), entry, weights, device, {"H": H, "D": D, "embed": embed})
            q, k, v = run(h)
            for name, got in zip(("q_proj", "k_proj", "v_proj"), (q, k, v)):
                with torch.no_grad():
                    gold = blk._split(getattr(blk, name)(h))
                worst = min(worst, pcc(gold, got))
        assert worst > 0.99, worst

        # assemble_layers composes runners into one forward (smoke: identity-style passthrough)
        model = assemble_layers([lambda t: t, lambda t: t])
        assert model(123) == 123
    finally:
        ttnn.close_device(device)
