import pytest
import torch
from models.experimental.opt_transfer.schema import FusionProposal, MemoryPlacement
from models.experimental.opt_transfer.codegen import build_fused_qkv


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _golden(x, w, b, H, D):
    out = x @ w.transpose(0, 1) + b
    B, S, _ = x.shape
    return out.reshape(B, S, H, D).transpose(1, 2)


@pytest.mark.device
@pytest.mark.parametrize("buffer", ["DRAM", "L1"])
def test_fused_qkv_pcc_neutral_under_placement(buffer):
    import ttnn

    B, S, H, D = 1, 64, 16, 64
    embed = H * D
    torch.manual_seed(0)
    x = torch.randn(B, S, embed)
    weights = {
        n: {"weight": torch.randn(embed, embed) * 0.02, "bias": torch.randn(embed) * 0.02}
        for n in ("q_proj", "k_proj", "v_proj")
    }
    goldens = {n: _golden(x, weights[n]["weight"], weights[n]["bias"], H, D) for n in weights}
    prop = FusionProposal(
        "qkv",
        "ttnn.experimental.nlp_create_qkv_heads",
        ["q_proj", "k_proj", "v_proj"],
        {"num_heads": H, "num_kv_heads": H, "transpose_k_heads": False},
        "concat_qkv",
        "",
        "x",
    )
    device = ttnn.open_device(device_id=0)
    try:
        run = build_fused_qkv(
            prop, weights, device, {"H": H, "D": D, "embed": embed}, placement=MemoryPlacement(buffer)
        )
        q, k, v = run(x)
        for n, got in zip(("q_proj", "k_proj", "v_proj"), (q, k, v)):
            pcc_val = _pcc(goldens[n], got)
            print(f"buffer={buffer} {n} PCC={pcc_val:.6f}")
            assert pcc_val > 0.99, f"PCC={pcc_val:.6f} for {n} with buffer={buffer}"
    finally:
        ttnn.close_device(device)
