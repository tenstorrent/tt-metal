import pytest
import torch
from models.experimental.opt_transfer.schema import FusionProposal
from models.experimental.opt_transfer.codegen import build_fused_qkv
from models.experimental.opt_transfer.codegen import register_emitter, build_fused


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _golden(x, w, b, H, D):
    out = x @ w.transpose(0, 1) + b
    Bb, S, _ = x.shape
    return out.reshape(Bb, S, H, D).transpose(1, 2)


@pytest.mark.device
def test_fused_qkv_matches_separate_projection_goldens():
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
        "qkv_merge",
        "ttnn.experimental.nlp_create_qkv_heads",
        ["q_proj", "k_proj", "v_proj"],
        {"num_heads": H, "num_kv_heads": H, "transpose_k_heads": False},
        "concat_qkv",
        "",
        "x",
    )
    device = ttnn.open_device(device_id=0)
    try:
        fused = build_fused_qkv(prop, weights, device, dims={"H": H, "D": D, "embed": embed})
        q, k, v = fused(x)
        pcc_q = _pcc(goldens["q_proj"], q)
        pcc_k = _pcc(goldens["k_proj"], k)
        pcc_v = _pcc(goldens["v_proj"], v)
        print(f"\nPCC q={pcc_q:.6f}  k={pcc_k:.6f}  v={pcc_v:.6f}")
        assert pcc_q > 0.99, f"q PCC {pcc_q} < 0.99"
        assert pcc_k > 0.99, f"k PCC {pcc_k} < 0.99"
        assert pcc_v > 0.99, f"v PCC {pcc_v} < 0.99"
    finally:
        ttnn.close_device(device)


def test_build_fused_dispatches_by_fused_op():
    @register_emitter("test.op")
    def _e(proposal, entry, weights, device, dims, placement=None):
        return lambda x: ("ran", x)

    p = FusionProposal("e", "test.op", ["n"], {}, None, "", "")
    runner = build_fused(p, entry=None, weights={}, device=None, dims={})
    assert runner(7) == ("ran", 7)


def test_build_fused_unknown_op_raises():
    p = FusionProposal("e", "ttnn.not_registered", ["n"], {}, None, "", "")
    with pytest.raises(KeyError):
        build_fused(p, None, {}, None, {})
