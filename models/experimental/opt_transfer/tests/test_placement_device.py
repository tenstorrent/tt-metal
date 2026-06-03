import pytest
import torch
from models.experimental.opt_transfer.codegen import build_fused_qkv
from models.experimental.opt_transfer.placement import L1Budget, decide_placement, tensor_bytes
from models.experimental.opt_transfer.schema import FusionProposal, MemoryPlacement, PlacementObservation


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


# ---------------------------------------------------------------------------
# MP6 — size-aware L1/DRAM placement driven by decide_placement + build_fused_qkv
# ---------------------------------------------------------------------------


def _obs_L1_if_small():
    return PlacementObservation(
        op="ttnn.experimental.nlp_create_qkv_heads",
        tensor_role="qkv_out",
        size_descriptor={"dims": "[seq, hidden]"},
        memory_config={"buffer": "L1", "layout": "interleaved", "shard_spec_template": None},
        program_config=None,
        condition={"var": "seq", "op": "<=", "value": 1024},
        source="dots.ocr",
    )


def test_decision_is_size_aware():
    # tiny aggregate budget so a large tensor is forced to DRAM by the backstop
    budget = L1Budget(per_core_bytes=64 * 1024, num_cores=8)  # ~256 KB aggregate @ safety 0.5
    H, D, embed = 16, 64, 1024
    small = decide_placement([_obs_L1_if_small()], tensor_bytes([64, embed], "bf16"), {"seq": 64}, budget)
    large = decide_placement([_obs_L1_if_small()], tensor_bytes([8192, embed], "bf16"), {"seq": 8192}, budget)
    assert small.buffer == "L1"  # small + donor-L1 + fits
    assert large.buffer == "DRAM"  # over budget AND seq>1024 -> DRAM


@pytest.mark.device
def test_placement_path_runs_both_on_device():
    import ttnn

    H, D, embed = 16, 64, 1024
    torch.manual_seed(0)
    weights = {
        n: {"weight": torch.randn(embed, embed) * 0.02, "bias": torch.randn(embed) * 0.02}
        for n in ("q_proj", "k_proj", "v_proj")
    }
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
        for buf, S in (("L1", 64), ("DRAM", 64)):
            x = torch.randn(1, S, embed)
            run = build_fused_qkv(
                prop, weights, device, {"H": H, "D": D, "embed": embed}, placement=MemoryPlacement(buf)
            )
            q, k, v = run(x)  # both placements execute on device without error
            assert q.shape == (1, H, S, D)
    finally:
        ttnn.close_device(device)
