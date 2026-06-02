import torch
from models.experimental.opt_transfer.transforms import apply_transform


def test_concat_qkv_stacks_weights_and_biases():
    w = {n: {"weight": torch.randn(1024, 1024), "bias": torch.randn(1024)} for n in ("q_proj", "k_proj", "v_proj")}
    out = apply_transform("concat_qkv", w, order=["q_proj", "k_proj", "v_proj"])
    assert out["weight"].shape == (3072, 1024)
    assert out["bias"].shape == (3072,)
    assert torch.allclose(out["weight"][:1024], w["q_proj"]["weight"])


def test_identity_passthrough():
    w = {"x": {"weight": torch.randn(4, 4), "bias": None}}
    assert apply_transform("identity", w, order=["x"])["weight"].shape == (4, 4)
