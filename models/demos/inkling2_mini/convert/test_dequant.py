# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Expert dequant vs transformers' ``Inkling2LoopDequantizeExperts`` (+ ``Interleave(dim=1)`` for w13), 2 experts.

Needs a transformers with Inkling2Loop (no ttnn, no device):
    PYTHONPATH=<worktree>:<worktree>/.t1-pydeps /work/inkling-bringup/ref-venv/bin/python -m pytest \
        --noconftest -c /dev/null models/demos/inkling2_mini/convert/test_dequant.py
The real-checkpoint cases read ``$INKLING_CKPT`` (default /work/models/Inkling-2-Mini) and skip when absent.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from models.demos.inkling2_mini.convert.source import deinterleave_gate_up, dequant_experts

core = pytest.importorskip("transformers.core_model_loading")
if not hasattr(core, "Inkling2LoopDequantizeExperts"):
    pytest.skip("transformers here has no Inkling2Loop (needs >= 5.24)", allow_module_level=True)
CKPT = Path(os.environ.get("INKLING_CKPT", "/work/models/Inkling-2-Mini"))
SRC = ["w", "w_scale", "w_amax", "w_denominator"]


def reference(weight, scale, amax, denominator, is_w13):
    op = core.Inkling2LoopDequantizeExperts()
    inputs = {"w": weight, "w_scale": scale}
    if amax is not None:
        inputs |= {"w_amax": amax, "w_denominator": denominator}
    out = op.convert(inputs, SRC, ["out"])["out"]
    if is_w13:
        out = core.Interleave(dim=1).convert({"out": out}, ["out"], ["out"])["out"]
    return out


def ours(weight, scale, amax, denominator, is_w13):
    out = dequant_experts(weight, scale, amax, denominator)
    return deinterleave_gate_up(out) if is_w13 else out


def check(weight, scale, amax, denominator, is_w13):
    ref = reference(weight, scale, amax, denominator, is_w13).float()
    got = ours(weight, scale, amax, denominator, is_w13).float()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    # minimax divides by denominator/amax where transformers multiplies by amax/denominator: fp32 rounding may
    # move a value by one bf16 ulp, nothing more.
    torch.testing.assert_close(got, ref, rtol=2**-7, atol=0)
    assert pcc > 0.99999, pcc
    return (got == ref).float().mean().item()


def synthetic(nvfp4, rows, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    if nvfp4:
        weight = torch.randint(0, 256, (2, rows, k // 2), dtype=torch.uint8, generator=g)
        scale = (torch.rand(2, rows, k // 16, generator=g) * 400 + 1).to(torch.float8_e4m3fn)
        amax = torch.rand(2, generator=g) + 0.05
        return weight, scale, amax, torch.tensor([2688.0])
    weight = (torch.randn(2, rows, k, generator=g) * 50).to(torch.float8_e4m3fn)
    scale = torch.randint(110, 130, (2, rows, k // 32), dtype=torch.uint8, generator=g)
    return weight, scale, None, None


@pytest.mark.parametrize("nvfp4", [True, False], ids=["nvfp4", "mxfp8"])
@pytest.mark.parametrize("is_w13", [True, False], ids=["w13", "w2"])
def test_synthetic(nvfp4, is_w13):
    check(*synthetic(nvfp4, 2304, 2304 if is_w13 else 1152), is_w13)


def _open_ckpt():
    index = CKPT / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip(f"no checkpoint at {CKPT}")
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if not all((CKPT / s).exists() for s in shards):
        pytest.skip("checkpoint shards not complete")
    from models.demos.inkling2_mini.convert.source import Checkpoint

    return Checkpoint(CKPT)


@pytest.mark.parametrize("layer", [0, 20], ids=["L0-nvfp4", "L20-mxfp8"])
@pytest.mark.parametrize("name", ["w13", "w2"])
def test_checkpoint(layer, name):
    ckpt = _open_ckpt()
    base = f"model.llm.layers.{layer}.mlp.experts.{name}_weight"
    weight, scale = ckpt.experts(base, 0, 2), ckpt.experts(base + "_scale", 0, 2)
    amax = ckpt.get(base + "_amax")[:2] if ckpt.has(base + "_amax") else None
    denominator = ckpt.get(base + "_outer_scale_denominator") if amax is not None else None
    assert (amax is not None) == (layer < 20)
    exact = check(weight, scale, amax, denominator, name == "w13")
    assert exact > 0.99, exact


BF16_STORE = Path(os.environ.get("INKLING_BF16_STORE", "/work/models/Inkling-2-Mini-bf16"))


@pytest.mark.parametrize("layer", [0, 20], ids=["L0-nvfp4", "L20-mxfp8"])
@pytest.mark.parametrize("name", ["w13", "w2"])
def test_bf16_store_matches_dequant(layer, name):
    """The --bf16-store source (cache/v1-b8) holds the same expert values as our dequant of the checkpoint."""
    if not BF16_STORE.exists():
        pytest.skip(f"no bf16 store at {BF16_STORE}")
    ckpt = _open_ckpt()
    from models.demos.inkling2_mini.convert.source import Checkpoint

    store = Checkpoint(BF16_STORE)
    base = f"model.llm.layers.{layer}.mlp.experts.{name}_weight"
    amax = ckpt.get(base + "_amax")[:2] if ckpt.has(base + "_amax") else None
    den = ckpt.get(base + "_outer_scale_denominator") if amax is not None else None
    got = ours(ckpt.experts(base, 0, 2), ckpt.experts(base + "_scale", 0, 2), amax, den, name == "w13").float()
    key = f"model.language_model.layers.{layer}.mlp.experts.{'gate_up_proj' if name == 'w13' else 'down_proj'}"
    torch.testing.assert_close(got, store.experts(key, 0, 2).float(), rtol=2**-7, atol=0)
