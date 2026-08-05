# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The CFM estimator UNet -- `02_plan.md` P3's last and largest piece.

Three tiers, cheapest first:

1. **Structure**, from the weight names alone. Catches a miscounted ModuleList
   without loading anything.
2. **Graph**, via `tt/flow/reference.py` -- a pure-torch reimplementation driven
   only by the flat export. If this matches the captured golden then the
   architecture is right and any device miss is a TTNN question, not an
   architecture question. That split is what makes the device tier cheap to debug.
3. **Device**, the TTNN UNet against the same golden.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.weights import default_weights_path

FLOW_WEIGHTS = default_weights_path().replace("hift_", "flow_")
PREFIX = "decoder.estimator"

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
needs_weights = pytest.mark.skipif(
    not os.path.exists(FLOW_WEIGHTS),
    reason="run scripts/export_weights.py --module flow in the CosyVoice venv first",
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "flow.cfm_estimator.npz")),
    reason="run scripts/gen_golden.py in the CosyVoice venv first",
)


def _flat_weights() -> dict[str, torch.Tensor]:
    with np.load(FLOW_WEIGHTS) as z:
        return {k: torch.from_numpy(np.ascontiguousarray(z[k])).float() for k in z.files if k.startswith(PREFIX)}


# --------------------------------------------------------------------------
# host tier -- structure
# --------------------------------------------------------------------------
@needs_weights
def test_estimator_shape_matches_the_config():
    """channels [256,256], n_blocks 4, num_mid_blocks 12, 8 heads x 64, in 320 -> out 80."""
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(FLOW_WEIGHTS).sub(PREFIX)
    assert bag.sub("down_blocks").children() == 2
    assert bag.sub("mid_blocks").children() == 12
    assert bag.sub("up_blocks").children() == 2
    for group, n in (("down_blocks", 2), ("mid_blocks", 12), ("up_blocks", 2)):
        for i in range(n):
            assert bag.sub(f"{group}.{i}.1").children() == 4, (group, i)

    assert bag.sub("time_mlp.linear_1").tensor("weight").shape == (1024, 320)
    assert bag.sub("down_blocks.0.0.block1.block.0").tensor("weight").shape == (256, 320, 3)
    assert bag.sub("up_blocks.0.0.block1.block.0").tensor("weight").shape == (256, 512, 3)
    assert bag.sub("final_proj").tensor("weight").shape == (80, 256, 1)
    # 8 heads x 64 = 512 inner, projected back to 256
    assert bag.sub("mid_blocks.0.1.0.attn1.to_q").tensor("weight").shape == (512, 256)
    assert not bag.sub("mid_blocks.0.1.0.attn1.to_q").has("bias"), "attention_bias is False upstream"
    assert bag.sub("mid_blocks.0.1.0.attn1.to_out.0").has("bias")


@needs_weights
def test_feed_forward_is_gelu_not_geglu():
    """`act_fn: 'gelu'` in cosyvoice.yaml reads like it might select GEGLU, and does
    not -- diffusers' FeedForward tests "gelu" in a bare `if` ahead of the `elif`
    chain. GEGLU would make the projection 2x wide to carry its gate; 1024 = 4*256
    exactly, so this is a plain GELU."""
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(FLOW_WEIGHTS).sub(PREFIX)
    proj = bag.sub("mid_blocks.0.1.0.ff.net.0.proj").tensor("weight")
    out = bag.sub("mid_blocks.0.1.0.ff.net.2").tensor("weight")
    assert proj.shape == (1024, 256), "GEGLU would be (2048, 256)"
    assert out.shape == (256, 1024)


@needs_golden
def test_classifier_free_guidance_batching_is_two_rows():
    """The estimator is always called on a batch of 2: row 0 conditioned, row 1 with
    mu, spks and cond zeroed. That is what makes it the RTF hot spot -- 10 Euler
    steps means 20 forward passes -- and it is why `batch=2` is the default here."""
    g = load_golden("flow.cfm_estimator")
    x, mu, spks, cond = (as_torch(g[f"call0.in_{n}"]) for n in ("x", "mu", "spks", "cond"))
    assert x.shape[0] == 2
    assert torch.equal(x[0], x[1]), "both CFG rows evaluate the same sample"
    for name, tensor in (("mu", mu), ("spks", spks), ("cond", cond)):
        assert bool((tensor[1] == 0).all()), f"{name} row 1 should be the null condition"
        assert not bool((tensor[0] == 0).all()), f"{name} row 0 should be conditioned"


@needs_golden
def test_mask_is_all_ones_so_masking_is_a_no_op():
    """`TtConditionalDecoder` rejects a mask rather than carrying an unverified path.
    This pins the assumption that makes that legal."""
    g = load_golden("flow.cfm_estimator")
    assert bool((as_torch(g["call0.in_mask"]) == 1).all())


@needs_golden
def test_sinusoidal_time_embedding_matches_the_schedule():
    """The 10 timesteps follow the cosine scheduler, `1 - cos(t*pi/2)`, not a linear
    ramp -- so step 0 is exactly 0 and the spacing widens. Getting this wrong would
    still produce audio, just the wrong audio."""
    import math

    from models.demos.cosyvoice.tt.flow.reference import sinusoidal_pos_emb

    g = load_golden("flow.cfm_estimator")
    got = torch.tensor([float(as_torch(g[f"call{i}.in_t"])[0]) for i in range(10)])
    span = torch.linspace(0, 1, 11)
    want = 1 - torch.cos(span * 0.5 * math.pi)
    assert torch.allclose(got, want[:10], atol=1e-6), (got, want[:10])

    emb = sinusoidal_pos_emb(got[:1], 320)
    assert emb.shape == (1, 320)
    assert torch.allclose(emb[0, :160], torch.zeros(160)), "t=0 makes every sine term zero"
    assert torch.allclose(emb[0, 160:], torch.ones(160)), "...and every cosine term one"


# --------------------------------------------------------------------------
# host tier -- the graph
# --------------------------------------------------------------------------
@needs_weights
@needs_golden
@pytest.mark.parametrize("call", [0, 5, 9])
def test_torch_reference_reproduces_the_golden(call):
    """No device. If this passes, the architecture is right.

    The residual `max|d|` here is the *export*, not the graph: `--fp16` stores any
    tensor over 65536 elements as float16, which is every conv and linear in this
    UNet. fp16 carries a 10-bit mantissa against bfloat16's 8, so it is strictly
    finer than what the device will hold and costs nothing downstream.
    """
    from models.demos.cosyvoice.tt.flow.reference import conditional_decoder

    w = _flat_weights()
    g = load_golden("flow.cfm_estimator")
    with torch.no_grad():
        got = conditional_decoder(
            w,
            as_torch(g[f"call{call}.in_x"]),
            as_torch(g["call0.in_mu"]),
            as_torch(g[f"call{call}.in_t"]),
            as_torch(g["call0.in_spks"]),
            as_torch(g["call0.in_cond"]),
        )
    want = as_torch(g[f"call{call}.out_dphi_dt"])
    p = pcc(got, want)
    print(f"\n  torch reference call{call}: PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape
    assert p >= 0.9999, p


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
@needs_weights
@needs_golden
@needs_l1_small
@pytest.mark.parametrize("call", [0, 9])
def test_device_estimator_matches_golden(device, call):
    """The whole UNet on device: 16 ResnetBlock1D and 64 BasicTransformerBlock.

    Inputs are transposed to channels-last once on the way in and the output once
    on the way out -- the 32 `rearrange`s the reference does between its
    convolutions and its transformers do not exist here.
    """
    import ttnn
    from models.demos.cosyvoice.tt.flow.estimator import TtConditionalDecoder
    from models.demos.cosyvoice.tt.weights import WeightBag

    g = load_golden("flow.cfm_estimator")
    x = as_torch(g[f"call{call}.in_x"])  # [2, 80, T]
    mu = as_torch(g["call0.in_mu"])
    t = as_torch(g[f"call{call}.in_t"])  # [2]
    spks = as_torch(g["call0.in_spks"])  # [2, 80]
    cond = as_torch(g["call0.in_cond"])
    want = as_torch(g[f"call{call}.out_dphi_dt"])
    b, _, length = x.shape

    bag = WeightBag.load(FLOW_WEIGHTS).sub(PREFIX)
    model = TtConditionalDecoder(device, bag)

    def cl(v):  # [B, C, T] -> [B, T, C]
        return ttnn.from_torch(
            v.permute(0, 2, 1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    out = model(
        cl(x),
        cl(mu),
        ttnn.from_torch(t.reshape(b, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        spks=ttnn.from_torch(spks.reshape(b, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        cond=cl(cond),
        batch=b,
    )
    got = ttnn.to_torch(out).float().permute(0, 2, 1)

    p = pcc(got, want)
    print(f"\n  CFM estimator call{call}, T={length}, batch={b}")
    print(f"  PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.99, p
