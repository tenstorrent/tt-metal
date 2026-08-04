# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8c: the MiniMax-H3 visual VAE's 36-layer ViT decoder.

The production decode shape is fixed by tiling: one call is always a
``(1, 24, 7, 16, 16)`` latent tile, i.e. ``7*16*16 = 1792`` patches plus a 5-token
suffix, so every test here runs at that shape.

Ordered cheapest-first, and the first two need no device at all:

* **RoPE**, host: the tables must be bit-exact against the reference module, and the
  permuted ``alt_complex_rotate90`` form must reproduce the reference rotation exactly.
  This is the decoder's riskiest detail -- only 48 of each head's 64 lanes rotate, and
  the pairing is ``(i, i+24)``, so the usual full-width RoPE op is simply wrong here.
* **swiglu half order**, host: the checkpoint's ``ff.net.0.proj`` packs ``[value; gate]``
  and tt_dit's swiglu wants the same, so no swap is needed -- unlike the H3 *DiT*, where
  a recorded amendment says the halves must be swapped. Applying that amendment here
  would corrupt every FFN, so the order is asserted rather than assumed.
* **one block** (which is where a missing ``scale1``/``scale2`` shows
  up as PCC near zero), then the **full 36 layers**.
"""

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3TransformerBlock, MiniMaxH3ViTDecoder3d, unpatchify
from ....models.vae.minimax_h3.rope_minimax_h3 import (
    head_lane_permutation,
    permuted_rotate,
    position_grid,
    reference_rotate,
    rope_tables,
)
from ....utils.check import assert_quality

SINGLE_DEVICE = [pytest.param((1, 1), {}, id="single_device")]

# The production decode unit: one 256x256 pixel tile over one 7-latent-frame chunk.
LATENT_FRAMES, LATENT_H, LATENT_W = 7, 16, 16
NUM_PATCHES = LATENT_FRAMES * LATENT_H * LATENT_W
LATENT_CHANNELS = 24
DIM, NUM_HEADS, HEAD_DIM = 2048, 32, 64
NUM_REGISTER_TOKENS = 4
NUM_SUFFIX_TOKENS = NUM_REGISTER_TOKENS + 1
EPS = 1e-5
ROPE_THETA, ROPE_DIM_RATIO = 100.0, 0.75


def _reference(name):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    attribute = getattr(ref, name, None)
    if attribute is None:
        pytest.skip(f"{name} missing -- diffusers is not at the pinned MiniMax-H3 commit")
    return attribute


def _reference_rope(num_suffix_tokens: int):
    """The reference module's own ``(cos, sin)`` for the production latent shape."""
    rope_cls = _reference("MiniMaxH3VideoRotaryPosEmbed")
    module = rope_cls(int(HEAD_DIM * ROPE_DIM_RATIO), theta=ROPE_THETA)
    positions = position_grid(LATENT_FRAMES, LATENT_H, LATENT_W)
    positions = torch.cat([positions, positions.new_zeros((num_suffix_tokens, 3))], dim=0).unsqueeze(0)
    cos, sin = module(positions)
    return cos[0, :, 0, :], sin[0, :, 0, :]


def test_rope_tables_are_bit_exact():
    """Host-only: our cos/sin must equal the reference module's exactly."""
    reference_cos, reference_sin = _reference_rope(NUM_SUFFIX_TOKENS)
    cos, sin = rope_tables(
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_suffix_tokens=NUM_SUFFIX_TOKENS,
        attention_head_dim=HEAD_DIM,
        rope_dim_ratio=ROPE_DIM_RATIO,
        theta=ROPE_THETA,
        permuted=False,
    )
    assert cos.shape == reference_cos.shape, f"{tuple(cos.shape)} != {tuple(reference_cos.shape)}"
    assert torch.equal(cos, reference_cos), f"cos differs by {(cos - reference_cos).abs().max()}"
    assert torch.equal(sin, reference_sin), f"sin differs by {(sin - reference_sin).abs().max()}"


def test_permuted_rope_matches_reference_rotation():
    """Host-only: lane permute + rot90 == the reference's half-split rotation.

    Also pins the two properties that make the no-slice trick valid: the pass-through
    lanes are untouched, and the suffix rows are the identity.
    """
    reference_cos, reference_sin = _reference_rope(NUM_SUFFIX_TOKENS)
    torch.manual_seed(0)
    total = NUM_PATCHES + NUM_SUFFIX_TOKENS
    x = torch.randn(1, total, NUM_HEADS, HEAD_DIM)

    expected = reference_rotate(x, reference_cos.unsqueeze(1), reference_sin.unsqueeze(1))

    permutation = head_lane_permutation(HEAD_DIM, ROPE_DIM_RATIO)
    cos, sin = rope_tables(
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_suffix_tokens=NUM_SUFFIX_TOKENS,
        attention_head_dim=HEAD_DIM,
        rope_dim_ratio=ROPE_DIM_RATIO,
        theta=ROPE_THETA,
        permuted=True,
    )
    rotated = permuted_rotate(x.index_select(-1, permutation), cos.unsqueeze(1), sin.unsqueeze(1))
    actual = rotated.index_select(-1, torch.argsort(permutation))

    assert torch.equal(actual, expected), f"rotation differs by {(actual - expected).abs().max()}"

    rotary_dim = reference_cos.shape[-1]
    assert torch.equal(actual[..., rotary_dim:], x[..., rotary_dim:]), "pass-through lanes were modified"
    assert torch.equal(actual[:, -NUM_SUFFIX_TOKENS:], x[:, -NUM_SUFFIX_TOKENS:]), "suffix rows are not identity"


def test_swiglu_half_order_needs_no_swap():
    """Host-only: the checkpoint packs ``[value; gate]``, matching tt_dit -- no swap.

    A recorded amendment says the H3 *DiT*'s ``fc1`` halves must be swapped. That came from
    the raw MiniMax layout, not the diffusers-converted one, so applying it to the VAE
    decoder would silently corrupt every FFN. Exactly one of the two orders must match.
    """
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.attention import FeedForward

    torch.manual_seed(1)
    dim, mult = 256, 4
    reference = FeedForward(dim, mult=mult, activation_fn="swiglu", bias=True).eval()
    x = torch.randn(2, 8, dim)
    with torch.no_grad():
        expected = reference(x)

    projected = reference.net[0].proj(x)
    first, second = projected.chunk(2, dim=-1)
    value_times_silu_gate = reference.net[2](first * torch.nn.functional.silu(second))
    gate_times_silu_value = reference.net[2](second * torch.nn.functional.silu(first))

    assert torch.allclose(
        value_times_silu_gate, expected, atol=1e-5
    ), "first half is not the value: tt_dit's [value; gate] convention would be wrong here"
    assert not torch.allclose(
        gate_times_silu_value, expected, atol=1e-5
    ), "both orders matched, so this test cannot detect a swap"


def _to_device(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_transformer_block(mesh_device):
    """One block: LayerScale, the weight-only RMS norms, and the SwiGLU FFN together.

    A missing or mis-shaped ``scale1``/``scale2`` shows up here as PCC near zero rather
    than as a subtle drift, which is exactly why this is a separate gate from attention.
    """
    block_cls = _reference("MiniMaxH3VideoTransformerBlock")
    torch.manual_seed(3)
    total = NUM_PATCHES + NUM_SUFFIX_TOKENS

    reference = block_cls(dim=DIM, heads=NUM_HEADS, dim_head=HEAD_DIM, ffn_mult=4, eps=EPS, bias=True).eval()
    # scale1/scale2 initialise to zeros, which would make the block the identity and hide
    # any error in the attention or FFN. Give them realistic non-zero values.
    with torch.no_grad():
        reference.scale1.normal_(0, 0.1)
        reference.scale2.normal_(0, 0.1)

    x = torch.randn(1, total, DIM)
    reference_cos, reference_sin = _reference_rope(NUM_SUFFIX_TOKENS)
    with torch.no_grad():
        expected = reference(x, (reference_cos.view(1, total, 1, -1), reference_sin.view(1, total, 1, -1)))

    tt_block = MiniMaxH3TransformerBlock(
        DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM, ffn_mult=4, eps=EPS, mesh_device=mesh_device
    )
    tt_block.load_torch_state_dict(dict(reference.state_dict()))

    cos, sin = rope_tables(
        LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_suffix_tokens=NUM_SUFFIX_TOKENS,
        attention_head_dim=HEAD_DIM,
        rope_dim_ratio=ROPE_DIM_RATIO,
        theta=ROPE_THETA,
        permuted=True,
    )
    actual = ttnn.to_torch(
        tt_block(
            _to_device(x, mesh_device),
            _to_device(cos.view(1, 1, total, HEAD_DIM), mesh_device),
            _to_device(sin.view(1, 1, total, HEAD_DIM), mesh_device),
        )
    ).float()

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.998)


@pytest.mark.parametrize("num_layers", [pytest.param(36, id="full_36_layers")])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decoder(mesh_device, num_layers):
    """The decoder on the production latent tile, against the reference decoder.

    Covers ``proj_in``, the fused suffix constant, the RoPE constants, ``norm_out``,
    ``proj_out`` and the unpatchify tail, at the full 2.4 B parameters.
    """
    decoder_cls = _reference("MiniMaxH3VideoViTDecoder3d")
    torch.manual_seed(4)

    reference = decoder_cls(
        in_channels=LATENT_CHANNELS,
        out_channels=3,
        patch_size=16,
        patch_size_t=4,
        num_layers=num_layers,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        num_register_tokens=NUM_REGISTER_TOKENS,
        ffn_mult=4,
        rope_theta=ROPE_THETA,
        rope_dim_ratio=ROPE_DIM_RATIO,
        norm_eps=EPS,
    ).eval()
    with torch.no_grad():
        for block in reference.transformer_blocks:
            block.scale1.normal_(0, 0.1)
            block.scale2.normal_(0, 0.1)

    z = torch.randn(1, LATENT_CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W)
    with torch.no_grad():
        expected = reference(z)

    tt_decoder = MiniMaxH3ViTDecoder3d(
        num_frames=LATENT_FRAMES,
        height=LATENT_H,
        width=LATENT_W,
        in_channels=LATENT_CHANNELS,
        out_channels=3,
        num_layers=num_layers,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_register_tokens=NUM_REGISTER_TOKENS,
        rope_theta=ROPE_THETA,
        rope_dim_ratio=ROPE_DIM_RATIO,
        eps=EPS,
        mesh_device=mesh_device,
    )
    tt_decoder.load_torch_state_dict(dict(reference.state_dict()))

    # The caller owns the latent-to-token flatten, mirroring the reference's own permute.
    tokens = z.permute(0, 2, 3, 4, 1).reshape(1, NUM_PATCHES, LATENT_CHANNELS)
    out_tokens = ttnn.to_torch(tt_decoder(_to_device(tokens, mesh_device))).float()
    actual = unpatchify(out_tokens, num_frames=LATENT_FRAMES, height=LATENT_H, width=LATENT_W, out_channels=3)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99)
