# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 DiT port: host-side exactness checks and small-config PCC against diffusers.

The host tests need no device and no checkpoint. The device tests build a tiny config with
random weights, copy them into a diffusers block, and compare -- covering the pieces of the
port that have no in-tree reference: ConcatLastDim's hand-written backward, the tanh gelu
routing, gate broadcasting, and ttnn.chunk on a 6-wide dim.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models.wan2_2 import (
    _CONV3D_ALIGNMENT,
    WanConfig,
    WanTransformer3D,
    WanTransformerBlock,
    assert_conv3d_patch_embed_is_frozen,
    build_rope_params,
    build_tables,
    conv3d_patch_embed,
    grid_size,
    patch_features,
    patchify,
    patchify_output_order,
    prepare_conv3d_patch_weight,
    timestep_features,
    to_ndhwc,
    to_ttml_name,
    unpatchify,
)
from ttml.models.wan2_2.patch_embed import conv3d_weight_to_linear
from ttml.models.wan2_2.weights import to_ttml_array

torch = pytest.importorskip("torch")

# Small enough to run in seconds; head_dim 32 keeps a whole tile, S=64 is two tiles.
SMALL = WanConfig(
    dim=64,
    ffn_dim=128,
    num_layers=1,
    num_heads=2,
    patch_size=(1, 2, 2),
    in_channels=4,
    out_channels=4,
    text_dim=64,
    freq_dim=32,
    cross_attn_norm=True,
    eps=1e-6,
    rope_max_seq_len=128,
)
LATENT = (1, SMALL.in_channels, 1, 16, 16)
TEXT_LEN = 32


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def to_ttml(arr, dtype=ttnn.bfloat16, requires_grad: bool = False):
    """from_numpy defaults requires_grad to False, so gradient checks must opt in."""
    tensor = ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), ttnn.Layout.TILE, dtype)
    if requires_grad:
        tensor.set_requires_grad(True)
    return tensor


def backward_from(tensor) -> None:
    """Reduce to a scalar first: backward() on a non-scalar seeds no gradient."""
    ttml.ops.unary.mean(tensor).backward(False)


def np_of(tensor) -> np.ndarray:
    """float32 numpy from either a ttml tensor or a raw ttnn one.

    get_grad() yields a raw ttnn tensor, whose to_numpy takes no dtype; a bf16 tensor with
    no fp32 master copy also cannot be read by numpy directly. Discriminate the same way
    ttml.autograd.function does, via get_requires_grad.
    """
    if hasattr(tensor, "get_requires_grad"):
        return np.asarray(tensor.to_numpy(new_type=ttnn.float32))
    return ttnn.to_torch(tensor).float().numpy()


def grad_of(tensor, label: str = "tensor") -> np.ndarray:
    """get_grad() returns nullptr when uninitialised and to_numpy() on that segfaults."""
    assert tensor.is_grad_initialized(), f"{label} received no gradient"
    return np_of(tensor.get_grad())


# ---------------------------------------------------------------------------
# host only -- exact, no device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent", [(1, 16, 1, 64, 64), (1, 16, 4, 90, 160), (1, 16, 1, 32, 32)])
def test_rope_tables_match_diffusers(latent):
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    head_dim, patch, max_seq = 128, (1, 2, 2), 1024
    cos, sin = build_tables(head_dim=head_dim, patch_size=patch, latent_shape=latent, max_seq_len=max_seq)

    reference = WanRotaryPosEmbed(head_dim, patch, max_seq, 10000.0)
    with torch.no_grad():
        ref_cos, ref_sin = reference(torch.zeros(latent))
    ref_cos = ref_cos.permute(0, 2, 1, 3).float().numpy()
    ref_sin = ref_sin.permute(0, 2, 1, 3).float().numpy()

    assert cos.shape == ref_cos.shape
    np.testing.assert_allclose(cos, ref_cos, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(sin, ref_sin, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("timesteps", [[0.0], [999.0], [875.0, 1.0, 500.0]])
def test_timestep_features_match_diffusers(timesteps):
    from diffusers.models.embeddings import get_timestep_embedding

    mine = timestep_features(timesteps, 256)
    reference = get_timestep_embedding(
        torch.tensor(timesteps, dtype=torch.float32), 256, flip_sin_to_cos=True, downscale_freq_shift=0
    ).numpy()
    # Large timesteps make sin/cos argument-sensitive in fp32; far below bf16 resolution.
    np.testing.assert_allclose(mine, reference, atol=1e-4)


def test_patchify_equals_conv3d():
    """Wan's patch embed is a Conv3d with stride == kernel, i.e. a linear map over patches."""
    torch.manual_seed(0)
    patch = (1, 2, 2)
    latent = torch.randn(2, 16, 1, 8, 8, dtype=torch.float64)
    conv = torch.nn.Conv3d(16, 32, kernel_size=patch, stride=patch, dtype=torch.float64)
    with torch.no_grad():
        reference = conv(latent).flatten(2).transpose(1, 2).numpy()

    weight = conv3d_weight_to_linear(conv.weight.detach().numpy())[0, 0]
    mine = patchify(latent.numpy(), patch) @ weight.T + conv.bias.detach().numpy()
    np.testing.assert_allclose(mine[:, 0], reference, atol=1e-12)


def test_to_ndhwc_is_a_pure_permutation():
    """conv3d takes the raw latent as (B, F, H, W, C) instead of patchify's tokens."""
    rng = np.random.default_rng(2)
    latent = rng.standard_normal((2, 4, 1, 8, 8)).astype(np.float32)
    ndhwc = to_ndhwc(latent)
    assert ndhwc.shape == (2, 1, 8, 8, 4)
    np.testing.assert_allclose(ndhwc.transpose(0, 4, 1, 2, 3), latent, atol=0.0)


def test_conv3d_alignment_divides_wan_in_channels():
    """16, not ttnn's default 32, so Wan's 16-channel latents need no zero-padded upload."""
    assert WanConfig().in_channels % _CONV3D_ALIGNMENT == 0


def test_unpatchify_inverts_output_order():
    rng = np.random.default_rng(0)
    patch = (1, 2, 2)
    latent = rng.standard_normal((2, 4, 1, 8, 8)).astype(np.float32)
    grid = grid_size(latent.shape, patch)
    tokens = patchify_output_order(latent, patch)
    np.testing.assert_allclose(unpatchify(tokens, patch, grid, 4), latent, atol=0.0)


def test_patch_orders_differ_but_permute_the_same_values():
    rng = np.random.default_rng(1)
    latent = rng.standard_normal((1, 4, 1, 8, 8)).astype(np.float32)
    channel_major = patchify(latent, (1, 2, 2))
    channel_minor = patchify_output_order(latent, (1, 2, 2))
    assert not np.allclose(channel_major, channel_minor)
    np.testing.assert_allclose(np.sort(channel_major, axis=None), np.sort(channel_minor, axis=None))


def test_checkpoint_name_mapping():
    assert to_ttml_name("patch_embedding.weight") == "patch_embed.weight"
    assert to_ttml_name("blocks.0.attn1.to_out.0.weight") == "blocks.0.attn1.to_out.weight"
    assert to_ttml_name("blocks.7.ffn.net.0.proj.bias") == "blocks.7.ffn.ff1.bias"
    assert to_ttml_name("blocks.7.ffn.net.2.weight") == "blocks.7.ffn.ff2.weight"
    for unchanged in ("blocks.0.attn1.norm_q.weight", "blocks.0.norm2.bias", "scale_shift_table", "proj_out.weight"):
        assert to_ttml_name(unchanged) == unchanged


def test_to_ttml_array_rejects_shape_mismatch(expect_error):
    with expect_error(ValueError, "model wants"):
        to_ttml_array("proj_out.weight", np.zeros((8, 4), np.float32), (1, 1, 8, 5))


# ---------------------------------------------------------------------------
# device -- the pieces with no in-tree reference
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_gelu_tanh_matches_torch_both_directions():
    from ttml.models.wan2_2.transformer import GeluTanh

    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, 1, 32, 64)).astype(np.float32) * 2.0

    x = to_ttml(data, requires_grad=True)
    out = GeluTanh.apply(x)
    backward_from(out)

    ref_in = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    ref_out = torch.nn.functional.gelu(ref_in, approximate="tanh")
    ref_out.mean().backward()

    assert pcc(np_of(out), ref_out.detach().numpy()) > 0.999
    assert pcc(grad_of(x, "gelu input"), ref_in.grad.numpy()) > 0.999


@pytest.mark.requires_device
def test_split_heads_matches_torch_both_directions():
    """The only hand-written backward in the port. Failure is silent, so check it directly."""
    from ttml.models.wan2_2.attention import SplitHeads

    num_heads, head_dim, seq = 2, 32, 32
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, 1, seq, num_heads * head_dim)).astype(np.float32)
    # Weight the output so a wrong permutation shows up as a wrong gradient, not just a shape.
    weights = rng.standard_normal((1, num_heads, seq, head_dim)).astype(np.float32)

    x = to_ttml(data, requires_grad=True)
    heads = SplitHeads.apply(x, num_heads)
    assert tuple(heads.shape()) == (1, num_heads, seq, head_dim)

    reference_in = torch.tensor(data, requires_grad=True)
    reference = reference_in.reshape(1, seq, num_heads, head_dim).permute(0, 2, 1, 3)
    assert pcc(np_of(heads), reference.detach().numpy()) > 0.999

    backward_from(heads * to_ttml(weights))
    (reference * torch.tensor(weights)).mean().backward()
    assert pcc(grad_of(x, "split-heads input"), reference_in.grad.numpy()) > 0.999


@pytest.mark.requires_device
def test_chunk_six_wide_tile_dim():
    """The modulation tensor is 6 wide on a tiled axis; 6 is not a multiple of 32."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, 1, 6, 64)).astype(np.float32)
    value = to_ttml(data).get_value()
    chunks = ttnn.chunk(value, 6, dim=2)
    assert len(chunks) == 6
    for index, chunk in enumerate(chunks):
        got = np_of(ttml.autograd.create_tensor(chunk, False))
        np.testing.assert_allclose(np.asarray(got).reshape(-1)[:64], data[0, 0, index], atol=1e-2)


@pytest.mark.requires_device
def test_gate_broadcasts_over_sequence():
    """gate is (B,1,1,dim) and multiplies a (B,1,S,dim) activation."""
    rng = np.random.default_rng(0)
    activation = rng.standard_normal((1, 1, 64, 64)).astype(np.float32)
    gate = rng.standard_normal((1, 1, 1, 64)).astype(np.float32)
    product = to_ttml(activation) * to_ttml(gate)
    assert pcc(np_of(product), activation * gate) > 0.999


# ---------------------------------------------------------------------------
# device -- block against diffusers at a small config
# ---------------------------------------------------------------------------


def _diffusers_block():
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock as TorchBlock

    torch.manual_seed(0)
    block = TorchBlock(
        dim=SMALL.dim,
        ffn_dim=SMALL.ffn_dim,
        num_heads=SMALL.num_heads,
        qk_norm="rms_norm_across_heads",
        cross_attn_norm=SMALL.cross_attn_norm,
        eps=SMALL.eps,
    ).eval()
    return block


def _copy_weights(torch_block, ttml_block) -> int:
    params = dict(ttml_block.named_parameters())
    unfilled, unmapped = set(params), []
    for key, tensor in torch_block.state_dict().items():
        name = to_ttml_name(key)
        target = params.get(name)
        if target is None:
            unmapped.append(f"{key} -> {name}")
            continue
        value = to_ttml_array(name, tensor.float().numpy(), tuple(target.shape()))
        target.set_value(to_ttml(value).get_value())
        unfilled.discard(name)
    assert not unmapped, f"checkpoint keys with no destination: {unmapped}"
    assert not unfilled, f"parameters never filled: {sorted(unfilled)}"
    return len(params)


@pytest.mark.requires_device
def test_block_forward_pcc_against_diffusers():
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    torch_block = _diffusers_block()
    ttml_block = WanTransformerBlock(SMALL)
    _copy_weights(torch_block, ttml_block)

    grid = grid_size(LATENT, SMALL.patch_size)
    seq_len = grid[0] * grid[1] * grid[2]
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, seq_len, SMALL.dim)).astype(np.float32)
    context = rng.standard_normal((1, TEXT_LEN, SMALL.dim)).astype(np.float32)
    temb = rng.standard_normal((1, 6, SMALL.dim)).astype(np.float32)

    rope = WanRotaryPosEmbed(SMALL.head_dim, SMALL.patch_size, SMALL.rope_max_seq_len, 10000.0)
    with torch.no_grad():
        rotary = rope(torch.zeros(LATENT))
        reference = torch_block(torch.tensor(hidden), torch.tensor(context), torch.tensor(temb), rotary).numpy()

    rope_params = build_rope_params(
        head_dim=SMALL.head_dim,
        patch_size=SMALL.patch_size,
        latent_shape=LATENT,
        max_seq_len=SMALL.rope_max_seq_len,
    )
    out = ttml_block(
        to_ttml(hidden[:, None]),
        None,
        to_ttml(context[:, None]),
        to_ttml(temb[:, None]),
        rope_params,
    )
    # bf16 activations plus diffusers' fp32 layernorm put the floor here, not correctness.
    assert pcc(np_of(out), reference) > 0.99


@pytest.mark.requires_device
def test_block_backward_reaches_adapter_inputs():
    """A gradient must arrive at every projection LoRA would wrap, including to_k and to_v."""
    ttml_block = WanTransformerBlock(SMALL)
    grid = grid_size(LATENT, SMALL.patch_size)
    seq_len = grid[0] * grid[1] * grid[2]
    rng = np.random.default_rng(0)

    rope_params = build_rope_params(
        head_dim=SMALL.head_dim,
        patch_size=SMALL.patch_size,
        latent_shape=LATENT,
        max_seq_len=SMALL.rope_max_seq_len,
    )
    out = ttml_block(
        to_ttml(rng.standard_normal((1, 1, seq_len, SMALL.dim)).astype(np.float32)),
        None,
        to_ttml(rng.standard_normal((1, 1, TEXT_LEN, SMALL.dim)).astype(np.float32)),
        to_ttml(rng.standard_normal((1, 1, 6, SMALL.dim)).astype(np.float32)),
        rope_params,
    )
    backward_from(out)

    params = dict(ttml_block.named_parameters())
    for name in (
        "attn1.to_q.weight",
        "attn1.to_k.weight",
        "attn1.to_v.weight",
        "attn1.to_out.weight",
        "attn2.to_k.weight",
        "attn2.to_v.weight",
        "ffn.ff1.weight",
        "ffn.ff2.weight",
    ):
        assert name in params, f"{name} missing; LoRA targets would not match"
        grad = grad_of(params[name], name)
        assert np.isfinite(grad).all(), f"{name} gradient has non-finite values"
        assert np.abs(grad).max() > 0.0, f"{name} received no gradient"


@pytest.mark.requires_device
@pytest.mark.parametrize("temb_scale", [1.0, 0.1])
def test_block_stage_by_stage_pcc(temb_scale):
    """Locate divergence: report PCC after each sub-step of the block.

    temb_scale probes whether random modulation is the cause: gamma = 1 + scale with
    scale ~ N(0,1) can land near zero, which amplifies bf16 error. Real timestep
    embeddings are far better behaved than random ones.
    """
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    torch_block = _diffusers_block()
    ttml_block = WanTransformerBlock(SMALL)
    _copy_weights(torch_block, ttml_block)

    grid = grid_size(LATENT, SMALL.patch_size)
    seq_len = grid[0] * grid[1] * grid[2]
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, seq_len, SMALL.dim)).astype(np.float32)
    context = rng.standard_normal((1, TEXT_LEN, SMALL.dim)).astype(np.float32)
    temb = (rng.standard_normal((1, 6, SMALL.dim)) * temb_scale).astype(np.float32)

    rope = WanRotaryPosEmbed(SMALL.head_dim, SMALL.patch_size, SMALL.rope_max_seq_len, 10000.0)
    rope_params = build_rope_params(
        head_dim=SMALL.head_dim,
        patch_size=SMALL.patch_size,
        latent_shape=LATENT,
        max_seq_len=SMALL.rope_max_seq_len,
    )

    stages = {}
    with torch.no_grad():
        rotary = rope(torch.zeros(LATENT))
        t_hidden = torch.tensor(hidden)
        t_context = torch.tensor(context)
        shift, scale, gate, c_shift, c_scale, c_gate = (
            torch_block.scale_shift_table + torch.tensor(temb).float()
        ).chunk(6, dim=1)

        t_n1 = torch_block.norm1(t_hidden.float()) * (1 + scale) + shift
        t_a1 = torch_block.attn1(t_n1, None, None, rotary)
        t_res1 = t_hidden.float() + t_a1 * gate
        t_n2 = torch_block.norm2(t_res1.float())
        t_a2 = torch_block.attn2(t_n2, t_context, None, None)
        t_res2 = t_res1 + t_a2
        t_n3 = torch_block.norm3(t_res2.float()) * (1 + c_scale) + c_shift
        t_ff = torch_block.ffn(t_n3)
        t_out = t_res2.float() + t_ff.float() * c_gate

    x = to_ttml(hidden[:, None])
    prompt = to_ttml(context[:, None])
    gamma1, beta1, gate1, gamma3, beta3, gate3 = ttml_block._modulation(to_ttml(temb[:, None]))

    n1 = ttml.ops.layernorm.layernorm(x, gamma1, beta1)
    stages["norm1"] = pcc(np_of(n1), t_n1.numpy())

    a1 = ttml_block.attn1(n1, rope_params=rope_params)
    stages["attn1"] = pcc(np_of(a1), t_a1.numpy())

    res1 = x + a1 * gate1
    stages["residual1"] = pcc(np_of(res1), t_res1.numpy())

    n2 = ttml_block.norm2(res1)
    stages["norm2"] = pcc(np_of(n2), t_n2.numpy())

    a2 = ttml_block.attn2(n2, context=prompt)
    stages["attn2"] = pcc(np_of(a2), t_a2.numpy())

    res2 = res1 + a2
    n3 = ttml.ops.layernorm.layernorm(res2, gamma3, beta3)
    stages["norm3"] = pcc(np_of(n3), t_n3.numpy())

    ff = ttml_block.ffn(n3)
    stages["ffn"] = pcc(np_of(ff), t_ff.numpy())

    out = res2 + ff * gate3
    stages["output"] = pcc(np_of(out), t_out.numpy())

    print(f"\ntemb_scale={temb_scale}")
    for name, value in stages.items():
        print(f"  {name:10s} pcc = {value:.6f}")

    for name, value in stages.items():
        assert value > 0.99, f"divergence starts at {name}: pcc {value:.6f} (see printed stages)"


@pytest.mark.requires_device
def test_rope_op_uses_interleaved_convention():
    """Which pairing does the rope kernel implement: adjacent pairs or split halves?

    Built with ttml's own build_rope_params so this isolates the kernel's convention from
    our 3D table construction. Wan (and trans_mat) assume adjacent pairs.
    """
    heads, head_dim, seq = 2, 32, 32
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, heads, seq, head_dim)).astype(np.float32)

    params = ttml.ops.rope.build_rope_params(sequence_length=seq, head_dim=head_dim, theta=10000.0)
    out = np_of(ttml.ops.rope.rope(to_ttml(x), params, 0))

    # Rebuild the same angles gen_freqs uses: inv_freq[i] = theta ** -(2*floor(i/2)/D)
    pair = (np.arange(head_dim) // 2) * (2.0 / head_dim)
    angles = np.arange(seq)[:, None] * (1.0 / np.power(10000.0, pair))[None, :]
    cos, sin = np.cos(angles), np.sin(angles)

    interleaved = np.empty_like(x)
    interleaved[..., 0::2] = x[..., 0::2] * cos[..., 0::2] - x[..., 1::2] * sin[..., 0::2]
    interleaved[..., 1::2] = x[..., 0::2] * sin[..., 0::2] + x[..., 1::2] * cos[..., 0::2]

    half = head_dim // 2
    split = np.concatenate(
        [
            x[..., :half] * cos[..., :half] - x[..., half:] * sin[..., :half],
            x[..., half:] * cos[..., :half] + x[..., :half] * sin[..., :half],
        ],
        axis=-1,
    )

    pcc_interleaved, pcc_split = pcc(out, interleaved), pcc(out, split)
    print(f"\nrope kernel vs interleaved pairing = {pcc_interleaved:.6f}")
    print(f"rope kernel vs split halves      = {pcc_split:.6f}")
    assert pcc_interleaved > 0.99, (
        f"the rope kernel does not use adjacent-pair rotation "
        f"(interleaved {pcc_interleaved:.4f} vs split {pcc_split:.4f}); "
        f"the 3D tables in rope.py must be reordered to match"
    )


# ---------------------------------------------------------------------------
# device -- the conv3d patch embed
# ---------------------------------------------------------------------------

# SMALL's in_channels=4 is not a multiple of _CONV3D_ALIGNMENT, so the conv3d tests need
# their own config.
CONV_SMALL = WanConfig(
    dim=64,
    ffn_dim=128,
    num_layers=1,
    num_heads=2,
    patch_size=(1, 2, 2),
    in_channels=16,
    out_channels=16,
    text_dim=64,
    freq_dim=32,
    cross_attn_norm=True,
    eps=1e-6,
    rope_max_seq_len=128,
)
CONV_LATENT = (1, CONV_SMALL.in_channels, 1, 8, 8)


def _conv_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    latent = (rng.standard_normal(CONV_LATENT) * 0.5).astype(np.float32)
    text = (rng.standard_normal((1, 1, 8, CONV_SMALL.text_dim)) * 0.5).astype(np.float32)
    rope = build_rope_params(
        head_dim=CONV_SMALL.head_dim,
        patch_size=CONV_SMALL.patch_size,
        latent_shape=CONV_LATENT,
        max_seq_len=CONV_SMALL.rope_max_seq_len,
    )
    return latent, text, rope


def _to_ttml_ndhwc(latent: np.ndarray):
    """conv3d wants its activation row-major, unlike every other input in these tests."""
    return ttml.autograd.Tensor.from_numpy(to_ndhwc(latent), ttnn.Layout.ROW_MAJOR, ttnn.bfloat16)


@pytest.mark.requires_device
def test_conv3d_patch_embed_matches_linear_path():
    """conv3d over the raw latent reproduces patchify + the linear patch embed it replaces.

    Reference is the float64 linear path, already pinned to torch.nn.Conv3d by
    test_patchify_equals_conv3d. Tolerance is bf16's, not the op's.
    """
    rng = np.random.default_rng(0)
    patch = (1, 2, 2)
    c_in, dim = _CONV3D_ALIGNMENT, 64
    latent = (rng.standard_normal((1, c_in, 1, 8, 8)) * 0.5).astype(np.float32)
    weight = (rng.standard_normal((dim, c_in, *patch)) * 0.1).astype(np.float32)
    bias = (rng.standard_normal(dim) * 0.1).astype(np.float32)

    reference = patchify(latent, patch).astype(np.float64) @ conv3d_weight_to_linear(weight)[0, 0].astype(
        np.float64
    ).T + bias.astype(np.float64)

    device = ttml.autograd.AutoContext.get_instance().get_device()
    weight_host = ttnn.from_torch(torch.from_numpy(weight), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    prepared = prepare_conv3d_patch_weight(weight_host, device)

    got = conv3d_patch_embed(
        _to_ttml_ndhwc(latent), prepared, patch, dim, bias=to_ttml(bias.reshape(1, 1, 1, dim)).get_value()
    )
    mine = np_of(got)

    assert not got.get_requires_grad(), "the conv3d patch embed must stay a graph leaf"
    assert mine.shape == reference.shape, f"{mine.shape} != {reference.shape}"
    assert pcc(mine, reference) > 0.999, f"PCC {pcc(mine, reference):.6f}"


@pytest.mark.requires_device
def test_conv3d_patch_embed_rejects_unaligned_channels(expect_error):
    """ttnn reports this as a confusing patch-size mismatch; the wrapper names the cause."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    patch, c_in, dim = (1, 2, 2), _CONV3D_ALIGNMENT // 2, 64
    weight = np.zeros((dim, c_in, *patch), np.float32)
    weight_host = ttnn.from_torch(torch.from_numpy(weight), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    latent_dev = _to_ttml_ndhwc(np.zeros((1, c_in, 1, 8, 8), np.float32))
    with expect_error(ValueError, "divisible by"):
        conv3d_patch_embed(latent_dev, prepare_conv3d_patch_weight(weight_host, device), patch, dim)


@pytest.mark.requires_device
def test_conv3d_patch_embed_matches_linear_end_to_end():
    """A whole expert forward is unchanged by swapping the patch embed for conv3d.

    Also pins token order: conv3d must emit f,h,w so RoPE lines up. A token-order bug leaves
    the isolated op's PCC intact and only shows up here.
    """
    latent, text, rope = _conv_inputs()
    model = WanTransformer3D(CONV_SMALL)
    model.eval()

    context = ttml.autograd.AutoContext.get_instance()
    context.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    try:
        linear = np_of(model(to_ttml(patchify(latent, CONV_SMALL.patch_size)), [500.0], to_ttml(text), rope))
        context.reset_graph()

        model.enable_conv3d_patch_embed()
        conv = np_of(model(_to_ttml_ndhwc(latent), [500.0], to_ttml(text), rope))
    finally:
        context.set_gradient_mode(ttml.autograd.GradMode.ENABLED)

    assert conv.shape == linear.shape, f"{conv.shape} != {linear.shape}"
    assert pcc(conv, linear) > 0.999, f"PCC {pcc(conv, linear):.6f}"


def test_conv3d_patch_embed_refuses_an_adapted_patch_embed(expect_error):
    """The conv3d path has no backward, so a LoRA'd patch_embed must be rejected, not run."""

    class _Adapted:
        def named_parameters(self):
            return [("patch_embed.weight", None), ("patch_embed.lora_A", None)]

    with expect_error(RuntimeError, "no conv3d backward"):
        assert_conv3d_patch_embed_is_frozen(_Adapted())


@pytest.mark.requires_device
def test_conv3d_patch_embed_survives_the_lora_wrapper():
    """Training wraps the expert in a LoraModel, so the conv3d path must work through it.

    The wrapper forwards no attribute lookups, so enable_conv3d_patch_embed() must be called
    on the inner model -- which still changes the wrapper's forward. That is what this pins.
    """
    # Inlined rather than imported from the example's utils/lora_targets.py: putting that
    # directory on sys.path would shadow train.py/pipeline.py for the whole pytest session.
    attn_targets = [r"blocks\.\d+\.attn[12]\.to_[qkv]", r"blocks\.\d+\.attn[12]\.to_out"]

    inner = WanTransformer3D(CONV_SMALL)
    wrapper = ttml.modules.LoraModel(
        inner,
        ttml.modules.LoraConfig(
            rank=8, alpha=8.0, target_modules=attn_targets, lora_dropout=0.0, use_rslora=False, verbose=False
        ),
    )
    assert not hasattr(wrapper, "enable_conv3d_patch_embed"), "wrapper gained the method; simplify train.py"

    # The frozen check must see the injected adapters through the inner model, or it is inert.
    assert any("lora" in name for name, _ in inner.named_parameters())
    assert_conv3d_patch_embed_is_frozen(inner)
    inner.enable_conv3d_patch_embed()
    assert inner.uses_conv3d_patch_embed

    latent, text, rope = _conv_inputs()
    context = ttml.autograd.AutoContext.get_instance()
    context.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    try:
        out = np_of(wrapper(_to_ttml_ndhwc(latent), [500.0], to_ttml(text), rope))
    finally:
        context.set_gradient_mode(ttml.autograd.GradMode.ENABLED)

    assert out.shape == (1, 1, 16, patch_features(CONV_SMALL.out_channels, CONV_SMALL.patch_size))
