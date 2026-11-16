# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for Wan2.2-Animate (no ttnn, no device, no weights download).

Every reformulation the TT implementation relies on is checked here against the Diffusers
reference ``WanAnimateTransformer3DModel`` (diffusers==0.38.0, the version tt-metal pins), at a
scaled-down random-init config. If one of these fails, the device tests cannot pass either.

    pytest models/tt_dit/experimental/tests/test_wan_animate_cpu.py -q
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import torch

# Import the torch-only helper by path: `models.tt_dit.experimental` pulls in ttnn-backed modules.
_SPEC = importlib.util.spec_from_file_location(
    "wan_animate_reference_math",
    pathlib.Path(__file__).resolve().parents[1] / "wan_animate" / "reference_math.py",
)
rm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rm)

diffusers = pytest.importorskip("diffusers")
from diffusers.models.transformers.transformer_wan_animate import WanAnimateTransformer3DModel  # noqa: E402

# Tiny config: TT-friendly (dim % (32 * tp=4) == 0, heads % 4 == 0, head_dim == 128 like the real model).
TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=128,
    in_channels=36,
    latent_channels=16,
    out_channels=16,
    text_dim=256,
    freq_dim=256,
    ffn_dim=1024,
    num_layers=5,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    image_dim=128,
    added_kv_proj_dim=512,
    rope_max_seq_len=1024,
    motion_encoder_size=64,
    motion_style_dim=128,
    motion_dim=20,
    motion_encoder_dim=128,
    face_encoder_hidden_dim=256,
    face_encoder_num_heads=4,
    inject_face_latents_blocks=5,
    motion_encoder_batch_size=8,
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.fixture(scope="module")
def model() -> WanAnimateTransformer3DModel:
    torch.manual_seed(0)
    m = WanAnimateTransformer3DModel(**TINY).eval()
    # Make zero-initialised parameters non-trivial so the checks exercise them.
    with torch.no_grad():
        m.face_encoder.padding_tokens.normal_()
        m.pose_patch_embedding.bias.normal_()
    return m


def test_pose_patch_embedding_fusion(model):
    torch.manual_seed(1)
    t, h, w = 5, 8, 12
    x = torch.randn(1, 36, t + 1, h, w)
    pose = torch.randn(1, 16, t, h, w)

    with torch.no_grad():
        ref = model.patch_embedding(x)
        ref[:, :, 1:] += model.pose_patch_embedding(pose)

        weight, bias = rm.fuse_pose_patch_embedding(
            model.patch_embedding.weight,
            model.patch_embedding.bias,
            model.pose_patch_embedding.weight,
            model.pose_patch_embedding.bias,
        )
        x_ext = torch.cat([x, rm.pose_conditioning(pose)], dim=1)
        out = torch.nn.functional.conv3d(x_ext, weight, bias, stride=(1, 2, 2))

    assert x_ext.shape[1] == 36 + rm.POSE_EXTRA_CHANNELS
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("face_frames", [17, 77])
def test_face_encoder_matmul_form(model, face_frames):
    torch.manual_seed(2)
    motion = torch.randn(1, face_frames, TINY["motion_encoder_dim"])
    with torch.no_grad():
        ref = model.face_encoder(motion)
        out = rm.face_encoder_matmul_reference(model.face_encoder, motion)
    # 77 face frames -> 20 latent-rate motion frames (+ the zero frame the transformer prepends = 21).
    assert ref.shape[1] == (face_frames - 1) // 4 + 1
    assert ref.shape[2] == TINY["face_encoder_num_heads"] + 1
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_causal_conv_selectors_padding_is_inert():
    # Zero-padded rows/cols must not change the logical result (the device works on tile-padded shapes).
    torch.manual_seed(3)
    conv = torch.nn.Conv1d(8, 16, 3, stride=2)
    x = torch.randn(39, 8)
    ref = conv(torch.nn.functional.pad(x.T[None], (2, 0), mode="replicate"))[0].T
    sel = rm.causal_conv_selectors(39, stride=2, rows=32, cols=64)
    x_pad = torch.cat([x, torch.randn(64 - 39, 8) * 1e3])  # garbage in padded rows
    w = rm.conv1d_weight_as_linear(conv.weight)
    y = sum((sel[k] @ x_pad) @ w[k].T for k in range(3)) + conv.bias
    torch.testing.assert_close(y[: ref.shape[0]], ref, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(("latent_frames", "h", "w"), [(6, 8, 12), (21, 6, 10)])
def test_face_adapter_as_masked_sdpa(model, latent_frames, h, w):
    torch.manual_seed(4)
    dim = TINY["num_attention_heads"] * TINY["attention_head_dim"]
    tokens_per_frame = (h // 2) * (w // 2)
    hidden = torch.randn(1, latent_frames * tokens_per_frame, dim)
    motion = torch.randn(1, latent_frames, rm.FACE_TOKENS_PER_FRAME, dim)
    motion[:, 0] = 0  # the transformer prepends a zero frame for the reference latent
    adapter = model.face_adapter[0]
    with torch.no_grad():
        ref = adapter(hidden, motion)
        out = rm.face_adapter_masked_reference(adapter, hidden, motion, tokens_per_frame=tokens_per_frame)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_face_mask_padding_rows_have_a_key():
    mask = rm.face_attention_mask(num_frames=3, tokens_per_frame=4, padded_seq_len=32)
    assert mask.shape == (1, 1, 32, 256)
    finite = torch.isfinite(mask[0, 0])
    assert finite.any(dim=-1).all(), "a fully masked row would make softmax NaN"
    assert finite.sum(dim=-1)[:12].eq(rm.FACE_TOKENS_PER_FRAME).all()


def test_image_cross_attention_with_padded_masked_keys(model):
    """attn2 with image K/V: padding 257 image tokens to 512 + additive mask is exact."""
    torch.manual_seed(5)
    dim = 512
    block = model.blocks[0]
    attn = block.attn2
    s = 48
    hidden = torch.randn(1, s, dim)
    image_ctx = torch.randn(1, rm.IMAGE_CONTEXT_TOKENS, dim)
    text_ctx = torch.randn(1, 512, dim)
    with torch.no_grad():
        ref = attn(hidden, torch.cat([image_ctx, text_ctx], dim=1))

        heads = attn.heads
        q = attn.norm_q(attn.to_q(hidden)).unflatten(-1, (heads, -1)).transpose(1, 2)
        k = attn.norm_k(attn.to_k(text_ctx)).unflatten(-1, (heads, -1)).transpose(1, 2)
        v = attn.to_v(text_ctx).unflatten(-1, (heads, -1)).transpose(1, 2)
        img = rm.pad_image_context(image_ctx)[0]
        ki = attn.norm_added_k(attn.add_k_proj(img)).unflatten(-1, (heads, -1)).transpose(1, 2)
        vi = attn.add_v_proj(img).unflatten(-1, (heads, -1)).transpose(1, 2)
        mask = rm.image_attention_mask(padded_seq_len=s)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        o = sdpa(q, k, v) + sdpa(q, ki, vi, attn_mask=mask)
        out = attn.to_out[0](o.transpose(1, 2).flatten(2))
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_full_model_reference_runs_and_is_deterministic(model):
    """One tiny forward through the Diffusers reference (the same call the device test uses)."""
    torch.manual_seed(6)
    t, h, w = 5, 8, 12
    face_frames = (t - 1) * 4 + 1
    inputs = dict(
        hidden_states=torch.randn(1, 36, t + 1, h, w),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 512, TINY["text_dim"]),
        encoder_hidden_states_image=torch.randn(1, rm.IMAGE_CONTEXT_TOKENS, TINY["image_dim"]),
        pose_hidden_states=torch.randn(1, 16, t, h, w),
        face_pixel_values=torch.rand(1, 3, face_frames, 64, 64) * 2 - 1,
        return_dict=False,
    )
    with torch.no_grad():
        a = model(**inputs)[0]
        b = model(**inputs)[0]
    assert a.shape == (1, 16, t + 1, h, w)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b)


def test_relighting_lora_merge_matches_runtime_lora(model):
    torch.manual_seed(7)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    r, dim = 8, 512
    lora = {
        "base_model.model.blocks.0.self_attn.q.lora_A.weight": torch.randn(r, dim) * 0.01,
        "base_model.model.blocks.0.self_attn.q.lora_B.weight": torch.randn(dim, r) * 0.01,
        "base_model.model.blocks.0.cross_attn.k_img.lora_A.weight": torch.randn(r, dim) * 0.01,
        "base_model.model.blocks.0.cross_attn.k_img.lora_B.weight": torch.randn(dim, r) * 0.01,
        "base_model.model.blocks.0.ffn.2.lora_A.weight": torch.randn(r, TINY["ffn_dim"]) * 0.01,
        "base_model.model.blocks.0.ffn.2.lora_B.weight": torch.randn(dim, r) * 0.01,
    }
    assert rm.merge_relighting_lora(state, lora) == 3
    x = torch.randn(3, dim)
    w0 = model.blocks[0].attn1.to_q.weight
    a = lora["base_model.model.blocks.0.self_attn.q.lora_A.weight"]
    b = lora["base_model.model.blocks.0.self_attn.q.lora_B.weight"]
    torch.testing.assert_close(x @ state["blocks.0.attn1.to_q.weight"].T, x @ w0.T + (x @ a.T) @ b.T)
    assert not torch.equal(state["blocks.0.attn2.add_k_proj.weight"], model.blocks[0].attn2.add_k_proj.weight)
    with pytest.raises(ValueError, match="unrecognized"):
        rm.map_relighting_lora({"base_model.model.face_adapter.0.q.lora_A.weight": a})


def test_relighting_lora_covers_every_target_module():
    # 40 blocks x 12 Linear targets x (A, B) = 960 tensors in the shipped adapter.
    keys = {}
    for i in range(40):
        for name in rm._LORA_MODULE_MAP:  # noqa: SLF001
            for ab in "AB":
                keys[f"base_model.model.blocks.{i}.{name}.lora_{ab}.weight"] = torch.zeros(1)
    assert len(keys) == 960
    assert len(rm.map_relighting_lora(keys)) == 480


def test_segment_math_matches_diffusers_pipeline():
    from diffusers.pipelines.wan.pipeline_wan_animate import WanAnimatePipeline

    for n in (1, 2, 30, 76, 77, 78, 153, 154, 200):
        frames = list(range(n))
        segs, pad = rm.num_segments_and_padding(n, 77, 1)
        if n > 1:
            assert rm.reflect_pad_frames(frames, n + pad) == WanAnimatePipeline.pad_video_frames(None, frames, n + pad)
        # Every segment generates 76 new frames after the first frame; the loop covers all frames.
        assert segs * 76 + 1 >= n
        assert math.isclose((n + pad - 1) / 76, segs) or n == 1


def test_i2v_mask_matches_diffusers_pipeline():
    from diffusers.pipelines.wan.pipeline_wan_animate import WanAnimatePipeline

    class _Shim:
        vae_scale_factor_temporal = 4

    ref = WanAnimatePipeline.get_i2v_mask(_Shim(), 1, 20, 6, 10, mask_len=1, device="cpu")
    torch.testing.assert_close(rm.i2v_mask(20, 6, 10, 1), ref)
    px = (torch.rand(1, 1, 77, 6, 10) > 0.5).float()
    ref = WanAnimatePipeline.get_i2v_mask(_Shim(), 1, 20, 6, 10, mask_len=0, mask_pixel_values=px, device="cpu")
    torch.testing.assert_close(rm.i2v_mask(20, 6, 10, 0, px), ref)


class _StubVAE:
    """Deterministic stand-in for AutoencoderKLWan.encode: temporal stride 4 (frame 0 kept), 8x8 avg pool,
    3 -> 16 channels by a fixed projection. Good enough to check the conditioning *assembly*."""

    dtype = torch.float32

    class config:  # noqa: N801
        z_dim = 16
        latents_mean = [0.1 * i for i in range(16)]
        latents_std = [1.0 + 0.05 * i for i in range(16)]

    def __init__(self):
        g = torch.Generator().manual_seed(11)
        self.proj = torch.randn(16, 3, generator=g)

    def raw(self, x):
        x = x[:, :, ::4]
        x = torch.nn.functional.avg_pool3d(x, (1, 8, 8))
        return torch.einsum("oc,bcthw->bothw", self.proj, x)

    def encode(self, x):
        mean = self.raw(x)

        class _Dist:
            def mode(self):
                return mean

        class _Out:
            latent_dist = _Dist()

        return _Out()

    def normalized(self, x):
        mean = torch.tensor(self.config.latents_mean).view(1, 16, 1, 1, 1)
        std = torch.tensor(self.config.latents_std).view(1, 16, 1, 1, 1)
        return (self.raw(x) - mean) / std


@pytest.mark.parametrize("mode", ["animate", "replace"])
@pytest.mark.parametrize("start", [0, 76])
@pytest.mark.parametrize("prev_len", [1, 5])
def test_segment_conditioning_matches_diffusers_pipeline(mode, start, prev_len):
    from diffusers.pipelines.wan.pipeline_wan_animate import WanAnimatePipeline as P

    class Shim:
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8
        get_i2v_mask = P.get_i2v_mask
        prepare_pose_latents = P.prepare_pose_latents
        prepare_prev_segment_cond_latents = P.prepare_prev_segment_cond_latents
        prepare_reference_image_latents = P.prepare_reference_image_latents

    shim = Shim()
    shim.vae = vae = _StubVAE()
    torch.manual_seed(12)
    L, H, W = 77, 32, 48
    h, w = H // 8, W // 8
    image = torch.rand(1, 3, H, W) * 2 - 1
    pose = torch.rand(1, 3, L, H, W) * 2 - 1
    bg = torch.rand(1, 3, L, H, W) * 2 - 1 if mode == "replace" else None
    mask = (torch.rand(1, 1, L, H, W) > 0.5).float() if mode == "replace" else None
    prev_frames = torch.rand(1, 3, L, H, W) * 2 - 1 if start > 0 else None

    ref = shim.prepare_reference_image_latents(image, 1, device="cpu")
    pose_lat = shim.prepare_pose_latents(pose, 1, device="cpu")
    prev = shim.prepare_prev_segment_cond_latents(
        prev_frames[:, :, -prev_len:] if start > 0 else None,
        background_video=bg,
        mask_video=mask,
        batch_size=1,
        segment_frame_length=L,
        start_frame=start,
        height=H,
        width=W,
        prev_segment_cond_frames=prev_len,
        task=mode,
        device="cpu",
    )
    expected = torch.cat([torch.cat([ref, prev], dim=2), rm.pose_conditioning(pose_lat)], dim=1)

    y_ref = torch.cat([rm.i2v_mask(1, h, w, 1), vae.normalized(image[:, :, None])], dim=1)
    got = rm.segment_conditioning(
        encode=vae.normalized,
        y_ref=y_ref,
        pose_segment=pose,
        prev_frames=prev_frames,
        start=start,
        mode=mode,
        prev_len=prev_len,
        background_segment=bg,
        mask_segment=mask,
    )
    assert got.shape == (1, 37, (L - 1) // 4 + 2, h, w)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
