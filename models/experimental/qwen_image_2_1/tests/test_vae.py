# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen-Image-2.1 VAE decoder: the torch reference against the diffusers golden, and
the TTNN decoder against the torch reference, block by block and end to end.

    # CPU only (no device needed)
    tt-metal/python_env/bin/python -m pytest models/experimental/qwen_image_2_1/tests/test_vae.py -v -s -k "not tt_"
    # device tests (on an exclusively reserved card)
    python -m pytest models/experimental/qwen_image_2_1/tests/test_vae.py -v -s

Env: QWEN_VAE_ATTN=matmul|sdpa picks the mid-block attention implementation (default sdpa),
QWEN_VAE_BLOCK_HW sets the spatial size of the per-block device tests (default 32).
"""
import os
import time

import pytest
import torch

from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR, VAE
from models.experimental.qwen_image_2_1.reference import torch_vae
from models.experimental.qwen_image_2_1.reference.torch_vae import dup_up_fast, dup_up_reference, load_golden, pcc

BLOCK_HW = int(os.environ.get("QWEN_VAE_BLOCK_HW", "32"))
# (in_dim, out_dim, factor_t) of the five decoder up blocks; block 4 has no shortcut.
DUP_UP_CASES = [(1152, 1152, 2), (1152, 1152, 2), (1152, 576, 2), (576, 288, 1)]


# --------------------------------------------------------------------------------- host-only tests


def test_dup_up_matches_diffusers():
    """`dup_up_fast` (and the literal 5D transcription) must equal diffusers' own DupUp3D."""
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import DupUp3D
    except ImportError:  # pragma: no cover
        pytest.skip("diffusers without autoencoder_kl_wan.DupUp3D")

    torch.manual_seed(0)
    for in_c, out_c, factor_t in DUP_UP_CASES:
        # Same channel ratio at 1/8 the width keeps the test instant.
        ci, co = in_c // 8, out_c // 8
        x4 = torch.randn(1, ci, 5, 7)
        x5 = x4.unsqueeze(2)
        ref = DupUp3D(ci, co, factor_t=factor_t, factor_s=2)(x5, first_chunk=True)
        assert torch.equal(ref, dup_up_reference(x5, ci, co, factor_t, 2, True))
        assert torch.equal(ref[:, :, 0], dup_up_fast(x4, ci, co, factor_t))
        print(f"dup_up {in_c}->{out_c} factor_t={factor_t}: exact match")


def test_torch_reference_matches_golden(torch_ref):
    """fp32 CPU reference vs the bf16 GPU diffusers decode of the same latent."""
    golden = load_golden()
    if golden is None:
        pytest.skip(f"no {GOLDENS_DIR}/vae.pt")
    vae_in, vae_out = golden
    t0 = time.time()
    out = torch_ref(vae_in.float())
    print(f"\ntorch reference decode (fp32 CPU): {time.time() - t0:.1f}s")
    assert tuple(out.shape) == (1, VAE.out_channels, 1024, 1024), out.shape
    assert out.min() >= -1.0 and out.max() <= 1.0
    p = pcc(out, vae_out.float())
    print(f"PCC(torch reference, golden) = {p:.6f}  mean|d| = {(out - vae_out.float()).abs().mean():.5f}")
    assert p > 0.99


# ------------------------------------------------------------------------------------------ shared


@pytest.fixture(scope="module")
def ckpt():
    from models.experimental.qwen_image_2_1.common.weights import vae_ckpt

    return vae_ckpt()


@pytest.fixture(scope="module")
def torch_ref(ckpt):
    t0 = time.time()
    m = torch_vae.load_decoder(ckpt)
    print(f"\ntorch reference weights loaded in {time.time() - t0:.1f}s")
    return m


@pytest.fixture(scope="module")
def dev():
    from models.experimental.qwen_image_2_1.common.device import close_device, open_device

    d = open_device()
    yield d
    close_device(d)


@pytest.fixture(scope="module")
def tt_model(dev, ckpt):
    import ttnn  # noqa: F401
    from models.experimental.qwen_image_2_1.tt.vae import QwenImageVAEDecoder, VAEPrecision

    prec = VAEPrecision(attn_impl=os.environ.get("QWEN_VAE_ATTN", VAEPrecision().attn_impl))
    t0 = time.time()
    m = QwenImageVAEDecoder(dev, ckpt, prec)
    print(f"\nTTNN decoder weights loaded to device in {time.time() - t0:.1f}s")
    return m


def _to_dev(dev, x_bchw):
    """torch `[1, C, H, W]` -> (device `[1, 1, H*W, align32(C)]` TILE bf16, H, W)."""
    import ttnn
    from models.experimental.qwen_image_2_1.tt.vae import align32

    _, c, h, w = x_bchw.shape
    cp = align32(c)
    flat = torch.zeros(1, 1, h * w, cp)
    flat[0, 0, :, :c] = x_bchw[0].permute(1, 2, 0).reshape(h * w, c)
    return (
        ttnn.from_torch(
            flat.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        h,
        w,
    )


def _from_dev(t, h, w, c):
    """device `[1, 1, H*W, C_pad]` -> torch `[1, C, H, W]` fp32."""
    import ttnn

    x = ttnn.to_torch(t).float().reshape(1, h, w, -1)[..., :c]
    return x.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------------- device: pieces


def test_tt_conv2d(dev, ckpt, torch_ref):
    """A 3x3 conv with unaligned channels (144 -> 4 head conv) and an aligned one."""
    from models.experimental.qwen_image_2_1.tt.vae import TTConv2d, VAEPrecision

    torch.manual_seed(0)
    prec = VAEPrecision()
    for name, ref_mod, cin in [
        ("decoder.conv_in", torch_ref.decoder.conv_in, 64),
        ("decoder.conv_out", torch_ref.decoder.conv_out, 144),
        ("decoder.up_blocks.4.resnets.0.conv1", torch_ref.decoder.up_blocks[4].resnets[0].conv1, 288),
    ]:
        pad_out = name != "decoder.conv_out"
        conv = TTConv2d(
            dev,
            ckpt.get(name + ".weight", torch.float32),
            ckpt.get(name + ".bias", torch.float32),
            padding=1,
            prec=prec,
            pad_out_channels=pad_out,
        )
        x = torch.randn(1, cin, BLOCK_HW, BLOCK_HW)
        xd, h, w = _to_dev(dev, x)
        out, oh, ow = conv(xd, h, w)
        got = _from_dev(out, oh, ow, ref_mod.out_channels)
        want = ref_mod(x)
        p = pcc(got, want)
        print(f"{name}: {tuple(got.shape)} pcc={p:.5f}")
        assert p > 0.995


def test_tt_conv2d_geometry_eviction_releases_buffers_and_preserves_outputs(dev):
    import ttnn
    from models.experimental.qwen_image_2_1.tt.vae import TTConv2d, VAEPrecision

    generator = torch.Generator().manual_seed(71)
    weight = torch.randn(32, 32, 3, 3, generator=generator) * 0.05
    bias = torch.randn(32, generator=generator) * 0.01
    conv = TTConv2d(dev, weight, bias, padding=1, prec=VAEPrecision())
    geometries = [(32, 32), (32, 64), (64, 32), (64, 64), (32, 96)]
    inputs = {size: torch.randn(1, 32, *size, generator=generator).to(torch.bfloat16).float() for size in geometries}
    first_output = None
    victim = None
    try:
        # The fifth geometry evicts the first; returning to it must prepare weights again.
        for size in [*geometries, geometries[-1], geometries[0]]:
            x = inputs[size]
            xd, h, w = _to_dev(dev, x)
            out = None
            try:
                out, oh, ow = conv(xd, h, w)
                actual = _from_dev(out, oh, ow, 32)
                expected = torch.nn.functional.conv2d(
                    x, weight.to(torch.bfloat16).float(), bias.to(torch.bfloat16).float(), padding=1
                )
                assert pcc(actual, expected) > 0.999
                assert len(conv.prepared) <= 4
                if first_output is None:
                    first_output = actual.clone()
                    victim = conv.prepared[size]
                elif size == geometries[0]:
                    assert torch.equal(actual, first_output)
                if size == geometries[-1]:
                    assert not any(tensor.is_allocated() for tensor in victim)
                assert all(tensor.is_allocated() for pair in conv.prepared.values() for tensor in pair)
            finally:
                if out is not None:
                    ttnn.deallocate(out)
                ttnn.deallocate(xd)
    finally:
        for pair in conv.prepared.values():
            for tensor in pair:
                ttnn.deallocate(tensor)


def test_tt_rms_norm(dev, ckpt, torch_ref):
    """RMS norm for an aligned (1152) and an unaligned (144) channel count."""
    from models.experimental.qwen_image_2_1.tt.vae import TTRmsNorm, VAEPrecision

    torch.manual_seed(0)
    prec = VAEPrecision()
    for key, ref_mod, c in [
        ("decoder.mid_block.resnets.0.norm1", torch_ref.decoder.mid_block.resnets[0].norm1, 1152),
        ("decoder.norm_out", torch_ref.decoder.norm_out, 144),
    ]:
        norm = TTRmsNorm(dev, ckpt.get(key + ".gamma", torch.float32), prec)
        x = torch.randn(1, c, BLOCK_HW, BLOCK_HW)
        xd, h, w = _to_dev(dev, x)
        got = _from_dev(norm(xd), h, w, c)
        want = ref_mod(x)
        p = pcc(got, want)
        print(f"{key} (C={c}): pcc={p:.6f} max|d|={(got - want).abs().max():.4f}")
        assert p > 0.999


@pytest.mark.parametrize("implementation", ["matmul", "sdpa"])
def test_attention_keeps_checkpoint_weights_unchanged(dev, ckpt, implementation):
    from models.experimental.qwen_image_2_1.tt.vae import TTAttention, VAEPrecision

    prefix = "decoder.mid_block.attentions.0."
    original = {name: ckpt.get(prefix + name).clone() for name in ("to_qkv.weight", "to_qkv.bias")}
    TTAttention(dev, ckpt, prefix, 1152, VAEPrecision(attn_impl=implementation))
    for name, expected in original.items():
        assert torch.equal(ckpt.get(prefix + name), expected)


def test_tt_dup_up(dev):
    """`dup_up_tt` must reproduce `dup_up_fast` exactly (it is a pure gather, so bf16 is lossless)."""
    from models.experimental.qwen_image_2_1.tt.vae import dup_up_tt

    torch.manual_seed(0)
    for in_c, out_c, factor_t in [(1152, 1152, 2), (1152, 576, 2), (576, 288, 1)]:
        x = torch.randn(1, in_c, BLOCK_HW, BLOCK_HW).to(torch.bfloat16).float()
        xd, h, w = _to_dev(dev, x)
        out, oh, ow = dup_up_tt(xd, h, w, in_c, out_c, factor_t)
        got = _from_dev(out, oh, ow, out_c)
        want = dup_up_fast(x, in_c, out_c, factor_t)
        print(
            f"dup_up_tt {in_c}->{out_c} ft={factor_t}: {tuple(got.shape)} exact={torch.equal(got, want)} pcc={pcc(got, want):.6f}"
        )
        assert torch.equal(got, want)


@pytest.mark.parametrize("height,width", [(32, 32), (720, 368), (368, 720)])
def test_tt_dup_up_aspect_ratios(dev, height, width):
    """The final channel/row shuffle must fit L1 and preserve the reference for wide images."""
    import ttnn
    from models.experimental.qwen_image_2_1.tt.vae import dup_up_tt

    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, 576, height, width, generator=generator, dtype=torch.bfloat16)
    xd, h, w = _to_dev(dev, x)
    out = None
    try:
        out, oh, ow = dup_up_tt(xd, h, w, 576, 288, 1)
        actual = _from_dev(out, oh, ow, 288)
        expected = dup_up_reference(x.unsqueeze(2), 576, 288, 1)[:, :, 0].float()
        assert torch.equal(actual, expected)
    finally:
        if out is not None:
            ttnn.deallocate(out)
        ttnn.deallocate(xd)


def test_tt_resblock(dev, ckpt, torch_ref, tt_model):
    """Residual blocks with and without a conv shortcut, including the 144-channel stage."""
    cases = [
        (
            "decoder.mid_block.resnets.0.",
            torch_ref.decoder.mid_block.resnets[0],
            1152,
            1152,
            tt_model.mid_block.resnets[0],
        ),
        (
            "decoder.up_blocks.2.resnets.0.",
            torch_ref.decoder.up_blocks[2].resnets[0],
            1152,
            576,
            tt_model.up_blocks[2].resnets[0],
        ),
        (
            "decoder.up_blocks.4.resnets.0.",
            torch_ref.decoder.up_blocks[4].resnets[0],
            288,
            144,
            tt_model.up_blocks[4].resnets[0],
        ),
        (
            "decoder.up_blocks.4.resnets.1.",
            torch_ref.decoder.up_blocks[4].resnets[1],
            144,
            144,
            tt_model.up_blocks[4].resnets[1],
        ),
    ]
    torch.manual_seed(0)
    for prefix, ref_mod, cin, cout, tt_block in cases:
        x = torch.randn(1, cin, BLOCK_HW, BLOCK_HW)
        xd, h, w = _to_dev(dev, x)
        out, oh, ow = tt_block(xd, h, w)
        got = _from_dev(out, oh, ow, cout)
        want = ref_mod(x)
        p = pcc(got, want)
        print(f"{prefix} ({cin}->{cout}): pcc={p:.5f} max|d|={(got - want).abs().max():.4f}")
        assert p > 0.99


def test_tt_attention(dev, torch_ref, tt_model):
    """Mid-block attention at the real 64x64 = 4096 token count, head_dim 1152."""
    torch.manual_seed(0)
    x = torch.randn(1, 1152, 64, 64) * 0.5
    xd, h, w = _to_dev(dev, x)
    out, oh, ow = tt_model.mid_block.attn(xd, h, w)
    got = _from_dev(out, oh, ow, 1152)
    want = torch_ref.decoder.mid_block.attentions[0](x)
    p = pcc(got, want)
    print(f"mid attention ({tt_model.prec.attn_impl}): pcc={p:.5f} max|d|={(got - want).abs().max():.4f}")
    assert p > 0.99


def test_tt_mid_block(dev, torch_ref, tt_model):
    torch.manual_seed(0)
    x = torch.randn(1, 1152, 64, 64) * 0.5
    xd, h, w = _to_dev(dev, x)
    out, oh, ow = tt_model.mid_block(xd, h, w)
    got = _from_dev(out, oh, ow, 1152)
    want = torch_ref.decoder.mid_block(x)
    p = pcc(got, want)
    print(f"mid block: pcc={p:.5f}")
    assert p > 0.99


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_tt_up_block(dev, torch_ref, tt_model, idx):
    """Each up block (3 residual blocks + upsample conv + DupUp3D shortcut) at a reduced size."""
    torch.manual_seed(idx)
    tt_block = tt_model.up_blocks[idx]
    x = torch.randn(1, tt_block.in_dim, BLOCK_HW, BLOCK_HW) * 0.5
    xd, h, w = _to_dev(dev, x)
    out, oh, ow = tt_block(xd, h, w)
    got = _from_dev(out, oh, ow, tt_block.out_dim)
    want = torch_ref.decoder.up_blocks[idx](x)
    assert tuple(got.shape) == tuple(want.shape), (got.shape, want.shape)
    p = pcc(got, want)
    print(
        f"up_block {idx} ({tt_block.in_dim}->{tt_block.out_dim}, {tuple(got.shape)[2:]}): pcc={p:.5f} max|d|={(got - want).abs().max():.4f}"
    )
    assert p > 0.99


# ------------------------------------------------------------------------------ device: full decode


@pytest.fixture(scope="module")
def latent():
    golden = load_golden()
    if golden is None:
        torch.manual_seed(0)
        return torch.randn(1, VAE.z_dim, 1, 64, 64).to(torch.bfloat16), None
    return golden


def test_tt_full_decode(dev, tt_model, torch_ref, latent):
    """The real 64x64 -> 1024x1024 decode on device, against the torch reference and the golden."""
    vae_in, vae_out = latent
    t0 = time.time()
    got = tt_model.decode(vae_in)
    eager_cold = time.time() - t0
    got, eager_warm = tt_model.time_decode(vae_in, iters=2)
    print(f"\nTT decode: {eager_cold:.2f}s cold (incl. compile), {eager_warm:.3f}s warm (eager)")
    assert tuple(got.shape) == (1, VAE.out_channels, 1024, 1024), got.shape
    assert got.min() >= -1.0 and got.max() <= 1.0

    want = torch_ref(vae_in.float())
    p_ref = pcc(got, want)
    print(
        f"PCC(TT, torch reference) = {p_ref:.6f}  mean|d|={(got - want).abs().mean():.5f}  max|d|={(got - want).abs().max():.4f}"
    )
    if vae_out is not None:
        print(f"PCC(TT, golden)          = {pcc(got, vae_out.float()):.6f}")

    out_png = os.path.join(GOLDENS_DIR, "vae_tt.png")
    try:
        rgba, rgb = torch_vae.to_pil(got)
        rgb.save(out_png)
        rgba.save(out_png.replace(".png", "_rgba.png"))
        print(f"wrote {out_png}")
    except Exception as e:  # pragma: no cover - PIL missing / read-only goldens dir
        print(f"could not write {out_png}: {e}")
    assert p_ref > 0.98


def test_tt_decode_traced(dev, tt_model, torch_ref, latent):
    """Capture the whole decode in a metal trace and replay it."""
    import ttnn

    vae_in, _ = latent
    try:
        t0 = time.time()
        tt_model.capture_trace(vae_in)
        print(f"\ntrace capture: {time.time() - t0:.2f}s")
    except RuntimeError as e:
        tt_model.release_trace()
        pytest.skip(f"trace capture failed: {e}")
    try:
        got = tt_model.decode_traced(vae_in)
        best = float("inf")
        for _ in range(3):
            ttnn.synchronize_device(dev)
            t0 = time.time()
            got = tt_model.decode_traced(vae_in)
            ttnn.synchronize_device(dev)
            best = min(best, time.time() - t0)
        want = torch_ref(vae_in.float())
        p = pcc(got, want)
        print(f"traced decode: {best:.3f}s  PCC(TT traced, torch reference) = {p:.6f}")
        assert p > 0.98
    finally:
        tt_model.release_trace()
