# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen-Image-2.1 VAE encoder: the torch reference against the diffusers golden, and
the TTNN encoder against the torch reference, block by block and end to end.

    # CPU only (no device needed)
    tt-metal/python_env/bin/python -m pytest models/experimental/qwen_image_2_1/tests/test_vae_encoder.py -v -s -k "not tt_"
    # device tests (on an exclusively reserved card)
    python -m pytest models/experimental/qwen_image_2_1/tests/test_vae_encoder.py -v -s

Env: QWEN_VAE_ATTN=matmul|sdpa picks the mid-block attention implementation (default matmul),
QWEN_VAE_BLOCK_HW sets the spatial size of the per-block device tests (default 32).
"""

import os
import time

import pytest
import torch

from models.experimental.qwen_image_2_1.reference import torch_vae_encoder as ref
from models.experimental.qwen_image_2_1.reference.torch_vae import pcc
from models.experimental.qwen_image_2_1.reference.torch_vae_encoder import (
    AVG_DOWN_CASES,
    ENC,
    avg_down_fast,
    avg_down_reference,
    load_golden,
)

BLOCK_HW = int(os.environ.get("QWEN_VAE_BLOCK_HW", "32"))
IMG_HW = int(os.environ.get("QWEN_VAE_IMG_HW", "1024"))
LATENT_HW = IMG_HW // 16


# --------------------------------------------------------------------------------- host-only tests


def test_avg_down_matches_diffusers():
    """`avg_down_fast` (and the literal 5D transcription) must equal diffusers' own AvgDown3D."""
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D
    except ImportError:  # pragma: no cover
        pytest.skip("diffusers without autoencoder_kl_wan.AvgDown3D")

    torch.manual_seed(0)
    for in_c, out_c, factor_t, factor_s in AVG_DOWN_CASES:
        # Same channel ratio at 1/8 the width keeps the test instant.
        ci, co = in_c // 8, out_c // 8
        x4 = torch.randn(1, ci, 6, 8)
        x5 = x4.unsqueeze(2)
        want = AvgDown3D(ci, co, factor_t=factor_t, factor_s=factor_s)(x5)
        assert torch.equal(want, avg_down_reference(x5, ci, co, factor_t, factor_s))
        assert torch.equal(want[:, :, 0], avg_down_fast(x4, ci, co, factor_t, factor_s))
        print(f"avg_down {in_c}->{out_c} factor_t={factor_t} factor_s={factor_s}: exact match")


def test_avg_down_weight_matches_reference():
    """The 2x2 stride-2 convolution the TT shortcut uses must reproduce `avg_down_fast`."""
    import torch.nn.functional as F

    from models.experimental.qwen_image_2_1.tt.vae_encoder import avg_down_weight

    torch.manual_seed(0)
    for in_c, out_c, factor_t, factor_s in AVG_DOWN_CASES:
        if factor_s == 1:
            continue  # the identity shortcut needs no convolution
        ci, co = in_c // 8, out_c // 8
        x = torch.randn(1, ci, 6, 8)
        got = F.conv2d(x, avg_down_weight(ci, co, factor_t), stride=2)
        want = avg_down_fast(x, ci, co, factor_t, factor_s)
        print(f"avg_down_weight {in_c}->{out_c} factor_t={factor_t}: max|d|={(got - want).abs().max():.3e}")
        assert torch.allclose(got, want, atol=1e-6)


def test_torch_reference_matches_golden(torch_ref, golden):
    """fp32 CPU reference vs the bf16 GPU diffusers encode of the same image."""
    t0 = time.time()
    with torch.no_grad():
        mean, logvar = torch_ref.encode(golden["vae_in"].float())
    print(f"\ntorch reference encode (fp32 CPU): {time.time() - t0:.1f}s")
    assert tuple(mean.shape) == (1, ENC.z_dim, LATENT_HW, LATENT_HW), mean.shape

    want_mode = golden["latent_mode"].float()[:, :, 0]
    p = pcc(mean, want_mode)
    print(f"PCC(torch reference, golden mode)   = {p:.6f}  mean|d| = {(mean - want_mode).abs().mean():.5f}")
    print(f"PCC(torch reference, golden logvar) = {pcc(logvar, golden['latent_logvar'].float()[:, :, 0]):.6f}")
    p_norm = pcc(ref.normalize_latent(mean), golden["latent_normalized"].float()[:, :, 0])
    print(f"PCC(torch reference, golden normalized) = {p_norm:.6f}")
    assert p > 0.99


# ------------------------------------------------------------------------------------------ shared


@pytest.fixture(scope="module")
def ckpt():
    from models.experimental.qwen_image_2_1.common.weights import vae_ckpt

    return vae_ckpt()


@pytest.fixture(scope="module")
def torch_ref(ckpt):
    t0 = time.time()
    m = ref.load_encoder(ckpt)
    print(f"\ntorch reference weights loaded in {time.time() - t0:.1f}s")
    return m


@pytest.fixture(scope="module")
def golden():
    g = load_golden()
    if g is None:
        pytest.skip("no goldens/edit/vae_encode.pt")
    return g


@pytest.fixture(scope="module")
def dev():
    from models.experimental.qwen_image_2_1.common.device import close_device, open_device

    d = open_device()
    yield d
    close_device(d)


@pytest.fixture(scope="module")
def tt_model(dev, ckpt):
    import ttnn  # noqa: F401
    from models.experimental.qwen_image_2_1.tt.vae_encoder import QwenImageVAEEncoder, VAEPrecision

    prec = VAEPrecision(attn_impl=os.environ.get("QWEN_VAE_ATTN", VAEPrecision().attn_impl))
    t0 = time.time()
    m = QwenImageVAEEncoder(dev, ckpt, prec)
    print(f"\nTTNN encoder weights loaded to device in {time.time() - t0:.1f}s")
    return m


@pytest.fixture(scope="module")
def ref_stages(torch_ref, golden):
    """The fp32 reference encode of the golden image with every stage output tapped."""
    taps = {}

    def tap(name):
        return lambda mod, args, out, name=name: taps.__setitem__(name, out.detach())

    enc = torch_ref.encoder
    hooks = [enc.conv_in.register_forward_hook(tap("conv_in"))]
    hooks += [b.register_forward_hook(tap(f"down{i}")) for i, b in enumerate(enc.down_blocks)]
    hooks += [
        enc.mid_block.register_forward_hook(tap("mid")),
        enc.conv_out.register_forward_hook(tap("conv_out")),
        torch_ref.quant_conv.register_forward_hook(tap("quant")),
    ]
    t0 = time.time()
    with torch.no_grad():
        mean, logvar = torch_ref.encode(golden["vae_in"].float())
    print(f"\ntorch reference staged encode: {time.time() - t0:.1f}s")
    for h in hooks:
        h.remove()
    taps["mean"], taps["logvar"] = mean, logvar
    return taps


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


def test_tt_conv_strided(dev, ckpt, torch_ref):
    """The 3x3 stride-2 downsamplers: asymmetric `(0, 1, 0, 1)` padding folded into `ttnn.conv2d`."""
    from models.experimental.qwen_image_2_1.tt.vae_encoder import TTConvStrided, VAEPrecision

    torch.manual_seed(0)
    prec = VAEPrecision()
    for i, dim in [(0, 96), (2, 384)]:
        key = f"encoder.down_blocks.{i}.downsampler.resample.1."
        conv = TTConvStrided(
            dev,
            ckpt.get(key + "weight", torch.float32),
            ckpt.get(key + "bias", torch.float32),
            stride=(2, 2),
            padding=(0, 1, 0, 1),
            prec=prec,
        )
        x = torch.randn(1, dim, BLOCK_HW, BLOCK_HW)
        xd, h, w = _to_dev(dev, x)
        out, oh, ow = conv(xd, h, w)
        got = _from_dev(out, oh, ow, dim)
        want = torch_ref.encoder.down_blocks[i].downsampler(x)
        assert tuple(got.shape) == tuple(want.shape), (got.shape, want.shape)
        p = pcc(got, want)
        print(f"{key}: {tuple(got.shape)} pcc={p:.5f} max|d|={(got - want).abs().max():.4f}")
        assert p > 0.995


def test_tt_avg_down(dev, torch_ref):
    """The AvgDown3D shortcut as a fixed 2x2 stride-2 convolution, against `avg_down_fast`."""
    from models.experimental.qwen_image_2_1.tt.vae_encoder import TTConvStrided, VAEPrecision, avg_down_weight

    torch.manual_seed(0)
    prec = VAEPrecision()
    for in_c, out_c, factor_t, factor_s in AVG_DOWN_CASES:
        if factor_s == 1:
            continue
        conv = TTConvStrided(
            dev,
            avg_down_weight(in_c, out_c, factor_t),
            torch.zeros(out_c),
            stride=(2, 2),
            padding=(0, 0),
            prec=prec,
        )
        x = torch.randn(1, in_c, BLOCK_HW, BLOCK_HW).to(torch.bfloat16).float()
        xd, h, w = _to_dev(dev, x)
        out, oh, ow = conv(xd, h, w)
        got = _from_dev(out, oh, ow, out_c)
        want = avg_down_fast(x, in_c, out_c, factor_t, factor_s)
        p = pcc(got, want)
        zeros_ok = True if factor_t == 1 else bool((got[:, 0::2] == 0).all())
        print(
            f"avg_down_tt {in_c}->{out_c} ft={factor_t}: pcc={p:.6f} max|d|={(got - want).abs().max():.5f} even_channels_zero={zeros_ok}"
        )
        assert zeros_ok
        assert p > 0.999


def test_tt_resblock(dev, torch_ref, tt_model):
    """Encoder residual blocks with and without a conv shortcut."""
    cases = [
        ("encoder.down_blocks.0.resnets.0.", torch_ref.encoder.down_blocks[0].resnets[0], 96, 96, 0, 0),
        ("encoder.down_blocks.1.resnets.0.", torch_ref.encoder.down_blocks[1].resnets[0], 96, 192, 1, 0),
        ("encoder.down_blocks.3.resnets.0.", torch_ref.encoder.down_blocks[3].resnets[0], 384, 768, 3, 0),
        ("encoder.down_blocks.4.resnets.1.", torch_ref.encoder.down_blocks[4].resnets[1], 768, 768, 4, 1),
    ]
    torch.manual_seed(0)
    for prefix, ref_mod, cin, cout, bi, ri in cases:
        tt_block = tt_model.down_blocks[bi].resnets[ri]
        x = torch.randn(1, cin, BLOCK_HW, BLOCK_HW)
        xd, h, w = _to_dev(dev, x)
        out, oh, ow = tt_block(xd, h, w)
        got = _from_dev(out, oh, ow, cout)
        want = ref_mod(x)
        p = pcc(got, want)
        print(f"{prefix} ({cin}->{cout}): pcc={p:.5f} max|d|={(got - want).abs().max():.4f}")
        assert p > 0.99


def test_tt_attention(dev, torch_ref, tt_model):
    """Mid-block attention at the real 64x64 = 4096 token count, head_dim 768."""
    torch.manual_seed(0)
    x = torch.randn(1, ENC.dims[-1], LATENT_HW, LATENT_HW) * 0.5
    xd, h, w = _to_dev(dev, x)
    out, oh, ow = tt_model.mid_block.attn(xd, h, w)
    got = _from_dev(out, oh, ow, ENC.dims[-1])
    want = torch_ref.encoder.mid_block.attentions[0](x)
    p = pcc(got, want)
    print(f"mid attention ({tt_model.prec.attn_impl}): pcc={p:.5f} max|d|={(got - want).abs().max():.4f}")
    assert p > 0.99


def test_tt_mid_block(dev, torch_ref, tt_model):
    torch.manual_seed(0)
    x = torch.randn(1, ENC.dims[-1], LATENT_HW, LATENT_HW) * 0.5
    xd, h, w = _to_dev(dev, x)
    out, oh, ow = tt_model.mid_block(xd, h, w)
    got = _from_dev(out, oh, ow, ENC.dims[-1])
    want = torch_ref.encoder.mid_block(x)
    p = pcc(got, want)
    print(f"mid block: pcc={p:.5f}")
    assert p > 0.99


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_tt_down_block(dev, torch_ref, tt_model, idx):
    """Each down block (2 residual blocks + stride-2 conv + AvgDown3D shortcut) at a reduced size."""
    torch.manual_seed(idx)
    tt_block = tt_model.down_blocks[idx]
    x = torch.randn(1, tt_block.in_dim, BLOCK_HW, BLOCK_HW) * 0.5
    xd, h, w = _to_dev(dev, x)
    out, oh, ow = tt_block(xd, h, w)
    got = _from_dev(out, oh, ow, tt_block.out_dim)
    want = torch_ref.encoder.down_blocks[idx](x)
    assert tuple(got.shape) == tuple(want.shape), (got.shape, want.shape)
    p = pcc(got, want)
    print(
        f"down_block {idx} ({tt_block.in_dim}->{tt_block.out_dim}, {tuple(got.shape)[2:]}): "
        f"pcc={p:.5f} max|d|={(got - want).abs().max():.4f}"
    )
    assert p > 0.99


# ----------------------------------------------------------------------------- device: full encode


def test_tt_stage_pcc(dev, tt_model, ref_stages, golden):
    """Walk the real 1024x1024 encode stage by stage, reporting the PCC of each against torch."""
    import ttnn

    m = tt_model
    x, h, w = m.upload(golden["vae_in"])
    results = []

    def record(name, t, th, tw, c):
        got = _from_dev(t, th, tw, c)
        want = ref_stages[name]
        assert tuple(got.shape) == tuple(want.shape), (name, got.shape, want.shape)
        results.append((name, tuple(got.shape)[1:], pcc(got, want), float((got - want).abs().max())))

    y, h, w = m.conv_in(x, h, w)
    ttnn.deallocate(x)
    record("conv_in", y, h, w, ENC.dims[0])
    for i, block in enumerate(m.down_blocks):
        y, h, w = block(y, h, w)
        record(f"down{i}", y, h, w, ENC.dims[i + 1])
    y, h, w = m.mid_block(y, h, w)
    record("mid", y, h, w, ENC.dims[-1])
    n = m.norm_out(y)
    ttnn.deallocate(y)
    s = ttnn.silu(n)
    ttnn.deallocate(n)
    y, h, w = m.conv_out(s, h, w)
    ttnn.deallocate(s)
    record("conv_out", y, h, w, ENC.latent_channels)
    out, h, w = m.quant_conv(y, h, w)
    ttnn.deallocate(y)
    record("quant", out, h, w, ENC.latent_channels)
    ttnn.deallocate(out)

    print("\nper-stage PCC(TT, torch reference) at 1024x1024:")
    for name, shape, p, dmax in results:
        print(f"  {name:9s} {str(shape):18s} pcc={p:.6f}  max|d|={dmax:.4f}")
    worst = min(results, key=lambda r: r[2])
    assert worst[2] > 0.98, worst


def test_tt_full_encode(dev, tt_model, ref_stages, golden):
    """The real 1024x1024 -> 64x64 encode on device, against the torch reference and the golden."""
    m = tt_model
    vae_in = golden["vae_in"]

    t0 = time.time()
    got = m.encode(vae_in)
    cold = time.time() - t0
    got, warm = m.time_encode(vae_in, iters=3)
    print(f"\nTT encode: {cold:.2f}s cold (incl. compile), {warm:.3f}s warm (eager)")
    assert tuple(got.shape) == (1, ENC.z_dim, LATENT_HW, LATENT_HW), got.shape

    want = ref_stages["mean"]
    p_ref = pcc(got, want)
    print(
        f"PCC(TT, torch reference) = {p_ref:.6f}  mean|d|={(got - want).abs().mean():.5f}  "
        f"max|d|={(got - want).abs().max():.4f}"
    )
    gm = golden["latent_mode"].float()[:, :, 0]
    print(f"PCC(TT, golden mode)     = {pcc(got, gm):.6f}")
    print(
        f"PCC(TT, golden normalized) = {pcc(ref.normalize_latent(got), golden['latent_normalized'].float()[:, :, 0]):.6f}"
    )

    _, logvar = m.encode(vae_in, return_logvar=True)
    print(f"PCC(TT logvar, torch reference) = {pcc(logvar, ref_stages['logvar']):.6f}")

    assert p_ref > 0.98
    assert warm < 2.0, f"encode took {warm:.3f}s"


def test_tt_encoder_leaves_ckpt_intact(dev, ckpt):
    """Building the TT encoder must not edit the shared memory-mapped checkpoint tensors.

    `safe_open(...).get_tensor` hands back the mapped buffer itself for an fp32 shard, and
    `tt.vae.TTAttention` folds the softmax scale into the Q rows of `to_qkv.weight` in place. Left
    unguarded that scales `to_qkv.weight` by `1 / sqrt(768)` for every later reader, which showed up
    as a torch reference whose mid block and everything after it silently disagreed with the device.
    """
    from models.experimental.qwen_image_2_1.tt.vae_encoder import QwenImageVAEEncoder, VAEPrecision

    key = "encoder.mid_block.attentions.0.to_qkv.weight"
    before = ckpt.get(key, torch.float32).clone()
    QwenImageVAEEncoder(dev, ckpt, VAEPrecision(attn_impl="matmul"))
    after = ckpt.get(key, torch.float32)
    print(f"\n{key}: sum before={before.double().sum():.6f} after={after.double().sum():.6f}")
    assert torch.equal(before, after), "the TT encoder mutated the shared checkpoint"


def _prepared_keys(m):
    """Every `(layer, (h, w))` a `ttnn.conv2d` has cached device weights for."""
    keys = set()
    stack = [("conv_in", m.conv_in), ("conv_out", m.conv_out), ("quant_conv", m.quant_conv)]
    for i, b in enumerate(m.down_blocks):
        for j, rn in enumerate(b.resnets):
            stack += [(f"d{i}.r{j}.conv1", rn.conv1), (f"d{i}.r{j}.conv2", rn.conv2)]
            if rn.shortcut is not None:
                stack.append((f"d{i}.r{j}.shortcut", rn.shortcut))
        if b.down_conv is not None:
            stack += [(f"d{i}.down", b.down_conv), (f"d{i}.avg", b.avg_conv)]
    for j, rn in enumerate(m.mid_block.resnets):
        stack += [(f"mid.r{j}.conv1", rn.conv1), (f"mid.r{j}.conv2", rn.conv2)]
    for name, conv in stack:
        keys |= {(name, hw) for hw in conv.prepared}
    return keys


def test_tt_warmup_then_encode(dev, ckpt, ref_stages, golden):
    """The pipeline's order: build, `warmup(h, w)`, then encode.

    Every persistent device buffer has to exist before a trace is captured, so the encode that
    follows a warmup must not prepare any new convolution weights -- and must still be correct.
    """
    from models.experimental.qwen_image_2_1.tt.vae_encoder import QwenImageVAEEncoder, VAEPrecision

    prec = VAEPrecision(attn_impl=os.environ.get("QWEN_VAE_ATTN", VAEPrecision().attn_impl))
    m = QwenImageVAEEncoder(dev, ckpt, prec)
    assert not _prepared_keys(m), "a freshly built encoder should hold no prepared weights yet"

    t0 = time.time()
    m.warmup(IMG_HW, IMG_HW)
    warm_s = time.time() - t0
    after_warmup = _prepared_keys(m)
    print(f"\nwarmup({IMG_HW}, {IMG_HW}): {warm_s:.2f}s, {len(after_warmup)} prepared convolutions")
    assert after_warmup

    got, best = m.time_encode(golden["vae_in"], iters=2)
    assert _prepared_keys(m) == after_warmup, "the encode prepared weights the warmup missed"
    p = pcc(got, ref_stages["mean"])
    print(f"encode after warmup: {best:.3f}s  PCC(TT, torch reference) = {p:.6f}")
    assert p > 0.98
