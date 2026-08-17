# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""EDM (Karras et al. 2022, arXiv:2206.00364) training + sampling, host side.

The network F is trained in EDM's F-space, where the loss is EXACTLY
unweighted MSE — so the device graph is identical to the DDPM path and all
recipe machinery stays in numpy:

    y      = x + n,  n ~ N(0, sigma^2),  sigma ~ LogNormal(P_mean, P_std)
    input  = c_in(sigma) * y            (patchified)
    cond   = c_noise(sigma) = ln(sigma)/4   (fed through the timestep-feature MLP)
    target = (x - c_skip(sigma) * y) / c_out(sigma)   (patchified)
    loss   = MSE(F(input, cond, class), target)

Denoiser reconstruction: D(y; sigma) = c_skip * y + c_out * F(c_in * y, c_noise).
Sampling: EDM deterministic Heun over the rho-spaced sigma schedule.

Official CIFAR-10 class-conditional reference (DDPM++ UNet): FID 1.79.
This module reproduces the METHODOLOGY on the DiT backbone.
"""

from __future__ import annotations

import numpy as np

from diffusion import patchify, unpatchify, timestep_features, one_hot

SIGMA_DATA = 0.5
P_MEAN = -1.2
P_STD = 1.2
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
# c_noise = ln(sigma)/4 spans roughly [-1.6, 1.1]; scale into the sinusoidal
# embedder's useful range (it was built for integer timesteps 0..1000).
CNOISE_FEAT_SCALE = 250.0


def precond_coeffs(sigma: np.ndarray):
    """c_skip, c_out, c_in, c_noise for sigma [B] (fp32)."""
    s2 = sigma**2
    d2 = SIGMA_DATA**2
    c_skip = d2 / (s2 + d2)
    c_out = sigma * SIGMA_DATA / np.sqrt(s2 + d2)
    c_in = 1.0 / np.sqrt(s2 + d2)
    c_noise = np.log(sigma) / 4.0
    return c_skip, c_out, c_in, c_noise


def sample_sigma(rng: np.random.Generator, batch: int) -> np.ndarray:
    return np.exp(rng.normal(P_MEAN, P_STD, size=batch)).astype(np.float32)


def make_edm_batch_image(
    images: np.ndarray,  # [B, C, H, W] fp32 in [-1, 1]
    labels: np.ndarray,  # [B] int
    model_dim: int,
    rng: np.random.Generator,
    cfg_drop_prob: float = 0.0,
    null_class: int | None = None,
    hflip: bool = False,
):
    """Unpatchified EDM batch for image-space (UNet) backbones.

    Returns (net_in [B,C,H,W], cnoise_feats [B,1,1,model_dim],
    labels_onehot [B,1,1,K], target [B,C,H,W]) — same math as
    make_edm_batch but leaves images in [B,C,H,W].
    """
    b = images.shape[0]
    if hflip:
        flip = rng.random(b) < 0.5
        images = images.copy()
        images[flip] = images[flip][..., ::-1]

    sigma = sample_sigma(rng, b)
    c_skip, c_out, c_in, c_noise = precond_coeffs(sigma)
    bc = lambda v: v.reshape(b, 1, 1, 1).astype(np.float32)

    n = rng.standard_normal(images.shape).astype(np.float32) * bc(sigma)
    y = images + n
    net_in = bc(c_in) * y
    target = (images - bc(c_skip) * y) / bc(c_out)

    if null_class is not None and cfg_drop_prob > 0:
        drop = rng.random(b) < cfg_drop_prob
        labels = np.where(drop, null_class, labels)

    feats = timestep_features(c_noise * CNOISE_FEAT_SCALE, model_dim)
    onehot = one_hot(labels, (null_class if null_class is not None else int(labels.max())) + 1)
    return net_in.astype(np.float32), feats.astype(np.float32), onehot, target.astype(np.float32)


def make_edm_batch(
    images: np.ndarray,  # [B, C, H, W] fp32 in [-1, 1]
    labels: np.ndarray,  # [B] int
    patch: int,
    model_dim: int,
    rng: np.random.Generator,
    cfg_drop_prob: float = 0.0,
    null_class: int | None = None,
    hflip: bool = False,
):
    """Returns (input_tokens, cnoise_feats, labels_onehot, target_tokens)."""
    net_in, feats, onehot, target = make_edm_batch_image(
        images, labels, model_dim, rng, cfg_drop_prob=cfg_drop_prob, null_class=null_class, hflip=hflip
    )
    return patchify(net_in, patch), feats, onehot, patchify(target, patch)


def nchw_to_nhwc_tokens(x: np.ndarray) -> np.ndarray:
    """[B,C,H,W] -> [B,1,H*W,C] channels-last tokens (device UNet activation form)."""
    b, c, h, w = x.shape
    return np.ascontiguousarray(x.transpose(0, 2, 3, 1)).reshape(b, 1, h * w, c)


def nhwc_tokens_to_nchw(t: np.ndarray, h: int, w: int) -> np.ndarray:
    """[B,1,H*W,C] -> [B,C,H,W]."""
    b, _, _, c = t.shape
    return np.ascontiguousarray(t.reshape(b, h, w, c).transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# SongUNet block plan (shared by reference_unet.py and edm_unet.py)
# ---------------------------------------------------------------------------


class BlockSpec:
    """One residual block of the DDPM++ SongUNet.

    resample: None | 'down' (avgpool 2x2 BEFORE the block) | 'up'
              (nearest-2x upsample BEFORE the block).
    skip_in:  channels concatenated from the encoder skip stack before the
              block (0 = no skip consumed). Decoder only.
    res:      spatial resolution the block runs at (after any resample).
    """

    __slots__ = ("cin", "cout", "res", "attn", "resample", "skip_in")

    def __init__(self, cin, cout, res, attn, resample=None, skip_in=0):
        self.cin, self.cout, self.res = cin, cout, res
        self.attn, self.resample, self.skip_in = attn, resample, skip_in

    def __repr__(self):
        return (
            f"BlockSpec(cin={self.cin}, cout={self.cout}, res={self.res}, "
            f"attn={self.attn}, resample={self.resample}, skip_in={self.skip_in})"
        )


def build_unet_plan(
    img_resolution: int = 32,
    model_channels: int = 128,
    channel_mult: tuple = (2, 2, 2),
    num_blocks: int = 4,
    attn_resolutions: tuple = (16,),
):
    """Encoder/decoder block plans for the DDPM++ SongUNet (EDM CIFAR-10 config).

    Mirrors EDM's SongUNet layout with encoder_type/decoder_type='standard':
    conv_in, then per level: [down-block (levels>0)] + num_blocks blocks;
    middle pair (attn + plain) at the lowest resolution; decoder per level:
    [up-block (levels < deepest)] + (num_blocks+1) skip-concat blocks.
    The skip stack holds conv_in's output plus every encoder block output.

    Returns (enc_specs, dec_specs). conv_in / out_conv are not in the plan.
    """
    enc, skips = [], [model_channels]  # conv_in output is the first skip
    cout = model_channels
    for level, mult in enumerate(channel_mult):
        res = img_resolution >> level
        if level > 0:
            enc.append(BlockSpec(cout, cout, res, attn=False, resample="down"))
            skips.append(cout)
        for _ in range(num_blocks):
            cin, cout = cout, model_channels * mult
            enc.append(BlockSpec(cin, cout, res, attn=res in attn_resolutions))
            skips.append(cout)

    dec = []
    for level in reversed(range(len(channel_mult))):
        res = img_resolution >> level
        if level == len(channel_mult) - 1:
            dec.append(BlockSpec(cout, cout, res, attn=True))  # middle 'in0'
            dec.append(BlockSpec(cout, cout, res, attn=False))  # middle 'in1'
        else:
            dec.append(BlockSpec(cout, cout, res, attn=False, resample="up"))
        for _ in range(num_blocks + 1):
            s = skips.pop()
            cin, cout = cout + s, model_channels * channel_mult[level]
            dec.append(BlockSpec(cin, cout, res, attn=res in attn_resolutions, skip_in=s))
    assert not skips, f"unconsumed encoder skips: {skips}"
    return enc, dec


def sigma_schedule(steps: int = 18) -> np.ndarray:
    i = np.arange(steps, dtype=np.float64)
    s = (SIGMA_MAX ** (1 / RHO) + i / (steps - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    return np.append(s, 0.0).astype(np.float32)  # sigma_N = 0


def heun_sample(
    denoise,  # callable: (y [B,C,H,W] fp32, sigma scalar) -> D(y; sigma) [B,C,H,W]
    batch: int,
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    steps: int = 18,
):
    """EDM deterministic sampler (Algorithm 1, no stochastic churn)."""
    sig = sigma_schedule(steps)
    x = rng.standard_normal((batch, *shape)).astype(np.float32) * sig[0]
    for i in range(steps):
        s_cur, s_next = float(sig[i]), float(sig[i + 1])
        d_cur = (x - denoise(x, s_cur)) / s_cur
        x_next = x + (s_next - s_cur) * d_cur
        if s_next > 0:
            d_next = (x_next - denoise(x_next, s_next)) / s_next
            x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
        x = x_next
    return x


def make_model_denoiser(model_forward, patch: int, model_dim: int, labels_onehot: np.ndarray, channels=3, size=32):
    """Wrap a token-space F-network forward into an image-space denoiser.

    model_forward: callable(tokens [B,1,T,in], feats [B,1,1,D], onehot) -> tokens.
    """

    def denoise(y: np.ndarray, sigma: float) -> np.ndarray:
        b = y.shape[0]
        s = np.full(b, sigma, dtype=np.float32)
        c_skip, c_out, c_in, c_noise = precond_coeffs(s)
        bc = lambda v: v.reshape(b, 1, 1, 1).astype(np.float32)
        tokens = patchify(bc(c_in) * y, patch)
        feats = timestep_features(c_noise * CNOISE_FEAT_SCALE, model_dim).astype(np.float32)
        f = model_forward(tokens, feats, labels_onehot)
        f_img = unpatchify(f, patch, channels, size, size)
        return bc(c_skip) * y + bc(c_out) * f_img

    return denoise
