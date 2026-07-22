# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 HiFi-GAN vocoder generator (Block 4).

Self-contained op-for-op (torch + F only) reimplementation of coqui's
TTS.vocoder.models.hifigan_generator.HifiganGenerator, as configured inside HifiDecoder,
with weight-norm folded into plain conv weights.

Block boundary (the port target): z [1,1024,L] + d-vector g [1,512,1] -> waveform [1,1,L*256].
The latents->z step in HifiDecoder.forward (two F.interpolate resamplings: ar_mel_length /
output_hop = 1024/256 = 4x, then 24000/22050 sample-rate) has no weights and runs on host.

Generator config (HifiDecoder): in=1024, upsample_initial=512, upsample_factors=[8,8,2,2],
upsample_kernels=[16,16,4,4], resblock kernels [3,7,11] / dilations [[1,3,5]]*3, cond_channels=512,
conv_pre/conv_post weight-norm removed, conv_post bias=False, d-vector conditioning in each up layer.

Flow:
  o = conv_pre(z) [k7,p3];  o = o + cond_layer(g) [k1, broadcast over time]
  for i in 0..3:
      o = leaky_relu(o, 0.1); o = ConvTranspose1d ups[i]; o = o + conds[i](g)
      o = mean_j ResBlock1(o, kernel[j], dil[j])          # MRF
  o = leaky_relu(o)  # NB: DEFAULT slope 0.01 here, not 0.1
  o = conv_post(o) [k7,p3,no-bias]; o = tanh(o)
ResBlock1: for (c1,c2): x = x + c2(lrelu(c1(lrelu(x)))); c1 dilated, c2 dilation 1.

Run (needs repo root on PYTHONPATH):
    PYTHONPATH=<repo> python models/experimental/xtts_v2/reference/xtts_hifigan_ref.py
"""

import os

import torch
import torch.nn.functional as F

from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT, load_full_state, pcc

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "hifigan")
WD_PREFIX = "hifigan_decoder.waveform_decoder."

LRELU = 0.1
IN_CHANNELS = 1024
COND_CHANNELS = 512
UPS = [(16, 8, 4), (16, 8, 4), (4, 2, 1), (4, 2, 1)]  # (kernel, stride, padding); padding=(k-u)//2
RES_K = [3, 7, 11]
RES_D = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def _pad(k, d):  # coqui get_padding
    return (k * d - d) // 2


def load_hifigan_state(ckpt_path=DEFAULT_CKPT):
    """waveform_decoder weights with torch weight-norm folded (dim=0), keyed relative to the generator.

    ups.* and resblocks.* are stored as weight_norm parametrizations (original0=g, original1=v);
    we fold them back to plain `.weight`. conv_pre/conv_post/cond_layer/conds are already plain."""
    full = load_full_state(ckpt_path)
    raw = {k[len(WD_PREFIX) :]: v.float() for k, v in full.items() if k.startswith(WD_PREFIX)}
    out = {}
    for k, v in raw.items():
        if "parametrizations.weight.original1" in k:  # v (direction)
            base = k.replace(".parametrizations.weight.original1", "")
            g = raw[base + ".parametrizations.weight.original0"]  # g (magnitude)
            out[base + ".weight"] = torch._weight_norm(v, g, 0)  # exact weight_norm fold (dim=0)
        elif "parametrizations.weight.original0" in k:
            continue
        else:
            out[k] = v
    return out


def _resblock(x, w, ridx, k, dils):  # ResBlock1
    for j in range(3):
        xt = F.leaky_relu(x, LRELU)
        xt = F.conv1d(
            xt,
            w[f"resblocks.{ridx}.convs1.{j}.weight"],
            w[f"resblocks.{ridx}.convs1.{j}.bias"],
            dilation=dils[j],
            padding=_pad(k, dils[j]),
        )
        xt = F.leaky_relu(xt, LRELU)
        xt = F.conv1d(
            xt,
            w[f"resblocks.{ridx}.convs2.{j}.weight"],
            w[f"resblocks.{ridx}.convs2.{j}.bias"],
            dilation=1,
            padding=_pad(k, 1),
        )
        x = xt + x
    return x


@torch.no_grad()
def generator(z, w, g):  # z [1,1024,L], g [1,512,1] -> [1,1,L*256]
    o = F.conv1d(z, w["conv_pre.weight"], w["conv_pre.bias"], padding=3)
    o = o + F.conv1d(g, w["cond_layer.weight"], w["cond_layer.bias"])  # k1, broadcasts over time
    for i in range(len(UPS)):
        k, s, p = UPS[i]
        o = F.leaky_relu(o, LRELU)
        o = F.conv_transpose1d(o, w[f"ups.{i}.weight"], w[f"ups.{i}.bias"], stride=s, padding=p)
        o = o + F.conv1d(g, w[f"conds.{i}.weight"], w[f"conds.{i}.bias"])  # per-layer d-vector conditioning
        z_sum = None
        for j in range(len(RES_K)):
            r = _resblock(o, w, i * len(RES_K) + j, RES_K[j], RES_D[j])
            z_sum = r if z_sum is None else z_sum + r
        o = z_sum / len(RES_K)  # MRF average
    o = F.leaky_relu(o)  # DEFAULT slope 0.01 here (coqui quirk), not LRELU
    o = F.conv1d(o, w["conv_post.weight"], None, padding=3)  # bias=False
    return torch.tanh(o)


def make_synthetic_inputs(n_latent=32, seed=0):
    """Deterministic generator input z [1,1024,L] and d-vector g [1,512,1]."""
    gz = torch.Generator().manual_seed(seed)
    gg = torch.Generator().manual_seed(seed + 1)
    z = torch.randn(1, IN_CHANNELS, n_latent, generator=gz)
    g = torch.randn(1, COND_CHANNELS, 1, generator=gg)
    return z, g


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--n-latent", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[ref] loading vocoder weights from {args.ckpt}")
    w = load_hifigan_state(args.ckpt)
    z, g = make_synthetic_inputs(args.n_latent)
    wav = generator(z, w, g)  # [1, 1, L*256]
    print(f"[ref] z {tuple(z.shape)}, g {tuple(g.shape)} -> wav {tuple(wav.shape)} (min {wav.min():.3f}, max {wav.max():.3f})")

    torch.save(z, os.path.join(args.out, "z.pt"))
    torch.save(g, os.path.join(args.out, "g.pt"))
    torch.save(wav, os.path.join(args.out, "wav.pt"))
    torch.save({"n_latent": args.n_latent, "upsample": 256}, os.path.join(args.out, "meta.pt"))
    print(f"[ref] wrote vocoder goldens to {args.out}")


if __name__ == "__main__":
    main()
