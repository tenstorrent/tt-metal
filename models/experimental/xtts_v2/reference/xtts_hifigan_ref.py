# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 HiFi-GAN vocoder generator (Block 4). Mirrors coqui
TTS.vocoder.models.hifigan_generator.HifiganGenerator op-for-op, with weight-norm folded.

Generator (input z [1,1024,L], d-vector g [1,512,1]):
  o = conv_pre(z) [k7,p3];  o = o + cond_layer(g) [k1, broadcast over time]
  for i in 0..3 (upsamples):
      o = leaky_relu(o, 0.1)
      o = ConvTranspose1d ups[i]           # 512->256->128->64->32, strides 8,8,2,2
      o = o + conds[i](g)                  # per-layer d-vector conditioning
      o = mean_j ResBlock1(o, kernel[j], dil[j])   # MRF, kernels [3,7,11], dil [1,3,5]
  o = leaky_relu(o, 0.1); o = conv_post(o) [k7,p3,no-bias]; o = tanh(o)  -> wav [1,1,L*256]

ResBlock1: for (c1,c2): x = x + c2(lrelu(c1(lrelu(x)))); c1 dilated, c2 dilation 1.
"""

import os

import torch
import torch.nn.functional as F

from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT, load_full_state

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "hifigan")
LRELU = 0.1
UPS = [(16, 8, 4), (16, 8, 4), (4, 2, 1), (4, 2, 1)]  # (kernel, stride, padding)
RES_K = [3, 7, 11]
RES_D = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def _pad(k, d):
    return (k * d - d) // 2


def load_hifigan_state(ckpt_path=DEFAULT_CKPT):
    """waveform_decoder weights, weight-norm folded, keyed relative to the generator."""
    full = load_full_state(ckpt_path)
    pref = "hifigan_decoder.waveform_decoder."
    raw = {k[len(pref) :]: v.float() for k, v in full.items() if k.startswith(pref)}
    out = {}
    keys = set(raw)
    for k, v in raw.items():
        if "parametrizations.weight.original1" in k:  # weight_norm v
            base = k.replace(".parametrizations.weight.original1", "")
            g = raw[base + ".parametrizations.weight.original0"]  # weight_g
            # exact PyTorch weight_norm fold (dim=0), bit-matches coqui's parametrization
            out[base + ".weight"] = torch._weight_norm(v, g, 0)
        elif "parametrizations.weight.original0" in k:
            continue
        else:
            out[k] = v
    return out


class HifiganReference:
    def __init__(self, ckpt_path=DEFAULT_CKPT):
        self.w = load_hifigan_state(ckpt_path)

    def _resblock(self, x, ridx, k, dils):
        w = self.w
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

    def __call__(self, z, g):  # z [1,1024,L], g [1,512,1] -> [1,1,L*256]
        w = self.w
        with torch.no_grad():
            o = F.conv1d(z, w["conv_pre.weight"], w["conv_pre.bias"], padding=3)
            o = o + F.conv1d(g, w["cond_layer.weight"], w["cond_layer.bias"])  # k1, broadcast over time
            for i in range(4):
                k, s, p = UPS[i]
                o = F.leaky_relu(o, LRELU)
                o = F.conv_transpose1d(o, w[f"ups.{i}.weight"], w[f"ups.{i}.bias"], stride=s, padding=p)
                o = o + F.conv1d(g, w[f"conds.{i}.weight"], w[f"conds.{i}.bias"])
                zsum = None
                for j in range(3):
                    r = self._resblock(o, i * 3 + j, RES_K[j], RES_D[j])
                    zsum = r if zsum is None else zsum + r
                o = zsum / 3
            o = F.leaky_relu(o)  # NB: coqui uses the DEFAULT slope 0.01 here (not LRELU_SLOPE 0.1)
            o = F.conv1d(o, w["conv_post.weight"], None, padding=3)  # bias=False
            return torch.tanh(o)


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


if __name__ == "__main__":
    ref = HifiganReference()
    z = torch.load(os.path.join(GOLDEN_DIR, "z.pt"))
    if z.dim() == 2:
        z = z.unsqueeze(0)
    g = torch.load(os.path.join(GOLDEN_DIR, "g.pt"))
    wav_g = torch.load(os.path.join(GOLDEN_DIR, "wav.pt"))
    wav = ref(z, g)
    print(f"wav {tuple(wav.shape)} vs golden {tuple(wav_g.shape)}  PCC = {_pcc(wav, wav_g):.6f}")
