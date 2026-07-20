# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the XTTS-v2 ResNet speaker encoder (Block 2, "Branch B" -> d-vector).

Reference: models/experimental/xtts_v2/reference/xtts_speaker_ref.py

Block boundary: input = log-mel `logmel` [1, 64, T] (the mel front-end / STFT stays on CPU,
as it is not a TTNN op and python_env has no torchaudio); output = d-vector [1, 512] which
the caller reshapes to the [1, 512, 1] speaker_embedding.

Pipeline (all on device from logmel):
    logmel [1,64,T]
      -> InstanceNorm1d(64) over time (biased var, eps 1e-5), unsqueeze -> NHWC [1,64,T,1]
      -> conv1 (1->32, 3x3, s1, p1) -> relu -> bn1                      -> [1,64,T,32]
      -> layer1 (3x SEBasicBlock, 32, s1)                              -> [1,64,T,32]
      -> layer2 (4x, 64, s2)                                           -> [1,32,T/2,64]
      -> layer3 (6x, 128, s2)                                          -> [1,16,T/4,128]
      -> layer4 (3x, 256, s2)                                          -> [1,8,T/8,256]
      -> reshape to [1, T'', 2048]  (feature = ch*8 + mel, channel-major, mel-minor)
      -> attention (Conv1d 2048->128 -> relu -> BN1d(128) -> Conv1d 128->2048 -> softmax/time)
      -> ASP pooling: mu = sum(x*w,t); sg = sqrt(clamp(sum(x^2*w,t) - mu^2, 1e-5)); cat -> [1,4096]
      -> fc (4096 -> 512) -> L2 normalize -> [1, 512]

Convs use `ttnn.conv2d` in channels-last (NHWC) interleaved layout; BatchNorm (eval) is folded
to a per-channel affine (scale = w/sqrt(var+eps), shift = b - mean*scale) applied as a
broadcast multiply+add on the last (channel) dim. SE global-avg-pool is a mean over the H*W
tokens; its two FC layers + attention Conv1d(1x1) layers + fc are all plain matmuls.
"""

import ttnn

import torch

from models.experimental.xtts_v2.reference.xtts_speaker_ref import (
    LAYERS,
    NUM_FILTERS,
    load_speaker_state,
)

BN_EPS = 1e-5
IN_EPS = 1e-5  # InstanceNorm1d eps
OUTMAP = 8  # input_dim / 8
FEAT = NUM_FILTERS[3] * OUTMAP  # 2048


def _compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _bn_affine(w, b, rm, rv, eps=BN_EPS):
    """Fold BatchNorm eval params to (scale, shift): y = x*scale + shift."""
    scale = w / torch.sqrt(rv + eps)
    shift = b - rm * scale
    return scale, shift


def preprocess_speaker_parameters(device, ckpt_path=None, conv_dtype=ttnn.bfloat16, lin_dtype=ttnn.bfloat16):
    w = load_speaker_state(ckpt_path) if ckpt_path else load_speaker_state()

    def conv_w(name):  # conv weight [out,in,kh,kw] -> ttnn host row-major
        return ttnn.from_torch(w[name], dtype=conv_dtype)

    def bias4(x):  # conv bias -> [1,1,1,out]
        return ttnn.from_torch(x.reshape(1, 1, 1, -1), dtype=conv_dtype)

    def affine_nhwc(scale, shift):  # per-channel affine tiles [1,1,1,C] for NHWC
        return {
            "scale": ttnn.from_torch(
                scale.reshape(1, 1, 1, -1), dtype=conv_dtype, layout=ttnn.TILE_LAYOUT, device=device
            ),
            "shift": ttnn.from_torch(
                shift.reshape(1, 1, 1, -1), dtype=conv_dtype, layout=ttnn.TILE_LAYOUT, device=device
            ),
        }

    def bn_nhwc(prefix):
        scale, shift = _bn_affine(
            w[f"{prefix}.weight"], w[f"{prefix}.bias"], w[f"{prefix}.running_mean"], w[f"{prefix}.running_var"]
        )
        return affine_nhwc(scale, shift)

    def linT(name):  # nn.Linear weight [out,in] -> ttnn [in,out]
        return ttnn.from_torch(w[name].t().contiguous(), dtype=lin_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def conv1d_linT(name):  # Conv1d weight [out,in,1] -> ttnn [in,out] (1x1 conv == linear)
        return ttnn.from_torch(
            w[name].squeeze(-1).t().contiguous(), dtype=lin_dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def vec(x):  # bias [n] -> ttnn [1,n]
        return ttnn.from_torch(x.reshape(1, -1), dtype=lin_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    p = {
        "conv1_w": conv_w("conv1.weight"),
        "conv1_b": bias4(w["conv1.bias"]),
        "bn1": bn_nhwc("bn1"),
        "layers": [],
    }

    def se_params(prefix):
        return {
            "fc0_w": linT(f"{prefix}.se.fc.0.weight"),
            "fc0_b": vec(w[f"{prefix}.se.fc.0.bias"]),
            "fc2_w": linT(f"{prefix}.se.fc.2.weight"),
            "fc2_b": vec(w[f"{prefix}.se.fc.2.bias"]),
        }

    inplanes = NUM_FILTERS[0]
    for li, (planes, nblocks) in enumerate(zip(NUM_FILTERS, LAYERS)):
        stride = 1 if li == 0 else 2
        blocks = []
        for bi in range(nblocks):
            s = stride if bi == 0 else 1
            in_ch = inplanes if bi == 0 else planes
            prefix = f"layer{li + 1}.{bi}"
            blk = {
                "in": in_ch,
                "planes": planes,
                "stride": s,
                "conv1_w": conv_w(f"{prefix}.conv1.weight"),
                "bn1": bn_nhwc(f"{prefix}.bn1"),
                "conv2_w": conv_w(f"{prefix}.conv2.weight"),
                "bn2": bn_nhwc(f"{prefix}.bn2"),
                "se": se_params(prefix),
                "downsample": False,
            }
            if f"{prefix}.downsample.0.weight" in w:
                blk["downsample"] = True
                blk["ds_w"] = conv_w(f"{prefix}.downsample.0.weight")
                ds_scale, ds_shift = _bn_affine(
                    w[f"{prefix}.downsample.1.weight"],
                    w[f"{prefix}.downsample.1.bias"],
                    w[f"{prefix}.downsample.1.running_mean"],
                    w[f"{prefix}.downsample.1.running_var"],
                )
                blk["ds_bn"] = affine_nhwc(ds_scale, ds_shift)
            blocks.append(blk)
        inplanes = planes
        p["layers"].append(blocks)

    # attention (ASP) + fc
    a_scale, a_shift = _bn_affine(
        w["attention.2.weight"], w["attention.2.bias"], w["attention.2.running_mean"], w["attention.2.running_var"]
    )
    p["attn"] = {
        "w1": conv1d_linT("attention.0.weight"),
        "b1": vec(w["attention.0.bias"]),
        "bn_scale": ttnn.from_torch(a_scale.reshape(1, 1, -1), dtype=lin_dtype, layout=ttnn.TILE_LAYOUT, device=device),
        "bn_shift": ttnn.from_torch(a_shift.reshape(1, 1, -1), dtype=lin_dtype, layout=ttnn.TILE_LAYOUT, device=device),
        "w2": conv1d_linT("attention.3.weight"),
        "b2": vec(w["attention.3.bias"]),
    }
    p["fc_w"] = linT("fc.weight")
    p["fc_b"] = vec(w["fc.bias"])
    return p


class TTNNSpeakerEncoder:
    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.cc = _compute_config()
        self.conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

    # ---- primitives ----------------------------------------------------------
    def _conv2d(self, x, weight, bias, in_ch, out_ch, H, W, stride, pad, ksize):
        out, [ho, wo], [_, _] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=in_ch,
            out_channels=out_ch,
            device=self.device,
            kernel_size=(ksize, ksize),
            stride=(stride, stride),
            padding=(pad, pad),
            batch_size=1,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=self.cc,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)  # sharded -> interleaved
        out = ttnn.reshape(out, (1, ho, wo, out_ch))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return out, ho, wo

    def _bn(self, x, bn):  # per-channel affine on NHWC last dim
        return ttnn.add(ttnn.multiply(x, bn["scale"]), bn["shift"])

    def _se(self, x, se, C, H, W):
        xr = ttnn.reshape(x, (1, H * W, C))
        m = ttnn.mean(xr, dim=1, keepdim=True)  # [1,1,C] global avg pool
        m = ttnn.reshape(m, (1, C))
        h = ttnn.linear(m, se["fc0_w"], bias=se["fc0_b"], compute_kernel_config=self.cc)
        h = ttnn.relu(h)
        h = ttnn.linear(h, se["fc2_w"], bias=se["fc2_b"], compute_kernel_config=self.cc)
        s = ttnn.sigmoid(h)
        s = ttnn.reshape(s, (1, 1, 1, C))
        return ttnn.multiply(x, s)  # broadcast over H,W

    def _block(self, x, blk, H, W):
        planes = blk["planes"]
        s = blk["stride"]
        residual = x
        out, ho, wo = self._conv2d(x, blk["conv1_w"], None, blk["in"], planes, H, W, s, 1, 3)
        out = ttnn.relu(out)
        out = self._bn(out, blk["bn1"])
        out, ho, wo = self._conv2d(out, blk["conv2_w"], None, planes, planes, ho, wo, 1, 1, 3)
        out = self._bn(out, blk["bn2"])
        out = self._se(out, blk["se"], planes, ho, wo)
        if blk["downsample"]:
            residual, _, _ = self._conv2d(x, blk["ds_w"], None, blk["in"], planes, H, W, s, 0, 1)
            residual = self._bn(residual, blk["ds_bn"])
        out = ttnn.add(out, residual)
        out = ttnn.relu(out)
        return out, ho, wo

    def _instancenorm(self, logmel):  # [1,64,T] -> NHWC [1,64,T,1]
        mean = ttnn.mean(logmel, dim=2, keepdim=True)  # [1,64,1] over time
        xc = ttnn.subtract(logmel, mean)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=2, keepdim=True)  # biased
        inv = ttnn.rsqrt(ttnn.add(var, IN_EPS))
        y = ttnn.multiply(xc, inv)  # [1,64,T]
        _, C, T = y.shape
        return ttnn.reshape(y, (1, C, T, 1)), C, T

    # ---- forward -------------------------------------------------------------
    def __call__(self, logmel, return_intermediates=False):
        """logmel: ttnn tensor [1, 64, T] (TILE). Returns d-vector [1, 512]."""
        inter = {}
        x, H, W = self._instancenorm(logmel)  # [1,64,T,1]
        inter["instancenorm"] = x

        x, H, W = self._conv2d(x, self.p["conv1_w"], self.p["conv1_b"], 1, NUM_FILTERS[0], H, W, 1, 1, 3)
        x = ttnn.relu(x)
        x = self._bn(x, self.p["bn1"])
        inter["conv1"] = x

        for li, blocks in enumerate(self.p["layers"]):
            for blk in blocks:
                x, H, W = self._block(x, blk, H, W)
            inter[f"layer{li + 1}"] = x

        # x: NHWC [1, H=8, W=T'', C=256] -> [1, T'', 2048] feature = c*8 + m
        C = NUM_FILTERS[3]
        xp = ttnn.permute(x, (0, 2, 3, 1))  # [1, W, C, H]
        x_tf = ttnn.reshape(xp, (1, W, FEAT))  # feature index = c*8 + m
        inter["reshape"] = x_tf

        a = self.p["attn"]
        h = ttnn.linear(x_tf, a["w1"], bias=a["b1"], compute_kernel_config=self.cc)  # [1,W,128]
        h = ttnn.relu(h)
        h = ttnn.add(ttnn.multiply(h, a["bn_scale"]), a["bn_shift"])  # BN1d(128)
        h = ttnn.linear(h, a["w2"], bias=a["b2"], compute_kernel_config=self.cc)  # [1,W,2048]
        # softmax over time (dim=1) -> permute to feature-first, softmax last dim
        hf = ttnn.permute(h, (0, 2, 1))  # [1,2048,W]
        w_ft = ttnn.softmax(hf, dim=-1, compute_kernel_config=self.cc)
        inter["attn_w"] = w_ft

        x_ft = ttnn.permute(x_tf, (0, 2, 1))  # [1,2048,W]
        mu = ttnn.sum(ttnn.multiply(x_ft, w_ft), dim=2, keepdim=True)  # [1,2048,1]
        sq = ttnn.sum(ttnn.multiply(ttnn.multiply(x_ft, x_ft), w_ft), dim=2, keepdim=True)
        var = ttnn.subtract(sq, ttnn.multiply(mu, mu))
        var = ttnn.add(ttnn.relu(ttnn.subtract(var, 1e-5)), 1e-5)  # clamp(min=1e-5)
        sg = ttnn.sqrt(var)
        mu = ttnn.reshape(mu, (1, FEAT))
        sg = ttnn.reshape(sg, (1, FEAT))
        emb_in = ttnn.concat([mu, sg], dim=1)  # [1,4096]
        inter["pool"] = emb_in

        emb = ttnn.linear(emb_in, self.p["fc_w"], bias=self.p["fc_b"], compute_kernel_config=self.cc)  # [1,512]
        inter["fc"] = emb
        # L2 normalize over dim=1
        nrm = ttnn.sqrt(ttnn.sum(ttnn.multiply(emb, emb), dim=1, keepdim=True))  # [1,1]
        emb = ttnn.multiply(emb, ttnn.reciprocal(nrm))  # broadcast
        inter["emb"] = emb
        if return_intermediates:
            return emb, inter
        return emb
