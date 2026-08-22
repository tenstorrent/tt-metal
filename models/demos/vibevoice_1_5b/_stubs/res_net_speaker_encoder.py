# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `res_net_speaker_encoder` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.speaker_encoder`, a
`TTS.encoder.models.resnet.ResNetSpeakerEncoder` (H/ASP, encoder_type="ASP",
log_input=True, use_torch_spec=True). Its `forward(x, l2_norm=False)` takes a
RAW WAVEFORM `(N, 1, T)` and produces a `(N, 512)` speaker embedding:

    x = x.squeeze(1)                                   # (N, T)
    x = torch_spec(x)                                  # PreEmphasis -> MelSpectrogram -> (N, 64, F)
    x = (x + 1e-6).log()                               # log_input
    x = instancenorm(x).unsqueeze(1)                   # per-channel norm over time -> (N, 1, 64, F)
    x = bn1(relu(conv1(x)))                            # (N, 32, 64, F)
    x = layer1..layer4(x)                              # SE-ResNet, 3× stride-2 -> (N, 256, 8, F/8)
    x = x.reshape(N, -1, F')                           # (N, 2048, F')
    w = attention(x)                                   # Conv1d-ReLU-BN-Conv1d-Softmax(time)
    mu = sum(x*w, t); sg = sqrt(sum(x^2*w, t) - mu^2)  # ASP statistics pooling
    x = fc(cat([mu, sg]))                              # (N, 512)

Everything runs natively on device:
  * PreEmphasis  -> ttnn slice/concat + scaled subtract (no host op).
  * STFT/mel     -> windowed DFT and mel projection are pure `ttnn.matmul`;
                    only the reflect boundary pad (center=True) is host-side
                    data movement, exactly as the graduated `mel_spectrogram`
                    port does (ttnn has no reflect pad).
  * conv/bn/relu -> `ttnn.conv2d` with BatchNorm (eval) FOLDED into the
                    preceding conv weights where no nonlinearity intervenes,
                    and applied as a per-channel affine otherwise.
  * SE block     -> global-avg-pool (`ttnn.mean`) + two `ttnn.matmul` +
                    `ttnn.sigmoid`, broadcast-scaling the activation.
  * attention    -> 1×1 Conv1d == per-time-step `ttnn.matmul` over channels.
  * ASP pooling  -> `ttnn.sum` / elementwise / `ttnn.sqrt`.
  * fc           -> `ttnn.matmul`.

The STFT/mel matmuls and all pooling/statistics run in float32; the conv stack
runs bf16 activations with HiFi4 + fp32 accumulation for a clean PCC.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs.instance_norm1d import build as _b_inorm
from models.demos.vibevoice_1_5b._stubs.mel_spectrogram import build as _b_mel
from models.demos.vibevoice_1_5b._stubs.pre_emphasis import build as _b_pre
from models.demos.vibevoice_1_5b._stubs.s_e_basic_block import build as _b_block

HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind all trained weights (BN folded) and return a native ttnn forward closure."""
    import torch

    se = torch_module
    eps_bn = 1e-5

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=False,
    )
    ACT = ttnn.bfloat16

    def f32(t):
        return ttnn.as_tensor(
            t.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ---- mel front-end: graduated leaf stubs (PreEmphasis -> MelSpectrogram -> InstanceNorm1d) ----
    pre_emph = _b_pre(device, se.torch_spec[0])
    mel_fe = _b_mel(device, se.torch_spec[1])
    inorm = _b_inorm(device, se.instancenorm)

    # ---- helpers to fold / extract BatchNorm ----
    def bn_scale_shift(bn):
        a = (bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)).float()
        b = (bn.bias.detach() - bn.running_mean.detach() * a).float()
        return a, b  # [C]

    def conv_w(W):
        return ttnn.as_tensor(
            W.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def conv_b(b):
        return ttnn.as_tensor(
            b.reshape(1, 1, 1, -1).contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def affine_t(a):
        return ttnn.as_tensor(
            a.reshape(1, 1, 1, -1).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ---- conv1 (has bias) -> relu -> bn1 (affine, relu intervenes so NOT folded) ----
    c1_w = conv_w(se.conv1.weight.detach().float())
    c1_b = conv_b(se.conv1.bias.detach().float())
    c1 = dict(w=c1_w, b=c1_b, k=3, s=1, p=1, ic=1, oc=se.conv1.out_channels)
    bn1_a, bn1_b = bn_scale_shift(se.bn1)
    bn1_at, bn1_bt = affine_t(bn1_a), affine_t(bn1_b)

    # ---- SE-ResNet blocks: graduated s_e_basic_block leaf stubs (NCHW) ----
    block_stubs = []
    for layer in (se.layer1, se.layer2, se.layer3, se.layer4):
        block_stubs.append([_b_block(device, b) for b in layer])

    # ---- attention (Conv1d 1x1 == matmul over channels) ----
    att = se.attention
    a0 = att[0]  # Conv1d(2048,128,1)
    a2 = att[2]  # BatchNorm1d(128)
    a3 = att[3]  # Conv1d(128,2048,1)
    att0_w = f32(a0.weight.detach().squeeze(-1).t())  # [2048,128]
    att0_b = ttnn.as_tensor(
        a0.bias.detach().reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    att_bn_a, att_bn_b = bn_scale_shift(a2)
    att_bn_at = ttnn.as_tensor(
        att_bn_a.reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    att_bn_bt = ttnn.as_tensor(
        att_bn_b.reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    att3_w = f32(a3.weight.detach().squeeze(-1).t())  # [128,2048]
    att3_b = ttnn.as_tensor(
        a3.bias.detach().reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    fc_w = f32(se.fc.weight.detach().t())  # [4096,512]
    fc_b = ttnn.as_tensor(
        se.fc.bias.detach().reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ---------------- conv helper ----------------
    def run_conv(x_nhwc, spec, H, W):
        out, [oh, ow] = ttnn.conv2d(
            input_tensor=x_nhwc,
            weight_tensor=spec["w"],
            in_channels=spec["ic"],
            out_channels=spec["oc"],
            device=device,
            bias_tensor=spec["b"],
            kernel_size=(spec["k"], spec["k"]),
            stride=(spec["s"], spec["s"]),
            padding=(spec["p"], spec["p"]),
            dilation=(1, 1),
            batch_size=1,
            input_height=H,
            input_width=W,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=ACT,
        )
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.reshape(out, (1, oh, ow, spec["oc"]))
        return out, oh, ow

    def affine(x, a, b):
        return ttnn.add(ttnn.multiply(x, a), b)

    def to_conv_in(x):
        # elementwise ops leave x as bf16 tile [1,H,W,C]; conv2d wants row-major
        return ttnn.to_layout(ttnn.typecast(x, ACT), ttnn.ROW_MAJOR_LAYOUT)

    # ---------------- forward ----------------
    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.as_tensor(
                x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        # (N,1,T) -> (1,T)
        x = ttnn.reshape(x, (1, -1))

        # ---- PreEmphasis (graduated leaf: pre_emphasis) ----
        y = pre_emph(x)  # [1, T]

        # ---- MelSpectrogram STFT + mel projection (graduated leaf: mel_spectrogram) ----
        mel = mel_fe(y)  # [1, 64, F]

        # ---- log_input ----
        mel = ttnn.log(ttnn.add(mel, 1e-6))

        # ---- InstanceNorm1d over time (graduated leaf: instance_norm1d) ----
        h = inorm(mel)  # [1, 64, F]
        n_mels = int(h.shape[1])
        F = int(h.shape[-1])

        # unsqueeze(1) -> NCHW (1,1,64,F) -> NHWC (1,64,F,1)
        h = ttnn.reshape(h, (1, n_mels, F, 1))
        H, W = n_mels, F

        # ---- conv1 -> relu -> bn1 ----
        out, H, W = run_conv(to_conv_in(h), c1, H, W)
        out = ttnn.typecast(out, ttnn.float32)
        out = ttnn.relu(out)
        out = affine(out, bn1_at, bn1_bt)

        # ---- SE-ResNet layers (graduated leaf: s_e_basic_block, NCHW) ----
        out = ttnn.permute(out, (0, 3, 1, 2))  # NHWC -> NCHW for the block stubs
        for layer in block_stubs:
            for block in layer:
                out = block(out)  # NCHW -> NCHW

        # out: NCHW [1,C,H,W] float32.  C=256, H=8, W=13
        C = int(out.shape[1])
        H = int(out.shape[2])
        W = int(out.shape[3])
        # merge (C,H) -> reshape (1, C*H, W) -> time-major (1, W, C*H)
        merged = ttnn.reshape(out, (1, C * H, W))  # [1, 2048, W]
        xt = ttnn.permute(merged, (0, 2, 1))  # [1, W, 2048]  (time-major)
        CH = C * H

        # ---- attention ----
        w = ttnn.matmul(xt, att0_w, compute_kernel_config=compute_config)  # [1,W,128]
        w = ttnn.add(w, att0_b)
        w = ttnn.relu(w)
        w = ttnn.add(ttnn.multiply(w, att_bn_at), att_bn_bt)  # BN1d affine
        w = ttnn.matmul(w, att3_w, compute_kernel_config=compute_config)  # [1,W,2048]
        w = ttnn.add(w, att3_b)
        w = ttnn.softmax(w, dim=1)  # softmax over time

        # ---- ASP statistics pooling (over time dim=1) ----
        xw = ttnn.multiply(xt, w)
        mu = ttnn.sum(xw, dim=1)  # [1,2048]
        x2w = ttnn.multiply(ttnn.multiply(xt, xt), w)
        s2 = ttnn.sum(x2w, dim=1)  # [1,2048]
        var2 = ttnn.subtract(s2, ttnn.multiply(mu, mu))
        var2 = ttnn.clamp(var2, 1e-5, None)
        sg = ttnn.sqrt(var2)
        emb = ttnn.concat([mu, sg], dim=1)  # [1,4096]

        # ---- fc ----
        emb = ttnn.reshape(emb, (1, 1, 2 * CH))
        y_out = ttnn.matmul(emb, fc_w, compute_kernel_config=compute_config)  # [1,1,512]
        y_out = ttnn.add(y_out, fc_b)
        y_out = ttnn.reshape(y_out, (1, se.proj_dim))
        return y_out

    return forward


def res_net_speaker_encoder(*args, **kwargs):
    raise RuntimeError(
        "res_net_speaker_encoder requires build(device, torch_module) to bind the "
        "trained STFT/mel/conv/attention/fc weights; the bare callable has no parameters."
    )
