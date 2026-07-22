# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN port of HunyuanImage-3.0's gen_image conv heads (host-glue lever #1).

Stage 1 (this file): `final_layer` (UNetUp) — the VELOCITY head, download-side.
Running it on TT keeps the transformer hidden ON-DEVICE (download only the small
velocity [B,32,64,64] instead of the [B,4116,4096] hidden). Stage 2 = patch_embed
(upload-side). See HOST_GLUE_PORT_PLAN.md.

Mirrors the model's OWN VAE ResBlock (wormhole/tt/vae/ttnn_vae_resnet.py) +
grafts the z_image adaLN scale/shift for the timestep (AdaGN). DRAFT — conv/GN
layouts marked `# VERIFY`; validate with test_host_glue_pcc.py (PCC vs host >= 0.99)
before wiring into the loop.

torch final_layer = UNetUp(patch_size=1, in=4096, hidden=1024, out=32, emb=4096, out_norm=True):
  model[0] = ResBlock(4096->1024, emb=4096):
    in_layers  = GroupNorm(32,4096) + SiLU + Conv2d(4096->1024,k3,p1)
    emb_layers = SiLU + Linear(4096 -> 2*1024)
    out_layers = GroupNorm(32,1024) [+AdaGN scale/shift] + SiLU + Conv2d(1024->1024,k3,p1)
    skip       = Conv2d(4096->1024,k1)
  model[1] = GroupNorm(32,1024) + SiLU + Conv2d(1024->32,k3,p1)
  forward(x[B,4096tok,4096], emb[B,4096], h=w=64): rearrange->[B,4096,64,64]; ResBlock; out-norm -> [B,32,64,64]
"""
from __future__ import annotations

import torch

import ttnn

GN_GROUPS = 32
GN_EPS = 1e-5  # upstream normalization() = nn.GroupNorm(32, C) -> torch default eps=1e-5 (NOT the VAE's 1e-6)


def _repl(device, t, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    kw = {}
    try:
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except Exception:
        pass
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)


def _conv_cfg(device):
    cc = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat8_b, deallocate_activation=False)
    cc.enable_act_double_buffer = True
    cc.enable_weights_double_buffer = True
    comp = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return cc, comp


class _Conv:
    """3x3 (or 1x1) conv over a [1,1,H*W,C] NHWC-flat activation. Weights prepared+cached
    on first call (return_weights_and_bias)."""

    def __init__(self, device, tconv, H, W, k=3, p=1):
        self.device, self.H, self.W = device, H, W
        self.cin = int(tconv.weight.shape[1])
        self.cout = int(tconv.weight.shape[0])
        self.k, self.p = k, p
        self.w = _repl(device, tconv.weight, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)  # OIHW raw # VERIFY
        b = tconv.bias.reshape(1, 1, 1, -1) if tconv.bias is not None else torch.zeros(1, 1, 1, self.cout)
        self.b = _repl(device, b, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.cfg, self.comp = _conv_cfg(device)

    def __call__(self, x):  # x: [1,1,H*W,cin] NHWC-flat TILE
        y, [self.w, self.b] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.w,
            bias_tensor=self.b,
            in_channels=self.cin,
            out_channels=self.cout,
            batch_size=1,
            input_height=self.H,
            input_width=self.W,
            kernel_size=(self.k, self.k),
            stride=(1, 1),
            padding=(self.p, self.p),
            dilation=(1, 1),
            groups=1,
            device=self.device,
            conv_config=self.cfg,
            compute_config=self.comp,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        return y  # [1,1,H*W,cout]


class _GroupNorm:
    def __init__(self, device, tnorm, channels, core_grid=None):
        self.device, self.channels = device, channels
        self.core_grid = core_grid or ttnn.CoreGrid(y=8, x=8)
        ncores = self.core_grid.y
        self.mask = ttnn.to_device(
            ttnn.create_group_norm_input_mask(channels, GN_GROUPS, ncores, ttnn.bfloat16), device
        )
        self.w = _repl(
            device, ttnn.create_group_norm_weight_bias_rm(tnorm.weight, channels, ncores), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.bi = _repl(
            device, ttnn.create_group_norm_weight_bias_rm(tnorm.bias, channels, ncores), layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def __call__(self, x):  # x: [1,1,H*W,C] bf16
        return ttnn.group_norm(
            x,
            num_groups=GN_GROUPS,
            input_mask=self.mask,
            weight=self.w,
            bias=self.bi,
            epsilon=GN_EPS,
            core_grid=self.core_grid,
            dtype=ttnn.bfloat16,
            inplace=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class _ResBlockTT:
    def __init__(self, device, tres, H, W):
        self.device, self.H, self.W = device, H, W
        self.cin = int(tres.in_layers[2].weight.shape[1])
        self.cout = int(tres.in_layers[2].weight.shape[0])
        self.norm1 = _GroupNorm(device, tres.in_layers[0], self.cin)
        self.conv1 = _Conv(device, tres.in_layers[2], H, W)
        self.emb_w = _repl(device, tres.emb_layers[1].weight.t().contiguous())  # [emb, 2*out]
        self.emb_b = _repl(device, tres.emb_layers[1].bias.reshape(1, -1))
        self.norm2 = _GroupNorm(device, tres.out_layers[0], self.cout)
        self.conv2 = _Conv(device, tres.out_layers[3], H, W)
        self.skip = None if self.cin == self.cout else _Conv(device, tres.skip_connection, H, W, k=1, p=0)

    def __call__(self, x, emb):  # x [1,1,H*W,cin] bf16 ; emb [1,emb] bf16
        h = self.conv1(ttnn.silu(self.norm1(x)))  # [1,1,H*W,cout]
        e = ttnn.silu(emb)
        e = ttnn.linear(e, self.emb_w, bias=self.emb_b)  # [1, 2*cout]
        scale = ttnn.reshape(ttnn.slice(e, [0, 0], [1, self.cout]), [1, 1, 1, self.cout])
        shift = ttnn.reshape(ttnn.slice(e, [0, self.cout], [1, 2 * self.cout]), [1, 1, 1, self.cout])
        h = self.norm2(h)
        h = ttnn.add(ttnn.multiply(h, ttnn.add(scale, 1.0)), shift)  # AdaGN: GN(h)*(1+scale)+shift
        h = self.conv2(ttnn.silu(h))
        skip = self.skip(x) if self.skip is not None else x
        return ttnn.add(skip, h)


class FinalLayerTT:
    """final_layer (UNetUp) on TT: hidden [1,4096tok,4096] + emb [1,4096] -> velocity [1,32,64,64]."""

    def __init__(self, device, model, token_h=64, token_w=64):
        self.device, self.H, self.W = device, token_h, token_w
        fl = model.final_layer.float()
        self.res = _ResBlockTT(device, fl.model[0], token_h, token_w)
        self.on = _GroupNorm(device, fl.model[1][0], int(fl.model[1][2].weight.shape[1]))  # GroupNorm(1024)
        self.oconv = _Conv(device, fl.model[1][2], token_h, token_w)  # conv(1024->32)

    def __call__(self, hidden, emb):
        # hidden [1, H*W, 4096] tokens == [1,1,H*W,4096] NHWC-flat (B=1) -> reshape only (# VERIFY)
        x = ttnn.reshape(hidden, [1, 1, self.H * self.W, hidden.shape[-1]])
        x = self.res(x, emb)  # [1,1,H*W,1024]
        x = self.oconv(ttnn.silu(self.on(x)))  # [1,1,H*W,32]
        return x  # caller reshapes/permutes [1,1,H*W,32] -> [1,32,H,W]


def build_final_layer(device, model, token_h=64, token_w=64):
    return FinalLayerTT(device, model, token_h, token_w)


class PatchEmbedTT:
    """patch_embed (UNetDown) on TT (stage 2/3, upload-side): VAE latent [1,32,H,W] NCHW +
    emb [1,4096] -> image tokens [1, H*W, 4096]. Lets the image tokens be built ON-DEVICE from
    the small uploaded latent (~0.5 MB) instead of host convs + uploading [1,4116,4096] embeds.

    torch patch_embed = UNetDown(patch_size=1, in=32, hidden=1024, out=4096, emb=4096):
      model[0] = Conv2d(32->1024,k3,p1) ; model[1] = ResBlock(1024->4096) ; rearrange b c h w -> b (h w) c
    emb comes from time_embed(t) (NOT time_embed_2, which feeds final_layer)."""

    def __init__(self, device, model, token_h=64, token_w=64):
        self.device, self.H, self.W = device, token_h, token_w
        pe = model.patch_embed.float()
        self.in_ch = int(pe.model[0].weight.shape[1])  # VAE latent channels (32)
        self.conv0 = _Conv(device, pe.model[0], token_h, token_w)  # conv(32->1024)
        self.res = _ResBlockTT(device, pe.model[1], token_h, token_w)  # ResBlock(1024->4096)

    def __call__(self, latent, emb):  # latent: torch [1,32,H,W] NCHW ; emb: ttnn [1,4096]
        # ONLY place a NCHW->NHWC permute is needed (the initial latent); tokens come out NHWC-flat.
        x = latent.permute(0, 2, 3, 1).reshape(1, 1, self.H * self.W, self.in_ch).contiguous()
        x = _repl(self.device, x)
        x = self.conv0(x)  # [1,1,H*W,1024]
        x = self.res(x, emb)  # [1,1,H*W,4096]
        return ttnn.reshape(x, [1, self.H * self.W, x.shape[-1]])  # tokens [1,H*W,4096]


def build_patch_embed(device, model, token_h=64, token_w=64):
    return PatchEmbedTT(device, model, token_h, token_w)
