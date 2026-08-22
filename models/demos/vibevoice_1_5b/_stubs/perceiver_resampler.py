# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `perceiver_resampler` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_perceiver`, a
`TTS.tts.layers.xtts.perceiver_encoder.PerceiverResampler`
(dim=1024, depth=2, num_latents=32, dim_head=64, heads=8, ff_mult=4,
use_flash_attn=False, dim_context==dim so `proj_context` is Identity).

    latents = repeat(self.latents, "n d -> b n d")            # [1, 32, 1024]
    for attn, ff in layers:                                   # depth 2
        latents = attn(latents, x) + latents
        latents = ff(latents) + latents
    return self.norm(latents)                                 # RMSNorm

Attention (cross, `cross_attn_include_queries=True`, no biases):
    context = cat([latents, x], dim=-2)                       # [1, 32+T, 1024]
    q = latents @ Wq^T                                        # [1, 32, 512]
    k, v = (context @ Wkv^T).chunk(2, -1)                     # [1, 32+T, 512] each
    heads: [1, 8, ., 64];  sim = q·kᵀ * 64^-0.5;  softmax;  out = sim·v
    out = merge_heads @ Wo^T                                  # [1, 32, 1024]

FeedForward: Linear(1024->5460) -> GEGLU -> Linear(2730->1024) (both biased).
RMSNorm: F.normalize(x,-1) * sqrt(dim) * gamma == rms_norm(x) * gamma.

Captured: x `[1, 259, 1024]` -> `[1, 32, 1024]`. Everything runs natively in ttnn
(float32 + HiFi4 matmuls); the GEGLU reuses the graduated `g_e_g_l_u` port.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs.attend import build as _build_attend
from models.demos.vibevoice_1_5b._stubs.g_e_g_l_u import _geglu

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained perceiver weights and return a native ttnn forward closure."""

    m = torch_module.float()

    heads = int(m.layers[0][0].heads)
    # SDPA scaling (m.layers[i][0].scale == head_dim**-0.5) is applied inside the
    # graduated `attend` leaf stub, so it is no longer needed here.
    dim = int(m.latents.shape[-1])
    # q/k/v project to heads*dim_head (== 512 here), which can differ from `dim`.
    head_dim = int(m.layers[0][0].to_q.weight.shape[0]) // heads

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _w(t):
        return ttnn.as_tensor(
            t.detach().contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _wt(t):
        # nn.Linear weight [out, in] -> [in, out] for x @ W.
        return ttnn.as_tensor(
            t.detach().t().contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _b(t, n):
        return ttnn.as_tensor(
            t.detach().reshape(1, 1, n).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    latents0 = ttnn.as_tensor(
        m.latents.detach().reshape(1, *m.latents.shape).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    layers = []
    for attn, ff in m.layers:
        inner = attn.to_q.weight.shape[0]  # heads*head_dim
        layers.append(
            {
                "to_q": _wt(attn.to_q.weight),
                "to_kv": _wt(attn.to_kv.weight),
                "to_out": _wt(attn.to_out.weight),
                # scaled-dot-product core via the graduated leaf stub (scale = d**-0.5).
                "attend": _build_attend(device, attn.attend),
                "inner": int(inner),
                "fc1_w": _wt(ff[0].weight),
                "fc1_b": _b(ff[0].bias, ff[0].weight.shape[0]),
                "fc2_w": _wt(ff[2].weight),
                "fc2_b": _b(ff[2].bias, ff[2].weight.shape[0]),
            }
        )

    gamma = _b(m.norm.gamma, dim)
    norm_scale = float(m.norm.scale)  # == sqrt(dim); folded into gamma below
    # rms_norm(x)*gamma == F.normalize(x)*sqrt(dim)*gamma, so scale is implicit.

    def _split_heads(x, n):
        t = int(x.shape[1])
        x4 = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, t, heads, head_dim))
        return ttnn.to_layout(ttnn.permute(x4, (0, 2, 1, 3)), ttnn.TILE_LAYOUT)

    def _merge_heads(x):
        x = ttnn.permute(x, (0, 2, 1, 3))  # [1, T, H, hd]
        _, t, h, hd = x.shape
        return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, t, h * hd)), ttnn.TILE_LAYOUT)

    def forward(x, *args, **kwargs):
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        latents = latents0
        nq = int(latents.shape[1])

        for L in layers:
            inner = L["inner"]
            # cross-attention with queries included in the context
            context = ttnn.concat([latents, x], dim=1)  # [1, nq+T, dim]
            q = ttnn.matmul(latents, L["to_q"], compute_kernel_config=compute_config)  # [1, nq, inner]
            kv = ttnn.matmul(context, L["to_kv"], compute_kernel_config=compute_config)  # [1, nq+T, 2*inner]
            ctx_t = int(context.shape[1])
            k = ttnn.slice(kv, [0, 0, 0], [1, ctx_t, inner])
            v = ttnn.slice(kv, [0, 0, inner], [1, ctx_t, 2 * inner])

            qh = _split_heads(q, nq)  # [1, H, nq, hd]
            kh = _split_heads(k, ctx_t)  # [1, H, ctx_t, hd]
            vh = _split_heads(v, ctx_t)

            # scaled-dot-product attention via the graduated `attend` leaf stub
            # (plain SDPA, scale = hd**-0.5 == this layer's `scale`).
            out = L["attend"](qh, kh, vh)  # [1, H, nq, hd]
            out = _merge_heads(out)  # [1, nq, inner]
            attn_out = ttnn.matmul(out, L["to_out"], compute_kernel_config=compute_config)  # [1, nq, dim]
            latents = ttnn.add(attn_out, latents)

            # feed-forward
            h = ttnn.add(ttnn.matmul(latents, L["fc1_w"], compute_kernel_config=compute_config), L["fc1_b"])
            h = _geglu(h)
            h = ttnn.add(ttnn.matmul(h, L["fc2_w"], compute_kernel_config=compute_config), L["fc2_b"])
            latents = ttnn.add(h, latents)

        return ttnn.rms_norm(latents, epsilon=1e-8, weight=gamma)

    return forward


def perceiver_resampler(*args, **kwargs):
    raise RuntimeError(
        "perceiver_resampler requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
