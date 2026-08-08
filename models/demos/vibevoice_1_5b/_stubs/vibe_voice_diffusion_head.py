# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `vibe_voice_diffusion_head` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.prediction_head`, a
`vibevoice.modular.modular_vibevoice_diffusion_head.VibeVoiceDiffusionHead`:

    x = noisy_images_proj(noisy_images)   # Linear(latent_size, hidden), no bias
    t = t_embedder(timesteps)             # graduated `timestep_embedder`
    condition = cond_proj(condition)      # Linear(hidden, cond_dim), no bias
    c = condition + t
    for layer in layers:
        x = layer(x, c)                  # graduated `head_layer`
    x = final_layer(x, c)                # graduated `final_layer`
    return x

`noisy_images`/`condition`/`c`/`x` are rank-2 `(N, dim)` here (N independent
diffusion samples, no sequence axis), but `head_layer`/`final_layer`'s
already-graduated native ports were built (and PCC-verified) against a
rank-3 `(B, T, C)` activation shape — their `adaLN_modulation` split uses a
3-index `ttnn.slice`. Reshaping to `(1, N, C)` before calling them and back
to `(N, C)` after is a no-op on the math (N plays the role of T) and keeps
those ports unmodified.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs import _precision as prec
from models.demos.vibevoice_1_5b._stubs.final_layer import build as _build_final_layer
from models.demos.vibevoice_1_5b._stubs.head_layer import build as _build_head_layer
from models.demos.vibevoice_1_5b._stubs.timestep_embedder import build as _build_timestep_embedder

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_TILE = ttnn.TILE_LAYOUT


def build(device, torch_module):
    """Bind every child submodule's trained weights and return a native ttnn forward closure."""
    m = torch_module

    ni_w = m.noisy_images_proj.weight.detach().float()  # [hidden, latent]
    cond_w = m.cond_proj.weight.detach().float()  # [cond_dim, hidden]
    hidden_size = ni_w.shape[0]

    ni_w_t = prec.mm_weight(ni_w.t().contiguous(), device)
    cond_w_t = prec.mm_weight(cond_w.t().contiguous(), device)

    t_embedder_forward = _build_timestep_embedder(device, m.t_embedder)
    layer_forwards = [_build_head_layer(device, layer) for layer in m.layers]
    final_forward = _build_final_layer(device, m.final_layer)

    compute_config = prec.compute_config(device)

    def _to_ttnn_f32(t):
        if not isinstance(t, ttnn.Tensor):
            return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=_TILE, device=device)
        return ttnn.typecast(t, ttnn.float32) if t.get_dtype() != ttnn.float32 else t

    def forward(noisy_images, timesteps, condition, *args, **kwargs):
        noisy_images = _to_ttnn_f32(noisy_images)
        timesteps = _to_ttnn_f32(timesteps)
        condition = _to_ttnn_f32(condition)

        n = int(noisy_images.shape[0])

        x = prec.matmul(noisy_images, ni_w_t, compute_config)  # [N, hidden]
        t = t_embedder_forward(timesteps)  # [N, hidden]
        cond = prec.matmul(condition, cond_w_t, compute_config)  # [N, cond_dim]
        c = ttnn.add(cond, t, memory_config=_DRAM)

        x3 = ttnn.reshape(x, (1, n, hidden_size))
        c3 = ttnn.reshape(c, (1, n, hidden_size))
        for layer_forward in layer_forwards:
            x3 = layer_forward(x3, c3)
        out3 = final_forward(x3, c3)

        out_dim = int(out3.shape[-1])
        return ttnn.reshape(out3, (n, out_dim))

    return forward


def vibe_voice_diffusion_head(*args, **kwargs):
    raise RuntimeError(
        "vibe_voice_diffusion_head requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
