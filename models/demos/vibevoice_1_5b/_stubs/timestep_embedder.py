# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timestep_embedder` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.prediction_head.t_embedder`, a
`vibevoice.modular.modular_vibevoice_diffusion_head.TimestepEmbedder`:

    half = frequency_embedding_size // 2                      # 128
    freqs = exp(-log(max_period) * arange(half) / half)        # [half], constant
    args = t[:, None] * freqs[None]                             # [N, half]
    t_freq = cat([cos(args), sin(args)], dim=-1)                # [N, freq_dim=256]
    t_emb = mlp(t_freq)   # Linear(256, hidden, bias=False) -> SiLU -> Linear(hidden, hidden, bias=False)

`freqs` depends only on static config (frequency_embedding_size, max_period),
so it's precomputed on host and uploaded once at build time; the
outer-product `t[:, None] * freqs[None]` is a [N,1] x [1,half] matmul.
"""

from __future__ import annotations

import math

import ttnn
from models.demos.vibevoice_1_5b._stubs import _precision as prec
from models.demos.vibevoice_1_5b._stubs._trace_pad import cached_zeros

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_TILE = ttnn.TILE_LAYOUT
_MAX_PERIOD = 10000


def build(device, torch_module):
    """Bind the trained MLP weights + static frequency table, return a native ttnn forward closure."""
    m = torch_module
    freq_dim = int(m.frequency_embedding_size)
    half = freq_dim // 2

    import torch

    freqs = torch.exp(-math.log(_MAX_PERIOD) * torch.arange(start=0, end=half, dtype=torch.float32) / half)  # [half]

    linear1 = m.mlp[0]  # Linear(freq_dim, hidden, bias=False)
    linear2 = m.mlp[2]  # Linear(hidden, hidden, bias=False)
    w1 = linear1.weight.detach().float()  # [hidden, freq_dim]
    w2 = linear2.weight.detach().float()  # [hidden, hidden]

    # freqs outer-product stays fp32 (feeds cos/sin — precision sensitive, and it's a tiny [1,half] weight);
    # only the two MLP projections (w1/w2) take the perf-oriented precision mode.
    freqs_t = ttnn.from_torch(freqs.reshape(1, half).contiguous(), dtype=ttnn.float32, layout=_TILE, device=device)
    w1_t = prec.mm_weight(w1.t().contiguous(), device)
    w2_t = prec.mm_weight(w2.t().contiguous(), device)

    freq_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    compute_config = prec.compute_config(device)
    _zc: dict = {}  # cached zero-pad buffer (odd freq_dim); trace-capture safe

    def forward(t, *args, **kwargs):
        if t.get_dtype() != ttnn.float32:
            t = ttnn.typecast(t, ttnn.float32)
        n = int(t.shape[0])
        t_col = ttnn.reshape(t, (n, 1))
        outer = ttnn.matmul(t_col, freqs_t, compute_kernel_config=freq_config, memory_config=_DRAM)  # [N, half]
        cos = ttnn.cos(outer, memory_config=_DRAM)
        sin = ttnn.sin(outer, memory_config=_DRAM)
        emb = ttnn.concat([cos, sin], dim=-1, memory_config=_DRAM)  # [N, freq_dim]
        if freq_dim % 2:
            pad = cached_zeros(_zc, (n, 1), ttnn.float32, _TILE, device)
            emb = ttnn.concat([emb, pad], dim=-1, memory_config=_DRAM)

        h = prec.matmul(emb, w1_t, compute_config)
        h = ttnn.silu(h, memory_config=_DRAM)
        h = prec.matmul(h, w2_t, compute_config)
        return h

    return forward


def timestep_embedder(*args, **kwargs):
    raise RuntimeError(
        "timestep_embedder requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
