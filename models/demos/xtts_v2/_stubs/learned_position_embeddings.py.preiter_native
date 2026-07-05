# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `learned_position_embeddings` (coqui/XTTS-v2 `gpt.mel_pos_embedding`).

The submodule is a tortoise/XTTS `LearnedPositionEmbeddings`
(`TTS.tts.layers.tortoise.autoregressive.LearnedPositionEmbeddings`, relative=False):

    forward(x): sl = x.shape[1]; return self.emb(arange(0, sl))

i.e. it returns the first `sl` rows of the learned position-embedding table. The
`arange(0, sl)` lookup is a contiguous prefix slice of the embedding weight, so the
native op is just slicing the stored `[seq_len, model_dim]` weight to `[sl, model_dim]`.
"""

from __future__ import annotations

import ttnn


HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind the position-embedding table and return a native ttnn forward closure."""
    m = torch_module
    if getattr(m, "relative", False):
        # Relative mode picks a random offset per call (untestable); this port
        # only handles the deterministic absolute-position path XTTS uses.
        raise RuntimeError("learned_position_embeddings native port supports relative=False only")

    weight = ttnn.from_torch(
        m.emb.weight.detach().contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    seq_len, model_dim = m.emb.weight.shape

    def forward(x, *args, **kwargs):
        # sl = x.shape[1] (the sequence length being embedded).
        if isinstance(x, ttnn.Tensor):
            shp = list(x.shape)
        else:
            shp = list(x.shape)
        sl = int(shp[1]) if len(shp) > 1 else int(shp[0])
        return ttnn.slice(weight, [0, 0], [sl, model_dim])

    return forward


def learned_position_embeddings(*args, **kwargs):
    raise RuntimeError(
        "learned_position_embeddings requires build(device, torch_module) to bind the "
        "position-embedding table; the bare callable has no parameters."
    )
