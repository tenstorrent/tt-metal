# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `albert_embeddings` (hexgrad/Kokoro-82M
`bert.embeddings`, a HF `AlbertEmbeddings`).

Reference torch forward (token_type_ids default all-zeros, position_ids default
`arange(seq_len)`, dropout identity in eval):

    inputs_embeds = word_embeddings(input_ids)
    token_type_embeddings = token_type_embeddings(token_type_ids)   # all row 0
    embeddings = inputs_embeds + token_type_embeddings
    position_embeddings = position_embeddings(position_ids)         # rows 0..sl-1
    embeddings = embeddings + position_embeddings
    embeddings = LayerNorm(embeddings)                             # eps 1e-12, affine

Native ttnn: three `ttnn.embedding` gathers + adds + a LayerNorm over the last
(embedding) axis. The PCC harness hands `input_ids` in as a bf16 activation
(all token ids < vocab=178 < 256 survive bf16 exactly); we round it back to
integer indices for the gather. Computed in float32 for a clean PCC.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind the embedding tables / LayerNorm and return a native ttnn forward."""
    import torch

    m = torch_module
    eps = float(getattr(m.LayerNorm, "eps", 1e-12))

    # fp32 embedding tables for a one-hot matmul gather. bf16 `ttnn.embedding` is
    # a ~1e-3 PCC floor that compounds through the deep prosody chain and shifts
    # a borderline duration; the one-hot matmul is fp32-exact (bert_dur ~1e-6).
    def _emb_weight_fp32(w):
        return ttnn.from_torch(
            w.detach().contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    word_w_fp32 = _emb_weight_fp32(m.word_embeddings.weight)
    pos_w_fp32 = _emb_weight_fp32(m.position_embeddings.weight)
    _vocab = int(m.word_embeddings.weight.shape[0])
    _maxpos = int(m.position_embeddings.weight.shape[0])
    _cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _onehot_gather(ids_int, table, n_classes):
        B, T = ids_int.shape
        oh = torch.zeros(B, T, n_classes, dtype=torch.float32)
        oh.scatter_(2, ids_int.long().unsqueeze(-1), 1.0)
        oh_tt = ttnn.from_torch(
            oh.contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.matmul(oh_tt, table, compute_kernel_config=_cc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ln_w = ttnn.from_torch(
        m.LayerNorm.weight.detach().reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ln_b = ttnn.from_torch(
        m.LayerNorm.bias.detach().reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Row 0 of the token-type table (token_type_ids default to all zeros).
    ttype0 = ttnn.from_torch(
        m.token_type_embeddings.weight[0:1].detach().reshape(1, 1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _ids_int(x, T):
        t = ttnn.to_torch(x) if isinstance(x, ttnn.Tensor) else x
        t = t.reshape(-1, T) if t.ndim != 2 else t
        return t.round().to(torch.int32)

    def forward(input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None, *args, **kwargs):
        shp = list(input_ids.shape)
        B, T = int(shp[0]), int(shp[-1])

        ids = _ids_int(input_ids, T)
        we = _onehot_gather(ids, word_w_fp32, _vocab)  # [B, T, E] fp32

        pos_t = torch.arange(T, dtype=torch.int32).reshape(1, T)
        pe = _onehot_gather(pos_t, pos_w_fp32, _maxpos)  # [1, T, E] fp32

        emb = ttnn.add(we, ttype0)  # broadcast token-type row 0
        emb = ttnn.add(emb, pe)  # broadcast position embeddings over batch

        # LayerNorm over the last (embedding) axis, affine.
        mean = ttnn.mean(emb, dim=2, keepdim=True)
        xc = ttnn.subtract(emb, mean)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=2, keepdim=True)
        xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
        return ttnn.add(ttnn.multiply(xn, ln_w), ln_b)

    return forward


def albert_embeddings(*args, **kwargs):
    raise RuntimeError(
        "albert_embeddings requires build(device, torch_module) to bind the "
        "embedding tables; the bare callable has no parameters."
    )
