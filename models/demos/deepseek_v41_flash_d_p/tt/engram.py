# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Engram rows: the model's one host input besides the token ids (components.yaml engram_rows, tag CPU; owner rule).

Per chunk and engram layer, on the host: hash the chunk's natural-order token ids with the carried 3-id tail
(reference/ngram.py:EngramHasher, bit-equal to the checkpoint's inference/engram.py), read the memory-mapped fp8 rows
times their e8m0 group scales (reference/weights.py:EngramTable; never the whole 98 GB table), and send the result to
the device once: [1, 1, S/SP, 24 x 256] bf16 per chip, rows split over mesh axis 0 (SP), replicated over axis 1 (TP),
TILE, DRAM. bf16 holds e4m3 x 2^k exactly, so the device copy equals the reference's fp32 rows.

Rows at or past actual_end (engine pad, PAD_ID 0xFFFFFFFF) are not hashed: they are zero and never reach a cache;
the tail is the compressed ids of the last 3 real positions. The device keeps no engram state: the id history (tail)
is host state per slot, returned to the caller.

This runs in the device model's ``begin_chunk(tokens, start)`` (agent rule 5: a host input), never inside a layer.
"""

from __future__ import annotations

import torch

import ttnn

_HASHERS = {}  # model path -> EngramHasher (the compressed token map takes seconds to build)


def engram_hasher(model_path: str, cfg):
    """One EngramHasher per checkpoint (from its tokenizer and config)."""
    key = str(model_path)
    if key not in _HASHERS:
        from transformers import AutoTokenizer

        from models.demos.deepseek_v41_flash_d_p.reference.ngram import EngramHasher

        _HASHERS[key] = EngramHasher.build(cfg.raw, AutoTokenizer.from_pretrained(key))
    return _HASHERS[key]


class TtEngramRows:
    """Host hash + table lookup for one engram layer, and the one host -> device transfer of its rows."""

    def __init__(self, mesh, hasher, table, layer: int):
        self.mesh, self.hasher, self.table, self.layer = mesh, hasher, table, layer
        self.width = hasher.tail_len * hasher.n_heads * table.dim  # 3 x 8 x 256 = 6144
        self.vocab = int(hasher.token_map.shape[0])

    def new_tail(self) -> torch.Tensor:
        return self.hasher.new_tail()

    def host_rows(self, tokens: torch.Tensor, tail: torch.Tensor, valid: int | None = None):
        """tokens [S] natural order (pad past ``valid``), tail [3] -> (rows [S, width] bf16, new tail [3])."""
        ids = tokens.reshape(-1).long()
        s = ids.shape[0]
        if valid is None:  # the engine pads past actual_end with 0xFFFFFFFF (or any id outside the vocabulary)
            bad = ((ids < 0) | (ids >= self.vocab)).nonzero()
            valid = int(bad[0]) if bad.numel() else s
        valid = max(0, min(int(valid), s))
        rows = torch.zeros(s, self.width, dtype=torch.bfloat16)
        if valid == 0:
            return rows, tail.clone()
        h, new_tail = self.hasher.hashes(ids[:valid], self.layer, tail.long())
        rows[:valid] = self.table.rows(h).flatten(1).to(torch.bfloat16)  # exact (e4m3 x 2^k)
        return rows, new_tail

    def to_device(self, rows: torch.Tensor) -> ttnn.Tensor:
        """Host rows [S, width] -> device [1, 1, S/SP, width] bf16 per chip (rows over axis 0, replicated over 1)."""
        s, w = rows.shape
        sp = self.mesh.shape[0]
        assert s % sp == 0, f"{s} rows do not split over {sp} mesh rows"
        return ttnn.from_torch(
            rows.reshape(1, 1, s, w).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None)),
        )

    def __call__(self, tokens: torch.Tensor, tail: torch.Tensor, valid: int | None = None):
        """begin_chunk work for this layer: (device rows [1, 1, S/SP, width] bf16, new tail)."""
        rows, new_tail = self.host_rows(tokens, tail, valid)
        return self.to_device(rows), new_tail


def build_engram_rows(mesh, ck, cfg, layer: int, hasher=None) -> TtEngramRows:
    """``hasher``: an EngramHasher already built (the reference's), else one is built from the checkpoint."""
    from models.demos.deepseek_v41_flash_d_p.reference.weights import EngramTable

    assert cfg.is_engram(layer), f"layer {layer} has no engram"
    return TtEngramRows(mesh, hasher or engram_hasher(ck.path, cfg), EngramTable(ck, layer), layer)
