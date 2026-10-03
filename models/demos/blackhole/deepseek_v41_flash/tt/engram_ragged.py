# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Engram n-gram hash with PER-USER start positions (the checkpoint's ``NgramHashState.forward`` takes one scalar ``start_pos`` for the whole
batch; prompts of different lengths need one per user). Same arithmetic, same ``cache`` buffer, bit-identical to the reference for a uniform start."""

import torch


class RaggedNgramHash:
    def __init__(self, state):
        """state: the checkpoint's NgramHashState (``HostEngramRows.engram.hash``)."""
        self.st = state

    @torch.inference_mode()
    def __call__(self, input_ids: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """input_ids [B, L], start [B] int (position of column 0 of every user) -> hashes [B, L, n_engram_layers, n_hash_cols]."""
        st = self.st
        B, L = input_ids.shape
        compressed = st.token_map[input_ids]
        start = start.reshape(B, 1).long()
        positions = start + torch.arange(L).reshape(1, L)
        st.cache[torch.arange(B).reshape(B, 1), positions] = compressed
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(st.layout.max_ngram_size):
            source = st.cache[:B].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == st.DEAD)
            tokens.append(torch.where(blocked, st.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)
        products = tokens.unsqueeze(2) * st.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, st.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % st.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + st.offsets
