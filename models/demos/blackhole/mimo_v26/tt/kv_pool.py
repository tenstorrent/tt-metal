# SPDX-License-Identifier: Apache-2.0
"""KV pool / page-table layout shared by chunked prefill (tt/prefill.py) and decode. Spec: bringup/mimo_v26/notes/kv_pool.md."""
from dataclasses import dataclass

import torch

BLOCK = 64
SLOTS = 32  # decode user slots per mesh row


@dataclass(frozen=True)
class KVLayout:
    users: int  # active users (slots 0..users-1) per mesh row
    max_pos: int  # GA capacity per user, multiple of chunk
    chunk: int = 2048  # prefill chunk (tokens per mesh row per step)

    @property
    def ring_blocks(self):
        return self.chunk // BLOCK + 8

    @property
    def ga_nb(self):
        return self.max_pos // BLOCK

    def ga_table(self):
        nb, U, sc = self.ga_nb, self.users, self.chunk // BLOCK
        t = torch.zeros(SLOTS, nb, dtype=torch.int32)
        for s in range(SLOTS):
            t[s] = s * nb + torch.arange(nb) if s < U else U * nb + torch.arange(nb) % sc
        return t

    def swa_table(self):
        R, U = self.ring_blocks, self.users
        t = torch.zeros(SLOTS, R, dtype=torch.int32)
        for s in range(SLOTS):
            t[s] = (s if s < U else U) * R + torch.arange(R)
        return t

    def tables(self):
        return {"ga": self.ga_table(), "swa": self.swa_table()}

    def bytes_per_chip(self, n_ga=9, n_swa=39):
        ga = (self.users * self.ga_nb + self.chunk // BLOCK) * BLOCK * 192 * 2 * 2
        swa = (self.users + 1) * self.ring_blocks * BLOCK * 192 * 2 * 2
        return n_ga * ga + n_swa * swa
