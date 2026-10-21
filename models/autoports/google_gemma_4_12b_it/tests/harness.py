# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-boundary helpers: host<->device conversion, a host page allocator, and PCC bookkeeping."""

import json
import os
import time

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import FunctionalDecoder

PCC_BAR = 0.995
BLOCK = 64
CANARY_BLOCK = 0  # never handed to a request; unused page-table entries point here


def build_decoder(device, cfg, layer_idx, state_dict, num_blocks, **kwargs):
    return FunctionalDecoder.from_state_dict(
        state_dict, hf_config=cfg, layer_idx=layer_idx, mesh_device=device, num_blocks=num_blocks, **kwargs
    )


def to_tt(x, device, dtype=ttnn.bfloat16):
    """torch ``[1, S, H]`` -> device ``[1, 1, S, H]`` TILE."""
    return ttnn.from_torch(x.reshape(1, 1, *x.shape[-2:]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def to_tt_int(t, device):
    return ttnn.from_torch(t.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def from_tt(t):
    return ttnn.to_torch(t).float().reshape(1, t.shape[-2], t.shape[-1])


class Pages:
    """Host-side page allocator: shuffled physical blocks per user, canary block for unused entries."""

    def __init__(self, num_blocks, max_pages, num_users, seed=0):
        g = torch.Generator().manual_seed(seed)
        free = (torch.randperm(num_blocks - 1, generator=g) + 1).tolist()
        self.free = free
        self.table = torch.full((num_users, max_pages), CANARY_BLOCK, dtype=torch.int32)
        self.owned = [0] * num_users

    def ensure(self, user, length):
        need = -(-length // BLOCK)
        while self.owned[user] < need:
            self.table[user, self.owned[user]] = self.free.pop()
            self.owned[user] += 1

    def release(self, user):
        for p in range(self.owned[user]):
            self.free.insert(0, int(self.table[user, p]))
            self.table[user, p] = CANARY_BLOCK
        self.owned[user] = 0

    def tt(self, device, rows=None):
        t = self.table if rows is None else self.table[rows]
        return to_tt_int(t, device)


def canary_is_clean(decoder):
    """The canary block must never be written (padding / page-table addressing guard)."""
    for cache in decoder.kv_cache:
        block = ttnn.to_torch(ttnn.slice(cache, [CANARY_BLOCK, 0, 0, 0], [CANARY_BLOCK + 1, *list(cache.shape)[1:]]))
        if block.abs().max() != 0:
            return False
    return True


def record(name, **fields):
    """Append one JSON line of evidence when FD_EVIDENCE is set (path of a .jsonl file)."""
    path = os.environ.get("FD_EVIDENCE")
    fields = {"test": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    print("EVIDENCE", json.dumps(fields))
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(fields) + "\n")


def row_pcc(want, got):
    a = want.reshape(-1, want.shape[-1]).double()
    b = got.reshape(-1, got.shape[-1]).double()
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def check(name, want, got, bar=PCC_BAR, **fields):
    """PCC over the whole tensor and the worst row; both must clear the bar."""
    assert want.shape == got.shape, (want.shape, got.shape)
    assert torch.isfinite(got).all(), f"{name}: non-finite output"
    whole = pcc(want, got)
    worst = float(row_pcc(want, got).min())
    record(name, pcc=round(whole, 6), worst_row_pcc=round(worst, 6), rows=want.numel() // want.shape[-1], **fields)
    assert whole >= bar and worst >= bar, f"{name}: pcc {whole:.6f}, worst row {worst:.6f} < {bar}"
    return whole, worst
