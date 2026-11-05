# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side greedy candidate selection for DFlash2 (port of z-lab CandidateSelector.select at temperature 0)."""

import os

import torch
from safetensors import safe_open

from models.demos.muse_glimmer_30b.tt.model_config import dflash_checkpoint_dir


def merge_device_topk(values, ids, *, vocab_local, vocab_size, k=16):
    """values, ids: [D, rows, k_dev] per-device top-k (ids local to the device shard).
    Returns (scores [rows, k] float32, candidates [rows, k] int64): global top-k over valid ids (< vocab_size),
    sorted by score descending."""
    values = torch.as_tensor(values).float()
    ids = torch.as_tensor(ids).long()
    num_devices, rows, k_dev = values.shape
    offsets = torch.arange(num_devices, dtype=torch.long).view(num_devices, 1, 1) * vocab_local
    global_ids = ids + offsets
    values = values.masked_fill(global_ids >= vocab_size, float("-inf"))
    flat_values = values.permute(1, 0, 2).reshape(rows, num_devices * k_dev)
    flat_ids = global_ids.permute(1, 0, 2).reshape(rows, num_devices * k_dev)
    scores, index = torch.topk(flat_values, k, dim=-1)
    return scores, flat_ids.gather(-1, index)


class HostCandidateSelector:
    """DFlash2 candidate selector on the host: greedy `select` (z-lab CandidateSelector.select at temperature 0) and temperature `sample`."""

    def __init__(self, predecessor_codebook, successor_codebook):
        self.pred = predecessor_codebook.float()
        self.succ = successor_codebook.float()

    @classmethod
    def from_checkpoint(cls, ckpt_dir=None):
        path = os.path.join(ckpt_dir or dflash_checkpoint_dir(), "model.safetensors")
        with safe_open(path, framework="pt") as f:
            pred = f.get_tensor("candidate_selector.predecessor_codebook")
            succ = f.get_tensor("candidate_selector.successor_codebook")
        return cls(pred, succ)

    def _walk(self, scores, candidates, hidden_proj, anchor_id, pick):
        """Shared path walk: total = scores[p] + succ[cand[p]] @ (pred[prev] * hp[p]); pick(total) -> index."""
        scores = torch.as_tensor(scores).float()
        candidates = torch.as_tensor(candidates).long()
        hidden_proj = torch.as_tensor(hidden_proj).float()
        succ_rows = self.succ[candidates]  # [L, k, R]
        prev = int(anchor_id)
        path = []
        for p in range(candidates.shape[0]):
            query = self.pred[prev] * hidden_proj[p]  # [R]
            total = scores[p] + succ_rows[p] @ query  # [k]
            prev = int(candidates[p, pick(total)])
            path.append(prev)
        return path

    def select(self, scores, candidates, hidden_proj, anchor_id):
        """scores [L, k] float, candidates [L, k] int64, hidden_proj [L, R] (already projected), anchor_id int.
        For position p: s = scores[p] + (pred[prev] * hp[p]) @ succ[candidates[p]].T ; idx = argmax(s);
        prev = candidates[p, idx]; append prev. Returns the L path token ids."""
        return self._walk(scores, candidates, hidden_proj, anchor_id, lambda total: int(torch.argmax(total)))

    def sample(self, scores, candidates, hidden_proj, anchor_id, temperature, generator):
        """Same walk as select, but idx ~ softmax(total / temperature) (float32, no top-k/top-p).
        Returns (path list of L ints, q_probs [L, k] float32)."""
        rows = []

        def pick(total):
            q = torch.softmax(total.float() / temperature, dim=-1)
            rows.append(q)
            return int(torch.multinomial(q, 1, generator=generator))

        path = self._walk(scores, candidates, hidden_proj, anchor_id, pick)
        return path, torch.stack(rows)
