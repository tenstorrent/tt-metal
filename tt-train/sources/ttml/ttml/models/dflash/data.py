# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Speculators offline DFlash data on the host: rows, packing, collation.

Ports, for a single replica, the pieces of speculators ``train/`` a training step
depends on, so a TT run consumes the same packed rows as a GPU run:

* ``ArrowDataset`` + ``hs_<file_idx>.safetensors`` loading (``train/data.py``).
* ``MultipackDistributedBatchSamplerV2`` with ``num_replicas == 1``
  (``train/distributed_batch_sampler.py``): seeded per-epoch permutation, binary search
  for the longest run of samples that packs into one row, LPT order inside the row.
* ``create_collate_fn``: concatenation, truncation of the last sample, zero padding,
  ``document_ids`` (``-1`` for padding) and per-sample ``position_ids``.

Hidden-state files hold ``token_ids [S]`` and ``hidden_states [S, L, H]`` (aux layers,
then the final pre-norm state). Pure numpy apart from lazy ``datasets``/``safetensors``.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Packing (num_replicas == 1 port of the speculators multipack sampler)
# ---------------------------------------------------------------------------


class _Bin(NamedTuple):
    fill: int
    rank: int


def _lpt_packed_batch(lengths: np.ndarray, max_len: int, num_replicas: int, start_index: int, rank: int):
    local_batch = []
    heap = [_Bin(0, i) for i in range(num_replicas)]
    indices = np.argsort(lengths)[::-1]
    for idx, size in zip(indices, lengths[indices], strict=True):
        new_fill = heap[0].fill + size
        if new_fill > max_len:
            return None
        if heap[0].rank == rank:
            local_batch.append(start_index + idx)
        heapq.heapreplace(heap, _Bin(new_fill, heap[0].rank))
    return local_batch


def _assign_to_packed_batches(lengths: np.ndarray, max_len: int, rank: int, replicas: int) -> list:
    lengths_so_far = 0
    ind = 0
    result = []
    lengths_cumsum = np.cumsum(lengths)
    while True:
        if len(lengths) - ind < replicas:
            break
        left = 1
        right = 1 + np.searchsorted(lengths_cumsum[ind:], lengths_so_far + max_len * replicas, "right")
        batch = None
        while right - left > 1 and right > replicas:
            mid = (left + right) // 2
            batch = _lpt_packed_batch(lengths[ind : ind + mid], max_len, replicas, ind, rank)
            if batch is None:
                right = mid
            else:
                left = mid
        if batch is None:
            batch = _lpt_packed_batch(lengths[ind : ind + left], max_len, replicas, ind, rank)
        ind += left
        lengths_so_far = lengths_cumsum[ind - 1]
        result.append(batch)
    return result


def packed_batches(lengths, max_len: int, seed: int, epoch: int) -> list[np.ndarray]:
    """Dataset indices of each packed row for ``epoch`` (speculators sampler, 1 replica).

    Like the reference with ``truncate_long_samples=True`` (its default), lengths are
    clipped to ``max_len`` for packing; :func:`collate` truncates the sample itself.
    """
    lengths = np.clip(np.asarray(lengths), 0, max_len)
    valid = np.arange(len(lengths))
    rng = np.random.default_rng(seed=seed + epoch)
    indices = rng.permutation(valid)
    batches = _assign_to_packed_batches(lengths[indices], max_len, 0, 1)
    return [indices[np.asarray(b, dtype=np.int64)] for b in batches]


# ---------------------------------------------------------------------------
# Collation (speculators create_collate_fn, numpy)
# ---------------------------------------------------------------------------


def _slice_and_pad(x: np.ndarray, length: int) -> np.ndarray:
    x = x[:length]
    if x.shape[0] < length:
        pad = np.zeros((length - x.shape[0],) + x.shape[1:], dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    return x


def collate(samples: list[dict[str, np.ndarray]], max_len: int) -> dict[str, np.ndarray]:
    """Pack samples (in the given order) into one row of ``max_len`` tokens.

    Each sample has ``hidden_states [s, k*H]``, ``verifier_last_hidden_states [s, H]``,
    ``input_ids [s]``, ``loss_mask [s]``. Returns the same keys padded to ``max_len`` plus
    ``position_ids`` (restart per sample, 0 on padding) and ``document_ids`` (-1 padding).
    """
    out: dict[str, np.ndarray] = {}
    for key in ("hidden_states", "verifier_last_hidden_states", "input_ids", "loss_mask"):
        out[key] = _slice_and_pad(np.concatenate([s[key] for s in samples], axis=0), max_len)
    out["position_ids"] = _slice_and_pad(
        np.concatenate([np.arange(s["input_ids"].shape[0], dtype=np.int64) for s in samples]), max_len
    )
    lengths, cum = [], 0
    for s in samples:
        length = int(s["input_ids"].shape[0])
        if length + cum >= max_len:
            lengths.append(max_len - cum)
            break
        lengths.append(length)
        cum += length
    doc = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    out["document_ids"] = np.concatenate([doc, -np.ones(max_len - doc.size, dtype=np.int64)])
    return out


# ---------------------------------------------------------------------------
# Offline dataset (Arrow rows + hs_<idx>.safetensors)
# ---------------------------------------------------------------------------


class OfflineDFlashDataset:
    """speculators ``ArrowDataset`` with ``on_missing='raise'`` (no online generation).

    Args:
        datapath: ``prepare_data.py`` output directory (``datasets.save_to_disk``).
        hidden_states_path: directory of ``hs_<file_idx>.safetensors`` (default
            ``<datapath>/hidden_states``).
        split_ratio: as in speculators (``0.9`` train / ``-0.1`` validation split).
    """

    def __init__(self, datapath, hidden_states_path=None, split_ratio: float = 1.0):
        from datasets import load_from_disk  # noqa: PLC0415

        data = load_from_disk(str(datapath))
        n = len(data)
        if split_ratio == 1.0:
            file_indices = list(range(n))
        elif 1.0 > split_ratio > 0:
            split = int(n * split_ratio)
            data, file_indices = data.select(range(split)), list(range(split))
        elif -1.0 < split_ratio < 0:
            split = int(n * (1.0 + split_ratio))
            data = data.select(range(split, n))
            file_indices = list(range(split, n))
        else:
            raise ValueError("split_ratio must be in (-1.0, 1.0] excluding 0.0")
        self.data = data
        self.file_indices = file_indices
        self.hidden_states_path = Path(hidden_states_path) if hidden_states_path else Path(datapath) / "hidden_states"
        self.lengths = np.asarray(self.data.with_format(None)["seq_len"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        from safetensors.torch import load_file  # noqa: PLC0415  (numpy cannot hold bf16)

        file_idx = self.file_indices[index]
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"missing hidden states for row {index}: {path}")
        hs = load_file(str(path))
        states = hs["hidden_states"].float().numpy()  # [S, L, H]
        row = self.data[index]
        loss_mask = np.asarray(row["loss_mask"], dtype=np.float32)
        token_ids = hs["token_ids"].numpy().astype(np.int64)
        if token_ids.shape[0] != loss_mask.shape[0]:
            raise ValueError(f"{path}: {token_ids.shape[0]} tokens vs loss_mask {loss_mask.shape[0]}")
        return {
            "hidden_states": states[:, :-1].reshape(states.shape[0], -1),
            "verifier_last_hidden_states": states[:, -1],
            "input_ids": token_ids,
            "loss_mask": loss_mask,
        }

    def epoch_rows(self, max_len: int, seed: int, epoch: int) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        """Yield ``(dataset indices, collated row)`` for every packed row of ``epoch``."""
        for batch in packed_batches(self.lengths, max_len, seed, epoch):
            yield batch, collate([self[int(i)] for i in batch], max_len)
