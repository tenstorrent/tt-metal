# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""An LRU of MiniMax-H3 conditioner taps, in memory and optionally on disk.

The conditioner is the expensive half of a small-mesh request. On a mesh that cannot keep the
stages co-resident, encoding a prompt means uploading ~50 GB of Qwen3-VL, running it, and then
paying to bring the DiT back; on the host path it means a multi-second CPU forward. Either way it
is pure overhead whenever a prompt repeats -- a served endpoint re-runs the same prompt constantly
(retries, a fixed negative prompt, A/B sweeps over seed / steps / canvas) and the tap does not
depend on any of those.

The key is the exact conditioner input, not the prompt string: the padded ``input_ids``, the
``mm_token_type_ids`` that go with them, and a digest of the vision patches for fl2va/ref2va. Two
requests with the same text but different keyframes, a different pad length or a different
tokenizer therefore miss, which is the safe direction. Keying on the prompt string would collide
across all four.

Entries are stored as bf16: the tap is uploaded to the device as bf16 regardless, so keeping fp32
on disk would double the file for precision the consumer immediately discards.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import torch

# Default in-memory depth. A 512-token tap is 1 x 512 x 5376 bf16 = 5.5 MB, so 8 entries is ~44 MB
# of host RAM -- far below the noise on a 249 GB box, and enough to cover a served endpoint's
# working set of prompts.
DEFAULT_CAPACITY = 8


def prompt_cache_key(input_ids: torch.Tensor, type_ids: torch.Tensor, pixel_values=None) -> str:
    """Digest of the exact conditioner input. See the module docstring for what is and is not in it."""
    digest = hashlib.sha256()
    for name, tensor in (("ids", input_ids), ("types", type_ids)):
        digest.update(name.encode())
        if tensor is None:
            digest.update(b"none")
            continue
        # Shape goes in explicitly: the raw buffer of a [1, 64] and a [64] tensor are identical.
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().to(torch.int64).contiguous().numpy().tobytes())
    digest.update(b"pixels")
    if pixel_values is None:
        digest.update(b"none")
    else:
        digest.update(str(tuple(pixel_values.shape)).encode())
        # fp32 and not the source dtype, so a bf16 and an fp32 presentation of the same keyframe
        # cannot hash differently and re-encode.
        digest.update(pixel_values.detach().to(torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


class PromptEmbedCache:
    """LRU over ``prompt_cache_key`` -> tap tensor, with an optional disk tier behind it.

    The disk tier survives a process restart, which is the case that matters: a served container
    restarts far more often than a prompt changes. A disk hit is promoted into memory.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, disk_dir: str | os.PathLike | None = None) -> None:
        self.capacity = int(capacity)
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self._memory: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.disk_hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path | None:
        return None if self.disk_dir is None else self.disk_dir / f"{key}.pt"

    def get(self, key: str) -> torch.Tensor | None:
        if key in self._memory:
            self._memory.move_to_end(key)
            self.hits += 1
            return self._memory[key]
        path = self._path(key)
        if path is not None and path.is_file():
            try:
                embeds = torch.load(path, map_location="cpu")
            except Exception:
                # A truncated file from a killed process must not take the request down with it:
                # drop it and fall through to a miss, which rewrites it.
                path.unlink(missing_ok=True)
            else:
                self.disk_hits += 1
                self._insert(key, embeds)
                return embeds
        self.misses += 1
        return None

    def put(self, key: str, embeds: torch.Tensor) -> torch.Tensor:
        """Store ``embeds`` (cast to bf16) and return the stored tensor, which is what callers use."""
        stored = embeds.detach().to(torch.bfloat16).contiguous()
        self._insert(key, stored)
        path = self._path(key)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written to a sibling and renamed: two processes sharing a cache directory must never
            # see a half-written entry, and rename within a directory is atomic.
            scratch = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(stored, scratch)
            scratch.replace(path)
        return stored

    def _insert(self, key: str, embeds: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        self._memory[key] = embeds
        self._memory.move_to_end(key)
        while len(self._memory) > self.capacity:
            self._memory.popitem(last=False)

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "disk_hits": self.disk_hits,
            "misses": self.misses,
            "resident": len(self._memory),
            "capacity": self.capacity,
            "disk_dir": str(self.disk_dir) if self.disk_dir else None,
        }
