# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image and video rows for a prefill request, and the first token's sampling.

The Encode worker (tt-d-gen adapters/dynamo/tt_dynamo/media.py) leaves, in a shared-memory directory:
  <hash:016x>.px          64-byte little-endian header {b"TMM1", u32 version, u32 kind, u32 patches,
                          u32 patch_elems, u32 dtype (1 = bf16), u32 grid_t, u32 grid_h, u32 grid_w}, then
                          patches * patch_elems bf16 values in the processor's patch order.
  req_<media_tag:016x>.json  {"version": 1, "items": [{"hash", "kind", "grid": [t, h, w],
                          "runs": [[prompt_pos, row_offset, count], ...]}]}: merged row `row_offset + i` of the
                          item belongs at prompt position `prompt_pos + i`.
The engine names the request's media_tag in each chunk's metadata. MediaRows encodes an item when the first chunk
holding its rows runs (items behind a cached prefix are never encoded), keeps each tower call's merged rows on device
as a table with a zero row at the end, and adds them to a chunk's token embedding, whose rows at those positions are
zero (the ids there are out of vocabulary).
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.chunk_layout import rotated_chunk_positions
from models.demos.minimax_m3.tt.vision import max_batch_patches

PX_MAGIC = b"TMM1"
PX_HEADER = struct.Struct("<4s8I")
PX_HEADER_BYTES = 64
DTYPE_BF16 = 1


def media_dir() -> str:
    return os.environ.get("TT_M3_MM_DIR", "/dev/shm/tt_m3_mm")


def read_manifest(media_tag: int, root: str | None = None) -> dict:
    with open(os.path.join(root or media_dir(), f"req_{media_tag:016x}.json")) as f:
        manifest = json.load(f)
    if manifest.get("version") != 1:
        raise ValueError(f"media manifest {media_tag:016x}: unsupported version {manifest.get('version')}")
    return manifest


def read_patches(item_hash: str, root: str | None = None) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """[patches, patch_elems] bf16 and the item's (t, h, w) patch grid."""
    path = os.path.join(root or media_dir(), f"{item_hash}.px")
    # Read into a writable buffer, which torch can share without a copy.
    with open(path, "rb", buffering=0) as f:
        raw = bytearray(os.fstat(f.fileno()).st_size)
        f.readinto(raw)
    magic, version, _kind, patches, elems, dtype, t, h, w = PX_HEADER.unpack_from(raw, 0)
    if magic != PX_MAGIC or version != 1 or dtype != DTYPE_BF16:
        raise ValueError(f"{path}: not a v1 bf16 patch file")
    data = np.frombuffer(raw, dtype=np.uint16, count=patches * elems, offset=PX_HEADER_BYTES)
    pv = torch.from_numpy(data).view(torch.bfloat16).reshape(patches, elems)
    return pv, (t, h, w)


class MediaRows:
    """One request's media rows on the prefill mesh, encoded when the first chunk that holds them runs, and the
    chunk-wise lookup that adds them to the token embedding."""

    def __init__(self, tower, mesh_device, sp_axis: int, chunk_size: int):
        self.tower = tower
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.sp = mesh_device.shape[sp_axis]
        self.chunk_local = chunk_size // self.sp
        self.max_patches = max_batch_patches(self.sp)
        self.tag = 0
        self.items: list[dict] = []
        self.batches: list[dict] = []  # table [rows + 1, hidden / tp] row-major per device, last row zero; runs

    def release(self) -> None:
        for b in self.batches:
            b["table"].deallocate(True)
        self.items, self.batches, self.tag = [], [], 0

    def _load(self, media_tag: int) -> None:
        """The request's items as pieces of whole attention segments, each small enough for one tower call."""
        self.release()
        seg = self.tower.args.segment_max_frames
        for item in read_manifest(media_tag)["items"]:
            t, h, w = item["grid"]
            per_t, m = h * w, h * w // 4
            step = max(1, self.max_patches // per_t)
            if seg and t > seg:
                step = max(seg, step // seg * seg)
            for t0 in range(0, t, step):
                t1 = min(t, t0 + step)
                runs = [(p, o - t0 * m, c) for p, o, c in item["runs"] if t0 * m <= o < t1 * m]
                if not runs:
                    continue
                self.items.append(
                    dict(
                        hash=item["hash"],
                        grid=(t, h, w),
                        piece=(t1 - t0, h, w),
                        patch_range=(t0 * per_t, t1 * per_t),
                        patches=(t1 - t0) * per_t,
                        runs=runs,
                        encoded=False,
                        end=max(p + c for p, _, c in runs),
                    )
                )
        self.tag = media_tag

    def _encode_from(self, position: int) -> None:
        """Encode, in one tower call, the not yet encoded items still ahead of `position`, in prompt order."""
        todo, patches = [], 0
        for item in sorted((i for i in self.items if not i["encoded"] and i["end"] > position), key=lambda i: i["end"]):
            if todo and patches + item["patches"] > self.max_patches:
                break
            todo.append(item)
            patches += item["patches"]
        pvs, grids, files = [], [], {}
        for item in todo:
            if item["hash"] not in files:
                files[item["hash"]] = read_patches(item["hash"])
            pv, grid = files[item["hash"]]
            if grid != item["grid"]:
                raise ValueError(f"media {item['hash']}: grid {grid} does not match the manifest's {item['grid']}")
            lo, hi = item["patch_range"]
            pvs.append(pv[lo:hi])
            grids.append(item["piece"])
        batch, rows = self.tower(torch.cat(pvs), grids)
        rm = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        rows.deallocate(True)
        # A batch with no padding slices to the whole of `rm`, and the slice then shares its buffer.
        real = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, batch.rows, rm.shape[-1]))
        zero = ttnn.from_torch(
            torch.zeros(1, 1, 1, real.shape[-1], dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        table = ttnn.concat([real, zero], dim=2)
        for t in (real, rm, zero):
            if t.is_allocated():
                t.deallocate(True)
        runs, base = [], 0
        for item in todo:
            for pos, offset, count in item["runs"]:
                if offset + count > item["patches"] // 4:
                    raise ValueError(f"media {item['hash']}: run past its {item['patches'] // 4} rows")
                runs.append((pos, base + offset, count))
            base += item["patches"] // 4
            item["encoded"] = True
        self.batches.append(
            dict(table=ttnn.reshape(table, (table.shape[-2], table.shape[-1])), zero_row=base, runs=runs)
        )
        logger.info(f"media {self.tag:016x}: encoded {len(todo)} item(s), {base} rows, {batch.local} patches/device")

    def add(self, x_embd, media_tag: int, actual_start: int, actual_end: int):
        """x_embd plus the media rows of [actual_start, actual_end); x_embd unchanged when the chunk has none."""
        if media_tag != self.tag:
            self._load(media_tag)
        overlaps = lambda runs: any(p < actual_end and p + c > actual_start for p, _, c in runs)
        while any(not i["encoded"] and overlaps(i["runs"]) for i in self.items):
            self._encode_from(actual_start)
        flat = np.asarray(rotated_chunk_positions(actual_start, self.sp, self.chunk_local), dtype=np.int64)
        dims = [None, None]
        dims[self.sp_axis] = 0
        for b in self.batches:
            if not overlaps(b["runs"]):
                continue
            idx = np.full(flat.shape, b["zero_row"], dtype=np.int32)
            for pos, row, count in b["runs"]:
                mask = (flat >= max(pos, actual_start)) & (flat < min(pos + count, actual_end))
                idx[mask] = row + (flat[mask] - pos)
            tt_idx = ttnn.from_torch(
                torch.from_numpy(idx).reshape(self.sp, 1, self.chunk_local),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=self.mesh_device.shape, dims=tuple(dims)
                ),
            )
            rows = ttnn.embedding(tt_idx, b["table"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            tt_idx.deallocate(True)
            if len(rows.shape) == 3:
                rows = ttnn.unsqueeze_to_4D(rows)
            out = ttnn.add(x_embd, rows)
            rows.deallocate(True)
            x_embd.deallocate(True)
            x_embd = out
        return x_embd


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, seed: int, position: int) -> int:
    """One draw from [vocab] logits with the request's sampling params; temperature <= 0 is greedy."""
    logits = logits.float()
    if temperature <= 0.0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_p, dim=-1) - sorted_p < top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
        probs = probs / probs.sum()
    gen = torch.Generator().manual_seed((seed * 1_000_003 + position) & 0x7FFFFFFFFFFFFFFF)
    return int(torch.multinomial(probs, 1, generator=gen))
