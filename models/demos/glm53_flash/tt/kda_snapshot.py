# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA state snapshots of the slot-level prefix cache (device side; design: ``doc/serving/prefix_caching.md``).

The 34 KDA layers keep ONE recurrent state per user slot (``FunctionalDecoder.kda_batched``: ``recurrent``
``[B, H/cols, K, V]`` fp32 tiles + the conv taps ``[taps, B, W]`` bf16 tiles in block mode / ``[B, taps, W]``
row-major otherwise, per chip, B = users per row). The traced decode step advances EVERY slot's state in place
(idle slots included), so a finished request's state at a 32-aligned position survives only as a copy: the
served adapter snapshots a slot's carries at every 32-aligned decode crossing (the multi-turn hit position) and
at eager prefill chunk ends, and restores one before a prefix-cache hit continues the slot from that position.

Two stores behind one interface (``save(user, entry)`` / ``load(user, entry)``):

* ``DeviceKdaSnapshotRing``: a DRAM ring per KDA layer, ``users_per_row x entries`` entries per chip (every chip
  of a mesh row holds every user's heads: ``memory_plan.snapshot_bytes_per_chip``), allocated at build before any
  trace capture (``GLMMesh.allocations_frozen``). ``save`` is the traced prefill's slot write in reverse: the
  user's recurrent slice goes into the ring entry through ``paged_fill_cache`` on the ``[E*H*K/32, 1, 32, V]`` view
  (identity page table, the entry index a device scalar) and the conv row through the one-hot mask blend
  (``FunctionalDecoder._write_user_kda_state_slot``); ``load`` slices the entry and writes it into the live state
  with ``FunctionalDecoder._write_user_kda_state`` (the eager prefill's own slot write).
* ``HostKdaSnapshotStore``: the same slices read to the host (one tensor per chip, 35.2 MiB per chip per entry on
  the 8x4: 1.1 GiB of host memory per snapshot) and uploaded back with a per-chip shard mapper; for profiles whose
  DRAM holds no ring (the 1 M x 4 profile) at the cost of a PCIe round trip per crossing / hit.

Nothing here is traced; every call runs between replays (the eager ops of a save are enqueued after the decode
replay that produced the state, before the next replay reads the live state again).
TODO(device): the first (2, 4) block run (``tests/test_serving_prefix_cache_mesh.py``) confirms both stores.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState

TILE = 32


def _upload(mesh, t: torch.Tensor, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=None if mesh.is_single else mesh.replicate(),
    )


def _kda_layers(model) -> list:
    return [dec for dec in model.layers if getattr(dec, "kda_batched", None) is not None]


def _conv_masks(taps: int, n: int, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(taps, n, 1)
    mask[:, idx, 0] = 1.0
    return mask, 1.0 - mask


class DeviceKdaSnapshotRing:
    """DRAM ring of KDA snapshots: ``entries`` per user slot (entry index ``user * entries + e`` per chip)."""

    store = "device"

    def __init__(self, model, entries: int):
        if int(entries) < 1:
            raise ValueError(f"snapshot entries {entries} < 1")
        self.model = model
        self.mesh = model.mesh
        self.entries = int(entries)
        self.layers = _kda_layers(model)
        if not self.layers:
            raise RuntimeError("no KDA layer: nothing to snapshot")
        B = int(model.max_batch)
        self.users = B
        E = B * self.entries
        self.total_entries = E
        attn = self.layers[0]._require_weights().attn
        self.tiled = bool(attn.conv_tiled)
        self.taps = attn.kcfg.conv_kernel_size - 1
        self.rec: list[ttnn.Tensor] = []
        self.conv: list[ttnn.Tensor] = []
        for dec in self.layers:
            live = dec.kda_batched
            rec_shape = (E,) + tuple(int(v) for v in live.recurrent.shape)[1:]
            self.rec.append(
                ttnn.zeros(
                    rec_shape,
                    dtype=live.recurrent.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            cshape = [int(v) for v in live.convolution.shape]
            cshape[1 if self.tiled else 0] = E
            self.conv.append(
                ttnn.zeros(
                    tuple(cshape),
                    dtype=live.convolution.dtype,
                    layout=live.convolution.layout,
                    device=self.mesh.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        nblk = attn.H * attn.K // TILE
        rm, i32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        self.ring_pt = _upload(self.mesh, torch.arange(E * nblk, dtype=torch.int32).reshape(E, nblk), i32, rm)
        self.entry_idx = [_upload(self.mesh, torch.tensor([e], dtype=torch.int32), i32, rm) for e in range(E)]
        self.ring_mask, self.ring_keep = [], []
        if self.tiled:
            for e in range(E):
                mask, keep = _conv_masks(self.taps, E, e)
                self.ring_mask.append(_upload(self.mesh, mask, ttnn.bfloat16, ttnn.TILE_LAYOUT))
                self.ring_keep.append(_upload(self.mesh, keep, ttnn.bfloat16, ttnn.TILE_LAYOUT))

    def _index(self, user: int, entry: int) -> int:
        if not (0 <= user < self.users) or not (0 <= entry < self.entries):
            raise ValueError(f"snapshot (user {user}, entry {entry}) outside {self.users} x {self.entries}")
        return user * self.entries + entry

    def save(self, user: int, entry: int) -> None:
        """The live carries of slot ``user`` -> ring entry ``entry`` (every chip, its own heads)."""
        idx = self._index(user, entry)
        E = self.total_entries
        for k, dec in enumerate(self.layers):
            live = dec.kda_batched
            attn = dec._require_weights().attn
            _, H, K, V = (int(v) for v in live.recurrent.shape)
            rec_u = ttnn.slice(live.recurrent, [user, 0, 0, 0], [user + 1, H, K, V])
            ring_view = ttnn.experimental.view(self.rec[k], ttnn.Shape([E * H * K // TILE, 1, TILE, V]))
            new_view = ttnn.experimental.view(rec_u, ttnn.Shape([1, 1, H * K, V]))
            ttnn.experimental.paged_fill_cache(ring_view, new_view, self.ring_pt, batch_idx_tensor=self.entry_idx[idx])
            ttnn.deallocate(rec_u)
            if self.tiled:
                row = attn.user_conv_state(live, user)  # [1, taps, W] row-major
                tiles = attn.conv_to_tiles(row)  # [taps, 1, W] tiles
                ttnn.deallocate(row)
                ttnn.multiply(self.conv[k], self.ring_keep[idx], output_tensor=self.conv[k])
                new_rows = ttnn.multiply(tiles, self.ring_mask[idx])  # [taps,1,W] x [taps,E,1] -> [taps,E,W]
                ttnn.add(self.conv[k], new_rows, output_tensor=self.conv[k])
                ttnn.deallocate(new_rows)
                ttnn.deallocate(tiles)
            else:
                W = int(live.convolution.shape[-1])
                rows = [
                    ttnn.slice(live.convolution, [user, 0, 0], [user + 1, self.taps, W])
                    if i == idx
                    else ttnn.slice(self.conv[k], [i, 0, 0], [i + 1, self.taps, W])
                    for i in range(E)
                ]
                stacked = ttnn.concat(rows, dim=0)
                ttnn.copy(stacked, self.conv[k])
                ttnn.deallocate(stacked)
                for r in rows:
                    ttnn.deallocate(r)

    def load(self, user: int, entry: int) -> None:
        """Ring entry ``entry`` of slot ``user`` -> the live carries of slot ``user``."""
        idx = self._index(user, entry)
        for k, dec in enumerate(self.layers):
            live = dec.kda_batched
            attn = dec._require_weights().attn
            _, H, K, V = (int(v) for v in live.recurrent.shape)
            W = int(live.convolution.shape[-1])
            rec_e = ttnn.slice(self.rec[k], [idx, 0, 0, 0], [idx + 1, H, K, V])
            if self.tiled:
                rm = attn.conv_to_rm(self.conv[k])  # [E, taps, W] row-major
                row = ttnn.slice(rm, [idx, 0, 0], [idx + 1, self.taps, W])
                ttnn.deallocate(rm)
                conv_e = attn.conv_to_tiles(row)  # [taps, 1, W] tiles: what write_user_conv_state takes
                ttnn.deallocate(row)
            else:
                conv_e = ttnn.slice(self.conv[k], [idx, 0, 0], [idx + 1, self.taps, W])  # [1, taps, W] row-major
            dec._write_user_kda_state(user, KdaState(recurrent=rec_e, convolution=conv_e))  # deallocates both

    def release(self) -> None:
        for t in self.rec + self.conv + [self.ring_pt] + self.entry_idx + self.ring_mask + self.ring_keep:
            if t.is_allocated():
                ttnn.deallocate(t)
        self.rec, self.conv, self.entry_idx, self.ring_mask, self.ring_keep = [], [], [], [], []


class HostKdaSnapshotStore:
    """Host copies of KDA snapshots (``entries`` per user slot; no device memory): per layer the per-chip
    recurrent slice ``[chips, 1, H, K, V]`` fp32 and conv row ``[chips, 1, taps, W]`` bf16 (row-major)."""

    store = "host"

    def __init__(self, model, entries: int):
        if int(entries) < 1:
            raise ValueError(f"snapshot entries {entries} < 1")
        self.model = model
        self.mesh = model.mesh
        self.entries = int(entries)
        self.users = int(model.max_batch)
        self.layers = _kda_layers(model)
        if not self.layers:
            raise RuntimeError("no KDA layer: nothing to snapshot")
        self.tiled = bool(self.layers[0]._require_weights().attn.conv_tiled)
        self.data: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def _check(self, user: int, entry: int) -> None:
        if not (0 <= user < self.users) or not (0 <= entry < self.entries):
            raise ValueError(f"snapshot (user {user}, entry {entry}) outside {self.users} x {self.entries}")

    def _read(self, t: ttnn.Tensor) -> torch.Tensor:
        return torch.stack([ttnn.to_torch(d) for d in self.mesh.device_tensors(t)])

    def _mapper(self):
        return None if self.mesh.is_single else self.mesh.shard_2d(0, 0)  # chunk i of dim 0 -> chip i (row-major)

    def save(self, user: int, entry: int) -> None:
        self._check(user, entry)
        out = []
        for dec in self.layers:
            live = dec.kda_batched
            attn = dec._require_weights().attn
            _, H, K, V = (int(v) for v in live.recurrent.shape)
            rec_u = ttnn.slice(live.recurrent, [user, 0, 0, 0], [user + 1, H, K, V])
            rec_host = self._read(rec_u)
            ttnn.deallocate(rec_u)
            row = attn.user_conv_state(live, user)  # [1, taps, W] row-major
            conv_host = self._read(row)
            ttnn.deallocate(row)
            out.append((rec_host, conv_host))
        self.data[(user, entry)] = out

    def load(self, user: int, entry: int) -> None:
        self._check(user, entry)
        if (user, entry) not in self.data:
            raise KeyError(f"no host snapshot for (user {user}, entry {entry})")
        mapper = self._mapper()
        for dec, (rec_host, conv_host) in zip(self.layers, self.data[(user, entry)]):
            attn = dec._require_weights().attn
            rec = ttnn.from_torch(
                rec_host.reshape(-1, *rec_host.shape[2:]) if mapper is not None else rec_host[0],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            conv_rm = ttnn.from_torch(
                conv_host.reshape(-1, *conv_host.shape[2:]) if mapper is not None else conv_host[0],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            if self.tiled:
                conv = attn.conv_to_tiles(conv_rm)  # [taps, 1, W] tiles
                ttnn.deallocate(conv_rm)
            else:
                conv = conv_rm
            dec._write_user_kda_state(user, KdaState(recurrent=rec, convolution=conv))

    def release(self) -> None:
        self.data = {}
