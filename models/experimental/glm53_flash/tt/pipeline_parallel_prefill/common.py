# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the pipeline-parallel GLM-5.3-Flash prefill: the per-stage TP context, layout helpers and the
weight uploader.

Every pipeline stage is a ``1 x tp`` submesh. Inside a stage the layers are tensor parallel along mesh axis 1;
activations crossing a sublayer, layer or stage boundary are replicated bf16 ROW_MAJOR tensors (the residual
streams ``[1, hc, T, D]``), TILE layouts only live inside a sublayer's compute.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import torch

import ttnn

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TL = ttnn.TILE_LAYOUT


def hifi4(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def hifi2(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def lofi(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class TPContext:
    """One pipeline stage: a ``1 x tp`` submesh, its CCLs and its compute configs."""

    def __init__(self, device):
        self.device = device
        self.tp = device.get_num_devices()
        shape = tuple(device.shape)
        assert shape[0] == 1, f"a stage is a 1 x tp row submesh, got {shape}"
        self.hifi4 = hifi4(device)
        self.hifi2 = hifi2(device)
        self.lofi = lofi(device)

    # ---- mesh mappers
    def replicate(self):
        return ttnn.ReplicateTensorToMesh(self.device) if self.tp > 1 else None

    def shard(self, dim: int):
        return ttnn.ShardTensorToMesh(self.device, dim=dim) if self.tp > 1 else None

    # ---- CCLs
    def all_reduce(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.tp == 1:
            return x
        return ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=DRAM)

    def local_slice(self, x: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        """Each chip keeps its 1/tp slice of a replicated tensor along ``dim``."""
        if self.tp == 1:
            return x
        return ttnn.mesh_partition(x, dim=dim, cluster_axis=1, memory_config=DRAM)

    # ---- host <-> device
    def upload(self, x: torch.Tensor, dtype=ttnn.bfloat16, layout=TL) -> ttnn.Tensor:
        return ttnn.from_torch(
            x.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=DRAM,
            mesh_mapper=self.replicate(),
        )

    def zeros_rm(self, rows: int, width: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return self.upload(torch.zeros(1, 1, rows, width), dtype, RM)

    def download(self, x: ttnn.Tensor) -> torch.Tensor:
        """A replicated tensor back to the host (chip 0's copy)."""
        return ttnn.to_torch(ttnn.get_device_tensors(x)[0])

    def download_sharded(self, x: ttnn.Tensor, dim: int) -> torch.Tensor:
        return torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(x)], dim=dim)


def to_tile(x: ttnn.Tensor, dtype=None) -> ttnn.Tensor:
    if x.layout != TL:
        x = ttnn.to_layout(x, TL, memory_config=DRAM)
    if dtype is not None and x.dtype != dtype:
        x = ttnn.typecast(x, dtype, memory_config=DRAM)
    return x


def to_rm(x: ttnn.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    if x.dtype != dtype:
        x = ttnn.typecast(x, dtype, memory_config=DRAM)
    if x.layout != RM:
        x = ttnn.to_layout(x, RM, memory_config=DRAM)
    return x


def pad_rows(n: int, multiple: int = TILE) -> int:
    return -(-n // multiple) * multiple


class ChunkInfo:
    """Positions of one chunk -- rows [start, start + t) of the sequence, the first ``n_valid`` real -- and its
    position-only masks, built once per stage and shared by every layer of that stage.

    A chunk may start anywhere (a decode step is a chunk of one real row at ``start = position``). The attended KV is
    the ``start`` cached rows plus the chunk's ``t`` rows, padded to ``kv_pad`` (a tile multiple); the indexer scores
    ``pool_width`` (a tile multiple >= the complete pools) pool columns. Padded columns are never visible.
    """

    NEG = -1e9

    def __init__(self, ctx: TPContext, start: int, t: int, n_valid: int, kpool: int = 4):
        self.ctx = ctx
        self.start, self.t, self.n_valid, self.kpool = start, t, n_valid, kpool
        self.kv_len = start + t
        self.kv_pad = pad_rows(self.kv_len)
        self.pool_width = pad_rows(max((start + n_valid) // kpool, 1))
        self._cache = {}

    @property
    def aligned(self) -> bool:
        """Chunk and KV are SDPA-chunk (128) aligned and no KV padding is needed."""
        return self.start % 128 == 0 and self.t % 128 == 0

    def _get(self, name, build):
        if name not in self._cache:
            self._cache[name] = build()
        return self._cache[name]

    @property
    def last_pos(self) -> int:
        return self.start + self.n_valid - 1

    def valid(self) -> ttnn.Tensor:
        """fp32 [1, 1, t, 1]: 1 on real rows."""
        return self._get(
            "valid",
            lambda: self.ctx.upload((torch.arange(self.t) < self.n_valid).float().reshape(1, 1, -1, 1), ttnn.float32),
        )

    def _pos(self):
        return (self.start + torch.arange(self.t))[:, None]

    def causal(self) -> ttnn.Tensor:
        """bf16 [1, 1, t, kv_pad] additive causal mask."""

        def build():
            m = torch.arange(self.kv_pad)[None, :] <= self._pos()
            return self.ctx.upload(torch.where(m, 0.0, self.NEG).reshape(1, 1, self.t, self.kv_pad))

        return self._get("causal", build)

    def pool_valid(self) -> ttnn.Tensor:
        """fp32 [1, 1, t, pool_width]: pool p (tokens [kpool p, kpool (p + 1))) is selectable by the query."""

        def build():
            p = torch.arange(self.pool_width)[None, :]
            m = self.kpool * p + self.kpool - 1 <= self._pos()
            return self.ctx.upload(m.float().reshape(1, 1, self.t, -1), ttnn.float32)

        return self._get("pool_valid", build)

    def tail(self) -> ttnn.Tensor:
        """bf16 [1, 1, t, kv_pad]: the query's incomplete trailing pool [kpool floor((pos + 1) / kpool), pos]."""

        def build():
            pos = self._pos()
            j = torch.arange(self.kv_pad)[None, :]
            m = (j >= self.kpool * ((pos + 1) // self.kpool)) & (j <= pos)
            return self.ctx.upload(m.float().reshape(1, 1, self.t, self.kv_pad))

        return self._get("tail", build)


# ---------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------
class WeightLoader:
    """Uploads one layer's weights onto a stage, optionally through a converted-tensor disk cache.

    ``build`` thunks produce the host tensor (fp8-dequantized checkpoint values, already laid out for the device);
    they are only called on a cache miss, so a warm cache never touches the checkpoint.
    """

    def __init__(self, ctx: TPContext, cache_dir: Optional[Path], prefix: str):
        self.ctx = ctx
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.prefix = prefix
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _file(self, name: str, shard_dim) -> Optional[str]:
        """``ttnn.as_tensor`` cache stem (it appends ``_dtype_*_layout_*.tensorbin``)."""
        if self.cache_dir is None:
            return None
        placement = "rep" if shard_dim is None else f"tp{self.ctx.tp}d{shard_dim}"
        return str(self.cache_dir / f"{self.prefix}.{name}.{placement}")

    def __call__(
        self,
        name: str,
        build: Callable[[], torch.Tensor],
        *,
        dtype=ttnn.bfloat16,
        layout=TL,
        shard_dim: Optional[int] = None,
    ) -> ttnn.Tensor:
        ctx = self.ctx
        mapper = ctx.replicate() if shard_dim is None else ctx.shard(shard_dim)
        stem = self._file(name, shard_dim)
        hit = stem is not None and os.path.isfile(f"{stem}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")
        host = None
        if not hit:
            host = build().contiguous()
            if dtype == ttnn.bfloat16:
                host = host.to(torch.bfloat16)
            elif host.dtype != torch.bfloat16:
                host = host.float()
        return ttnn.as_tensor(
            host,
            dtype=dtype,
            layout=layout,
            device=ctx.device,
            memory_config=DRAM,
            cache_file_name=stem,
            mesh_mapper=mapper,
        )
