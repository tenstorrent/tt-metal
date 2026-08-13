# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 AttnRes TTNN query-cache contract.

The pure state-dict extraction and its upstream provenance anchor live in the CPU
reference module. This layer defines stable cache identity and grouped placement;
``TtAttnRes`` performs the actual explicit-path serialization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.reference.attn_res.attn_res import KimiK3AttnResHostQueries
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes


@dataclass(frozen=True)
class KimiK3AttnResDeviceQueries:
    """Placed queries consumed by a composed K3 segment."""

    layer_indices: tuple[int, ...]
    pre: tuple[ttnn.Tensor, ...]
    post: tuple[ttnn.Tensor, ...]
    output: ttnn.Tensor | None = None


class KimiK3AttnResQueryCache:
    """Persist and load folded AttnRes queries with stable model-layer names.

    Files use ``DumpTensorMode.LOCAL`` to preserve the host's mesh shards, so
    ``cache_path`` must be host-local in a multi-host run. Each process must use
    its own directory rather than concurrently writing a shared path.
    """

    @staticmethod
    def _file(
        op: TtAttnRes,
        cache_path: Path,
        cache_id: str,
        kind: str,
        layer_idx: int | None = None,
    ) -> Path:
        if not cache_id:
            raise ValueError("cache_id must identify the checkpoint or query content")
        suffix = kind if layer_idx is None else f"layer_{layer_idx}.{kind}"
        rows, cols = tuple(op.mesh_device.shape)
        identity = sha256(cache_id.encode()).hexdigest()[:16]
        return Path(cache_path) / (
            f"attn_res.id_{identity}.hidden_{op.hidden_size}.mesh_{rows}x{cols}.tp_axis_{op.tp_axis}."
            f"dtype_{op.dtype.name}.layout_{ttnn.TILE_LAYOUT.name}.{suffix}.tensorbin"
        )

    @classmethod
    def check_cache_complete(
        cls,
        op: TtAttnRes,
        cache_path: Path,
        layer_indices: Sequence[int],
        *,
        cache_id: str,
        include_output: bool = False,
    ) -> bool:
        stems = [
            cls._file(op, cache_path, cache_id, kind, layer_idx)
            for layer_idx in layer_indices
            for kind in ("q_pre", "q_post")
        ]
        if include_output:
            stems.append(cls._file(op, cache_path, cache_id, "q_out"))
        return all(file.is_file() for file in stems)

    @classmethod
    def build(
        cls,
        op: TtAttnRes,
        cache_path: Path,
        queries: KimiK3AttnResHostQueries,
        *,
        cache_id: str,
    ) -> None:
        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        lengths = {len(queries.layer_indices), len(queries.pre), len(queries.post)}
        if len(lengths) != 1:
            raise ValueError(
                "AttnRes host query lengths must match: "
                f"layers={len(queries.layer_indices)}, pre={len(queries.pre)}, post={len(queries.post)}"
            )
        for layer_idx, pre, post in zip(queries.layer_indices, queries.pre, queries.post):
            for kind, query in (("q_pre", pre), ("q_post", post)):
                file = cls._file(op, cache_path, cache_id, kind, layer_idx)
                op.cache_query(query, file)
        if queries.output is not None:
            file = cls._file(op, cache_path, cache_id, "q_out")
            op.cache_query(queries.output, file)

    @classmethod
    def load(
        cls,
        op: TtAttnRes,
        cache_path: Path,
        layer_indices: Sequence[int],
        *,
        cache_id: str,
        include_output: bool = False,
    ) -> KimiK3AttnResDeviceQueries:
        layer_indices = tuple(layer_indices)
        if not cls.check_cache_complete(
            op,
            cache_path,
            layer_indices,
            cache_id=cache_id,
            include_output=include_output,
        ):
            raise FileNotFoundError(f"incomplete Kimi K3 AttnRes query cache at {cache_path}")
        placed = []

        def load_one(kind: str, layer_idx: int | None = None):
            query = op.load_query(cls._file(op, cache_path, cache_id, kind, layer_idx))
            placed.append(query)
            return query

        try:
            pre = tuple(load_one("q_pre", layer_idx) for layer_idx in layer_indices)
            post = tuple(load_one("q_post", layer_idx) for layer_idx in layer_indices)
            output = load_one("q_out") if include_output else None
            for query in placed:
                op.prepare_query(query)
            return KimiK3AttnResDeviceQueries(layer_indices, pre, post, output)
        except Exception:
            for query in placed:
                op.release_query(query)
            raise


__all__ = [
    "KimiK3AttnResDeviceQueries",
    "KimiK3AttnResHostQueries",
    "KimiK3AttnResQueryCache",
]
