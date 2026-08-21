# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reader for the ``per_stream_safetensors_v1`` Kimi-K3 vLLM trace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

LAYOUT = "per_stream_safetensors_v1"


@dataclass(frozen=True)
class TraceStream:
    name: str
    path: Path
    shape: tuple[int, ...]
    dtype: str


class KimiK3Trace:
    """One trace directory, addressed by stream name.

    Every stream is a whole tensor in its own file, so a stream is read by slicing rows off it
    rather than by stitching chunks together. ``tensor_mapping.json`` is the only source of the
    stream-to-file mapping: the file stem usually equals the stream name but not always.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = json.loads((root / "tensor_mapping.json").read_text())
        layout = self.metadata.get("layout")
        if layout != LAYOUT:
            raise ValueError(f"{root} has layout {layout!r}, expected {LAYOUT!r}")
        self.streams = _expand_streams(root, self.metadata["tensor_streams"])

    @property
    def row_count(self) -> int:
        return int(self.metadata["n_rows"])

    def stream(self, name: str) -> TraceStream:
        stream = self.streams.get(name)
        if stream is None:
            raise KeyError(f"{self.root} has no stream {name!r}; known: {sorted(self.streams)}")
        return stream

    def rows(self, name: str, count: int, start: int = 0) -> torch.Tensor:
        """The first ``count`` rows of a stream from row ``start``, along the token axis."""
        stream = self.stream(name)
        if start < 0 or count <= 0 or start + count > stream.shape[0]:
            raise ValueError(f"rows [{start}, {start + count}) fall outside {name} {stream.shape}")
        with safe_open(str(stream.path), framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[start : start + count]


def _expand_streams(root: Path, entries: list[dict]) -> dict[str, TraceStream]:
    streams: dict[str, TraceStream] = {}
    for entry in entries:
        template = entry["stream"]
        layer_range = entry.get("layer_range")
        layers = range(layer_range[0], layer_range[1] + 1) if layer_range else [None]
        for layer in layers:
            name = template if layer is None else template.format(i=layer)
            path = entry["path"] if layer is None else entry["path"].format(i=layer)
            streams[name] = TraceStream(
                name=name,
                path=root / path,
                shape=tuple(entry["shape"]),
                dtype=entry["dtype"],
            )
    return streams
