# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-derived branch flags, from a key/shape index -- not the weights.

Some branches a dry run must take are decided by the *checkpoint*, not the
config: ``LtxTransformerBuilder.__init__`` (``transformer_ltx.py``) scans the
state-dict keys and reads one tensor's shape to set ``has_gate`` and
``cross_attention_adaln``. A weightless dry run that hardcodes those booleans
silently commits to one checkpoint's topology; get ``has_gate`` wrong and the
exact finding phase 5 reported disappears (roadmap blocker 38).

The fix is to derive them the way tt_dit does, from a *metadata-only* source:

* a real ``.safetensors`` file -- its 8-byte length prefix + JSON header carry
  every tensor's dtype and shape with **no tensor bytes read** (KB, not GB), so
  this needs the checkpoint present but never downloads or loads weights;
* a ``.safetensors.index.json`` sharded-checkpoint index -- keys only, no shapes;
* a *declared* manifest -- a key list (and any shapes the rule needs) written
  into the target, standing in for an index that is not on disk.

Whichever was used is reported with the run, the same honesty rule
``hostenv`` follows for its substitutions, so a report is never quietly built on
a stand-in that disagrees with the real checkpoint.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class CheckpointIndex:
    """Keys, and any tensor shapes we could read, plus where they came from."""

    keys: List[str]
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    source: str = "declared"  # human string, reported with the run

    def has_key(self, name: str) -> bool:
        return name in self.keys

    def any_key_contains(self, substr: str) -> bool:
        return any(substr in k for k in self.keys)

    def shape_of(self, name: str) -> Optional[Tuple[int, ...]]:
        return self.shapes.get(name)


def from_safetensors_header(path: str) -> CheckpointIndex:
    """Keys + shapes from a ``.safetensors`` header alone -- no tensor bytes.

    safetensors layout: ``<u64 header_len><header_len bytes of JSON><data>``.
    The JSON maps ``name -> {dtype, shape, data_offsets}`` (plus an optional
    ``__metadata__``). Reading it touches only the first few KB of the file.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    keys, shapes = [], {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        keys.append(name)
        if isinstance(spec, dict) and "shape" in spec:
            shapes[name] = tuple(int(d) for d in spec["shape"])
    return CheckpointIndex(keys=keys, shapes=shapes, source="safetensors header: %s" % path)


def from_index_json(path: str) -> CheckpointIndex:
    """Keys from a sharded-checkpoint ``model.safetensors.index.json``.

    The index carries a ``weight_map`` (key -> shard file) but no shapes, so
    shape-dependent flags fall back to their ``default`` in :func:`ltx_flags`.
    """
    with open(path, "r") as f:
        index = json.load(f)
    keys = list(index.get("weight_map", {}).keys())
    return CheckpointIndex(keys=keys, shapes={}, source="safetensors index: %s" % path)


def declared(keys: Sequence[str], shapes: Optional[Dict[str, Sequence[int]]] = None) -> CheckpointIndex:
    """A key manifest written into a target, standing in for a real index."""
    shp = {k: tuple(int(d) for d in v) for k, v in (shapes or {}).items()}
    return CheckpointIndex(keys=list(keys), shapes=shp, source="declared manifest")


# -- the detection rules, matching tt_dit exactly -----------------------------
# Kept next to the source they mirror so a drift in tt_dit's detection is a
# one-line change here, not a silently wrong flag.
LTX_ADALN_KEY = "model.diffusion_model.adaln_single.linear.weight"
LTX_GATE_MARKER = "to_gate_logits"


def ltx_flags(index: CheckpointIndex, inner_dim: int) -> Dict[str, object]:
    """``has_gate`` / ``cross_attention_adaln`` per ``LtxTransformerBuilder``.

    * ``has_gate = any("to_gate_logits" in k)`` -- pure key scan.
    * ``cross_attention_adaln``: if the adaln weight is present, its first
      dimension ``> 6 * inner_dim``; else ``True`` (tt_dit's fallback). When the
      key is present but no shape was available (index-json source), we cannot
      evaluate the ``> 6*inner_dim`` test, so the flag is reported as unknown and
      the caller keeps its declared default rather than guessing.
    """
    has_gate = index.any_key_contains(LTX_GATE_MARKER)
    if index.has_key(LTX_ADALN_KEY):
        shape = index.shape_of(LTX_ADALN_KEY)
        adaln = None if shape is None else bool(shape[0] > 6 * inner_dim)
    else:
        adaln = True  # tt_dit's fallback when the adaln weight is absent
    return {"has_gate": has_gate, "cross_attention_adaln": adaln}
