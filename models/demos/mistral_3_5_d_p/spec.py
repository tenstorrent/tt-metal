# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loader for the BINDING prefill spec (``Mistral-Medium-3.5-128B.spec.json``).

The spec is the highest-priority input to the bring-up: target HW, TP/SP, max_seq_len, chunk_size,
the dtype groups and the PCC bar. Every module reads its values from here rather than hard-coding
them, so a spec edit propagates instead of drifting.

Nothing readable from ``configs/<model>/config.json`` lives in the spec by design (dims, layer count,
attention family, rope parameters, vocab, norm eps, checkpoint quantization) — read those from
:mod:`models.demos.mistral_3_5_d_p.reference.mistral_config` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import ttnn

PACKAGE_DIR = Path(__file__).resolve().parent
SPEC_PATH = PACKAGE_DIR / "Mistral-Medium-3.5-128B.spec.json"

# ttnn dtype spellings accepted in the spec's dataformats block.
_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
}

# Mesh shape per target_hw, as (rows, cols) = (SP, TP) with tp_axis=1.
_TARGET_MESH = {
    "bh_galaxy": (4, 8),
    "wh_galaxy": (4, 8),
    "bh_loudbox": (1, 8),
    "bh_quietbox": (1, 4),
    "single_bh": (1, 1),
}


def _dtype(name: str, default):
    """Resolve a ttnn dtype from a spec spelling; an empty/absent override inherits ``default``."""
    if not name:
        return default
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"unknown dtype spelling {name!r} in the prefill spec; valid: {sorted(_DTYPES)}")


@dataclass(frozen=True)
class PrefillSpec:
    """The binding spec, resolved. Field names mirror the JSON."""

    model_name: str
    target_hw: str
    tp: int
    sp: int
    max_seq_len: int
    chunk_size: int
    pcc: float
    activation_dtype: object
    kv_cache_dtype: object
    attention_weight_dtype: object
    mlp_up_dtype: object
    mlp_gate_dtype: object
    mlp_down_dtype: object

    @property
    def cache_capacity(self) -> int:
        """Per-user KV-cache capacity in tokens: ``max_seq_len`` rounded UP to a whole number of chunks.

        The spec only requires ``max_seq_len % (32 * sp) == 0``, but the whole-cache block-cyclic
        indexed rope and the KV chunk address table both tile the cache by ``chunk_size``, so the
        allocated capacity must be a multiple of it (262144 is 51.2 chunks of 5120 -> 52 chunks =
        266240). The spec's ``max_seq_len`` stays the servable context; the extra tail is capacity
        only, never addressed by a request, and the runtime asserts chunks stay inside ``max_seq_len``.
        """
        return -(-self.max_seq_len // self.chunk_size) * self.chunk_size

    @property
    def seq_local(self) -> int:
        """Per-chip cache rows: the SP shard of :attr:`cache_capacity`."""
        return self.cache_capacity // self.sp

    @property
    def mesh_shape(self) -> tuple:
        """(rows, cols) == (SP, TP): TP on the cols (``tp_axis=1``), SP prefill on the rows."""
        return _TARGET_MESH[self.target_hw]

    @property
    def sp_axis(self) -> int:
        return 0

    @property
    def tp_axis(self) -> int:
        return 1

    def __post_init__(self):
        rows, cols = _TARGET_MESH[self.target_hw]
        # The mesh shape is derived from target_hw, so a spec whose tp/sp disagree with it is
        # self-contradictory: fail at load rather than silently sharding across the wrong axis.
        if (self.sp, self.tp) != (rows, cols):
            raise ValueError(
                f"spec parallelism sp={self.sp} tp={self.tp} does not match target_hw={self.target_hw!r} "
                f"mesh {(rows, cols)} (rows=SP, cols=TP)"
            )
        # Both values are block-cyclic addressing periods of the KV table: a misaligned one corrupts
        # addresses silently rather than raising, so check them here, once, at the source.
        for name, value in (("max_seq_len", self.max_seq_len), ("chunk_size", self.chunk_size)):
            if value % (ttnn.TILE_SIZE * self.sp) != 0:
                raise ValueError(f"spec {name}={value} must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * self.sp})")
        if self.cache_capacity % (ttnn.TILE_SIZE * self.sp) != 0:
            raise ValueError(
                f"cache_capacity={self.cache_capacity} must be a multiple of TILE_SIZE*sp "
                f"({ttnn.TILE_SIZE * self.sp}); seq_local must be tile-aligned"
            )


@lru_cache(maxsize=1)
def load_spec(path: str | Path = SPEC_PATH) -> PrefillSpec:
    """Parse + validate the prefill spec. Cached: the file does not change within a run."""
    raw = json.loads(Path(path).read_text())
    fmts = raw["dataformats"]
    act_default = _dtype(fmts["activations"].get("default", ""), ttnn.bfloat16)
    kv_default = _dtype(fmts["kv_cache"].get("default", ""), ttnn.bfloat8_b)
    w = fmts["weights"]
    w_default = _dtype(w.get("default", ""), ttnn.bfloat8_b)
    mlp = w.get("mlp") or {}
    return PrefillSpec(
        model_name=raw["model_name"],
        target_hw=raw["target_hw"],
        tp=raw["parallelism"]["tp"],
        sp=raw["parallelism"]["sp"],
        max_seq_len=raw["shapes"]["max_seq_len"],
        chunk_size=raw["shapes"]["chunk_size"],
        pcc=raw["acceptance"]["pcc"],
        activation_dtype=act_default,
        kv_cache_dtype=kv_default,
        attention_weight_dtype=_dtype(w.get("attention", ""), w_default),
        mlp_up_dtype=_dtype(mlp.get("up", ""), w_default),
        mlp_gate_dtype=_dtype(mlp.get("gate", ""), w_default),
        mlp_down_dtype=_dtype(mlp.get("down", ""), w_default),
    )


SPEC = load_spec()
