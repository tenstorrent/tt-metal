# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The binding prefill spec (``qwen_3_8_27b.spec.json``), parsed once.

The spec carries only what the HF config and the checkpoint cannot tell you: target hardware,
TP/SP, chunk size and sequence length, data formats and the two PCC thresholds. Everything that
IS readable from ``config.json`` lives in :mod:`reference.config` instead and is asserted there.

Every value here is binding: where a borrowed implementation disagrees, the spec wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parent / "qwen_3_8_27b.spec.json"

_TTNN_DTYPES = ("bfloat16", "bfloat8_b", "bfloat4_b", "float32")


@dataclass(frozen=True)
class Acceptance:
    """The two spec-wide PCC numbers, applied unchanged to every component test."""

    pcc_target: float
    pcc_lower_bound: float


@dataclass(frozen=True)
class PrefillSpec:
    model_name: str
    hf_repo: str
    target_hw: str
    tp: int
    sp: int
    max_seq_len: int
    chunk_size: int
    activation_dtype: str
    kv_cache_dtype: str
    weight_dtype: str
    attention_weight_dtype: str
    mlp_up_dtype: str
    mlp_gate_dtype: str
    mlp_down_dtype: str
    acceptance: Acceptance

    @property
    def mesh_shape(self) -> tuple[int, int]:
        """(rows, cols) = (sp, tp) — the graded mesh. SP is the row axis, TP the column axis."""
        return (self.sp, self.tp)

    def validate(self) -> None:
        align = 32 * self.sp
        assert self.max_seq_len % align == 0, f"max_seq_len {self.max_seq_len} % (32*sp={align}) != 0"
        assert self.chunk_size % align == 0, f"chunk_size {self.chunk_size} % (32*sp={align}) != 0"
        for name in ("activation_dtype", "kv_cache_dtype", "weight_dtype"):
            value = getattr(self, name)
            assert value in _TTNN_DTYPES, f"{name}={value!r} is not a ttnn dtype spelling"
        assert 0.0 < self.acceptance.pcc_lower_bound <= self.acceptance.pcc_target <= 1.0


def _group(dataformats: dict, group: str, *path: str) -> str:
    """Resolve one dataformat override, falling back to the group default when empty or absent."""
    node = dataformats[group]
    default = node["default"]
    if not path:
        return default
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return default
    return node or default


@lru_cache(maxsize=1)
def load_spec(path: Path | str = SPEC_PATH) -> PrefillSpec:
    with open(path) as f:
        raw = json.load(f)
    df = raw["dataformats"]
    spec = PrefillSpec(
        model_name=raw["model_name"],
        hf_repo=raw["hf_repo"],
        target_hw=raw["target_hw"],
        tp=raw["parallelism"]["tp"],
        sp=raw["parallelism"]["sp"],
        max_seq_len=raw["shapes"]["max_seq_len"],
        chunk_size=raw["shapes"]["chunk_size"],
        activation_dtype=_group(df, "activations"),
        kv_cache_dtype=_group(df, "kv_cache"),
        weight_dtype=_group(df, "weights"),
        attention_weight_dtype=_group(df, "weights", "attention"),
        mlp_up_dtype=_group(df, "weights", "mlp", "up"),
        mlp_gate_dtype=_group(df, "weights", "mlp", "gate"),
        mlp_down_dtype=_group(df, "weights", "mlp", "down"),
        acceptance=Acceptance(
            pcc_target=raw["acceptance"]["pcc_target"],
            pcc_lower_bound=raw["acceptance"]["pcc_lower_bound"],
        ),
    )
    spec.validate()
    return spec


def ttnn_dtype(name: str):
    """Map a spec dtype spelling to the ttnn enum. Imported lazily so ``spec.py`` stays host-only."""
    import ttnn

    return {
        "bfloat16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
        "float32": ttnn.float32,
    }[name]
