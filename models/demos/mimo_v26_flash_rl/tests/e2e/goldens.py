# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Readers for the MiMo-V2.6-Flash-RL prefill golden of the e2e test: HF transformers on an H200 (metadata device
"NVIDIA H200"), not the bring-up's own CPU reference. Root: $MIMO_E2E_GOLDENS, else
/mnt/models/agentic-prefill-goldens/XiaomiMiMo/MiMo-V2.6-Flash-RL.

    isl56320/   golden_schema 1 (models/common/hf_golden), revision dcd3fd83: bf16 compute, eager attention, every layer
                run as 5120-row chunks over a growing cache (so its first 5120 positions are a 5120-token prefill),
                wikitext with no prefix. Every position of every layer output and of the final norm output; fp32 logits
                of the last position only; attention K (after RoPE on dims [0, 64)) / V of every position of every
                layer (sliding layers too, although the device only needs their last 128).
        input_ids.safetensors        input_ids [56320] int32 (header metadata n_prefix_tokens 0)
        hidden/layer_XX.safetensors  model.layers.<i>.output [1, N, 4096] bf16 (all 48 layers)
        hidden/final.safetensors     model.last_hidden_state [1, N, 4096] bf16, logits [1, 1, 152576] fp32
        cache/layer_XX.safetensors   past_key_values.layers.<i>.keys [1, KV, N, 192] / values [1, KV, N, 128] bf16;
                                     KV 4 on the 9 full attention layers, 8 on the 39 sliding ones (MoE layers also
                                     store their router outputs, unused here)

Every tensor comes back in the layout tt/model.py reads the device back in: a layer output or the final hidden state
as [rows, 4096]; key / value [KV, n, width] (State.to_torch). Every read is a safetensors slice: a chunk of a 56k-token
file never loads the whole file, and nothing is cached here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_ROOT = "/mnt/models/agentic-prefill-goldens/XiaomiMiMo/MiMo-V2.6-Flash-RL"
ROOT_ENV = "MIMO_E2E_GOLDENS"


def golden_root() -> Path:
    return Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def read(path: Path, name: str, index: tuple | None = None) -> torch.Tensor:
    """Tensor ``name`` of the safetensors file ``path``, or only the slice ``index`` of it (a tuple of slices)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"golden file {path} is missing")
    with safe_open(str(path), framework="pt") as f:
        names = list(f.keys())
        if name not in names:
            raise KeyError(f"{path}: no tensor {name!r} (has {sorted(names)})")
        sl = f.get_slice(name)
        return sl[:] if index is None else sl[index]


class Golden:
    """isl56320/ (module docstring): ``n_tokens`` positions. A case may use its first ``n``: positions [0, n) of a
    causal prefill are what a prefill of n tokens computes."""

    layout = "hf_golden trace (golden_schema 1)"

    def __init__(self, root: Path, name: str):
        self.name = name
        self.dir = Path(root) / name
        self.meta_path = self.dir / "metadata.json"
        self.meta = json.loads(self.meta_path.read_text())
        self.n_tokens = int(self.meta["n_tokens"])

    def ids(self, n: int) -> torch.Tensor:
        """The first n prefill token ids, int64."""
        assert 0 < n <= self.n_tokens, f"{self.name} has {self.n_tokens} tokens, {n} asked"
        return read(self.dir / "input_ids.safetensors", "input_ids", (slice(0, n),)).long()

    def layer_kinds(self) -> list[str] | None:
        """The per-layer kinds the golden was computed with ("full_attention" / "sliding_attention"), if recorded."""
        kinds = self.meta.get("layer_kinds") or self.meta.get("layer_types")
        return list(kinds) if kinds else None

    def output_layers(self) -> list[int]:
        """The layers whose output this golden stores at every position."""
        kept = self.meta.get("hidden_layers")  # a trimmed trace keeps a few layers (a list or {"kept": [...]})
        if isinstance(kept, dict):
            kept = kept.get("kept")
        if kept is None:
            entry = (self.meta.get("tensors") or {}).get("model.layers.{i}.output") or {}
            kept = entry.get("layers", range(int(self.meta["num_layers"])))
        return sorted(int(i) for i in kept)

    def layer_out(self, i: int, s0: int, s1: int) -> torch.Tensor:
        p = self.dir / "hidden" / f"layer_{i:02d}.safetensors"
        return read(p, f"model.layers.{i}.output", (slice(0, 1), slice(s0, s1)))[0]

    def final(self, s0: int, s1: int) -> torch.Tensor:
        """The final norm output of positions [s0, s1), [s1 - s0, hidden] bf16."""
        p = self.dir / "hidden" / "final.safetensors"
        return read(p, "model.last_hidden_state", (slice(0, 1), slice(s0, s1)))[0]

    def logits_at(self, pos: int) -> torch.Tensor | None:
        """The stored logits of position ``pos``, [V] fp32, or None: the trace stores its last position's only."""
        if pos != self.n_tokens - 1:
            return None
        return read(self.dir / "hidden" / "final.safetensors", "logits").reshape(-1).float()

    def kv(self, i: int, n: int) -> dict[str, torch.Tensor]:
        """Attention layer i's key / value of positions [0, n), [KV, n, width] each."""
        p = self.dir / "cache" / f"layer_{i:02d}.safetensors"
        idx = (slice(0, 1), slice(None), slice(0, n))
        return {
            "key": read(p, f"past_key_values.layers.{i}.keys", idx)[0],
            "value": read(p, f"past_key_values.layers.{i}.values", idx)[0],
        }

    def provenance(self) -> dict:
        m = self.meta
        keys = (
            "golden_schema",
            "hf_repo",
            "revision",
            "model_type",
            "compute_dtype",
            "store_dtype",
            "fp32_tensors",
            "device",
            "attn_implementation",
            "attn_chunk",
            "prefill_chunk",
            "experts_implementation",
            "transformers_version",
            "torch_version",
            "created",
            "seconds",
            "source",
            "source_sha256",
        )
        out = {"name": self.name, "dir": str(self.dir), "layout": self.layout, "n_tokens": self.n_tokens}
        out.update({k: m[k] for k in keys if k in m})
        out["metadata_sha256"] = hashlib.sha256(self.meta_path.read_bytes()).hexdigest()
        with safe_open(str(self.dir / "input_ids.safetensors"), framework="pt") as f:
            ids_meta = f.metadata() or {}
        out["n_prefix_tokens"] = int(ids_meta.get("n_prefix_tokens", len(m.get("prefix_ids") or [])))
        if "written_by" in ids_meta:
            out["input_ids_written_by"] = ids_meta["written_by"]
        return out


def load(name: str = "isl56320", root: Path | None = None) -> Golden:
    """``isl56320`` (the only MiMo golden) under ``root`` (default golden_root())."""
    root = Path(root) if root is not None else golden_root()
    return Golden(root, name)
