# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Readers for the Qwen3.8-27B prefill goldens of the e2e test: HF transformers on the CPU (metadata device "cpu"), not
the bring-up's own CPU reference. Root: $QWEN38_E2E_GOLDENS, else /mnt/models/agentic-prefill-goldens/Qwen/Qwen3.8-27B.

    isl56320/   golden_schema 1 (models/common/hf_golden), revision 1d4bf0f2: bf16 compute, every layer run as 5120-row
                chunks over a growing cache (so its first 5120 positions are a 5120-token prefill), 55-token chat
                prefix + wikitext. Every position of every layer output and of the final norm output; fp32 logits of
                the last position only; attention K (after k_norm and RoPE) / V of every position; Gated DeltaNet
                conv / recurrent state after the last position only.
        input_ids.safetensors        input_ids [56320] int32 (header metadata n_prefix_tokens 55)
        hidden/layer_XX.safetensors  model.layers.<i>.output [1, N, 5120] bf16 (all 64 layers)
        hidden/final.safetensors     model.last_hidden_state [1, N, 5120] bf16, logits [1, 1, 248320] fp32
        cache/layer_XX.safetensors   past_key_values.layers.<i>.keys / values [1, 4, N, 256] bf16 (16 attention
                                     layers); conv_states [1, 10240, 4] bf16 (the last 4 pre-conv q|k|v inputs, oldest
                                     first), recurrent_states [1, 48, 128, 128] fp32 (48 Gated DeltaNet layers)
    isl10240/   the older whole-model layout: fp32 compute, one forward, wikitext with no prefix (header metadata
                n_prefix_tokens 0), so neither golden is a prefix of the other. No layer outputs, no logits.
        input_ids.safetensors        input_ids [10240] int32
        final_hidden.safetensors     final_hidden [1, 10240, 5120] bf16 (the final norm output)
        kv_cache/layer_<i>.safetensors  key_cache_layer_<i> / value_cache_layer_<i> [1, 4, 10240, 256] bf16;
                                     conv_state_layer_<i> [1, 10240, 3] bf16 (oldest first),
                                     recurrent_state_layer_<i> [1, 48, 128, 128] bf16

Every tensor comes back in the layout tt/model.py reads the device state back in (TtModelState.to_torch): a layer
output or the final hidden state as [rows, 5120]; key / value [n_kv, n, head_dim]; conv [conv_dim, n_inputs] (the last
pre-conv q|k|v inputs, oldest first: isl56320 stores four, the model's conv_k and its state's width, as
bringup/trace_hooks.py takes them; isl10240 stores the last three only, so a check compares the newest three columns);
recurrent [v_heads, k_dim, v_dim]. Every read is a safetensors slice: a chunk of a 56k-token file never loads the whole
file, and nothing is cached here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_ROOT = "/mnt/models/agentic-prefill-goldens/Qwen/Qwen3.8-27B"
ROOT_ENV = "QWEN38_E2E_GOLDENS"


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


def header(path: Path) -> dict:
    """{name: (dtype, shape)} of a safetensors file, nothing loaded."""
    with safe_open(str(path), framework="pt") as f:
        return {k: (f.get_slice(k).get_dtype(), list(f.get_slice(k).get_shape())) for k in list(f.keys())}


class Golden:
    """One golden directory of ``n_tokens`` positions. A case may use its first ``n``: positions [0, n) of a causal
    prefill are what a prefill of n tokens computes."""

    layout = ""

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
        """The per-layer kinds the golden was computed with ("linear_attention" / "full_attention"), if recorded."""
        kinds = self.meta.get("layer_kinds") or self.meta.get("layer_types")
        return list(kinds) if kinds else None

    def output_layers(self) -> list[int]:
        """The layers whose output this golden stores at every position."""
        return []

    def layer_out(self, i: int, s0: int, s1: int) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} stores no layer outputs")

    def final(self, s0: int, s1: int) -> torch.Tensor:
        """The final norm output of positions [s0, s1), [s1 - s0, hidden] bf16."""
        raise NotImplementedError

    def logits_at(self, pos: int) -> torch.Tensor | None:
        """The stored logits of position ``pos``, [V] fp32, or None when the golden has none there."""
        return None

    def kv(self, i: int, n: int) -> dict[str, torch.Tensor]:
        """Attention layer i's key / value of positions [0, n), [n_kv, n, head_dim] each."""
        raise NotImplementedError

    def gdn(self, i: int, n: int) -> dict[str, torch.Tensor] | None:
        """Gated DeltaNet layer i's state after position n - 1, or None when the golden does not hold it at n: conv
        [conv_dim, inputs stored] (oldest first), recurrent [v_heads, k_dim, v_dim]."""
        raise NotImplementedError

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
            "transformers_version",
            "torch_version",
            "gdn_implementation",
            "created",
            "seconds",
            "forward_seconds",
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


class TraceGolden(Golden):
    """isl56320/ (module docstring)."""

    layout = "hf_golden trace (golden_schema 1)"

    def output_layers(self) -> list[int]:
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
        p = self.dir / "hidden" / "final.safetensors"
        return read(p, "model.last_hidden_state", (slice(0, 1), slice(s0, s1)))[0]

    def logits_at(self, pos: int) -> torch.Tensor | None:
        if pos != self.n_tokens - 1:
            return None  # the trace stores the logits of its last position only
        return read(self.dir / "hidden" / "final.safetensors", "logits").reshape(-1).float()

    def kv(self, i: int, n: int) -> dict[str, torch.Tensor]:
        p = self.dir / "cache" / f"layer_{i:02d}.safetensors"
        idx = (slice(0, 1), slice(None), slice(0, n))
        return {
            "key": read(p, f"past_key_values.layers.{i}.keys", idx)[0],
            "value": read(p, f"past_key_values.layers.{i}.values", idx)[0],
        }

    def gdn(self, i: int, n: int) -> dict[str, torch.Tensor] | None:
        if n != self.n_tokens:
            return None  # the trace stores the fixed-size state after its last position only
        p = self.dir / "cache" / f"layer_{i:02d}.safetensors"
        return {
            "conv": read(p, f"past_key_values.layers.{i}.conv_states")[0],  # [conv_dim, 4], oldest first
            "recurrent": read(p, f"past_key_values.layers.{i}.recurrent_states")[0],
        }


class LegacyGolden(Golden):
    """isl10240/ (module docstring)."""

    layout = "whole-model forward (final_hidden + kv_cache/)"

    def final(self, s0: int, s1: int) -> torch.Tensor:
        return read(self.dir / "final_hidden.safetensors", "final_hidden", (slice(0, 1), slice(s0, s1)))[0]

    def kv(self, i: int, n: int) -> dict[str, torch.Tensor]:
        p = self.dir / "kv_cache" / f"layer_{i}.safetensors"
        idx = (slice(0, 1), slice(None), slice(0, n))
        return {"key": read(p, f"key_cache_layer_{i}", idx)[0], "value": read(p, f"value_cache_layer_{i}", idx)[0]}

    def gdn(self, i: int, n: int) -> dict[str, torch.Tensor] | None:
        if n != self.n_tokens:
            return None
        p = self.dir / "kv_cache" / f"layer_{i}.safetensors"
        return {"conv": read(p, f"conv_state_layer_{i}")[0], "recurrent": read(p, f"recurrent_state_layer_{i}")[0]}


KINDS = {"isl56320": TraceGolden, "isl10240": LegacyGolden}


def load(name: str, root: Path | None = None) -> Golden:
    """``isl56320`` or ``isl10240`` under ``root`` (default golden_root())."""
    root = Path(root) if root is not None else golden_root()
    return KINDS[name](root, name)
