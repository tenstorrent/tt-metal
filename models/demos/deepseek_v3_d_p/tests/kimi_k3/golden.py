# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reader for the Kimi-K3 vLLM golden traces (``per_stream_safetensors_v1``).

A third on-disk layout, distinct from the two ``test_prefill_transformer_chunked.py`` already
knows. DeepSeek's ``single_file`` writes one safetensors per layer holding every stream as a key;
Kimi-K2.6's ``chunked_group_a_v1`` writes each stream as a *directory* of row shards. K3 writes one
flat file per stream whose single tensor key is the file stem::

    decoder_io/decoder_output_layer_7.safetensors   -> key "decoder_output_layer_7"
    kda/kda_recurrent_state_layer_0.safetensors     -> key "kda_recurrent_state_layer_0"
    kv_cache/layer_7.safetensors                    -> key "kv_post_transform_layer_7"   (the one exception)

Every read goes through ``get_slice``. That is not an optimization: in the 1M trace a single
``decoder_output_layer_*`` file is ~14 GiB, so ``load_file`` would take the host down.

Two traces exist, and they are not interchangeable:

* ``kimi_k3_100k_vllm`` — 102400 tokens through a **5-layer** ``Kimi-K3-SLIM-5L-PARTIAL``
  checkpoint, with module-level streams (``kda/`` for layer 0, ``mla_io/`` + ``sdpa/`` for layer 3,
  ``moe_io/`` for all five) and per-640-token KDA state snapshots. The only source of KDA and
  AttnRes-read intermediates.
* ``k3_vllm_code_debug_1M`` — 1048559 tokens through the **full 93-layer** model, with
  ``decoder_output_layer_{0..24}``, KV for all 24 MLA layers, routing for all 92 MoE layers, and
  ``lm_head_logits``. The oracle for depth.

Both are causal, so a prefix of the token stream is a valid shorter run and every stream may be
sliced to the length under test.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

# The 5-layer module-level trace and the full-model depth trace.
GOLDEN_ROOT = Path("/mnt/models/deepseek-prefill-cache/golden")
TRACE_100K = GOLDEN_ROOT / "structured_traces" / "kimi_k3_100k_vllm"
TRACE_1M = GOLDEN_ROOT / "k3_vllm_code_debug_1M"

# `kv_cache/layer_N.safetensors` is the one stream whose file stem is not its tensor key; every
# other group names the file after the key it holds.
_KV_CACHE_KEY = "kv_post_transform_layer_{layer}"

# The multimodal checkpoint nests the decoder under `language_model.`; the dequantized export
# strips the wrapper. Both are on the box, and only the prefix differs.
CHECKPOINT_PREFIXES = ("language_model.model.", "model.")


@dataclass(frozen=True)
class GoldenTrace:
    """One trace directory, read a slice at a time."""

    path: Path

    @property
    def metadata(self) -> dict:
        with (self.path / "metadata.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    def token_ids(self, count: int, start: int = 0) -> torch.Tensor:
        """`count` prompt tokens beginning at `start`, as `[1, count]` int64.

        `start` is what a chunked run needs: chunk k of a 5120-token prefill wants
        `token_ids(5120, 5120 * k)`, and every stream this class reads is indexed the same way.
        """
        ids = self.metadata["token_ids"][start : start + count]
        if len(ids) < count:
            raise ValueError(
                f"{self.path.name} has {len(self.metadata['token_ids'])} tokens, " f"asked for {count} from {start}"
            )
        return torch.tensor([ids], dtype=torch.int64)

    def has(self, group: str, key: str) -> bool:
        return (self.path / group / f"{key}.safetensors").is_file()

    def rows(self, group: str, key: str, start: int = 0, end: int | None = None) -> torch.Tensor:
        """Rows `[start:end]` of one stream, as float32.

        `end=None` reads to the end of the tensor, which for the 1M trace means ~14 GiB — pass a
        bound unless the stream is known to be small.
        """
        path = self.path / group / f"{key}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"{self.path.name} has no {group}/{key}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            sliced = handle.get_slice(key)
            rows = sliced[start:] if end is None else sliced[start:end]
        return rows.to(torch.float32)

    def decoder_output(self, layer: int, start: int = 0, end: int | None = None) -> torch.Tensor:
        """The residual stream after `layer` — an AttnRes *running sum*, not a plain residual.

        Pinned by `test_golden_contract.py`: after layer 0's seal the running sum restarts, so this
        carries no embedding term.
        """
        return self.rows("decoder_io", f"decoder_output_layer_{layer}", start, end)

    def decoder_input(self, start: int = 0, end: int | None = None) -> torch.Tensor:
        """The stack's first live stream — bit-identical to `embed_tokens[token_ids]`."""
        return self.rows("decoder_io", "decoder_input_layer_0", start, end)

    def has_kv_cache(self, layer: int) -> bool:
        """Whether this trace records the KV slab for one MLA layer, by *model* layer index.

        Both traces record them, but only for the MLA layers within their depth, so callers check
        rather than assume — the same way `has` gates every other optional stream.
        """
        return (self.path / "kv_cache" / f"layer_{layer}.safetensors").is_file()

    def kv_cache(self, layer: int, start: int = 0, end: int | None = None) -> torch.Tensor:
        """`[tokens, kv_lora_rank + qk_rope_head_dim]` for one MLA layer, by *model* layer index.

        The one stream whose file stem is not its tensor key, hence its own reader.
        """
        path = self.path / "kv_cache" / f"layer_{layer}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"{self.path.name} has no kv_cache/layer_{layer}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            sliced = handle.get_slice(_KV_CACHE_KEY.format(layer=layer))
            rows = sliced[start:] if end is None else sliced[start:end]
        return rows.to(torch.float32)


def resolve_trace(default: Path) -> GoldenTrace | None:
    """`$KIMI_K3_GOLDEN_TRACE` if set, else `default` — or `None` when neither is on the box."""
    override = os.getenv("KIMI_K3_GOLDEN_TRACE")
    path = Path(override) if override else default
    return GoldenTrace(path) if (path / "metadata.json").is_file() else None


def resolve_checkpoint() -> Path | None:
    """The Kimi-K3 checkpoint named by `$KIMI_K3_HF_MODEL` / `$KIMI_K3_CKPT`, if it has an index."""
    for var in ("KIMI_K3_HF_MODEL", "KIMI_K3_CKPT"):
        value = os.getenv(var)
        if value and (Path(value) / "model.safetensors.index.json").is_file():
            return Path(value)
    return None


def checkpoint_prefix(checkpoint_dir: Path) -> str:
    """Which of the two key roots this checkpoint uses.

    The published MXFP4 checkpoint is a multimodal wrapper and spells everything
    `language_model.model.…`; the dequantized export drops the wrapper. Reading it off the index
    rather than guessing keeps one loader working against both.
    """
    with (checkpoint_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    for prefix in CHECKPOINT_PREFIXES:
        if f"{prefix}embed_tokens.weight" in weight_map:
            return prefix
    raise ValueError(f"{checkpoint_dir} uses neither known key root: tried {CHECKPOINT_PREFIXES}")


def load_checkpoint_tensors(checkpoint_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Read `names` from the shards the index puts each in, opening each shard once."""
    with (checkpoint_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    missing = sorted(name for name in names if name not in weight_map)
    if missing:
        raise ValueError(f"{checkpoint_dir} index is missing: {missing}")

    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)

    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def embedding_rows(checkpoint_dir: Path, token_ids: torch.Tensor) -> torch.Tensor:
    """`embed_tokens[token_ids]` read row-by-row, so the 2.3 GB table never lands in host memory."""
    prefix = checkpoint_prefix(checkpoint_dir)
    key = f"{prefix}embed_tokens.weight"
    with (checkpoint_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        shard = json.load(handle)["weight_map"][key]
    with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
        table = handle.get_slice(key)
        return torch.stack([table[int(i) : int(i) + 1].squeeze(0) for i in token_ids.flatten()])
