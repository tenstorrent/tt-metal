# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reader for the DeepSeek-V4 vLLM golden traces and for the native V4 checkpoint.

The traces use the same layout GLM's do: one directory per stream, split into
``rows_<s>_<e>.safetensors`` shards, each shard keying its tensor by the stream's own name. Reading
rows is therefore ``read_sharded_rows``' job; this module adds the manifest and the stream names.

What a V4 trace carries:

  * ``decoder_io`` -- ``decoder_input_layer_0`` and ``decoder_output_layer_{i}``, each
    ``[tokens, hc_mult * hidden]``. The hyper-connection streams come packed on the last dim, the
    same width ``TtV4Block`` takes and returns, but a TP upload still permutes them
    (``_pack_streams``) so each chip gets its hidden slice of every stream.
  * ``compressed_entries`` -- the layer's compressed KV after the whole prompt. Each layer compresses
    at its own rate, so the row count says which kind it is: over 56320 tokens, 440 rows is HCA
    (rate 128) and 14080 is CSA (rate 4).
  * ``expert_ids`` / ``expert_weights`` / ``router_logits`` -- which experts the reference picked per
    token, and the scores it picked them from.
  * ``indexer_key_cache`` -- the keys a CSA layer selects with.

A trace can hold ``decoder_output`` for only some layers. Teacher forcing feeds layer ``i`` the
output of layer ``i-1``, so it works only where that layer was kept, and ``layer_input`` says so
rather than reading a directory that is not there.

The checkpoint uses DeepSeek's own key names -- ``layers.3.attn.wq_a.weight`` where HF would write
``model.layers.3.self_attn.q_a_proj.weight`` -- and stores each expert as three matrices where the
reference packs them into one. ``v4_layer_from_checkpoint`` holds that translation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.tt.v4.weights import CHECKPOINT_INDEX as _INDEX
from models.demos.deepseek_v3_d_p.utils.test_utils import read_sharded_rows

GOLDEN_ROOT = Path("/mnt/models/deepseek-prefill-cache/golden")
_CHECKPOINT_ROOT = Path("/mnt/models/blaze/deepseek-ai")


@dataclass(frozen=True)
class GoldenVariant:
    """One model's trace, the checkpoint it was captured from, and the env vars that override them."""

    trace: Path
    checkpoint: Path
    trace_env: str
    ckpt_envs: tuple[str, ...]


# The dequantized exports, which are the ones a host loader can read; the fp8 originals need
# dequantizing first. Both traces carry the same streams, so one reader serves both.
V4_PRO = GoldenVariant(
    trace=GOLDEN_ROOT / "structured_traces" / "v4_pro_55K_partial_trace" / "trace_v4_pro_full",
    checkpoint=_CHECKPOINT_ROOT / "DeepSeek-V4-Pro-0813-dequantized",
    trace_env="V4_PRO_GOLDEN_TRACE",
    ckpt_envs=("V4_PRO_HF_MODEL", "V4_PRO_CKPT"),
)
V4_FLASH = GoldenVariant(
    trace=GOLDEN_ROOT / "structured_traces" / "v4_flash_55K_partial_trace" / "trace_v4_flash_full",
    checkpoint=_CHECKPOINT_ROOT / "DeepSeek-V4-Flash-0731-dequantized",
    trace_env="V4_FLASH_GOLDEN_TRACE",
    ckpt_envs=("V4_FLASH_HF_MODEL", "V4_FLASH_CKPT"),
)


@dataclass(frozen=True)
class GoldenTrace:
    """One trace directory, read a slice at a time."""

    path: Path

    @cached_property
    def index(self) -> dict:
        with (self.path / "index.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    @cached_property
    def metadata(self) -> dict:
        with (self.path / "metadata.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def streams(self) -> dict:
        return self.index["tensor_streams"]

    @cached_property
    def kept_layers(self) -> list[int]:
        """The layers whose output the trace holds, which is what bounds teacher forcing.

        Taken from the directories, not the manifest: v4_flash's index lists all 43 layers where the
        trace holds 10, so the manifest would promise layers that cannot be read.
        """
        kept = []
        for name in self.streams:
            if name.startswith("decoder_output_layer_") and (self.path / "decoder_io" / name).is_dir():
                kept.append(int(name.rsplit("_", 1)[1]))
        return sorted(kept)

    def row_count(self, stream: str) -> int:
        if stream not in self.streams:
            raise KeyError(f"{self.path.name} has no stream {stream}; kept layers are {self.kept_layers}")
        return int(self.streams[stream]["row_count"])

    def rows(self, group: str, stream: str, start: int = 0, end: int | None = None) -> torch.Tensor:
        """Rows ``[start:end]`` of one stream as float32, reading only the shards that overlap.

        ``end=None`` reads the whole stream. The decoder streams are 56320 x 28672, so pass a bound
        when only a chunk is wanted.
        """
        end = self.row_count(stream) if end is None else end
        return read_sharded_rows(self.path / group / stream, stream, start, end)

    def token_ids(self, count: int, start: int = 0) -> torch.Tensor:
        """``count`` prompt tokens from ``start``, as ``[1, count]`` int64.

        A hash-routed layer indexes its expert table with these, so a chunked run needs the chunk's
        own slice: chunk k of a 5120-token prefill wants ``token_ids(5120, 5120 * k)``.
        """
        ids = self.metadata["token_ids"][start : start + count]
        if len(ids) < count:
            raise ValueError(
                f"{self.path.name} has {len(self.metadata['token_ids'])} tokens, asked for {count} from {start}"
            )
        return torch.tensor([ids], dtype=torch.int64)

    def decoder_input(self, start: int = 0, end: int | None = None) -> torch.Tensor:
        """The embedding, already expanded to hc_mult streams -- what layer 0 is fed."""
        return self.rows("decoder_io", "decoder_input_layer_0", start, end)

    def decoder_output(self, layer: int, start: int = 0, end: int | None = None) -> torch.Tensor:
        """The packed streams leaving ``layer`` -- the truth a block test compares against."""
        return self.rows("decoder_io", f"decoder_output_layer_{layer}", start, end)

    def layer_input(self, layer: int, start: int = 0, end: int | None = None) -> torch.Tensor:
        """What teacher-forcing feeds ``layer``: layer 0's own stream, else layer-1's output."""
        if layer == 0:
            return self.decoder_input(start, end)
        if layer - 1 not in self.kept_layers:
            raise ValueError(
                f"teacher-forcing layer {layer} needs decoder_output_layer_{layer - 1}, which "
                f"{self.path.name} did not keep (kept: {self.kept_layers})"
            )
        return self.decoder_output(layer - 1, start, end)

    def compressed_entries(self, layer: int, count: int | None = None) -> torch.Tensor:
        """The layer's compressed KV, ``[entries, kv_single_dim]``.

        The stream holds the cache after the whole prompt, so a run of fewer chunks compares against
        the first ``ceil(tokens / rate)`` rows of it, which is what ``count`` is for.
        """
        return self.rows("compressed_entries", f"compressed_entries_layer_{layer}", 0, count)

    def expert_ids(self, layer: int, start: int = 0, end: int | None = None) -> torch.Tensor:
        """``[tokens, top_k]`` of the reference's chosen experts, to compare routing against.

        Cast back to int64: the stream is int32, and every row reader here widens to float32.
        """
        return self.rows("routing", f"expert_ids_layer_{layer}", start, end).to(torch.int64)

    def expert_weights(self, layer: int, start: int = 0, end: int | None = None) -> torch.Tensor:
        return self.rows("routing", f"expert_weights_layer_{layer}", start, end)

    def router_logits(self, layer: int, start: int = 0, end: int | None = None) -> torch.Tensor:
        """``[tokens, n_experts]`` raw gate scores, before the top-k picks from them."""
        return self.rows("routing", f"router_logits_layer_{layer}", start, end)


def resolve_trace(variant: GoldenVariant = V4_PRO) -> GoldenTrace | None:
    """The variant's trace env var if set, else its default -- or ``None`` when neither is on the box."""
    override = os.getenv(variant.trace_env)
    path = Path(override) if override else variant.trace
    return GoldenTrace(path) if (path / "index.json").is_file() else None


def resolve_checkpoint(variant: GoldenVariant = V4_PRO) -> Path | None:
    """The first of the variant's checkpoint env vars that names one, else its default.

    ``None`` when neither has an index on the box. A readable index does not mean readable shards:
    a checkpoint whose files are owner-only resolves here and fails on the first tensor read.
    """
    for var in variant.ckpt_envs:
        value = os.getenv(var)
        if value and (Path(value) / _INDEX).is_file():
            return Path(value)
    return variant.checkpoint if (variant.checkpoint / _INDEX).is_file() else None
