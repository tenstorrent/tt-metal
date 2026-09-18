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
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.utils.test_utils import read_sharded_rows

GOLDEN_ROOT = Path("/mnt/models/deepseek-prefill-cache/golden")
_CHECKPOINT_ROOT = Path("/mnt/models/blaze/deepseek-ai")
_INDEX = "model.safetensors.index.json"


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


def load_checkpoint_tensors(checkpoint_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Read ``names`` from the shards the index puts each in, opening each shard once.

    The export is one shard per layer, so a layer's ~1150 expert matrices sit in a single file of
    tens of GiB. Grouping by shard is therefore not tidiness: opening one per tensor would open and
    parse that file once for every matrix in it.
    """
    with (checkpoint_dir / _INDEX).open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    missing = sorted(name for name in names if name not in weight_map)
    if missing:
        raise ValueError(f"{checkpoint_dir} index is missing {len(missing)} keys, first: {missing[:4]}")

    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)

    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def v4_layer_from_checkpoint(
    config,
    layer_idx: int,
    checkpoint_dir: Path,
    expert_batch: int = 32,
) -> dict:
    """One decoder layer's real weights, in the dict shape ``build_v4_block_reference`` returns.

    The modules are built here rather than through that builder because the builder's whole job is
    to randomise them -- for a Pro layer, 25 billion elements of ``normal_`` over the expert
    matrices, all overwritten on the next line.

    Experts are read ``expert_batch`` at a time and copied in as they arrive; holding all 1152
    matrices in a dict first would keep two copies of a 50 GB layer alive at once.

    Name translation, with Pro's shapes (hidden 7168, head_dim 512, 128 heads):

        attn.wq_a            -> q_a_proj                  [1536, 7168]
        attn.q_norm          -> q_a_norm                  [1536]
        attn.wq_b            -> q_b_proj                  [65536, 1536]
        attn.wkv             -> kv_proj                   [512, 7168]
        attn.wo_a / wo_b     -> o_a_proj / o_b_proj       [16384, 4096] / [7168, 16384]
        attn.attn_sink       -> sinks                     [128]
        compressor.ape       -> compressor.position_bias  [128, 512]
        compressor.norm      -> compressor.kv_norm        [512]
        ffn.gate.bias        -> gate.e_score_correction_bias   (top-k layers)
        ffn.gate.tid2eid     -> gate.tid2eid                   (hash layers)
        ffn.experts.e.w1/w3  -> experts.gate_up_proj[e]    cat on dim 0, gate half first
        ffn.experts.e.w2     -> experts.down_proj[e]
        hc_{site}_fn/base/scale -> ref["{site}_hc"].fn/base/scale
    """
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.block import v4_block_modules

    inter = config.intermediate_size
    n_experts = config.num_local_experts
    prefix = f"layers.{layer_idx}."

    ref = v4_block_modules(config, layer_idx)
    attn, mlp = ref["attn"], ref["mlp"]

    flat = {
        "attn_norm.weight": ref["attn_norm"].weight,
        "ffn_norm.weight": ref["ffn_norm"].weight,
        "attn.attn_sink": attn.sinks,
        "attn.wq_a.weight": attn.q_a_proj.weight,
        "attn.q_norm.weight": attn.q_a_norm.weight,
        "attn.wq_b.weight": attn.q_b_proj.weight,
        "attn.wkv.weight": attn.kv_proj.weight,
        "attn.kv_norm.weight": attn.kv_norm.weight,
        "attn.wo_a.weight": attn.o_a_proj.weight,
        "attn.wo_b.weight": attn.o_b_proj.weight,
        "ffn.gate.weight": mlp.gate.weight,
        "ffn.shared_experts.w1.weight": mlp.shared_experts.gate_proj.weight,
        "ffn.shared_experts.w2.weight": mlp.shared_experts.down_proj.weight,
        "ffn.shared_experts.w3.weight": mlp.shared_experts.up_proj.weight,
    }
    if attn.compressor is not None:
        flat.update(
            {
                "attn.compressor.wkv.weight": attn.compressor.kv_proj.weight,
                "attn.compressor.wgate.weight": attn.compressor.gate_proj.weight,
                "attn.compressor.ape": attn.compressor.position_bias,
                "attn.compressor.norm.weight": attn.compressor.kv_norm.weight,
            }
        )
    # The router's second tensor says which kind of layer this is: a hash layer has the frozen
    # table, a top-k layer the selection bias.
    flat["ffn.gate.tid2eid" if mlp.is_hash else "ffn.gate.bias"] = (
        mlp.gate.tid2eid if mlp.is_hash else mlp.gate.e_score_correction_bias
    )
    for site in ("attn", "ffn"):
        hc = ref[f"{site}_hc"]
        flat[f"hc_{site}_fn"] = hc.fn
        flat[f"hc_{site}_base"] = hc.base
        flat[f"hc_{site}_scale"] = hc.scale

    loaded = load_checkpoint_tensors(checkpoint_dir, [prefix + key for key in flat])
    with torch.no_grad():
        for key, param in flat.items():
            value = loaded[prefix + key]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"{prefix}{key} is {tuple(value.shape)}, the module wants {tuple(param.shape)}")
            param.copy_(value)

        for start in range(0, n_experts, expert_batch):
            stop = min(start + expert_batch, n_experts)
            names = [
                f"{prefix}ffn.experts.{expert}.w{matrix}.weight"
                for expert in range(start, stop)
                for matrix in (1, 2, 3)
            ]
            batch = load_checkpoint_tensors(checkpoint_dir, names)
            for expert in range(start, stop):
                w1, w2, w3 = (batch[f"{prefix}ffn.experts.{expert}.w{m}.weight"] for m in (1, 2, 3))
                mlp.experts.gate_up_proj[expert, :inter].copy_(w1)
                mlp.experts.gate_up_proj[expert, inter:].copy_(w3)
                mlp.experts.down_proj[expert].copy_(w2)
            del batch

    return ref
