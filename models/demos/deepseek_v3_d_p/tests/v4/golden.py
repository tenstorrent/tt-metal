# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reader for the DeepSeek-V4 vLLM golden traces and for the native V4 checkpoint.

The traces are ``chunked_group_a_v1``, the layout ``test_prefill_block_chunked.py`` already reads for
GLM: one directory per stream, row-sharded ``rows_<s>_<e>.safetensors`` inside it, and the tensor key
inside each shard is the stream's own name. Row reads therefore go through ``read_sharded_rows``
rather than a second reader; what this module adds is the manifest, the stream names and the two
places where V4 differs from GLM.

What a V4 trace carries:

  * ``decoder_io`` -- ``decoder_input_layer_0`` and ``decoder_output_layer_{i}``, each
    ``[tokens, hc_mult * hidden]``. These are the PACKED hyper-connection streams, which is exactly
    ``TtV4Block``'s own input and output contract, so a layer's input needs no expansion.
  * ``compressed_entries`` -- the layer's compressed KV after the whole prompt,
    ``[tokens / rate, kv_single_dim]``. The rate is the layer's own, so the row count names the kind:
    440 rows over 56320 tokens is heavily-compressed (128), 14080 is compressed-sparse (4).
  * ``expert_ids`` / ``expert_weights`` / ``router_logits`` -- the reference's routing per token,
    which is what a routing disagreement has to be measured against.
  * ``indexer_key_cache`` -- the DSA indexer keys, for the compressed-sparse layers.

A partial trace keeps ``decoder_output`` for only some layers, so the usual alias
``decoder_input_layer_{i} := decoder_output_layer_{i-1}`` holds only where layer ``i-1`` was kept.
``layer_input`` refuses the layers it cannot serve rather than reading a stream that is not there.

The checkpoint uses DeepSeek's native key root -- ``layers.3.attn.wq_a.weight``, not HF's
``model.layers.3.self_attn.q_a_proj.weight`` -- and keeps each expert as three separate matrices
(``ffn.experts.{e}.w{1,2,3}``) where the reference packs them into one. ``layer_tensor_names`` spells
the set one layer needs; turning those into the reference module's parameters is the caller's
business, because that mapping is where the shapes have to be checked one by one.
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
V4_PRO_TRACE = GOLDEN_ROOT / "structured_traces" / "v4_pro_55K_partial_trace" / "trace_v4_pro_full"

# The dequantized export, which is the one a host-side loader can read; the fp8 original cannot be
# consumed without dequantizing it first.
V4_PRO_CHECKPOINT = Path("/mnt/models/blaze/deepseek-ai/DeepSeek-V4-Pro-0813-dequantized")

TRACE_ENV = "V4_PRO_GOLDEN_TRACE"
CKPT_ENVS = ("V4_PRO_HF_MODEL", "V4_PRO_CKPT")
_INDEX = "model.safetensors.index.json"

# One layer's non-expert parameters, by their checkpoint names. The hyper-connection triples are per
# site (attn, ffn), which is the shape TtV4Block's `mhc_weights` wants.
_LAYER_KEYS = (
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn.attn_sink",
    "attn.q_norm.weight",
    "attn.kv_norm.weight",
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.compressor.ape",
    "attn.compressor.norm.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
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

    @property
    def kept_layers(self) -> list[int]:
        """The layers whose decoder output the trace kept, which is what bounds teacher forcing."""
        return sorted(int(name.rsplit("_", 1)[1]) for name in self.streams if name.startswith("decoder_output_layer_"))

    def row_count(self, stream: str) -> int:
        if stream not in self.streams:
            raise KeyError(f"{self.path.name} has no stream {stream}; kept layers are {self.kept_layers}")
        return int(self.streams[stream]["row_count"])

    def rows(self, group: str, stream: str, start: int = 0, end: int | None = None) -> torch.Tensor:
        """Rows ``[start:end]`` of one stream as float32, read only from the shards that overlap.

        ``end=None`` means the whole stream, which the manifest already knows the length of -- the
        decoder streams are 56320 x 28672, so pass a bound whenever a chunk is what is wanted.
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
        """The stack's first live stream: the embedding already expanded to hc_mult streams."""
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
        """``[tokens, n_experts]`` raw gate scores -- the margins a flipped selection turns on."""
        return self.rows("routing", f"router_logits_layer_{layer}", start, end)


def resolve_trace(default: Path = V4_PRO_TRACE) -> GoldenTrace | None:
    """``$V4_PRO_GOLDEN_TRACE`` if set, else ``default`` -- or ``None`` when neither is on the box."""
    override = os.getenv(TRACE_ENV)
    path = Path(override) if override else default
    return GoldenTrace(path) if (path / "index.json").is_file() else None


def resolve_checkpoint(default: Path = V4_PRO_CHECKPOINT) -> Path | None:
    """``$V4_PRO_HF_MODEL`` / ``$V4_PRO_CKPT`` if either names a checkpoint, else ``default``.

    ``None`` when nothing on the box has an index, which is a skip and not a failure: a golden is a
    reference only for the weights it was captured from, so a run without them measures nothing.
    """
    for var in CKPT_ENVS:
        value = os.getenv(var)
        if value and (Path(value) / _INDEX).is_file():
            return Path(value)
    return default if (default / _INDEX).is_file() else None


def layer_tensor_names(layer: int, n_experts: int) -> list[str]:
    """Every checkpoint key one decoder layer needs, experts included.

    The hyper-connection parameters are named per site rather than nested, so they are spelled here
    instead of in ``_LAYER_KEYS``.
    """
    names = [f"layers.{layer}.{key}" for key in _LAYER_KEYS]
    names += [f"layers.{layer}.hc_{site}_{part}" for site in ("attn", "ffn") for part in ("fn", "base", "scale")]
    names += [
        f"layers.{layer}.ffn.experts.{expert}.w{matrix}.weight" for expert in range(n_experts) for matrix in (1, 2, 3)
    ]
    return names


def load_checkpoint_tensors(checkpoint_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Read ``names`` from the shards the index puts each in, opening each shard once.

    One Pro layer is ~1150 expert matrices spread over 66 shards, so the grouping is not tidiness:
    reopening a shard per tensor reads it from disk that many times.
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

    The modules are constructed here rather than through that builder because the builder's whole
    job is to randomise them: for a Pro layer that is a 25-billion-element ``normal_`` over the
    expert matrices, all of it overwritten on the next line.

    Experts are read ``expert_batch`` at a time and copied into the parameter as they arrive. Holding
    all 1152 matrices in a dict first would keep two full copies of a 50 GB layer alive at once.

    Name translation, with the shapes that pin it (Pro, hidden 7168, head_dim 512, 128 heads):

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
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Attention,
        DeepseekV4HyperConnection,
        DeepseekV4RMSNorm,
        DeepseekV4RotaryEmbedding,
        DeepseekV4SparseMoeBlock,
    )

    hidden, inter = config.hidden_size, config.intermediate_size
    n_experts = config.num_local_experts
    prefix = f"layers.{layer_idx}."

    attn = DeepseekV4Attention(config, layer_idx=layer_idx).eval()
    mlp = DeepseekV4SparseMoeBlock(config, layer_idx=layer_idx).eval()
    ref = {
        "attn": attn,
        "mlp": mlp,
        "attn_norm": DeepseekV4RMSNorm(hidden, eps=config.rms_norm_eps).eval(),
        "ffn_norm": DeepseekV4RMSNorm(hidden, eps=config.rms_norm_eps).eval(),
        "attn_hc": DeepseekV4HyperConnection(config).eval(),
        "ffn_hc": DeepseekV4HyperConnection(config).eval(),
    }
    if attn.compressor is None:
        # A sliding-only layer carries no rotary_emb of its own, the same gap build_v4_block_reference fills.
        attn.rotary_emb = DeepseekV4RotaryEmbedding(config)

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
    # The router's second tensor names which kind of layer this is: a frozen table, or the bias that
    # biases selection only.
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
