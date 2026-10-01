# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side loading of a native DeepSeek-V4 checkpoint into the shapes TtV4Transformer takes.

The checkpoint uses DeepSeek's own key names -- ``layers.3.attn.wq_a.weight`` where HF would write
``model.layers.3.self_attn.q_a_proj.weight`` -- and stores each expert as three matrices where the
reference packs them into one. ``v4_layer_from_checkpoint`` holds the per-layer translation and
``v4_model_from_checkpoint`` the model-level keys around it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open

CHECKPOINT_INDEX = "model.safetensors.index.json"

# Model-level keys outside ``layers.*``: the token embedding, the final norm, and the hyper-connection
# head that collapses the residual streams before it.
_EMBED = "embed.weight"
_NORM = "norm.weight"
_HC_HEAD = ("hc_head_fn", "hc_head_base", "hc_head_scale")


def load_checkpoint_tensors(checkpoint_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Read ``names`` from the shards the index puts each in, opening each shard once.

    The export is one shard per layer, so a layer's ~1150 expert matrices sit in a single file of
    tens of GiB. Grouping by shard is therefore not tidiness: opening one per tensor would open and
    parse that file once for every matrix in it.
    """
    with (checkpoint_dir / CHECKPOINT_INDEX).open(encoding="utf-8") as handle:
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


def v4_model_from_checkpoint(config, checkpoint_dir: Path, *, load_embed: bool, load_tail: bool) -> dict:
    """The model-level part of TtV4Transformer's ``state_dict``: embedding on the first rank, head and norm on the last."""
    names = ([_EMBED] if load_embed else []) + ([_NORM, *_HC_HEAD] if load_tail else [])
    if not names:
        return {}
    loaded = load_checkpoint_tensors(checkpoint_dir, names)
    state = {}
    if load_embed:
        state["embed_weight"] = loaded[_EMBED]
    if load_tail:
        state["norm_weight"] = loaded[_NORM]
        state["hc_head"] = tuple(loaded[key].float() for key in _HC_HEAD)
    return state


class V4CheckpointLayers(Sequence):
    """``state_dict["layers"]`` for a rank's slice, read from the checkpoint one layer at a time.

    A Pro layer is ~100 GB of fp32 reference modules, so the slice is never held at once: indexing
    builds that layer's modules, and TtV4Transformer drops them once its block is on device.
    """

    def __init__(self, config, checkpoint_dir: Path, first_layer_idx: int, num_layers: int):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.first_layer_idx = first_layer_idx
        self.num_layers = num_layers

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, local_idx: int) -> dict:
        from models.demos.deepseek_v3_d_p.reference.deepseek_v4.model import v4_layer_weights

        if not 0 <= local_idx < self.num_layers:
            raise IndexError(f"local layer {local_idx} outside a {self.num_layers}-layer slice")
        layer = v4_layer_from_checkpoint(self.config, self.first_layer_idx + local_idx, self.checkpoint_dir)
        return v4_layer_weights(layer, self.config)
