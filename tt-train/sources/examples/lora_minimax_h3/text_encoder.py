# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3's text conditioner on device: the Qwen3-VL 32B language tower, tensor parallel, truncated to the
first `text_encoder_layer` decoder layers, returning the un-normalised residual stream (HF `hidden_states[50]`).

Text-only input makes Qwen3-VL's interleaved mRoPE equal to plain RoPE (all three position axes carry the token
index), so the tower is the ttml Qwen3 architecture with the `model.language_model.*` checkpoint keys."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
import ttnn
import ttml
from ttml.models.qwen3 import Qwen3RMSNorm
from ttml.modules import (
    AbstractModuleBase,
    ColumnParallelLinear,
    FeatureParallelEmbedding,
    ModuleList,
    RowParallelLinear,
)

HF_PREFIX = "model.language_model."
TEXT_ENCODER_LAYER = 50


@dataclass(frozen=True)
class TextTowerConfig:
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_layers: int = TEXT_ENCODER_LAYER
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_seq_len: int = 512

    @classmethod
    def from_hf(cls, config_json: Path, num_layers: int = TEXT_ENCODER_LAYER, max_seq_len: int = 512):
        hf = json.loads(Path(config_json).read_text())
        tc = hf.get("text_config", hf)
        if tc["num_hidden_layers"] <= num_layers:
            raise ValueError(f"need more than {num_layers} decoder layers, checkpoint has {tc['num_hidden_layers']}")
        if tc.get("attention_bias", False):
            raise ValueError("attention bias is not supported")
        return cls(
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_layers=num_layers,
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            rope_theta=float(tc["rope_theta"]),
            max_seq_len=max_seq_len,
        )


class TextTowerAttention(AbstractModuleBase):
    def __init__(self, config: TextTowerConfig) -> None:
        super().__init__()
        tp = ttml.mesh().axis_size("tp")
        if config.num_attention_heads % tp or config.num_key_value_heads % tp:
            raise ValueError(
                f"heads {config.num_attention_heads}/{config.num_key_value_heads} not divisible by tp {tp}"
            )
        self.num_local_heads = config.num_attention_heads // tp
        self.num_local_kv_heads = config.num_key_value_heads // tp
        q_out = config.num_attention_heads * config.head_dim
        kv_out = config.num_key_value_heads * config.head_dim
        self.q_proj = ColumnParallelLinear(config.hidden_size, q_out, has_bias=False)
        self.kv_proj = ColumnParallelLinear(config.hidden_size, 2 * kv_out, has_bias=False)
        self.o_proj = RowParallelLinear(q_out, config.hidden_size, has_bias=False, input_is_parallel=True)
        self.q_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.rope_params = ttml.ops.rope.build_rope_params(
            sequence_length=config.max_seq_len,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            rope_scaling_params=ttml.ops.rope.RopeScalingParams(),
        )

    def forward(self, x, mask):
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        q, k, v = ttml.ops.multi_head_utils.grouped_heads_creation(q, kv, self.num_local_heads, self.num_local_kv_heads)
        q = ttml.ops.rope.rope(self.q_norm(q), self.rope_params, 0)
        k = ttml.ops.rope.rope(self.k_norm(k), self.rope_params, 0)
        attn = ttml.ops.attention.scaled_dot_product_attention(q, k, v, mask)
        return self.o_proj(ttml.ops.multi_head_utils.heads_fusion(attn))


class TextTowerMLP(AbstractModuleBase):
    def __init__(self, config: TextTowerConfig) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, has_bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, has_bias=False)
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, has_bias=False, input_is_parallel=True
        )

    def forward(self, x):
        return self.down_proj(ttml.ops.binary.mul(ttml.ops.unary.silu(self.gate_proj(x)), self.up_proj(x)))


class TextTowerLayer(AbstractModuleBase):
    def __init__(self, config: TextTowerConfig) -> None:
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TextTowerAttention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TextTowerMLP(config)

    def forward(self, x, mask):
        x = ttml.ops.binary.add(x, self.self_attn(self.input_layernorm(x), mask))
        return ttml.ops.binary.add(x, self.mlp(self.post_attention_layernorm(x)))


class TextTower(AbstractModuleBase):
    """`embed_tokens -> layers[:num_layers]`, no final norm, no head."""

    def __init__(self, config: TextTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = FeatureParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList([TextTowerLayer(config) for _ in range(config.num_layers)])

    def forward(self, ids, mask):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# --------------------------------------------------------------------------- weights


def _rows_half_to_interleaved(w: np.ndarray, n_heads: int) -> np.ndarray:
    """HF rotate-half row order -> ttml interleaved pairs, per head. 1-D or 2-D."""
    if w.ndim == 1:
        d = w.shape[0] // n_heads
        v = w.reshape(n_heads, 2, d // 2)
        return v.transpose(0, 2, 1).reshape(-1)
    rows, cols = w.shape
    d = rows // n_heads
    v = w.reshape(n_heads, 2, d // 2, cols)
    return v.transpose(0, 2, 1, 3).reshape(rows, cols)


def _fuse_kv_per_shard(k: np.ndarray, v: np.ndarray, tp: int) -> np.ndarray:
    """[K_s0, V_s0, K_s1, V_s1, ...] so a contiguous column shard holds [K_local | V_local]."""
    kv_out, cols = k.shape
    per = kv_out // tp
    return np.stack([k.reshape(tp, per, cols), v.reshape(tp, per, cols)], axis=1).reshape(2 * kv_out, cols)


def weight_plan(config: TextTowerConfig, root: str) -> dict[str, tuple[tuple[str, ...], int | None]]:
    """ttml param name -> (HF key(s) without the language-model prefix, sharded tensor dim or None)."""
    plan = {f"{root}/embed_tokens/weight": (("embed_tokens.weight",), 3)}
    for i in range(config.num_layers):
        h, t = f"layers.{i}.", f"{root}/layers/{i}/"
        plan[f"{t}self_attn/q_proj/weight"] = ((h + "self_attn.q_proj.weight",), 2)
        plan[f"{t}self_attn/kv_proj/weight"] = ((h + "self_attn.k_proj.weight", h + "self_attn.v_proj.weight"), 2)
        plan[f"{t}self_attn/o_proj/weight"] = ((h + "self_attn.o_proj.weight",), 3)
        plan[f"{t}self_attn/q_norm/weight"] = ((h + "self_attn.q_norm.weight",), None)
        plan[f"{t}self_attn/k_norm/weight"] = ((h + "self_attn.k_norm.weight",), None)
        plan[f"{t}mlp/gate_proj/weight"] = ((h + "mlp.gate_proj.weight",), 2)
        plan[f"{t}mlp/up_proj/weight"] = ((h + "mlp.up_proj.weight",), 2)
        plan[f"{t}mlp/down_proj/weight"] = ((h + "mlp.down_proj.weight",), 3)
        plan[f"{t}input_layernorm/weight"] = ((h + "input_layernorm.weight",), None)
        plan[f"{t}post_attention_layernorm/weight"] = ((h + "post_attention_layernorm.weight",), None)
    return plan


def prepare_weight(name: str, arrays: tuple[np.ndarray, ...], config: TextTowerConfig, tp: int) -> np.ndarray:
    """Host-side transform of the HF array(s) for one ttml parameter into its 4-D global bf16 layout."""
    if name.endswith("kv_proj/weight"):
        k = _rows_half_to_interleaved(arrays[0], config.num_key_value_heads)
        w = _fuse_kv_per_shard(k, arrays[1], tp)
    elif name.endswith("q_proj/weight"):
        w = _rows_half_to_interleaved(arrays[0], config.num_attention_heads)
    elif name.endswith("q_norm/weight") or name.endswith("k_norm/weight"):
        w = _rows_half_to_interleaved(arrays[0], 1)
    else:
        w = arrays[0]
    w = w.reshape(1, 1, 1, -1) if w.ndim == 1 else w.reshape(1, 1, *w.shape)
    return np.ascontiguousarray(w.astype(ml_dtypes.bfloat16, copy=False))


def load_text_tower(model: TextTower, weights_dir: Path, verbose: bool = True) -> None:
    """Fill `model` from the HF `text_encoder/` safetensors shards. Numpy only, one shard open at a time."""
    from safetensors import safe_open

    weights_dir = Path(weights_dir)
    index = json.loads((weights_dir / "model.safetensors.index.json").read_text())["weight_map"]
    params = model.parameters()
    root = next(iter(params)).split("/", 1)[0]
    mesh = ttml.mesh()
    tp = mesh.axis_size("tp")
    device = ttml.autograd.AutoContext.get_instance().get_device()
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)

    plan = weight_plan(model.config, root)
    missing = [n for n in plan if n not in params]
    if missing:
        raise KeyError(f"model has no parameters {missing[:5]}...; have {sorted(params)[:5]}...")

    by_shard: dict[str, list[tuple[str, tuple[str, ...], int | None]]] = {}
    for name, (hf_keys, tdim) in plan.items():
        shard = index[HF_PREFIX + hf_keys[0]]
        by_shard.setdefault(shard, []).append((name, hf_keys, tdim))

    loaded = 0
    for shard, entries in sorted(by_shard.items()):
        with safe_open(str(weights_dir / shard), framework="np") as f:
            for name, hf_keys, tdim in entries:
                arrays = []
                for key in hf_keys:
                    src = index[HF_PREFIX + key]
                    if src == shard:
                        arrays.append(f.get_tensor(HF_PREFIX + key))
                    else:
                        with safe_open(str(weights_dir / src), framework="np") as g:
                            arrays.append(g.get_tensor(HF_PREFIX + key))
                w = prepare_weight(name, tuple(arrays), model.config, tp)
                expected = list(params[name].shape())
                if tdim is not None:
                    expected[tdim] *= tp
                if list(w.shape) != expected:
                    raise ValueError(f"{name}: checkpoint {list(w.shape)} vs model {expected}")
                mapper = mesh.axis_mapper("tp", tdim=tdim) if tdim is not None else replicate
                params[name].assign(
                    ttml.autograd.Tensor.from_numpy(w, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
                )
                loaded += 1
        if verbose:
            print(f"  {shard}: {len(entries)} params ({loaded}/{len(plan)})", flush=True)
    if loaded != len(plan):
        raise RuntimeError(f"loaded {loaded} of {len(plan)} parameters")
