# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Model arguments, derived TP shapes, weight cache paths and lazy checkpoint loading for Muse-Glimmer-30B."""

import json
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from safetensors import safe_open

import ttnn

DEFAULT_CKPT_DIR = "/mnt/tt-data/Muse-Glimmer-30B"
_PREFIX = "model.language_model."
_LAYER_KEYS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.gate_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
]


def checkpoint_dir() -> str:
    return os.environ.get("HF_MODEL") or DEFAULT_CKPT_DIR


def _dtype_str(dtype) -> str:
    if dtype is None:
        return "none"
    names = {
        ttnn.bfloat16: "bf16",
        ttnn.bfloat8_b: "bfp8",
        ttnn.bfloat4_b: "bfp4",
        ttnn.float32: "fp32",
    }
    return names.get(dtype, str(dtype).split(".")[-1].lower())


@dataclass
class MuseGlimmerModelArgs:
    ckpt_dir: str
    # model config (config.json["text_config"])
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    post_norm_eps: float
    rope_theta: float
    sliding_window: int
    layer_types: list
    layer_rope_theta: list
    final_logit_softcapping: float
    qk_scale_factor: float
    output_multiplier: float
    max_position_embeddings: int
    bos_token_id: int
    eos_token_ids: list
    pad_token_id: Optional[int]
    # runtime
    tp: int = 1
    num_layers: Optional[int] = None
    max_batch_size: int = 32
    max_seq_len: int = 8192
    max_prefill_len: int = 8192
    block_size: int = 64
    attn_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b
    embed_dtype: ttnn.DataType = ttnn.bfloat16
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat16
    cache_dir: Optional[str] = None
    _handles: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _weight_map: Optional[dict] = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_checkpoint(cls, ckpt_dir=None, mesh_device=None, tp=None, **overrides) -> "MuseGlimmerModelArgs":
        ckpt_dir = str(ckpt_dir or checkpoint_dir())
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            top = json.load(f)
        d = dict(top.get("text_config", top))

        gen = {}
        gen_path = os.path.join(ckpt_dir, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                gen = json.load(f)

        if mesh_device is not None:
            shape = tuple(mesh_device.shape)
            assert shape[0] == 1, f"Muse replica needs a 1xTP mesh, got {shape}"
            tp = shape[1]
        else:
            tp = tp or 1

        n_layers = d["num_hidden_layers"]
        rope_theta = float(d.get("rope_parameters", {}).get("rope_theta", d.get("rope_theta", 500000.0)))
        layer_types = d.get("layer_types") or [
            "full_attention" if (n_layers - 1 - i) % 4 == 0 else "sliding_attention" for i in range(n_layers)
        ]
        layer_rope_theta = d.get("layer_rope_theta") or [
            0 if (n_layers - 1 - i) % 4 == 0 else rope_theta for i in range(n_layers)
        ]

        eos = gen.get("eos_token_id", d.get("eos_token_id"))
        eos = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        if "pad_token_id" in gen:
            pad = gen["pad_token_id"]
        else:
            pad = d.get("pad_token_id")

        args = cls(
            ckpt_dir=ckpt_dir,
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=n_layers,
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-5)),
            post_norm_eps=float(d.get("post_norm_eps", 1e-8)),
            rope_theta=rope_theta,
            sliding_window=d.get("sliding_window", 2048),
            layer_types=list(layer_types),
            layer_rope_theta=list(layer_rope_theta),
            final_logit_softcapping=float(d.get("final_logit_softcapping", 20.0)),
            qk_scale_factor=float(d.get("qk_scale_factor", 3.87)),
            output_multiplier=float(d.get("output_multiplier", 0.19611613513818404)),
            max_position_embeddings=d.get("max_position_embeddings", 131072),
            bos_token_id=gen.get("bos_token_id", d.get("bos_token_id")),
            eos_token_ids=eos,
            pad_token_id=pad,
            tp=tp,
        )
        for k, v in overrides.items():
            assert hasattr(args, k), f"Unknown MuseGlimmerModelArgs override: {k}"
            setattr(args, k, v)
        args._validate()
        return args

    def _validate(self):
        assert self.tp in (1, 2, 4, 8), f"Unsupported tp={self.tp}"
        assert self.num_attention_heads % self.tp == 0, "heads not divisible by tp"
        assert self.hidden_size % self.tp == 0, "hidden not divisible by tp"
        assert self.n_layers <= self.num_hidden_layers
        assert self.max_seq_len % 256 == 0, f"max_seq_len must be a multiple of 256, got {self.max_seq_len}"
        assert (
            self.max_seq_len <= self.max_position_embeddings
        ), f"max_seq_len {self.max_seq_len} exceeds max_position_embeddings {self.max_position_embeddings}"
        self.max_prefill_len = min(self.max_prefill_len, self.max_seq_len)
        _ = self.intermediate_local  # divisibility asserts

    @property
    def n_layers(self) -> int:
        n = self.num_layers or self.num_hidden_layers
        assert n <= self.num_hidden_layers
        return n

    @property
    def q_heads_local(self) -> int:
        return self.num_attention_heads // self.tp

    @property
    def kv_replicated(self) -> bool:
        return self.num_key_value_heads < self.tp

    @property
    def kv_heads_local(self) -> int:
        return 1 if self.kv_replicated else self.num_key_value_heads // self.tp

    def kv_head_for_rank(self, rank: int) -> int:
        return (rank * self.q_heads_local) * self.num_key_value_heads // self.num_attention_heads

    @property
    def q_width_local(self) -> int:
        return self.q_heads_local * self.head_dim

    @property
    def kv_width_local(self) -> int:
        return self.kv_heads_local * self.head_dim

    @property
    def qkv_width_local(self) -> int:
        return self.q_width_local + 2 * self.kv_width_local

    @property
    def qkvg_width_local(self) -> int:
        return self.qkv_width_local + self.q_width_local

    @property
    def hidden_local(self) -> int:
        return self.hidden_size // self.tp

    @property
    def intermediate_local(self) -> int:
        assert self.intermediate_size % self.tp == 0
        n = self.intermediate_size // self.tp
        assert n % 32 == 0, f"intermediate_local {n} not tile aligned"
        return n

    @property
    def intermediate_local_padded(self) -> int:
        """Per-rank MLP intermediate padded to a multiple of 512 (16 DRAM-sharded cores for down_proj K)."""
        return math.ceil(self.intermediate_local / 512) * 512

    @property
    def padded_vocab_size(self) -> int:
        return math.ceil(self.vocab_size / 2048) * 2048

    @property
    def vocab_local(self) -> int:
        return self.padded_vocab_size // self.tp

    @property
    def sdpa_scale(self) -> float:
        # Exact float32: chunked SDPA binds scale as a no-convert float32 and rejects other doubles.
        return float(torch.tensor(self.qk_scale_factor / math.sqrt(self.head_dim), dtype=torch.float32))

    @property
    def max_num_blocks_per_user(self) -> int:
        return math.ceil(self.max_seq_len / self.block_size)

    @property
    def max_num_blocks(self) -> int:
        return self.max_batch_size * self.max_num_blocks_per_user

    def is_sliding(self, i: int) -> bool:
        return self.layer_types[i] == "sliding_attention"

    def uses_rope(self, i: int) -> bool:
        return self.layer_rope_theta[i] != 0

    def weight_cache_dir(self) -> pathlib.Path:
        base = self.cache_dir or os.environ.get("TT_CACHE_PATH") or f"{self.ckpt_dir}/tt_cache"
        path = pathlib.Path(base) / f"tp{self.tp}"
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK):
                raise PermissionError(str(path))
        except OSError as e:
            home = os.environ.get("TT_METAL_HOME", ".")
            fallback = pathlib.Path(home) / "generated" / "muse_glimmer_tt_cache" / f"tp{self.tp}"
            logger.warning(f"Weight cache dir {path} not writable ({e}); falling back to {fallback}")
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        return path

    def cache_file(self, name: str, dtype=None, tag: str = "") -> str:
        tag_s = f"_{tag}" if tag else ""
        return str(self.weight_cache_dir() / f"{name}{tag_s}_{_dtype_str(dtype)}")

    def _get_weight_map(self) -> dict:
        if self._weight_map is None:
            with open(os.path.join(self.ckpt_dir, "model.safetensors.index.json")) as f:
                self._weight_map = json.load(f)["weight_map"]
        return self._weight_map

    def _load(self, key: str) -> torch.Tensor:
        fname = self._get_weight_map()[key]
        h = self._handles.get(fname)
        if h is None:
            h = safe_open(os.path.join(self.ckpt_dir, fname), framework="pt", device="cpu")
            self._handles[fname] = h
        return h.get_tensor(key).to(torch.bfloat16)

    def load_layer_state_dict(self, i: int) -> dict:
        assert 0 <= i < self.num_hidden_layers
        pre = f"{_PREFIX}layers.{i}."
        return {k: self._load(pre + k) for k in _LAYER_KEYS}

    def load_embedding_weight(self) -> torch.Tensor:
        return self._load(f"{_PREFIX}embed_tokens.weight")

    def load_final_norm_weight(self) -> torch.Tensor:
        return self._load(f"{_PREFIX}norm.weight")

    def load_lm_head_weight(self) -> torch.Tensor:
        return self._load("lm_head.weight")

    def reference_config(self):
        from models.demos.muse_glimmer_30b.reference.model import MuseGlimmerTextConfig

        cfg = MuseGlimmerTextConfig.from_json(self.ckpt_dir)
        n = self.n_layers
        cfg.num_hidden_layers = n
        cfg.layer_types = list(cfg.layer_types[:n])
        cfg.layer_rope_theta = list(cfg.layer_rope_theta[:n])
        return cfg
