# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`ModelArgs` — the **one** place a config value is read, a checkpoint is loaded, or a cache path is built.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaForCausalLM`'s config and checkpoint
layout (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`, `lm_head.weight`).
**Template:** `models/demos/gpt_oss_d_p/tt/model_config.py:30` (`load_state_dict:106`,
`weight_cache_path:157`, `get_state_dict_prefix:175`); second opinion
`models/demos/minimax_m3/tt/model_config.py:22`.

Recipe P1 trap 2 is why this file exists at all: the templates mix `hf_config` **object** access
(`models/demos/minimax_m3/tt/dense_mlp.py:47` does `hf_config.hidden_size`) with dict access, and a
silent mix is how a `None` dimension gets in. This package normalises **once, to a dict**, here, and
every module downstream takes that dict (`bringup_log/03_OUTLINE.md` §2.3).

**Four things from the template are deliberately not carried across.** Each is a measured trap, not
a style preference:

1. `models/demos/gpt_oss_d_p/tt/model_config.py:76`'s `getattr(self.hf_config, "rope_theta", …)` is
   the highest-severity silent-wrongness trap in the bring-up (`07_RISKS.md` R-005). Theta and the
   scaling parameters are read **only** by `tt/rope.py::rope_params`; this class exposes them by
   delegating to that function, so there is still exactly one reader.
2. `models/demos/gpt_oss_d_p/tt/model_config.py:160` defaults the weight-cache root to
   `self.model_path`, i.e. **into `$HF_MODEL`** — which on this box already holds `ttnn_cache/` and
   `P150/` from an unrelated package (`07_RISKS.md` R-003). `weight_cache_path` here **refuses** to
   fall back to the checkpoint directory (`DEC-048`).
3. `hf_config.head_dim` (`models/demos/gpt_oss_d_p/tt/model.py:64`) — Llama's `config.json` has no
   such key (`bringup_log/00_MODEL_CARD.md` §2). `head_dim` comes from
   `tt/config.py::derive_head_dim`, the package's one derivation (`DEC-020`, `DEC-032`).
4. **`map_hf_to_meta_keys` is NOT applied** (`DEC-046`), and neither is
   `convert_hf_qkv_to_meta_format` (`DEC-047`) — the Q/K Meta swizzle belongs to
   `tt/attention/weights.py`, and applying it twice is silently wrong.
"""

import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn

from .config import derive_head_dim
from .rope import rope_params

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The bundled config is byte-identical to the staged checkpoint's (`DEC-001`, asserted by
# `tests/unit/test_reference_model.py::test_bundled_config_matches_checkpoint`), so a dimension-only
# construction needs neither network nor weights.
BUNDLED_CONFIG_PATH = os.path.join(_PKG_ROOT, "configs", "Llama-3.1-8B-Instruct", "config.json")

# `ttnn.DataType` -> the tag that goes in the cache path. A tilized tensor is already sharded AND
# already cast, so both the dtype and the mesh shape must be in the path (`DEC-048`).
_DTYPE_TAG = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}

# HF checkpoint key layout, per module. Used both by the loader's own bookkeeping and by
# `G-WEIGHTS`'s "no missing, no unused" assertion, so the two cannot disagree.
_ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LAYER_NORMS = ("input_layernorm", "post_attention_layernorm")


class ModelArgs:
    """Normalised Llama-3.1-8B config + checkpoint loader + weight-cache paths."""

    def __init__(
        self,
        mesh_device,
        *,
        hf_config=None,
        model_path=None,
        max_seq_len=128 * 1024,
        max_batch_size=1,
    ):
        """
        Args:
            mesh_device: the open mesh. Its shape goes into every weight-cache path.
            hf_config: the raw `config.json` **dict**. `None` loads the bundled copy. A
                `transformers` config *object* is refused: on 5.12.1 it has no `rope_theta`
                attribute and `getattr` with a default silently substitutes a wrong theta
                (recipe P1 trap 1).
            model_path: checkpoint directory; defaults to `$HF_MODEL`. Read for weights only —
                never for the weight cache (`DEC-048`).
            max_seq_len: per-user KV-cache capacity in tokens.
            max_batch_size: user slots. Prefill here is one user per call
                (`tt/attention/kv_cache.py::write_kv_chunk`).
        """
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(mesh_device.shape)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        # Rows carry SP, so a batch wider than the mesh's rows is split across them; this bring-up
        # runs one user, and `max_local_batch_size` is what the modules see.
        self.max_local_batch_size = max_batch_size

        if hf_config is None:
            hf_config = self.load_bundled_config()
        if not isinstance(hf_config, dict):
            raise TypeError(
                f"hf_config must be the raw config.json dict, got {type(hf_config).__name__}. A "
                f"transformers config object has no rope_theta attribute on 5.12.1 and "
                f"getattr(cfg, 'rope_theta', DEFAULT) returns the DEFAULT silently (recipe P1 "
                f"trap 1, 07_RISKS.md R-005)."
            )
        self.hf_config = hf_config

        # --- the normalised dimensions. Read once, here, and asserted non-None. ---------------
        self.hidden_size = self._require("hidden_size")
        self.num_hidden_layers = self._require("num_hidden_layers")
        self.num_attention_heads = self._require("num_attention_heads")
        self.num_key_value_heads = self._require("num_key_value_heads")
        self.intermediate_size = self._require("intermediate_size")
        self.vocab_size = self._require("vocab_size")
        self.rms_norm_eps = self._require("rms_norm_eps")
        self.head_dim = derive_head_dim(hf_config)
        # Delegated, NOT re-read: `tt/rope.py` is the only reader of theta and the scaling
        # parameters, and it asserts all three non-None (`07_RISKS.md` R-005).
        self.rope_theta, self.rope_scale_factor, self.rope_orig_context_len = rope_params(hf_config)

        self.model_path = model_path if model_path is not None else os.getenv("HF_MODEL")
        self.model_name = Path(self.model_path).name if self.model_path else "Llama-3.1-8B-Instruct"

    # ------------------------------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def load_bundled_config() -> dict:
        """The bundled `config.json`, verbatim — no network, no checkpoint, no `transformers`."""
        with open(BUNDLED_CONFIG_PATH) as f:
            return json.load(f)

    def _require(self, key):
        """Read a config key and refuse `None`, rather than letting a `None` dim reach a module."""
        value = self.hf_config.get(key)
        assert value is not None, f"config.json is missing {key!r} — a None dimension would reach the device"
        return value

    # ------------------------------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=False) -> dict:
        """The checkpoint, straight from the safetensors shards, with HF key names **unchanged**.

        Read through `model.safetensors.index.json` rather than
        `AutoModelForCausalLM.from_pretrained` (the template's route,
        `models/demos/gpt_oss_d_p/tt/model_config.py:138`): Llama ships plain bf16 safetensors with
        nothing to dequantise, so instantiating a 16 GB HF model to read its `state_dict` would
        double the host footprint for no gain.

        **Keys stay HF-named.** Recipe P6.2 names `map_hf_to_meta_keys`
        (`models/tt_transformers/tt/load_checkpoints.py:800`), which renames `q_proj`->`wq`,
        `gate_proj`->`w1`, `input_layernorm`->`attention_norm` and so on. Every module in this
        package was built in P5 against **HF** names (`tt/attention/weights.py` splits
        `substate(state_dict, "q_proj")`, `tt/mlp.py` splits `"gate_proj"`), so applying that map
        here would make every key go missing. `DEC-046` records the deviation, and `G-WEIGHTS`
        keeps the equivalent negative control: feed the *mapped* keys and every module weight must
        go missing.

        `convert_to_meta_format=True` **refuses** (`DEC-047`): the Q/K Meta swizzle happens in
        `tt/attention/weights.py` so a weight cannot reach the device un-swizzled via a path that
        forgot, and applying `convert_hf_qkv_to_meta_format` here too would `reverse_permute`
        twice — which reads back plausibly and attends wrongly.
        """
        if convert_to_meta_format:
            raise NotImplementedError(
                "ModelArgs.load_state_dict(convert_to_meta_format=True) would apply the Q/K Meta "
                "swizzle a second time: tt/attention/weights.py already calls "
                "convert_hf_qkv_to_meta_format at load, so a double reverse_permute would produce "
                "a wrong RoPE with no exception. See DEC-047."
            )
        if dummy_weights:
            return {}
        return ModelArgs._load_safetensors(weights_path)

    @staticmethod
    def _load_safetensors(weights_path, prefixes=None) -> dict:
        """Load every shard the index names, optionally only the keys under `prefixes`."""
        from safetensors.torch import load_file

        weights_path = str(weights_path)
        index_path = os.path.join(weights_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"no safetensors index at {index_path}; is HF_MODEL a checkpoint directory?")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        wanted = {k: v for k, v in weight_map.items() if prefixes is None or any(k.startswith(p) for p in prefixes)}
        if not wanted:
            raise KeyError(f"no checkpoint keys match {prefixes}")

        state_dict = {}
        for shard in sorted(set(wanted.values())):
            shard_tensors = load_file(os.path.join(weights_path, shard))
            state_dict.update({k: shard_tensors[k] for k in wanted if wanted[k] == shard})
        logger.info(f"loaded {len(state_dict)} checkpoint tensors from {weights_path}")
        return state_dict

    def expected_state_dict_keys(self, n_layers=None) -> set:
        """Every HF checkpoint key this package's `Model` consumes, at `n_layers` layers.

        Not in `bringup_log/03_OUTLINE.md` §2.3's contract; added because `G-WEIGHTS` has to assert
        "no missing weights **and** no silently-unused weights" against something, and the honest
        something is a list derived from the same constants the loader itself uses (`DEC-046`). A
        hand-maintained list in the test would drift from the loader on the first rename.
        """
        n_layers = self.num_hidden_layers if n_layers is None else n_layers
        keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}."
            keys.update(f"{prefix}self_attn.{name}.weight" for name in _ATTN_PROJECTIONS)
            keys.update(f"{prefix}mlp.{name}.weight" for name in _MLP_PROJECTIONS)
            keys.update(f"{prefix}{name}.weight" for name in _LAYER_NORMS)
        return keys

    def get_state_dict_prefix(self, module_name, layer_idx=None) -> str:
        """The HF key prefix for one module. `("self_attn", 3)` -> `model.layers.3.self_attn.`."""
        if layer_idx is None:
            return f"model.{module_name}." if module_name else "model."
        base = f"model.layers.{layer_idx}."
        return f"{base}{module_name}." if module_name else base

    # ------------------------------------------------------------------------------------------
    # weight cache
    # ------------------------------------------------------------------------------------------
    def weight_cache_path(self, dtype, *, cache_root=None, create=True) -> Path:
        """`<$TT_CACHE_PATH>/tensor_cache_<dtype>_<mesh_shape>`. **Refuses** to use `$HF_MODEL`.

        The dtype and the mesh shape are both in the path because `ttnn.as_tensor` persists the
        **tilized, already-sharded, already-cast** tensor: a cache written at one mesh shape or one
        dtype is wrong at another, and the symptom is "one layer runs on garbage" three phases
        later (`BRINGUP_RECIPE.md:1073-1076`, Appendix B). The template defaults the root to the
        checkpoint directory (`models/demos/gpt_oss_d_p/tt/model_config.py:160`); this box's
        checkpoint already contains two foreign caches, so that default is refused rather than
        followed (`07_RISKS.md` R-003, `DEC-048`).
        """
        root = cache_root if cache_root is not None else os.getenv("TT_CACHE_PATH")
        if not root:
            raise ValueError(
                "weight_cache_path needs TT_CACHE_PATH (or an explicit cache_root). This package "
                "refuses to fall back to the checkpoint directory the way "
                "models/demos/gpt_oss_d_p/tt/model_config.py:160 does: $HF_MODEL already holds "
                "ttnn_cache/ and P150/ from another package (07_RISKS.md R-003, DEC-048)."
            )
        if dtype not in _DTYPE_TAG:
            raise ValueError(f"no cache tag for dtype {dtype}; add it to _DTYPE_TAG deliberately")
        path = Path(root) / f"tensor_cache_{_DTYPE_TAG[dtype]}_{self.mesh_shape[0]}x{self.mesh_shape[1]}"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self):
        return (
            f"ModelArgs(mesh={self.mesh_shape}, layers={self.num_hidden_layers}, "
            f"hidden={self.hidden_size}, heads={self.num_attention_heads}/{self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, vocab={self.vocab_size}, theta={self.rope_theta})"
        )


def torch_dtype_of(state_dict) -> torch.dtype | None:
    """The single dtype every tensor in `state_dict` shares, or `None` if they differ.

    Used by `G-WEIGHTS` to state the loader's input dtype as a measurement rather than an
    assumption: the checkpoint is `bfloat16` (`bringup_log/00_MODEL_CARD.md` §2) and the reference
    dtype policy casts to fp32 only on the *reference* side (`DEC-006`).
    """
    dtypes = {t.dtype for t in state_dict.values()}
    return next(iter(dtypes)) if len(dtypes) == 1 else None
