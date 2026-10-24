# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 dense decoder (``transformers.Qwen3ForCausalLM``) as a tt_dit text encoder.

FLUX.2 Klein encodes its prompt with Qwen3-4B (4B models) or Qwen3-8B (9B models) and feeds the DiT the hidden
states of three layers concatenated per token. Architecturally Qwen3 is the ``TransformerEncoder`` of
``encoders/transformer.py`` with GQA, per-head RMSNorm on q and k before RoPE (``attn_qk_norm``), SiLU-gated MLP,
no biases and tied embeddings; this module is the HF config and state-dict mapping, like ``Mistral3Encoder`` and
``SmolLm3Encoder``. The exact profile (bf16 weights, fp32 accumulation) is the default here: ``linear_dtype`` /
``attn_qkv_dtype`` are bf16, not the generic encoder's bfp8. (An fp32 reduce of the tensor-parallel partial sums of
o_proj / down_proj was measured on the 4B text encoder, job 146466: WORSE than bf16 partials, real-row rel_rmse 0.0085
-> 0.0120, pad block 0.9951 -> 0.9910; not used.)

Depth: a tap consumer needs only the decoder layers below its deepest tap (HF ``hidden_states[k]`` is the OUTPUT of
layer ``k - 1``), so :class:`Qwen3Checkpoint` takes ``num_layers`` / ``final_norm`` and builds, loads and caches that
prefix alone (the FLUX 3 Klein text encoder does the same: 32 of its 36 layers). The state keys of the layers above
and the final norm are dropped explicitly in :func:`route_qwen3_state_keys` (``Module.load_torch_state_dict`` is
strict); a reduced entry gets its own weight-cache subfolder (``<subfolder>_L<num_layers>``) so it never collides
with the full-depth one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

import torch
import transformers

import ttnn
from models.tt_dit.blocks.rope import RopeConfig

from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from ..transformer import StateConversion, TransformerEncoder, TransformerEncoderConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

EXACT_DTYPE = ttnn.bfloat16
EXACT_WEIGHT_CACHE_DTYPE = "bf16"
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_FINAL_NORM_KEY = "final_norm.weight"
_FINAL_LINEAR_KEY = "final_linear.weight"
_EMBEDDING_KEY = "token_embedding.weight"


def _rope_theta(hf_config: transformers.PretrainedConfig) -> float:
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    return float(hf_config.rope_theta)


class Qwen3Encoder(TransformerEncoder):
    @staticmethod
    def config_from_hf(
        hf_config: transformers.PretrainedConfig,
        *,
        num_layers: int | None = None,
        final_norm: bool = True,
        final_linear: bool = False,
        linear_dtype: ttnn.DataType = EXACT_DTYPE,
        linear_math_fidelity: ttnn.MathFidelity | None = None,
    ) -> TransformerEncoderConfig:
        """``num_layers`` (default: the config's ``num_hidden_layers``) keeps decoder layers ``[0, num_layers)``;
        ``final_norm=False`` drops the final RMSNorm (a hidden-state tap consumer reads the raw layer outputs)."""
        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types and not all(t == "full_attention" for t in layer_types):
            msg = f"expected all layer_types to be 'full_attention', got {set(layer_types)}"
            raise ValueError(msg)
        if getattr(hf_config, "use_sliding_window", False):
            msg = "sliding-window attention is not supported"
            raise ValueError(msg)
        depth = int(hf_config.num_hidden_layers)
        num_layers = depth if num_layers is None else int(num_layers)
        if not 1 <= num_layers <= depth:
            msg = f"num_layers {num_layers} outside [1, {depth}] (config num_hidden_layers)"
            raise ValueError(msg)
        return TransformerEncoderConfig(
            vocab_size=hf_config.vocab_size,
            head_size=hf_config.head_dim,
            embed_size=hf_config.hidden_size,
            ff_size=hf_config.intermediate_size,
            num_layers=num_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            norm_eps=hf_config.rms_norm_eps,
            attn_qkv_bias=bool(getattr(hf_config, "attention_bias", False)),
            attn_out_bias=bool(getattr(hf_config, "attention_bias", False)),
            attn_qk_norm=True,
            rope_config=RopeConfig(theta=_rope_theta(hf_config)),
            final_norm=final_norm,
            final_linear=final_linear,
            attn_qkv_dtype=linear_dtype,
            linear_dtype=linear_dtype,
            linear_math_fidelity=linear_math_fidelity,
        )

    @staticmethod
    def convert_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return STATE_CONVERSION.convert(state_dict)


STATE_CONVERSION = StateConversion(
    rename=[
        (r"^model\.embed_tokens", r"token_embedding"),
        (r"^model\.layers\.([0-9]+)\.self_attn\.([qkvo])_proj", r"layers.\1.attn.\2_proj"),
        (r"^model\.layers\.([0-9]+)\.self_attn\.q_norm", r"layers.\1.attn.q_norm"),
        (r"^model\.layers\.([0-9]+)\.self_attn\.k_norm", r"layers.\1.attn.k_norm"),
        (r"^model\.layers\.([0-9]+)\.mlp\.gate_proj", r"layers.\1.ff.gate"),
        (r"^model\.layers\.([0-9]+)\.mlp\.up_proj", r"layers.\1.ff.linear_in"),
        (r"^model\.layers\.([0-9]+)\.mlp\.down_proj", r"layers.\1.ff.linear_out"),
        (r"^model\.layers\.([0-9]+)\.post_attention_layernorm", r"layers.\1.ff_norm"),
        (r"^model\.layers\.([0-9]+)\.input_layernorm", r"layers.\1.attn_norm"),
        (r"^model\.norm\.weight", r"final_norm.weight"),
        (r"^lm_head\.weight", r"final_linear.weight"),
    ],
)


class Qwen3StateKeys(NamedTuple):
    """:func:`route_qwen3_state_keys`' groups (sorted): the keys the encoder loads and the ones dropped on purpose."""

    kept: tuple[str, ...]
    dropped_layers: tuple[str, ...]
    dropped_heads: tuple[str, ...]


def route_qwen3_state_keys(
    keys: Iterable[str],
    *,
    num_layers: int,
    final_norm: bool = True,
    final_linear: bool = False,
    num_hidden_layers: int | None = None,
) -> Qwen3StateKeys:
    """Route CONVERTED state keys (:meth:`Qwen3Encoder.convert_state` names) into kept / dropped layers (index >=
    ``num_layers``) / dropped heads (``final_norm.weight`` without ``final_norm``, ``final_linear.weight`` without
    ``final_linear``); raise ``ValueError`` naming every key that belongs to none of the groups. With
    ``num_hidden_layers`` (the config's depth) the layer indices are cross-checked too: kept must be exactly
    ``range(num_layers)`` and dropped exactly ``range(num_layers, num_hidden_layers)``, so an extra, renumbered or
    missing layer is reported instead of landing silently in the dropped group."""
    kept, dropped_layers, dropped_heads, unknown = [], [], [], []
    for key in keys:
        match = _LAYER_RE.match(key)
        if match is not None:
            (kept if int(match.group(1)) < num_layers else dropped_layers).append(key)
        elif key == _EMBEDDING_KEY:
            kept.append(key)
        elif key == _FINAL_NORM_KEY:
            (kept if final_norm else dropped_heads).append(key)
        elif key == _FINAL_LINEAR_KEY:
            (kept if final_linear else dropped_heads).append(key)
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"Qwen3 text encoder: state keys outside the routed groups (layers < {num_layers} / {_EMBEDDING_KEY}"
            f"{' / ' + _FINAL_NORM_KEY if final_norm else ''}{' / ' + _FINAL_LINEAR_KEY if final_linear else ''} kept, "
            f"layers >= {num_layers} and the unused heads dropped): {sorted(unknown)}"
        )
    if num_hidden_layers is not None:
        kept_indices = {int(m.group(1)) for m in map(_LAYER_RE.match, kept) if m is not None}
        dropped_indices = {int(_LAYER_RE.match(key).group(1)) for key in dropped_layers}
        if kept_indices != set(range(num_layers)):
            raise ValueError(f"Qwen3 text encoder: kept layer indices {sorted(kept_indices)} != range({num_layers})")
        if dropped_indices != set(range(num_layers, num_hidden_layers)):
            raise ValueError(
                f"Qwen3 text encoder: dropped layer indices {sorted(dropped_indices)} != "
                f"range({num_layers}, {num_hidden_layers}) (config num_hidden_layers {num_hidden_layers})"
            )
    return Qwen3StateKeys(tuple(sorted(kept)), tuple(sorted(dropped_layers)), tuple(sorted(dropped_heads)))


def qwen3_cache_subfolder(subfolder: str, *, num_layers: int, num_hidden_layers: int, final_norm: bool) -> str:
    """The weight-cache subfolder of a depth: the plain ``subfolder`` for the full checkpoint (all layers, final
    norm), ``<subfolder>_L<num_layers>`` for a reduced one (``_norm`` appended when the final norm is kept at a
    reduced depth), so no two configurations share a cache entry."""
    if num_layers == num_hidden_layers and final_norm:
        return subfolder
    return f"{subfolder}_L{num_layers}{'_norm' if final_norm else ''}"


class Qwen3Checkpoint:
    """A Qwen3 text-encoder checkpoint (``<repo>/text_encoder``): reads ``config.json`` up front, loads the torch
    weights lazily on a weight-cache miss in :meth:`build`."""

    def __init__(
        self,
        name: str,
        *,
        subfolder: str = "text_encoder",
        linear_dtype: ttnn.DataType = EXACT_DTYPE,
        linear_math_fidelity: ttnn.MathFidelity | None = None,
        num_layers: int | None = None,
        final_norm: bool = True,
    ) -> None:
        """``linear_math_fidelity`` raises the matmul fidelity of every linear (HiFi4 for the FLUX.2 Klein exact
        profile); ``None`` keeps the generic encoder's HiFi2. The weight cache is the same either way (bf16 weights).
        ``num_layers`` keeps decoder layers ``[0, num_layers)`` only (default: all) and ``final_norm=False`` drops the
        final RMSNorm: a hidden-state tap consumer whose deepest tap is ``hidden_states[k]`` needs ``num_layers=k``
        and no final norm (the generic forward then returns the raw output of layer ``k - 1`` for index ``k``)."""
        hf_config = transformers.AutoConfig.from_pretrained(name, subfolder=subfolder)
        self._name = name
        self._subfolder = subfolder
        self._linear_dtype = linear_dtype
        self.num_hidden_layers = int(hf_config.num_hidden_layers)
        self.config = Qwen3Encoder.config_from_hf(
            hf_config,
            num_layers=num_layers,
            final_norm=final_norm,
            linear_dtype=linear_dtype,
            linear_math_fidelity=linear_math_fidelity,
        )

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def cache_subfolder(self) -> str:
        """The cache subfolder this depth loads from (:func:`qwen3_cache_subfolder`)."""
        return qwen3_cache_subfolder(
            self._subfolder,
            num_layers=self.config.num_layers,
            num_hidden_layers=self.num_hidden_layers,
            final_norm=self.config.final_norm,
        )

    def build(
        self,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
        cache_name: str | None = None,
    ) -> Qwen3Encoder:
        """Build the encoder on ``device`` and load it through the weight cache. ``cache_name`` names the cache
        entry (``TT_DIT_CACHE_DIR/<cache_name>/<cache_subfolder>/...``); it defaults to the checkpoint name, which a
        caller holding a local snapshot PATH must replace with a plain name. With ``parallel_config.fsdp`` set
        the weights are sharded over that mesh axis and the cache key carries ``_FSDP``."""
        model = Qwen3Encoder(self.config, device=device, parallel_config=parallel_config, ccl_manager=ccl_manager)
        fsdp = parallel_config.fsdp if parallel_config is not None else None
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name if cache_name is None else cache_name,
            subfolder=self.cache_subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
            dtype=EXACT_WEIGHT_CACHE_DTYPE if self._linear_dtype == ttnn.bfloat16 else "bf8",
            is_fsdp=fsdp is not None and fsdp.factor > 1,
        )
        return model

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name, subfolder=self._subfolder, torch_dtype=torch.bfloat16
        )
        state = Qwen3Encoder.convert_state(torch_model.state_dict())
        keys = route_qwen3_state_keys(
            state,
            num_layers=self.config.num_layers,
            final_norm=self.config.final_norm,
            final_linear=self.config.final_linear,
            num_hidden_layers=self.num_hidden_layers,
        )
        return {key: state[key] for key in keys.kept}
