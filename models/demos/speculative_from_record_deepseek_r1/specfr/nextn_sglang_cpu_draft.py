# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NextN CPU draft adapter: manual MLA + MoE loading weights directly from safetensors.

Full SGLang MTP dataflow per draft step
---------------------------------------

1. ``embed_tokens(current_token)``
2. **Fusion**: ``eh_proj(cat(enorm(embed), hnorm(hidden)))``
3. ``DeepseekV3DecoderLayer`` — MLA attention (Multi-head Latent Attention) + MoE
4. ``shared_head.norm`` — RMS norm mapped from ``model.layers.0.shared_head.norm``
5. ``lm_head`` → draft logits

Weights are loaded from ``nextn_layer_parameters.safetensors`` (and optionally an
auxiliary embed/head file):

* MoE expert weights are kept in **FP8** on disk (~11 GB) and dequantized lazily
  during each forward via :func:`_dequant_on_the_fly`.
* Non-expert weights (fusion, MLA projections, norms) are loaded as **bfloat16**
  (~4 GB).
* Total ~19 GB, well within typical CPU RAM.

By default, drafting batches all active beams into **one MLA + MoE forward per
depth** (``EagleConfig.draft_sglang_cpu_batch_beams``).  Set ``False`` for a
per-beam loop (legacy; slower for ``depth > 1`` with many paths).

The hidden state fed to the **next** step's ``hnorm`` is the
**post-``shared_head.norm``** tensor (the same tensor ``lm_head`` sees), matching
SGLang's use of draft ``hidden_states`` after forward — **not** raw ``fused``,
and **not** logits from ``shared_head`` applied directly on ``fused``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from specfr.config import EagleConfig, PathProposal
from specfr.models_draft import (
    draft_branch_token_ids_from_logits,
    draft_requires_positive_top_k,
    truncate_beams_by_draft_confidence,
)

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None  # type: ignore[misc, assignment]

try:
    from specfr.dequantize import dequantize_tensor
except ImportError:
    dequantize_tensor = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(dtype)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


# ---------------------------------------------------------------------------
# Yarn RoPE helpers
# ---------------------------------------------------------------------------


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_seq_len: int
) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale: float, mscale_coeff: float) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale_coeff * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def _build_yarn_cos_sin(
    dim: int,
    max_seq_len: int,
    original_seq_len: int,
    rope_theta: float,
    rope_factor: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos / sin caches for Yarn-scaled RoPE.

    Returns ``(cos, sin)`` each of shape ``[max_seq_len, dim // 2]``.
    """
    freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if max_seq_len > original_seq_len:
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, dim, rope_theta, original_seq_len
        )
        smooth = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
        freqs = freqs / rope_factor * (1.0 - smooth) + freqs * smooth
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with DeepSeek interleaved permutation.

    Pairs consecutive elements ``(x[..., 0], x[..., 1])``, ``(x[..., 2], x[..., 3])``
    etc., matching the complex-multiplication form used in the reference model.

    Args:
        q: ``[B, S, H, rope_dim]``
        k: ``[B, S, 1_or_H, rope_dim]``
        cos: ``[max_seq_len, rope_dim // 2]``
        sin: ``[max_seq_len, rope_dim // 2]``
        position_ids: ``[B, S]``
    """
    cos_pos = cos[position_ids].unsqueeze(2)  # [B, S, 1, D/2]
    sin_pos = sin[position_ids].unsqueeze(2)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        x0 = xf[..., 0::2]
        x1 = xf[..., 1::2]
        out = torch.stack(
            [x0 * cos_pos - x1 * sin_pos, x0 * sin_pos + x1 * cos_pos], dim=-1
        )
        return out.flatten(-2).to(x.dtype)

    return _rotate(q), _rotate(k)


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _load_tensor(
    sf: Any,
    key: str,
    target_dtype: torch.dtype,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    """Load from a ``safe_open`` handle; dequantize FP8 when ``*_scale_inv`` exists."""
    t = sf.get_tensor(key)
    scale_key = key.replace(".weight", ".weight_scale_inv")
    if scale_key in sf.keys():
        if dequantize_tensor is None:
            raise ImportError(
                "specfr.dequantize.dequantize_tensor is required "
                "for FP8 weight loading"
            )
        inv = sf.get_tensor(scale_key)
        return dequantize_tensor(t, inv, block_shape).to(target_dtype)
    return t.to(target_dtype)


def _load_fp8_pair(
    sf: Any, key: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load FP8 weight and its ``scale_inv`` as-is (lazy dequant during forward)."""
    w = sf.get_tensor(key)
    scale_key = key.replace(".weight", ".weight_scale_inv")
    inv = sf.get_tensor(scale_key) if scale_key in sf.keys() else None
    return w, inv


def _dequant_on_the_fly(
    weight: torch.Tensor,
    inv_scale: torch.Tensor | None,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    """Dequantize an FP8 weight to bfloat16 for a single matmul."""
    if inv_scale is None:
        return weight.to(torch.bfloat16)
    if dequantize_tensor is None:
        raise ImportError(
            "specfr.dequantize.dequantize_tensor is required "
            "for FP8 dequantization"
        )
    return dequantize_tensor(weight, inv_scale, block_shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# MoE gate routing
# ---------------------------------------------------------------------------


def _moe_gate_forward(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor | None,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid gate routing with optional bias (noaux_tc for DeepSeek-R1).

    Returns ``(weights [tokens, top_k], indices [tokens, top_k])``.
    """
    scores = F.linear(hidden.float(), gate_weight.float())
    scores = torch.sigmoid(scores)
    original_scores = scores.clone()

    if gate_bias is not None:
        scores = scores + gate_bias

    if n_group > 1:
        scores_g = scores.view(hidden.size(0), n_group, -1)
        if gate_bias is None:
            group_scores = scores_g.amax(dim=-1)
        else:
            group_scores = scores_g.topk(2, dim=-1)[0].sum(dim=-1)
        top_groups = group_scores.topk(topk_group, dim=-1)[1]
        mask = torch.zeros_like(scores_g[..., 0]).scatter_(1, top_groups, True)
        scores = (scores_g * mask.unsqueeze(-1)).flatten(1)

    indices = torch.topk(scores, top_k, dim=-1)[1]
    weights = original_scores.gather(1, indices)

    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * routed_scaling_factor

    return weights.type_as(hidden), indices


# ---------------------------------------------------------------------------
# Weight key constants
# ---------------------------------------------------------------------------

_PREFIX = "model.layers.0."
_FUSION_KEYS = {
    "eh_proj": f"{_PREFIX}eh_proj.weight",
    "enorm": f"{_PREFIX}enorm.weight",
    "hnorm": f"{_PREFIX}hnorm.weight",
    "shared_head_norm": f"{_PREFIX}shared_head.norm.weight",
}
_SHARED_HEAD_HEAD_KEY = f"{_PREFIX}shared_head.head.weight"
_EMBED_KEYS = ("model.embed_tokens.weight", "embed_tokens.weight")


# ---------------------------------------------------------------------------
# NextNSglangCPUDraftAdapter
# ---------------------------------------------------------------------------


class NextNSglangCPUDraftAdapter:
    """Full SGLang MTP draft on CPU: fusion → MLA + MoE decoder → shared_head.norm → lm_head.

    Weights are loaded directly from safetensors (no HF ``AutoModel`` loading).
    Expert weights stay in FP8 and are dequantized per-forward; everything else
    is bfloat16 or float32.
    """

    def __init__(
        self,
        device: str = "cpu",
        torch_dtype: str = "float32",
        trust_remote_code: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.bound = False

        # ---- config scalars (set by _apply_config) ----
        self._hidden_size: int = 0
        self._num_heads: int = 0
        self._q_lora_rank: int = 0
        self._kv_lora_rank: int = 0
        self._qk_nope_head_dim: int = 0
        self._qk_rope_head_dim: int = 0
        self._v_head_dim: int = 0
        self._qk_head_dim: int = 0
        self._rms_eps: float = 1e-6
        self._vocab_size: int = 0
        self._softmax_scale: float = 0.0
        self._block_shape: tuple[int, int] = (128, 128)

        # MoE config
        self._n_routed_experts: int = 0
        self._n_shared_experts: int = 0
        self._n_activated_experts: int = 0
        self._n_group: int = 0
        self._topk_group: int = 0
        self._routed_scaling_factor: float = 1.0
        self._scoring_func: str = "sigmoid"
        self._moe_inter_dim: int = 0

        # ---- RoPE caches ----
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None

        # ---- fusion weights ----
        self._eh_proj_w: torch.Tensor | None = None
        self._enorm_w: torch.Tensor | None = None
        self._hnorm_w: torch.Tensor | None = None
        self._shared_head_norm_w: torch.Tensor | None = None

        # ---- decoder layer norms ----
        self._input_layernorm_w: torch.Tensor | None = None
        self._post_attn_layernorm_w: torch.Tensor | None = None

        # ---- MLA attention (bfloat16) ----
        self._q_a_proj_w: torch.Tensor | None = None
        self._q_a_layernorm_w: torch.Tensor | None = None
        self._q_b_proj_w: torch.Tensor | None = None
        self._kv_a_proj_w: torch.Tensor | None = None
        self._kv_a_layernorm_w: torch.Tensor | None = None
        self._kv_b_proj_w: torch.Tensor | None = None
        self._o_proj_w: torch.Tensor | None = None

        # ---- MoE gate ----
        self._gate_w: torch.Tensor | None = None
        self._gate_bias: torch.Tensor | None = None

        # ---- routed experts: list[(weight, scale_inv | None)] — FP8 lazy dequant ----
        self._expert_gate: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self._expert_up: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self._expert_down: list[tuple[torch.Tensor, torch.Tensor | None]] = []

        # ---- shared expert (FP8 lazy dequant) ----
        self._shared_gate: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self._shared_up: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self._shared_down: tuple[torch.Tensor, torch.Tensor | None] | None = None

        # ---- embed + head ----
        self._embed_weight: torch.Tensor | None = None
        self._lm_head_w: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        """Extract architecture hyper-parameters from ``config.json``."""
        self._hidden_size = int(cfg["hidden_size"])
        self._num_heads = int(cfg["num_attention_heads"])
        self._q_lora_rank = int(cfg.get("q_lora_rank", 1536))
        self._kv_lora_rank = int(cfg.get("kv_lora_rank", 512))
        self._qk_nope_head_dim = int(cfg.get("qk_nope_head_dim", 128))
        self._qk_rope_head_dim = int(cfg.get("qk_rope_head_dim", 64))
        self._v_head_dim = int(cfg.get("v_head_dim", 128))
        self._qk_head_dim = self._qk_nope_head_dim + self._qk_rope_head_dim
        self._rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
        self._vocab_size = int(cfg.get("vocab_size", 129280))

        # MoE
        self._n_routed_experts = int(cfg.get("n_routed_experts", 256))
        self._n_shared_experts = int(cfg.get("n_shared_experts", 1))
        self._n_activated_experts = int(cfg.get("num_experts_per_tok", 8))
        self._n_group = int(cfg.get("n_group", 8))
        self._topk_group = int(cfg.get("topk_group", 4))
        self._routed_scaling_factor = float(cfg.get("routed_scaling_factor", 2.5))
        self._scoring_func = str(cfg.get("scoring_func", "sigmoid"))
        self._moe_inter_dim = int(cfg.get("moe_intermediate_size", 2048))

        # FP8 block shape from quantization_config
        qc = cfg.get("quantization_config", {})
        if isinstance(qc, dict):
            bs = qc.get("weight_block_size")
            if isinstance(bs, (list, tuple)) and len(bs) == 2:
                self._block_shape = (int(bs[0]), int(bs[1]))

        # Softmax scale with Yarn mscale correction
        rope_scaling = cfg.get("rope_scaling") or {}
        rope_factor = float(rope_scaling.get("factor", 40))
        mscale_coeff = float(rope_scaling.get("mscale", 1.0))
        original_seq_len = int(
            rope_scaling.get("original_max_position_embeddings", 4096)
        )
        max_seq_len = int(cfg.get("max_position_embeddings", 163840))

        self._softmax_scale = self._qk_head_dim ** -0.5
        if max_seq_len > original_seq_len:
            m = _yarn_get_mscale(rope_factor, mscale_coeff)
            self._softmax_scale *= m * m

        logger.debug(
            "Config: hidden=%d heads=%d q_lora=%d kv_lora=%d "
            "nope=%d rope=%d v=%d experts=%d moe_inter=%d softmax_scale=%.6f",
            self._hidden_size,
            self._num_heads,
            self._q_lora_rank,
            self._kv_lora_rank,
            self._qk_nope_head_dim,
            self._qk_rope_head_dim,
            self._v_head_dim,
            self._n_routed_experts,
            self._moe_inter_dim,
            self._softmax_scale,
        )

    # ------------------------------------------------------------------ #
    # RoPE
    # ------------------------------------------------------------------ #

    def _build_rope(self, cfg: dict[str, Any]) -> None:
        """Build Yarn RoPE cos / sin caches and store on ``self.device``."""
        rope_scaling = cfg.get("rope_scaling") or {}
        cos, sin = _build_yarn_cos_sin(
            dim=self._qk_rope_head_dim,
            max_seq_len=int(cfg.get("max_position_embeddings", 163840)),
            original_seq_len=int(
                rope_scaling.get("original_max_position_embeddings", 4096)
            ),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_factor=float(rope_scaling.get("factor", 40)),
            beta_fast=float(rope_scaling.get("beta_fast", 32)),
            beta_slow=float(rope_scaling.get("beta_slow", 1)),
        )
        self._cos_cache = cos.to(self.device)
        self._sin_cache = sin.to(self.device)
        logger.debug(
            "Built Yarn RoPE cache: shape=%s on %s", list(cos.shape), self.device
        )

    # ------------------------------------------------------------------ #
    # Weight loading — decoder layer
    # ------------------------------------------------------------------ #

    def _load_all_weights(self, nextn_path: Path, cfg: dict[str, Any]) -> None:
        """Load fusion, MLA, norms, gate, routed experts, and shared expert.

        Non-expert weights are dequantized to ``bfloat16``.  Expert weights stay
        FP8 and are dequantized lazily during each forward call.
        """
        if safe_open is None:
            raise ImportError(
                "NextNSglangCPUDraftAdapter requires the `safetensors` package."
            )
        dev = str(self.device)
        bs = self._block_shape
        attn_prefix = f"{_PREFIX}self_attn."
        mlp_prefix = f"{_PREFIX}mlp."

        with safe_open(str(nextn_path), framework="pt", device=dev) as sf:
            all_keys = set(sf.keys())

            # -- fusion --
            missing_fusion = [
                k for k in _FUSION_KEYS.values() if k not in all_keys
            ]
            if missing_fusion:
                raise ValueError(
                    f"NextN safetensors missing fusion keys: {missing_fusion}"
                )
            self._eh_proj_w = _load_tensor(
                sf, _FUSION_KEYS["eh_proj"], torch.bfloat16, bs
            )
            self._enorm_w = sf.get_tensor(_FUSION_KEYS["enorm"]).float()
            self._hnorm_w = sf.get_tensor(_FUSION_KEYS["hnorm"]).float()
            self._shared_head_norm_w = sf.get_tensor(
                _FUSION_KEYS["shared_head_norm"]
            ).float()

            exp_in = 2 * self._hidden_size
            if self._eh_proj_w.shape[1] != exp_in:
                raise ValueError(
                    f"eh_proj in_features {self._eh_proj_w.shape[1]} != "
                    f"2 * hidden_size {exp_in}"
                )

            # -- decoder layer norms --
            self._input_layernorm_w = sf.get_tensor(
                f"{_PREFIX}input_layernorm.weight"
            ).float()
            self._post_attn_layernorm_w = sf.get_tensor(
                f"{_PREFIX}post_attention_layernorm.weight"
            ).float()

            # -- MLA attention (dequantized to bfloat16) --
            self._q_a_proj_w = _load_tensor(
                sf, f"{attn_prefix}q_a_proj.weight", torch.bfloat16, bs
            )
            self._q_a_layernorm_w = sf.get_tensor(
                f"{attn_prefix}q_a_layernorm.weight"
            ).float()
            self._q_b_proj_w = _load_tensor(
                sf, f"{attn_prefix}q_b_proj.weight", torch.bfloat16, bs
            )
            self._kv_a_proj_w = _load_tensor(
                sf, f"{attn_prefix}kv_a_proj_with_mqa.weight", torch.bfloat16, bs
            )
            self._kv_a_layernorm_w = sf.get_tensor(
                f"{attn_prefix}kv_a_layernorm.weight"
            ).float()
            self._kv_b_proj_w = _load_tensor(
                sf, f"{attn_prefix}kv_b_proj.weight", torch.bfloat16, bs
            )
            self._o_proj_w = _load_tensor(
                sf, f"{attn_prefix}o_proj.weight", torch.bfloat16, bs
            )

            logger.debug(
                "MLA shapes: q_a=%s q_b=%s kv_a=%s kv_b=%s o=%s",
                list(self._q_a_proj_w.shape),
                list(self._q_b_proj_w.shape),
                list(self._kv_a_proj_w.shape),
                list(self._kv_b_proj_w.shape),
                list(self._o_proj_w.shape),
            )

            # -- MoE gate --
            self._gate_w = sf.get_tensor(f"{mlp_prefix}gate.weight").float()
            bias_key = f"{mlp_prefix}gate.e_score_correction_bias"
            self._gate_bias = (
                sf.get_tensor(bias_key).float() if bias_key in all_keys else None
            )

            # -- routed experts (FP8, lazy dequant) --
            self._expert_gate = []
            self._expert_up = []
            self._expert_down = []
            for i in range(self._n_routed_experts):
                ep = f"{mlp_prefix}experts.{i}."
                self._expert_gate.append(
                    _load_fp8_pair(sf, f"{ep}gate_proj.weight")
                )
                self._expert_up.append(
                    _load_fp8_pair(sf, f"{ep}up_proj.weight")
                )
                self._expert_down.append(
                    _load_fp8_pair(sf, f"{ep}down_proj.weight")
                )
            if self._expert_gate:
                logger.debug(
                    "Expert[0] gate shape=%s dtype=%s",
                    list(self._expert_gate[0][0].shape),
                    self._expert_gate[0][0].dtype,
                )

            # -- shared expert (FP8) --
            sp = f"{mlp_prefix}shared_experts."
            self._shared_gate = _load_fp8_pair(sf, f"{sp}gate_proj.weight")
            self._shared_up = _load_fp8_pair(sf, f"{sp}up_proj.weight")
            self._shared_down = _load_fp8_pair(sf, f"{sp}down_proj.weight")

        logger.info(
            "Loaded decoder-layer weights from %s: "
            "MLA (bfloat16), %d routed experts (FP8 lazy dequant), "
            "shared expert (FP8 lazy dequant), block_shape=%s.",
            nextn_path.name,
            self._n_routed_experts,
            self._block_shape,
        )

    # ------------------------------------------------------------------ #
    # Weight loading — embed + head
    # ------------------------------------------------------------------ #

    def _load_embed_head(
        self, nextn_path: Path, aux_path: Path | None
    ) -> None:
        """Load ``embed_tokens`` and ``lm_head`` from the nextn file or an aux file."""
        if safe_open is None:
            raise ImportError("safetensors is required")
        dev = str(self.device)

        # Try the main nextn safetensors first
        with safe_open(str(nextn_path), framework="pt", device=dev) as sf:
            keys = set(sf.keys())
            for ek in _EMBED_KEYS:
                if ek in keys and self._embed_weight is None:
                    self._embed_weight = sf.get_tensor(ek).float()
                    logger.debug("Loaded embed_tokens from nextn file (key=%s)", ek)
                    break
            if _SHARED_HEAD_HEAD_KEY in keys and self._lm_head_w is None:
                self._lm_head_w = sf.get_tensor(_SHARED_HEAD_HEAD_KEY).float()
                logger.debug("Loaded lm_head from nextn file")

        # Fall back to auxiliary embed/head file
        if aux_path is not None and (
            self._embed_weight is None or self._lm_head_w is None
        ):
            aux = Path(aux_path).expanduser().resolve()
            if not aux.is_file():
                raise FileNotFoundError(
                    f"embed_head_aux_safetensors not found: {aux}"
                )
            from specfr.local_hf_snapshot import (
                load_nextn_mtp_auxiliary_safetensors,
            )

            embed, head = load_nextn_mtp_auxiliary_safetensors(aux)
            if self._embed_weight is None:
                self._embed_weight = embed.to(self.device).float()
                logger.debug("Loaded embed_tokens from aux file: %s", aux)
            if self._lm_head_w is None:
                self._lm_head_w = head.to(self.device).float()
                logger.debug("Loaded lm_head from aux file: %s", aux)

        if self._embed_weight is None or self._lm_head_w is None:
            from specfr.default_paths import DEFAULT_EMBED_HEAD_AUX_PATH
            if aux_path is None and DEFAULT_EMBED_HEAD_AUX_PATH.is_file():
                logger.info("Falling back to default embed/head aux: %s", DEFAULT_EMBED_HEAD_AUX_PATH)
                from specfr.local_hf_snapshot import (
                    load_nextn_mtp_auxiliary_safetensors,
                )
                embed, head = load_nextn_mtp_auxiliary_safetensors(DEFAULT_EMBED_HEAD_AUX_PATH)
                if self._embed_weight is None:
                    self._embed_weight = embed.to(self.device).float()
                if self._lm_head_w is None:
                    self._lm_head_w = head.to(self.device).float()

        if self._embed_weight is None:
            raise ValueError(
                "embed_tokens missing from nextn safetensors and no aux file "
                "provided. Use --embed-head-aux-safetensors or run "
                "scripts/materialize_nextn_embed_head_aux_from_r1_shards.py under speculative_from_record_deepseek_r1."
            )
        if self._lm_head_w is None:
            raise ValueError(
                "lm_head / shared_head.head missing from nextn safetensors and "
                "no aux file provided. Use --embed-head-aux-safetensors or run "
                "scripts/materialize_nextn_embed_head_aux_from_r1_shards.py under speculative_from_record_deepseek_r1."
            )

        logger.info(
            "Embed + head loaded: embed=%s lm_head=%s",
            list(self._embed_weight.shape),
            list(self._lm_head_w.shape),
        )

    # ------------------------------------------------------------------ #
    # Public binding
    # ------------------------------------------------------------------ #

    def bind_from_nextn_paths(
        self,
        *,
        nextn_safetensors: str | Path,
        embed_head_aux_safetensors: str | Path | None = None,
        nextn_config_dir: str | Path | None = None,
    ) -> None:
        """Load config, build RoPE, and load all weights from disk.

        Args:
            nextn_safetensors: path to ``nextn_layer_parameters.safetensors``
            embed_head_aux_safetensors: optional auxiliary file with embed + head
            nextn_config_dir: directory containing ``config.json``; defaults to
                the parent of *nextn_safetensors*.
        """
        if self.bound:
            return
        nextn_path = Path(nextn_safetensors).expanduser().resolve()
        if not nextn_path.is_file():
            raise FileNotFoundError(
                f"NextN safetensors not found: {nextn_path}"
            )
        cfg_dir = (
            Path(nextn_config_dir).expanduser().resolve()
            if nextn_config_dir
            else nextn_path.parent
        )

        from specfr.local_hf_snapshot import (
            read_snapshot_config,
        )

        cfg = read_snapshot_config(cfg_dir)

        self._apply_config(cfg)
        self._build_rope(cfg)
        self._load_all_weights(nextn_path, cfg)

        aux = (
            Path(embed_head_aux_safetensors).expanduser().resolve()
            if embed_head_aux_safetensors
            else None
        )
        self._load_embed_head(nextn_path, aux)

        self.bound = True
        logger.info(
            "Bound NextNSglangCPUDraftAdapter: nextn=%s config=%s "
            "(SGLang: fusion → MLA+MoE → shared_head.norm → lm_head on CPU)",
            nextn_path,
            cfg_dir,
        )

    # ================================================================== #
    #  Forward helpers                                                    #
    # ================================================================== #

    # ------------------------------------------------------------------ #
    # MLA attention
    # ------------------------------------------------------------------ #

    def _mla_forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Multi-head Latent Attention (naive path — no KV-absorption).

        Q path::

            hidden → q_a_proj → q_a_layernorm → q_b_proj
            → split q_nope [B,S,H,nope_dim] + q_pe [B,S,H,rope_dim]

        KV path::

            hidden → kv_a_proj → split compressed_kv + k_pe
            compressed_kv → kv_a_layernorm → kv_b_proj
            → split k_nope [B,S,H,nope_dim] + value_states [B,S,H,v_dim]

        RoPE on ``q_pe`` / ``k_pe``, assemble full Q and K, concat with KV
        cache, scaled dot-product attention, ``o_proj``.

        Returns ``(attn_output [B,S,H_dim], new_kv_cache)``.
        """
        bsz, seq_len, _ = hidden.shape

        cd = self._q_a_proj_w.dtype

        # ----- Q path -----
        q_a = F.linear(hidden.to(cd), self._q_a_proj_w)
        q_a = _rms_norm(q_a, self._q_a_layernorm_w, self._rms_eps)
        q = F.linear(q_a.to(cd), self._q_b_proj_w)
        q = q.view(bsz, seq_len, self._num_heads, self._qk_head_dim)
        q_nope, q_pe = q.split(
            [self._qk_nope_head_dim, self._qk_rope_head_dim], dim=-1
        )

        # ----- KV path -----
        kv_a = F.linear(hidden.to(cd), self._kv_a_proj_w)
        compressed_kv, k_pe = kv_a.split(
            [self._kv_lora_rank, self._qk_rope_head_dim], dim=-1
        )
        kv_normed = _rms_norm(compressed_kv, self._kv_a_layernorm_w, self._rms_eps)
        kv_b = F.linear(kv_normed.to(cd), self._kv_b_proj_w)
        kv_b = kv_b.view(
            bsz,
            seq_len,
            self._num_heads,
            self._qk_nope_head_dim + self._v_head_dim,
        )
        k_nope, value_states = kv_b.split(
            [self._qk_nope_head_dim, self._v_head_dim], dim=-1
        )

        # ----- RoPE on q_pe and k_pe -----
        k_pe = k_pe.unsqueeze(2)  # [B, S, 1, rope_dim]
        q_pe, k_pe = _apply_rotary_pos_emb(
            q_pe, k_pe, self._cos_cache, self._sin_cache, position_ids
        )

        # ----- assemble full Q and K -----
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self._num_heads, -1)], dim=-1
        )

        # Transpose to [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = value_states.transpose(1, 2)

        # ----- KV cache -----
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        # ----- scaled dot-product attention -----
        attn_weights = (
            torch.matmul(q, k.transpose(-2, -1)) * self._softmax_scale
        )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_output = torch.matmul(attn_weights, v)

        # ----- output projection -----
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, seq_len, self._num_heads * self._v_head_dim
        )
        attn_output = F.linear(attn_output.to(cd), self._o_proj_w)

        return attn_output, new_kv

    # ------------------------------------------------------------------ #
    # Expert dequant + MoE forward
    # ------------------------------------------------------------------ #

    def _dequant_expert(
        self, w: torch.Tensor, s: torch.Tensor | None
    ) -> torch.Tensor:
        """Dequantize one expert weight from FP8 to bfloat16."""
        return _dequant_on_the_fly(w, s, self._block_shape)

    def _moe_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """MoE forward: gate routing → routed expert MLPs → weighted sum + shared expert.

        Each expert's gate/up/down projections are dequantized from FP8 on the
        fly so only the active-expert weights occupy bfloat16 memory at a time.
        """
        shape = hidden.shape
        x = hidden.view(-1, self._hidden_size)

        weights, indices = _moe_gate_forward(
            x,
            self._gate_w,
            self._gate_bias,
            self._n_activated_experts,
            self._n_group,
            self._topk_group,
            self._routed_scaling_factor,
            self._scoring_func == "sigmoid",
        )

        x_f = x.float()
        y = torch.zeros_like(x_f)
        counts = torch.bincount(
            indices.flatten(), minlength=self._n_routed_experts
        ).tolist()

        for i in range(self._n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            x_i = x_f[idx]
            gw = self._dequant_expert(*self._expert_gate[i]).float()
            uw = self._dequant_expert(*self._expert_up[i]).float()
            dw = self._dequant_expert(*self._expert_down[i]).float()
            expert_out = F.linear(
                _silu(F.linear(x_i, gw)) * F.linear(x_i, uw), dw
            )
            y[idx] += expert_out * weights[idx, top, None].float()

        # Shared expert (same SwiGLU pattern)
        assert self._shared_gate is not None
        sg = self._dequant_expert(*self._shared_gate).float()
        su = self._dequant_expert(*self._shared_up).float()
        sd = self._dequant_expert(*self._shared_down).float()
        z = F.linear(_silu(F.linear(x_f, sg)) * F.linear(x_f, su), sd)

        return (y + z).to(hidden.dtype).view(shape)

    # ------------------------------------------------------------------ #
    # Decoder layer
    # ------------------------------------------------------------------ #

    def _decoder_layer_forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """input_layernorm → MLA (+ residual) → post_attention_layernorm → MoE (+ residual)."""
        residual = hidden
        hidden = _rms_norm(hidden, self._input_layernorm_w, self._rms_eps)
        hidden, new_kv = self._mla_forward(hidden, position_ids, kv_cache)
        hidden = residual + hidden.to(residual.dtype)

        residual = hidden
        hidden = _rms_norm(hidden, self._post_attn_layernorm_w, self._rms_eps)
        hidden = self._moe_forward(hidden)
        hidden = residual + hidden.to(residual.dtype)

        return hidden, new_kv

    # ------------------------------------------------------------------ #
    # Single-beam step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _one_step(
        self,
        h_side: torch.Tensor,
        token_id: int,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One draft step for a single beam.

        Returns ``(logits_1d [V], h_recurrence [H], new_kv)``.

        * **logits** = ``lm_head(shared_head.norm(decoder_layer(fused)))``
        * **h_recurrence** = post-``shared_head.norm`` hidden (input to ``lm_head``),
          fed into ``hnorm`` on the next step.
        """
        dev = self.device

        # Embed current token
        emb = self._embed_weight[token_id].unsqueeze(0)  # [1, H]

        # Fusion: enorm(embed) ‖ hnorm(hidden) → eh_proj
        e_normed = _rms_norm(emb.float(), self._enorm_w, self._rms_eps)
        h_normed = _rms_norm(
            h_side.to(dev).float().unsqueeze(0), self._hnorm_w, self._rms_eps
        )
        fused = F.linear(
            torch.cat([e_normed, h_normed], dim=-1).to(torch.bfloat16),
            self._eh_proj_w,
        )
        fused_seq = fused.unsqueeze(1)  # [1, 1, H]

        # Position from KV cache length
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        position_ids = torch.tensor([[past_len]], dtype=torch.long, device=dev)

        # Decoder layer
        dec_hidden, new_kv = self._decoder_layer_forward(
            fused_seq, position_ids, past_kv
        )

        # shared_head.norm → lm_head
        post_norm = _rms_norm(
            dec_hidden, self._shared_head_norm_w, self._rms_eps
        )
        logits = F.linear(post_norm.to(self._lm_head_w.dtype), self._lm_head_w)

        logits_1d = logits.squeeze(0).squeeze(0).float()
        h_recurrence = post_norm.squeeze(0).squeeze(0).float()

        return logits_1d, h_recurrence, new_kv

    # ------------------------------------------------------------------ #
    # Batched step (all beams at one depth)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _batch_one_step(
        self,
        beams: list[
            tuple[
                torch.Tensor,
                list[int],
                list[float],
                int,
                tuple[torch.Tensor, torch.Tensor] | None,
            ]
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Batch all beams at one depth into a single decoder-layer forward.

        Stacks fused hidden states and KV caches across beams, runs one
        ``_decoder_layer_forward`` on ``[B, 1, H]``, then returns:

        * ``logits``  — ``[B, V]``
        * ``h_recurrence`` — ``[B, H]`` (post ``shared_head.norm``)
        * ``new_kv`` — batched KV ``(k [B, heads, S+1, D], v [...])``
        """
        B = len(beams)
        dev = self.device

        fused_list: list[torch.Tensor] = []
        past_k_list: list[torch.Tensor] = []
        past_v_list: list[torch.Tensor] = []
        has_kv = False

        for h_side, _path, _probs, tid, past_kv in beams:
            emb = self._embed_weight[tid].unsqueeze(0)
            e_normed = _rms_norm(emb.float(), self._enorm_w, self._rms_eps)
            h_normed = _rms_norm(
                h_side.to(dev).float().unsqueeze(0),
                self._hnorm_w,
                self._rms_eps,
            )
            fused = F.linear(
                torch.cat([e_normed, h_normed], dim=-1).to(torch.bfloat16),
                self._eh_proj_w,
            )
            fused_list.append(fused)  # [1, H]

            if past_kv is not None:
                has_kv = True
                past_k_list.append(past_kv[0])
                past_v_list.append(past_kv[1])

        # Stack fused: each [1, H] → cat → [B, H] → unsqueeze → [B, 1, H]
        fused_batch = torch.cat(fused_list, dim=0).unsqueeze(1)

        past_len = 0
        batch_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        if has_kv:
            batch_kv = (
                torch.cat(past_k_list, dim=0),
                torch.cat(past_v_list, dim=0),
            )
            past_len = batch_kv[0].shape[2]

        position_ids = torch.full(
            (B, 1), past_len, dtype=torch.long, device=dev
        )

        dec_hidden, new_kv = self._decoder_layer_forward(
            fused_batch, position_ids, batch_kv
        )

        # shared_head.norm → lm_head
        post_norm = _rms_norm(
            dec_hidden, self._shared_head_norm_w, self._rms_eps
        )
        logits = F.linear(post_norm.to(self._lm_head_w.dtype), self._lm_head_w)  # [B, 1, V]

        logits_2d = logits.squeeze(1).float()  # [B, V]
        h_recurrence = post_norm.squeeze(1).float()  # [B, H]

        return logits_2d, h_recurrence, new_kv

    # ================================================================== #
    #  Draft interface                                                    #
    # ================================================================== #

    @torch.no_grad()
    def forward_draft(
        self,
        prefix_token_ids: Sequence[int],
        cfg: EagleConfig,
        *,
        decode_state: Any = None,
        base_adapter: Any = None,
        **kwargs: object,
    ) -> PathProposal:
        """Main draft entry point.

        Args:
            prefix_token_ids: committed token sequence (non-empty).
            cfg: speculative decoding controls.
            decode_state: must carry ``last_hidden_state`` from the base model.
            base_adapter: base model adapter (used only for guard check).
            **kwargs: ``draft_torch_generator`` for stochastic branching.

        Returns:
            :class:`PathProposal` with drafted paths and per-token draft probs.
        """
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        draft_mtp_greedy = getattr(cfg, "draft_mtp_greedy", False)
        if cfg.depth <= 0:
            return PathProposal(paths=[], draft_probs_per_path=None)
        if draft_requires_positive_top_k(cfg):
            return PathProposal(paths=[], draft_probs_per_path=None)
        if decode_state is None or base_adapter is None:
            return PathProposal(paths=[], draft_probs_per_path=None)
        if not self.bound:
            raise RuntimeError(
                "NextNSglangCPUDraftAdapter: call bind_from_nextn_paths first."
            )

        k = 1 if draft_mtp_greedy else cfg.top_k
        gen = kwargs.get("draft_torch_generator")
        gen_t = gen if isinstance(gen, torch.Generator) else None
        last_token_id = int(prefix_token_ids[-1])
        h0 = decode_state.last_hidden_state.squeeze(0).float()

        batch_depth = getattr(cfg, "draft_sglang_cpu_batch_beams", True)

        Beam = tuple[
            torch.Tensor,
            list[int],
            list[float],
            int,
            tuple[torch.Tensor, torch.Tensor] | None,
        ]
        beams: list[Beam] = [(h0.clone(), [], [], last_token_id, None)]

        for _ in range(cfg.depth):
            if not beams:
                break

            # ---------- batched path ----------
            if batch_depth and len(beams) > 0:
                logits_2d, h_rec_2d, new_kv = self._batch_one_step(beams)
                probs_2d = F.softmax(logits_2d.float(), dim=-1)

                next_beams: list[Beam] = []
                for i, (_h, appended, appended_probs, _tid, _pkv) in enumerate(
                    beams
                ):
                    topk_ids = draft_branch_token_ids_from_logits(
                        logits_2d[i], cfg, k, gen_t
                    )
                    if not topk_ids:
                        continue
                    for tok_id in topk_ids:
                        q = float(probs_2d[i, tok_id].item())
                        beam_kv = (
                            new_kv[0][i : i + 1].clone(),
                            new_kv[1][i : i + 1].clone(),
                        )
                        next_beams.append(
                            (
                                h_rec_2d[i].clone(),
                                appended + [tok_id],
                                appended_probs + [q],
                                tok_id,
                                beam_kv,
                            )
                        )
            # ---------- per-beam path ----------
            else:
                next_beams = []
                for h_side, appended, appended_probs, tid, past_kv in beams:
                    logits_1d, h_next, new_past = self._one_step(
                        h_side, tid, past_kv
                    )
                    probs = F.softmax(logits_1d.float(), dim=-1)
                    topk_ids = draft_branch_token_ids_from_logits(
                        logits_1d, cfg, k, gen_t
                    )
                    if not topk_ids:
                        continue
                    for tok_id in topk_ids:
                        q = float(probs[tok_id].item())
                        kv_branch = (
                            new_past[0].clone(),
                            new_past[1].clone(),
                        )
                        next_beams.append(
                            (
                                h_next.clone(),
                                appended + [tok_id],
                                appended_probs + [q],
                                tok_id,
                                kv_branch,
                            )
                        )

            next_beams = truncate_beams_by_draft_confidence(
                next_beams, cfg.max_paths, lambda b: b[2]
            )

            beams = next_beams

        paths = [p for _, p, _, _, _ in beams]
        draft_probs_per_path = [pr for _, _, pr, _, _ in beams]
        return PathProposal(paths=paths, draft_probs_per_path=draft_probs_per_path)

    def propose_paths(
        self,
        prefix_token_ids: Sequence[int],
        cfg: EagleConfig,
        **kwargs: object,
    ) -> PathProposal:
        """Alias for :meth:`forward_draft` (matches draft adapter protocol)."""
        return self.forward_draft(prefix_token_ids, cfg, **kwargs)
