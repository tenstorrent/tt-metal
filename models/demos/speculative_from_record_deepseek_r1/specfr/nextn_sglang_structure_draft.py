# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NextN draft aligned with **SGLang** ``DeepseekModelNextN`` (``deepseek_nextn.py``).

Per draft step (same dataflow as SGLang’s inner module + HF causal LM head):

1. ``embed_tokens(current_token)``
2. ``eh_proj(cat(RMSNorm_e(embed), RMSNorm_h(spec_hidden)))`` — fusion mats from the shard
3. ``DeepseekV3DecoderLayer`` (attention + MoE/MLP) on the fused tensor, with KV cache
4. ``model.norm`` — HF maps this from ``shared_head.norm`` in the NextN snapshot when available
5. ``lm_head`` → draft logits (SGLang applies the head in the outer causal LM)

The hidden fed into the **next** step’s ``hnorm`` branch is the **post-``model.norm``** tensor (the same
one ``lm_head`` sees), matching SGLang’s use of draft ``hidden_states`` after a forward — **not**
raw ``fused``, and **not** logits from ``shared_head`` applied directly on ``fused`` (that path is
:class:`NextNMTPHeadDraftAdapter` only).

Implementation fixes that stay on this graph: draft-local RoPE ``position_ids``, 4D causal masks,
and correct ``past_key_values`` seq-length parsing for single-layer ``(k, v)`` caches (see
:func:`_hf_past_kv_seq_len`).

Requires loading the full NextN HF model (heavy on CPU); prefer GPU + bf16 when possible.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from specfr.base_runtime import clone_past_key_values
from specfr.config import EagleConfig, PathProposal
from specfr.local_hf_snapshot import read_snapshot_config
from specfr.models_draft import (
    _rms_norm,
    draft_branch_token_ids_from_logits,
    draft_requires_positive_top_k,
    truncate_beams_by_draft_confidence,
)
from specfr.nextn_hf_nextn_model import (
    NextNFullHuggingfaceDraftAdapter,
    _causal_attention_mask_2d,
    _hf_past_kv_seq_len,
)

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None  # type: ignore[misc, assignment]

try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
except ImportError:  # pragma: no cover - transformers layout varies by version
    _prepare_4d_causal_attention_mask = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


def _attention_mask_4d_for_deepseek_layer(
    fused_seq: torch.Tensor,
    *,
    batch_size: int,
    query_seq_len: int,
    past_key_values_length: int,
) -> torch.Tensor:
    """Match ``DeepseekV3Model.forward``: eager MLA attention expects a 4D additive mask, not 2D."""
    if _prepare_4d_causal_attention_mask is None:
        raise ImportError(
            "transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask is required "
            "for NextNSglangStructureDraftAdapter."
        )
    attn_2d = _causal_attention_mask_2d(
        batch_size, query_seq_len, past_key_values_length, device=fused_seq.device
    )
    mask_4d = _prepare_4d_causal_attention_mask(
        attn_2d,
        (batch_size, query_seq_len),
        fused_seq,
        past_key_values_length,
    )
    if mask_4d is None:
        raise RuntimeError("_prepare_4d_causal_attention_mask returned None (unexpected).")
    return mask_4d


def _clone_layer_past(past: object) -> object:
    """Clone KV returned from a single ``DeepseekV3DecoderLayer`` (usually a ``(k, v)`` tensor pair)."""
    if past is None:
        return None
    if (
        isinstance(past, tuple)
        and len(past) == 2
        and all(isinstance(x, torch.Tensor) for x in past)
    ):
        return (past[0].clone(), past[1].clone())
    return clone_past_key_values(past)


_FUSION_KEYS = (
    "model.layers.0.eh_proj.weight",
    "model.layers.0.enorm.weight",
    "model.layers.0.hnorm.weight",
)


class NextNSglangStructureDraftAdapter:
    """SGLang ``DeepseekModelNextN``-style draft: fusion → decoder → ``model.norm`` → ``lm_head``."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        keep_fp8_quantization_config: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self._torch_dtype = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._local_files_only = local_files_only
        self._keep_fp8 = keep_fp8_quantization_config
        self.bound = False
        self._hf: NextNFullHuggingfaceDraftAdapter | None = None
        self._eh_proj_w: torch.Tensor | None = None
        self._enorm_w: torch.Tensor | None = None
        self._hnorm_w: torch.Tensor | None = None
        self._rms_eps: float = 1e-6
        self._hidden_size: int = 0

    def _load_fusion_from_file(self, nextn_safetensors: Path, dev: torch.device) -> None:
        if safe_open is None:
            raise ImportError("NextNSglangStructureDraftAdapter requires `safetensors`.")
        with safe_open(str(nextn_safetensors), framework="pt", device=str(dev)) as f:
            missing = [k for k in _FUSION_KEYS if k not in f.keys()]
            if missing:
                raise ValueError(f"NextN fusion weights missing keys: {missing}")
            self._eh_proj_w = f.get_tensor(_FUSION_KEYS[0]).to(dev).float()
            self._enorm_w = f.get_tensor(_FUSION_KEYS[1]).to(dev).float()
            self._hnorm_w = f.get_tensor(_FUSION_KEYS[2]).to(dev).float()

    def bind_from_nextn_paths(
        self,
        *,
        nextn_safetensors: str | Path,
        embed_head_aux_safetensors: str | Path | None = None,
        nextn_config_dir: str | Path | None = None,
        decoder_layer0_override_safetensors: str | Path | None = None,
    ) -> None:
        if self.bound:
            return
        nextn_path = Path(nextn_safetensors).expanduser().resolve()
        if not nextn_path.is_file():
            raise FileNotFoundError(f"NextN safetensors not found: {nextn_path}")
        cfg_dir = Path(nextn_config_dir).expanduser().resolve() if nextn_config_dir else nextn_path.parent

        dev = self.device
        self._load_fusion_from_file(nextn_path, dev)

        cfg = read_snapshot_config(cfg_dir)
        self._hidden_size = int(cfg["hidden_size"])
        self._rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
        exp_in = 2 * self._hidden_size
        if self._eh_proj_w is not None and self._eh_proj_w.shape[1] != exp_in:
            raise ValueError(
                f"eh_proj in_features {self._eh_proj_w.shape[1]} != 2*hidden_size {exp_in}"
            )

        self._hf = NextNFullHuggingfaceDraftAdapter(
            model_id_or_path=str(cfg_dir),
            device=str(dev),
            torch_dtype=self._torch_dtype,
            trust_remote_code=self._trust_remote_code,
            local_files_only=self._local_files_only,
            keep_fp8_quantization_config=self._keep_fp8,
            embed_head_aux_safetensors=embed_head_aux_safetensors,
            decoder_layer0_override_safetensors=decoder_layer0_override_safetensors,
        )
        self.bound = True
        logger.info(
            "Bound NextNSglangStructureDraftAdapter: fusion=%s HF_dir=%s (SGLang: fusion→layer0→model.norm→lm_head)",
            nextn_path,
            cfg_dir,
        )

    def _model_dtype(self) -> torch.dtype:
        assert self._hf is not None
        return next(self._hf.model.parameters()).dtype

    @torch.no_grad()
    def _one_step(
        self,
        h_side: torch.Tensor,
        token_id: int,
        past_key_values: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor, object | None]:
        """Returns (logits_1d [V], h_recurrence [H], new_past_kv).

        **Logits:** ``lm_head(model.norm(decoder_layer(fused)))`` — SGLang outer NextN path.

        **Recurrence:** ``h_recurrence`` is the **post-``model.norm``** hidden (input to ``lm_head``),
        fed into ``hnorm`` on the next step — same role as SGLang draft ``hidden_states`` after forward.
        """
        assert self._hf is not None
        assert self._eh_proj_w is not None and self._enorm_w is not None and self._hnorm_w is not None
        causal_lm = self._hf.model
        inner = causal_lm.model
        dev = self.device
        md = self._model_dtype()

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=dev)
        emb = inner.embed_tokens(input_ids).to(md)  # [1,1,H]

        e_flat = emb.squeeze(0).float()  # [1,H]
        h_flat = h_side.to(dev).float().unsqueeze(0)  # [1,H]
        e_normed = _rms_norm(e_flat, self._enorm_w.to(dev), self._rms_eps)
        h_normed = _rms_norm(h_flat, self._hnorm_w.to(dev), self._rms_eps)
        fused = F.linear(
            torch.cat([e_normed.to(md), h_normed.to(md)], dim=-1),
            self._eh_proj_w.to(md),
        )
        fused_seq = fused.unsqueeze(1)  # [1,1,H]

        past_len = _hf_past_kv_seq_len(past_key_values)
        position_ids = torch.tensor([[past_len]], dtype=torch.long, device=dev)
        b, nq = fused_seq.shape[0], fused_seq.shape[1]
        attn_4d = _attention_mask_4d_for_deepseek_layer(
            fused_seq,
            batch_size=b,
            query_seq_len=nq,
            past_key_values_length=past_len,
        )

        layer = inner.layers[0]
        layer_out = layer(
            fused_seq,
            attention_mask=attn_4d,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=True,
        )
        dec_hidden = layer_out[0]
        new_past = layer_out[1] if len(layer_out) > 1 else None

        post_norm = inner.norm(dec_hidden)
        logits = causal_lm.lm_head(post_norm)
        logits_1d = logits.squeeze(0).squeeze(0)
        h_recurrence = post_norm.squeeze(0).squeeze(0).float()
        return logits_1d, h_recurrence, new_past

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
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        draft_mtp_greedy = getattr(cfg, "draft_mtp_greedy", False)
        if cfg.depth <= 0:
            return PathProposal(paths=[], draft_probs_per_path=None)
        if draft_requires_positive_top_k(cfg):
            return PathProposal(paths=[], draft_probs_per_path=None)
        if decode_state is None or base_adapter is None:
            return PathProposal(paths=[], draft_probs_per_path=None)
        if not self.bound or self._hf is None:
            raise RuntimeError("NextNSglangStructureDraftAdapter: call bind_from_nextn_paths first.")

        k = 1 if draft_mtp_greedy else cfg.top_k
        gen = kwargs.get("draft_torch_generator")
        gen_t = gen if isinstance(gen, torch.Generator) else None
        last_token_id = int(prefix_token_ids[-1])
        h0 = decode_state.last_hidden_state.squeeze(0).float()

        # (h_side for hnorm branch, path, probs, last_tid, past_kv)
        Beam = tuple[torch.Tensor, list[int], list[float], int, object | None]
        beams: list[Beam] = [(h0.clone(), [], [], last_token_id, None)]

        for _ in range(cfg.depth):
            if not beams:
                break
            next_beams: list[Beam] = []
            for h_side, appended, appended_probs, tid, past_kv in beams:
                logits_1d, h_next, new_past = self._one_step(h_side, tid, past_kv)
                probs = F.softmax(logits_1d.float(), dim=-1)
                topk_ids = draft_branch_token_ids_from_logits(logits_1d, cfg, k, gen_t)
                if not topk_ids:
                    continue
                for tok_id in topk_ids:
                    q = float(probs[tok_id].item())
                    kv_branch = _clone_layer_past(new_past)
                    next_beams.append(
                        (h_next.clone(), appended + [tok_id], appended_probs + [q], tok_id, kv_branch)
                    )

            next_beams = truncate_beams_by_draft_confidence(next_beams, cfg.max_paths, lambda b: b[2])
            beams = next_beams

        paths = [p for _, p, _, _, _ in beams]
        draft_probs_per_path = [pr for _, _, pr, _, _ in beams]
        return PathProposal(paths=paths, draft_probs_per_path=draft_probs_per_path)

    def propose_paths(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        return self.forward_draft(prefix_token_ids, cfg, **kwargs)
