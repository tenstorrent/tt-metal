# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None  # type: ignore[misc, assignment]

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig, PathProposal
from models.demos.speculative_deepseek_r1_broad.local_hf_snapshot import (
    load_nextn_mtp_auxiliary_safetensors,
    read_snapshot_config,
)

if TYPE_CHECKING:
    from models.demos.speculative_deepseek_r1_broad.models_base import DecodeState, DeepSeekBaseAdapter

logger = logging.getLogger(__name__)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(dtype)


def _clone_hf_kv(past_key_values: tuple) -> tuple:
    return tuple(
        tuple(t.clone() for t in layer_kv)
        for layer_kv in past_key_values
    )


def _draft_path_rank_key(appended_probs: Sequence[float]) -> tuple[float, float, float]:
    """Descending sort key: best peak step prob, then last step, then sum (tie-break).

    So a path with a weak early token but a strong later token can outrank a uniformly mediocre path.
    """
    if not appended_probs:
        return (float("-inf"), float("-inf"), float("-inf"))
    p = list(appended_probs)
    return (max(p), p[-1], sum(p))


def draft_requires_positive_top_k(cfg: EagleConfig) -> bool:
    """Non-greedy draft needs ``top_k > 0`` unless using nucleus mode (where ``top_k <= 0`` = full nucleus)."""
    if getattr(cfg, "draft_mtp_greedy", False):
        return False
    if getattr(cfg, "draft_branching", "top_k") == "temperature_top_p":
        return False
    return cfg.top_k <= 0


def truncate_beams_by_draft_confidence(
    beams: list,
    max_paths: int,
    get_appended_probs: Callable[[object], Sequence[float]],
) -> list:
    """Keep up to ``max_paths`` beams with highest draft-confidence keys (see :func:`_draft_path_rank_key`)."""
    if max_paths <= 0 or len(beams) <= max_paths:
        return beams
    return sorted(
        beams,
        key=lambda b: _draft_path_rank_key(get_appended_probs(b)),
        reverse=True,
    )[:max_paths]


def draft_branch_token_ids_from_logits(
    logits_1d: torch.Tensor,
    cfg: EagleConfig,
    k: int,
    generator: torch.Generator | None,
) -> list[int]:
    """Pick candidate token ids for one draft expansion step.

    **Mode ``draft_branching == "top_k"`` (default):** take the ``k`` largest logits.
    ``k`` must be **> 0** here.

    **Mode ``draft_branching == "temperature_top_p"``:** build the **top-p nucleus** on ``softmax(logits/T)``.

    * If ``k > 0``: sample ``min(k, nucleus_size)`` **distinct** tokens without replacement
      (``torch.multinomial``).
    * If ``k <= 0``: take **every** token in the nucleus (sorted by descending tempered prob).
      This matches “top-p only” without an extra top-k cap on branch count. Can be **very** wide
      and slow; pair with ``max_paths <= 0`` to disable beam pruning if you want the full cross
      product (see ``truncate_beams_by_draft_confidence``).

    T **≤ 0** falls back to **argmax** (single branch). **Probabilistic verification** still uses
    **untempered** ``softmax(logits)`` for ``q`` in ``min(1, q/p)``.

    Runs nucleus steps on **CPU** for stable ``torch.Generator`` behavior.
    """
    branching = getattr(cfg, "draft_branching", "top_k")
    if branching not in ("top_k", "temperature_top_p"):
        raise ValueError(f"Unknown draft_branching={branching!r}; expected 'top_k' or 'temperature_top_p'.")
    logits_flat = logits_1d.float().reshape(-1)
    vocab = int(logits_flat.numel())
    k_raw = int(k)

    if branching != "temperature_top_p":
        if k_raw <= 0:
            return []
        ki = min(k_raw, vocab)
        _, idx = torch.topk(logits_flat, k=ki, dim=-1)
        return [int(x) for x in idx.tolist()]

    temp = float(getattr(cfg, "draft_temperature", 0.6))
    top_p = float(getattr(cfg, "draft_top_p", 0.95))
    top_p = min(max(top_p, 1e-6), 1.0)

    if temp <= 0:
        return [int(torch.argmax(logits_flat, dim=-1).item())]

    probs = F.softmax(logits_flat / temp, dim=-1).cpu()
    sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_p, dim=-1)
    ge = cumsum >= top_p
    if bool(ge.any().item()):
        n_keep = int(ge.nonzero(as_tuple=False)[0].item()) + 1
    else:
        n_keep = int(sorted_p.numel())
    n_keep = max(1, min(n_keep, int(sorted_p.numel())))
    slice_p = sorted_p[:n_keep].clone()
    slice_p = slice_p / (slice_p.sum() + 1e-20)

    if k_raw <= 0:
        return [int(sorted_i[j].item()) for j in range(n_keep)]

    n_draw = min(min(k_raw, vocab), n_keep)
    draws = torch.multinomial(slice_p, num_samples=n_draw, replacement=False, generator=generator)
    return [int(sorted_i[int(j)].item()) for j in draws.tolist()]


# ---------------------------------------------------------------------------
# TraditionalDraftAdapter — separate small model as draft (same tokenizer)
# ---------------------------------------------------------------------------


class TraditionalDraftAdapter:
    """Traditional speculative decoding with a separate small HF model.

    Uses HF KV cache: the prefix is processed once, then each depth step
    feeds only the new token with past_key_values. No prefix recomputation.

    Recommended pairing for DeepSeek-R1-0528:
      jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0  (0.6B, vocab-transplanted Qwen2.5)
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
        trust_remote_code: bool = False,
    ) -> None:
        self.device = torch.device(device)
        resolved_dtype = torch.float32
        if torch_dtype.lower() == "float16":
            resolved_dtype = torch.float16
        elif torch_dtype.lower() == "bfloat16":
            resolved_dtype = torch.bfloat16
        logger.info("Loading traditional draft model '%s' on device=%s dtype=%s", model_id, device, torch_dtype)
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=resolved_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.model_id = model_id
        logger.info("Traditional draft model '%s' loaded: %d params", model_id, sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def forward_draft(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        if cfg.depth <= 0 or draft_requires_positive_top_k(cfg):
            return PathProposal(paths=[], draft_probs_per_path=None)

        gen = kwargs.get("draft_torch_generator")

        prefix = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=self.device)
        out = self.model(input_ids=prefix, use_cache=True)
        prefix_logits = out.logits[:, -1, :]
        prefix_probs = F.softmax(prefix_logits.float(), dim=-1)
        prefix_kv = out.past_key_values

        topk_ids = draft_branch_token_ids_from_logits(
            prefix_logits[0], cfg, cfg.top_k, gen if isinstance(gen, torch.Generator) else None,
        )
        if not topk_ids:
            return PathProposal(paths=[], draft_probs_per_path=None)
        Beam = tuple[list[int], list[float], tuple]
        beams: list[Beam] = []
        for tok_id in topk_ids:
            q = float(prefix_probs[0, tok_id].item())
            beams.append(([tok_id], [q], _clone_hf_kv(prefix_kv)))

        beams = truncate_beams_by_draft_confidence(beams, cfg.max_paths, lambda b: b[1])

        for depth_step in range(1, cfg.depth):
            if not beams:
                break
            next_beams: list[Beam] = []
            for appended, probs, kv in beams:
                new_token = torch.tensor([[appended[-1]]], dtype=torch.long, device=self.device)
                out = self.model(input_ids=new_token, past_key_values=kv, use_cache=True)
                logits = out.logits[:, -1, :]
                logits_probs = F.softmax(logits.float(), dim=-1)
                new_kv = out.past_key_values
                cand_ids = draft_branch_token_ids_from_logits(
                    logits[0], cfg, cfg.top_k, gen if isinstance(gen, torch.Generator) else None,
                )
                if not cand_ids:
                    continue
                for tok_id in cand_ids:
                    q = float(logits_probs[0, tok_id].item())
                    next_beams.append((appended + [tok_id], probs + [q], _clone_hf_kv(new_kv)))
            next_beams = truncate_beams_by_draft_confidence(next_beams, cfg.max_paths, lambda b: b[1])
            beams = next_beams

        paths = [appended for appended, _, _ in beams]
        draft_probs_per_path = [probs for _, probs, _ in beams]
        return PathProposal(paths=paths, draft_probs_per_path=draft_probs_per_path)

    def propose_paths(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        return self.forward_draft(prefix_token_ids, cfg, **kwargs)


# ---------------------------------------------------------------------------
# BaseAsDraftAdapter — uses base model as its own draft (for testing)
# ---------------------------------------------------------------------------


class BaseAsDraftAdapter:
    """Draft adapter that reuses the base model for proposals (testing only)."""

    def __init__(self, base_adapter: "DeepSeekBaseAdapter") -> None:
        self.base = base_adapter
        self.model = base_adapter.model
        self.device = base_adapter.device
        logger.info("Using base-as-draft adapter")

    @torch.no_grad()
    def forward_draft(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        if cfg.depth <= 0 or draft_requires_positive_top_k(cfg):
            return PathProposal(paths=[], draft_probs_per_path=None)

        gen = kwargs.get("draft_torch_generator")

        prefix = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=self.device)
        out = self.model(input_ids=prefix, use_cache=True)
        prefix_logits = out.logits[:, -1, :]
        prefix_probs = F.softmax(prefix_logits.float(), dim=-1)
        prefix_kv = out.past_key_values

        topk_ids = draft_branch_token_ids_from_logits(
            prefix_logits[0], cfg, cfg.top_k, gen if isinstance(gen, torch.Generator) else None,
        )
        if not topk_ids:
            return PathProposal(paths=[], draft_probs_per_path=None)
        Beam = tuple[list[int], list[float], tuple]
        beams: list[Beam] = []
        for tok_id in topk_ids:
            q = float(prefix_probs[0, tok_id].item())
            beams.append(([tok_id], [q], _clone_hf_kv(prefix_kv)))

        beams = truncate_beams_by_draft_confidence(beams, cfg.max_paths, lambda b: b[1])

        for depth_step in range(1, cfg.depth):
            if not beams:
                break
            next_beams: list[Beam] = []
            for appended, probs, kv in beams:
                new_token = torch.tensor([[appended[-1]]], dtype=torch.long, device=self.device)
                out = self.model(input_ids=new_token, past_key_values=kv, use_cache=True)
                logits = out.logits[:, -1, :]
                logits_probs = F.softmax(logits.float(), dim=-1)
                new_kv = out.past_key_values
                cand_ids = draft_branch_token_ids_from_logits(
                    logits[0], cfg, cfg.top_k, gen if isinstance(gen, torch.Generator) else None,
                )
                if not cand_ids:
                    continue
                for tok_id in cand_ids:
                    q = float(logits_probs[0, tok_id].item())
                    next_beams.append((appended + [tok_id], probs + [q], _clone_hf_kv(new_kv)))
            next_beams = truncate_beams_by_draft_confidence(next_beams, cfg.max_paths, lambda b: b[1])
            beams = next_beams

        paths = [appended for appended, _, _ in beams]
        draft_probs_per_path = [probs for _, probs, _ in beams]
        return PathProposal(paths=paths, draft_probs_per_path=draft_probs_per_path)

    def propose_paths(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        return self.forward_draft(prefix_token_ids, cfg, **kwargs)


# ---------------------------------------------------------------------------
# Eagle3HiddenStateDraftAdapter — EAGLE3 hidden-state draft head
# ---------------------------------------------------------------------------


class Eagle3HiddenStateDraftAdapter:
    """EAGLE3 hidden-state draft head (e.g. Yuhuili LLaMA-8B or eigen R1-0528).

    Drafting is fully self-contained: the midlayer's own hidden state and KV
    cache propagate across depths. NO base model forward calls during drafting.
    Only the initial hidden state comes from the base model's decode state.

    Supports any EAGLE3 checkpoint with compatible weight layout (fc, norm,
    lm_head, d2t, optional midlayer) and config (eagle_aux_hidden_state_layer_ids).
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        del torch_dtype
        self.device = torch.device(device)
        logger.info("Loading EAGLE3 draft head '%s' on device=%s", model_id, device)
        try:
            weights_path = hf_hub_download(model_id, "model.safetensors")
            from safetensors.torch import load_file
            state = load_file(weights_path, device=str(self.device))
        except Exception:
            weights_path = hf_hub_download(model_id, "pytorch_model.bin")
            state = torch.load(weights_path, map_location=self.device)
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected state format in '{model_id}': {type(state)}")

        self.feature_layer_ids: list[int] | None = None
        try:
            import json
            cfg_path = hf_hub_download(model_id, "config.json")
            with open(cfg_path) as f:
                eagle_cfg = json.load(f)
            eagle_extra = eagle_cfg.get("eagle_config", {})
            layer_ids = eagle_extra.get("eagle_aux_hidden_state_layer_ids")
            if isinstance(layer_ids, list) and len(layer_ids) == 3:
                self.feature_layer_ids = [int(x) for x in layer_ids]
                logger.info("EAGLE3 feature layer IDs from config: %s", self.feature_layer_ids)
        except Exception:
            pass

        required_keys = ("fc.weight", "norm.weight", "lm_head.weight", "d2t")
        missing = [k for k in required_keys if k not in state]
        if missing:
            raise ValueError(
                f"Draft checkpoint '{model_id}' missing EAGLE3 keys: {missing}. "
                "That model is a traditional full draft (use --draft-mode draft_r1). "
                "For EAGLE3 hidden-state head use --draft-model-preset eagle3_8b or eagle3_r1."
            )

        self.fc_weight = state["fc.weight"].to(self.device).float()
        self.norm_weight = state["norm.weight"].to(self.device).float()
        self.lm_head_weight = state["lm_head.weight"].to(self.device).float()
        self.d2t = state["d2t"].to(self.device).long()

        self.hidden_size = int(self.norm_weight.shape[0])
        self.fc_input_size = int(self.fc_weight.shape[1])

        has_midlayer = any(k.startswith("midlayer.") for k in state)
        if has_midlayer:
            self._load_midlayer(state)
            logger.info(
                "Loaded EAGLE3 draft head with midlayer: hidden=%d, heads=%d, kv_heads=%d, inter=%d, fc_in=%d",
                self.hidden_size, self.num_heads, self.num_kv_heads,
                self.intermediate_size, self.fc_input_size,
            )
        else:
            self.has_midlayer = False
            logger.info(
                "Loaded EAGLE3 draft head (no midlayer): hidden=%d, fc_in=%d",
                self.hidden_size, self.fc_input_size,
            )

        self.embed_tokens_weight: torch.Tensor | None = None
        # Default True: released EAGLE3 checkpoints (yuhuili, eigen) were trained with FC at every depth.
        # Set False for paper formulation (FC only at depth 0) via --no-eagle3-fc-every-depth.
        self.use_fc_at_every_depth: bool = True

    def _load_midlayer(self, state: dict) -> None:
        self.has_midlayer = True
        g = lambda k: state[k].to(self.device).float()

        self.mid_hidden_norm_w = g("midlayer.hidden_norm.weight")
        self.mid_input_ln_w = g("midlayer.input_layernorm.weight")
        self.mid_post_attn_ln_w = g("midlayer.post_attention_layernorm.weight")

        self.mid_q_proj_w = g("midlayer.self_attn.q_proj.weight")
        self.mid_k_proj_w = g("midlayer.self_attn.k_proj.weight")
        self.mid_v_proj_w = g("midlayer.self_attn.v_proj.weight")
        self.mid_o_proj_w = g("midlayer.self_attn.o_proj.weight")

        self.mid_gate_proj_w = g("midlayer.mlp.gate_proj.weight")
        self.mid_up_proj_w = g("midlayer.mlp.up_proj.weight")
        self.mid_down_proj_w = g("midlayer.mlp.down_proj.weight")

        q_out_dim = self.mid_q_proj_w.shape[0]
        k_out_dim = self.mid_k_proj_w.shape[0]
        for candidate_hd in [128, 64, 96, 256]:
            if q_out_dim % candidate_hd == 0 and k_out_dim % candidate_hd == 0:
                self.head_dim = candidate_hd
                break
        else:
            self.head_dim = 128
        self.num_heads = q_out_dim // self.head_dim
        self.num_kv_heads = k_out_dim // self.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.intermediate_size = self.mid_gate_proj_w.shape[0]
        self.rms_eps = 1e-5

        inv_freq = 1.0 / (500000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self._inv_freq = inv_freq.to(self.device)

    def bind_base_embeddings(self, base_adapter: "DeepSeekBaseAdapter") -> None:
        if self.embed_tokens_weight is not None:
            return
        if self.feature_layer_ids is not None and hasattr(base_adapter, "set_eagle3_layer_indices"):
            base_adapter.set_eagle3_layer_indices(self.feature_layer_ids)
        try:
            model = base_adapter.model
            if hasattr(model, "inner"):
                model = model.inner
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                self.embed_tokens_weight = model.model.embed_tokens.weight.data.to(self.device).float()
            elif hasattr(model, "embed_tokens"):
                self.embed_tokens_weight = model.embed_tokens.weight.data.to(self.device).float()
            elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                self.embed_tokens_weight = model.transformer.wte.weight.data.to(self.device).float()
            if self.embed_tokens_weight is not None:
                logger.info("Bound base embed_tokens: shape=%s", list(self.embed_tokens_weight.shape))
        except Exception as exc:
            logger.warning("Could not bind base embeddings: %s", exc)

    def _get_token_embedding(self, token_id: int) -> torch.Tensor:
        if self.embed_tokens_weight is not None and 0 <= token_id < self.embed_tokens_weight.shape[0]:
            return self.embed_tokens_weight[token_id]
        return torch.zeros(self.hidden_size, device=self.device)

    def _apply_rope(self, x: torch.Tensor, position: int) -> torch.Tensor:
        t = torch.tensor([position], dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, self._inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    def _midlayer_forward(
        self,
        hidden: torch.Tensor,
        token_emb: torch.Tensor,
        position: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden

        h_normed = _rms_norm(hidden, self.mid_hidden_norm_w, self.rms_eps)
        e_normed = _rms_norm(token_emb, self.mid_input_ln_w, self.rms_eps)
        cat_input = torch.cat([e_normed, h_normed], dim=-1)

        bsz = cat_input.shape[0]
        q = F.linear(cat_input, self.mid_q_proj_w).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = F.linear(cat_input, self.mid_k_proj_w).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = F.linear(cat_input, self.mid_v_proj_w).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, position)
        k_new = self._apply_rope(k_new, position)

        if kv_cache is not None:
            k_all = torch.cat([kv_cache[0], k_new], dim=2)
            v_all = torch.cat([kv_cache[1], v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new
        new_kv_cache = (k_all, v_all)

        k_for_attn, v_for_attn = k_all, v_all
        if self.num_kv_groups > 1:
            k_for_attn = k_for_attn.repeat_interleave(self.num_kv_groups, dim=1)
            v_for_attn = v_for_attn.repeat_interleave(self.num_kv_groups, dim=1)

        attn_w = torch.matmul(q, k_for_attn.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_w, v_for_attn)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, 1, self.hidden_size)
        attn_out = F.linear(attn_out, self.mid_o_proj_w)
        hidden = residual + attn_out.squeeze(1)

        residual = hidden
        h2 = _rms_norm(hidden, self.mid_post_attn_ln_w, self.rms_eps)
        mlp_out = F.linear(F.silu(F.linear(h2, self.mid_gate_proj_w)) * F.linear(h2, self.mid_up_proj_w), self.mid_down_proj_w)
        hidden = residual + mlp_out
        return hidden, new_kv_cache

    def _draft_idx_to_target(self, draft_idx: int) -> int:
        return draft_idx + int(self.d2t[draft_idx].item())

    def _eagle3_step(
        self,
        hidden: torch.Tensor,
        token_id: int,
        position: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        depth_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Run one EAGLE3 draft step. Returns (logits, output_hidden, updated_kv_cache).

        FC usage (depth_idx is the speculative depth 0, 1, 2, ...):
        - depth_idx == 0: always apply FC. Input is either 3-layer (fc_input_size) from base
          or 1-layer (hidden_size) fallback; output is always FC(.) -> hidden_size.
        - depth_idx >= 1: input is previous step output (hidden_size). If use_fc_at_every_depth
          then FC(cat(h,h,h)); else (paper) use h as-is and skip FC.
        """
        h = hidden.float()
        last_dim = h.shape[-1]

        # Depth 0: never skip FC — first-token logits must not depend on use_fc_at_every_depth
        if depth_idx == 0:
            if last_dim == self.fc_input_size:
                projected = F.linear(h, self.fc_weight)
            elif last_dim == self.hidden_size:
                projected = F.linear(torch.cat([h, h, h], dim=-1), self.fc_weight)
            else:
                inp = h[..., : self.fc_input_size] if last_dim > self.fc_input_size else F.pad(h, (0, self.fc_input_size - last_dim))
                projected = F.linear(inp, self.fc_weight)
        # Depth >= 1: optional FC per use_fc_at_every_depth
        elif last_dim == self.hidden_size:
            if self.use_fc_at_every_depth:
                projected = F.linear(torch.cat([h, h, h], dim=-1), self.fc_weight)
            else:
                projected = h
        else:
            inp = h[..., : self.fc_input_size] if last_dim > self.fc_input_size else F.pad(h, (0, self.fc_input_size - last_dim))
            projected = F.linear(inp, self.fc_weight)

        new_kv = None
        if self.has_midlayer and self.embed_tokens_weight is not None and self.embed_tokens_weight.shape[-1] == self.hidden_size:
            if projected.dim() == 1:
                projected = projected.unsqueeze(0)
            emb = self._get_token_embedding(token_id).unsqueeze(0)
            projected, new_kv = self._midlayer_forward(projected, emb, position, kv_cache)

        normed = _rms_norm(projected, self.norm_weight, self.rms_eps if self.has_midlayer else 1e-6)
        logits = F.linear(normed, self.lm_head_weight)
        return logits, projected, new_kv

    def _filter_to_base_vocab(self, token_ids: Sequence[int], base_adapter: "DeepSeekBaseAdapter | None") -> list[int]:
        if base_adapter is None:
            return list(token_ids)
        vocab_size = int(base_adapter.model.config.vocab_size)
        return [int(tid) for tid in token_ids if 0 <= int(tid) < vocab_size]

    @torch.no_grad()
    def forward_draft(
        self,
        prefix_token_ids: Sequence[int],
        cfg: EagleConfig,
        *,
        decode_state: "DecodeState | None" = None,
        base_adapter: "DeepSeekBaseAdapter | None" = None,
        **kwargs: object,
    ) -> PathProposal:
        """Draft using only EAGLE3 head — NO base model calls during drafting.

        The midlayer propagates its own hidden state and KV cache across depths.
        Only the initial hidden state comes from the base model's decode state.
        """
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        if cfg.depth <= 0 or draft_requires_positive_top_k(cfg):
            return PathProposal(paths=[], draft_probs_per_path=None)
        if decode_state is None or base_adapter is None:
            return PathProposal(paths=[], draft_probs_per_path=None)

        gen = kwargs.get("draft_torch_generator")

        if self.has_midlayer:
            self.bind_base_embeddings(base_adapter)

        position = len(prefix_token_ids)
        last_token_id = int(prefix_token_ids[-1])

        use_multi = decode_state.multi_layer_hidden is not None
        init_hidden = decode_state.multi_layer_hidden if use_multi else decode_state.last_hidden_state
        if not use_multi and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "EAGLE3 draft: multi_layer_hidden is None, using last_hidden_state (1-layer). "
                "First-token logits still use FC via cat([h,h,h])."
            )

        MidKV = tuple[torch.Tensor, torch.Tensor] | None
        Beam = tuple[torch.Tensor, list[int], list[float], int, MidKV]

        beams: list[Beam] = [(init_hidden.squeeze(0), [], [], last_token_id, None)]

        for depth_idx in range(cfg.depth):
            if not beams:
                break
            next_beams: list[Beam] = []
            for draft_hidden, appended, appended_probs, tid, mid_kv in beams:
                logits, out_hidden, new_kv = self._eagle3_step(
                    draft_hidden, tid, position + depth_idx, mid_kv, depth_idx=depth_idx,
                )
                draft_flat = logits.reshape(-1)
                top_draft_ids = draft_branch_token_ids_from_logits(
                    draft_flat,
                    cfg,
                    cfg.top_k,
                    gen if isinstance(gen, torch.Generator) else None,
                )
                draft_probs = F.softmax(logits.float(), dim=-1)
                for draft_idx in top_draft_ids:
                    target_tid = self._draft_idx_to_target(draft_idx)
                    filt = self._filter_to_base_vocab([target_tid], base_adapter)
                    if not filt:
                        continue
                    target_tid = filt[0]
                    if draft_probs.dim() == 1:
                        q = float(draft_probs[draft_idx].item())
                    else:
                        q = float(draft_probs[0, draft_idx].item())
                    cloned_kv = (new_kv[0].clone(), new_kv[1].clone()) if new_kv is not None else None
                    next_beams.append((out_hidden.squeeze(0), appended + [target_tid], appended_probs + [q], target_tid, cloned_kv))
            next_beams = truncate_beams_by_draft_confidence(next_beams, cfg.max_paths, lambda b: b[2])
            beams = next_beams

        paths = [path for _, path, _, _, _ in beams]
        draft_probs_per_path = [probs for _, _, probs, _, _ in beams]
        return PathProposal(paths=paths, draft_probs_per_path=draft_probs_per_path)

    def propose_paths(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        return self.forward_draft(prefix_token_ids, cfg, **kwargs)


# ---------------------------------------------------------------------------
# MTPDraftAdapter — shared MTP-head math (weights come from NextNMTPHeadDraftAdapter only)
# ---------------------------------------------------------------------------


class MTPDraftAdapter:
    """Shared **fusion + head** MTP step (:meth:`_mtp_forward_batch`) used by :class:`NextNMTPHeadDraftAdapter`.

    This path applies ``eh_proj`` / ``enorm`` / ``hnorm`` then ``shared_head.norm`` + ``shared_head.head``.
    It **does not** run the ``DeepseekV3DecoderLayer`` (MoE/MLA) that SGLang's ``DeepseekModelNextN`` runs
    after fusion — see :class:`NextNSglangStructureDraftAdapter` in ``nextn_sglang_structure_draft.py``.

    Weights must be bound via :meth:`NextNMTPHeadDraftAdapter.bind_from_nextn_paths` before use.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.bound = False
        self._enorm_w: torch.Tensor | None = None
        self._hnorm_w: torch.Tensor | None = None
        self._eh_proj_w: torch.Tensor | None = None
        self._shared_head_norm_w: torch.Tensor | None = None
        self._shared_head_w: torch.Tensor | None = None
        self._embed_weight: torch.Tensor | None = None
        self._hidden_size: int = 0
        self._rms_eps: float = 1e-6

    def _mtp_forward_batch(
        self, hidden_states: torch.Tensor, token_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched MTP step. Returns (logits [B, vocab], output_hidden [B, hidden])."""
        bsz = hidden_states.shape[0]
        dev = self.device
        hidden_states = hidden_states.to(dev)
        embs = torch.stack([self._embed_weight[tid] for tid in token_ids]).float()
        h = hidden_states.float()

        e_normed = _rms_norm(embs, self._enorm_w, self._rms_eps)
        h_normed = _rms_norm(h, self._hnorm_w, self._rms_eps)
        fused = F.linear(torch.cat([e_normed, h_normed], dim=-1), self._eh_proj_w)
        logits = F.linear(_rms_norm(fused, self._shared_head_norm_w, self._rms_eps), self._shared_head_w)
        return logits, fused

    @torch.no_grad()
    def forward_draft(
        self,
        prefix_token_ids: Sequence[int],
        cfg: EagleConfig,
        *,
        decode_state: "DecodeState | None" = None,
        base_adapter: "DeepSeekBaseAdapter | None" = None,
        **kwargs: object,
    ) -> PathProposal:
        """Autoregressive MTP drafting with 1 batched MTP call per depth.

        Greedy (draft_mtp_greedy): argmax at each step → single path of length depth.
        Non-greedy: top_k at each step → up to top_k^depth paths (capped by max_paths, keeping
        beams with best peak per-step draft probability; see :func:`truncate_beams_by_draft_confidence`).
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
                "MTP draft weights are not bound. Use NextNMTPHeadDraftAdapter.bind_from_nextn_paths(...) "
                "or bind_from_nextn_hub(...) for record + Hub NextN weights (no full DeepSeek load)."
            )

        k = 1 if draft_mtp_greedy else cfg.top_k
        gen = kwargs.get("draft_torch_generator")
        last_token_id = int(prefix_token_ids[-1])
        Beam = tuple[torch.Tensor, list[int], list[float], int]
        beams: list[Beam] = [(decode_state.last_hidden_state.squeeze(0), [], [], last_token_id)]

        for depth_idx in range(cfg.depth):
            if not beams:
                break
            hidden_batch = torch.stack([h for h, _, _, _ in beams])
            tid_batch = [tid for _, _, _, tid in beams]

            logits_batch, hidden_out_batch = self._mtp_forward_batch(hidden_batch, tid_batch)
            probs_batch = F.softmax(logits_batch.float(), dim=-1)

            next_beams: list[Beam] = []
            for i, (_, appended, appended_probs, _) in enumerate(beams):
                cand_ids = draft_branch_token_ids_from_logits(
                    logits_batch[i],
                    cfg,
                    k,
                    gen if isinstance(gen, torch.Generator) else None,
                )
                if not cand_ids:
                    continue
                for tok_id in cand_ids:
                    q = float(probs_batch[i, tok_id].item())
                    next_beams.append((hidden_out_batch[i], appended + [tok_id], appended_probs + [q], tok_id))

            next_beams = truncate_beams_by_draft_confidence(next_beams, cfg.max_paths, lambda b: b[2])
            # Below threshold: still verify paths at the current depth; only stop adding more tokens.
            thr = getattr(cfg, "draft_extend_min_p_max", None)
            if thr is not None and next_beams:
                step_max = max(b[2][-1] for b in next_beams)
                if step_max < thr:
                    if depth_idx == 0:
                        beams = next_beams
                    break
            beams = next_beams

        paths = [path for _, path, _, _ in beams]
        draft_probs_per_path = [probs for _, _, probs, _ in beams]
        return PathProposal(paths=paths, draft_probs_per_path=draft_probs_per_path)

    def propose_paths(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        return self.forward_draft(prefix_token_ids, cfg, **kwargs)


# ---------------------------------------------------------------------------
# NextN MTP head (lmsys/DeepSeek-R1-NextN): fusion shard + embed/head from shard or optional aux file
# ---------------------------------------------------------------------------

_NEXTN_MTP_KEYS_REQUIRED = (
    "model.layers.0.eh_proj.weight",
    "model.layers.0.enorm.weight",
    "model.layers.0.hnorm.weight",
    "model.layers.0.shared_head.norm.weight",
)
_NEXTN_SHARED_HEAD_HEAD_KEY = "model.layers.0.shared_head.head.weight"
_NEXTN_EMBED_KEYS = ("model.embed_tokens.weight", "embed_tokens.weight")


class NextNMTPHeadDraftAdapter(MTPDraftAdapter):
    """Draft = **NextN MTP fusion** mats + ``shared_head`` (norm + lm), **without** the draft decoder layer.

    Token embeddings and ``shared_head.head`` may appear **inside** the fusion file (if your
    checkpoint includes them) or in a separate small ``.safetensors`` passed to
    ``embed_head_aux_safetensors`` (see :func:`load_nextn_mtp_auxiliary_safetensors`).

    SGLang's ``DeepseekModelNextN`` runs **fusion → ``DeepseekV2DecoderLayer`` → norm → head**; this class
    stops after fusion and uses ``shared_head`` directly. For the full order on CPU, use
    :class:`NextNSglangStructureDraftAdapter`.
    """

    @torch.no_grad()
    def forward_draft(
        self,
        prefix_token_ids: Sequence[int],
        cfg: EagleConfig,
        *,
        decode_state: "DecodeState | None" = None,
        base_adapter: "DeepSeekBaseAdapter | None" = None,
        **kwargs: object,
    ) -> PathProposal:
        if not self.bound:
            raise RuntimeError(
                "NextNMTPHeadDraftAdapter is not bound. Call bind_from_nextn_paths(...) or "
                "bind_from_nextn_hub(...) before running the engine."
            )
        return super().forward_draft(
            prefix_token_ids,
            cfg,
            decode_state=decode_state,
            base_adapter=base_adapter,
            **kwargs,
        )

    def _bind_nextn_fusion_from_file(self, nextn_safetensors: Path, dev: torch.device) -> None:
        self._embed_weight = None
        with safe_open(str(nextn_safetensors), framework="pt", device=str(dev)) as f:
            missing = [k for k in _NEXTN_MTP_KEYS_REQUIRED if k not in f.keys()]
            if missing:
                raise ValueError(f"NextN weights missing expected keys: {missing}")
            self._eh_proj_w = f.get_tensor("model.layers.0.eh_proj.weight").to(dev).float()
            self._enorm_w = f.get_tensor("model.layers.0.enorm.weight").to(dev).float()
            self._hnorm_w = f.get_tensor("model.layers.0.hnorm.weight").to(dev).float()
            self._shared_head_norm_w = f.get_tensor("model.layers.0.shared_head.norm.weight").to(dev).float()
            if _NEXTN_SHARED_HEAD_HEAD_KEY in f.keys():
                self._shared_head_w = f.get_tensor(_NEXTN_SHARED_HEAD_HEAD_KEY).to(dev).float()
            else:
                self._shared_head_w = None
            for ek in _NEXTN_EMBED_KEYS:
                if ek in f.keys():
                    self._embed_weight = f.get_tensor(ek).to(dev).float()
                    break

    def _apply_config_embed_head_validate(self, cfg: dict, _dev: torch.device) -> None:
        n_mtp = int(cfg.get("num_nextn_predict_layers", 0))
        if n_mtp == 0:
            raise ValueError("config.json must have num_nextn_predict_layers > 0")
        self._hidden_size = int(cfg["hidden_size"])
        self._rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
        exp_in = 2 * self._hidden_size
        if self._eh_proj_w.shape[1] != exp_in:
            raise ValueError(
                f"eh_proj weight in_features {self._eh_proj_w.shape[1]} != 2*hidden {exp_in}"
            )
        if self._shared_head_w is None:
            raise ValueError(
                "shared_head.head missing: add it to the NextN fusion file or pass embed_head_aux_safetensors"
            )
        if self._shared_head_w.shape[1] != self._hidden_size:
            raise ValueError(
                f"shared_head.head in_features {self._shared_head_w.shape[1]} != hidden_size {self._hidden_size}"
            )
        if self._embed_weight.shape[0] != self._shared_head_w.shape[0]:
            logger.warning(
                "embed vocab size %d != shared_head.out_features %d — logits may be inconsistent",
                self._embed_weight.shape[0],
                self._shared_head_w.shape[0],
            )

    def bind_from_nextn_paths(
        self,
        *,
        nextn_safetensors: str | Path,
        embed_head_aux_safetensors: str | Path | None = None,
        nextn_config_dir: str | Path | None = None,
    ) -> None:
        """Load NextN fusion from disk; optional aux file supplies embed/head if not in the fusion shard."""
        if self.bound:
            return
        if safe_open is None:
            raise ImportError(
                "NextNMTPHeadDraftAdapter requires the `safetensors` package "
                "(e.g. pip install safetensors)."
            )
        dev = self.device
        nextn_path = Path(nextn_safetensors).expanduser().resolve()
        if not nextn_path.is_file():
            raise FileNotFoundError(f"NextN safetensors not found: {nextn_path}")
        cfg_dir = Path(nextn_config_dir).expanduser().resolve() if nextn_config_dir else nextn_path.parent
        cfg = read_snapshot_config(cfg_dir)

        self._bind_nextn_fusion_from_file(nextn_path, dev)
        if embed_head_aux_safetensors is not None:
            aux_path = Path(embed_head_aux_safetensors).expanduser().resolve()
            if not aux_path.is_file():
                raise FileNotFoundError(f"embed_head_aux_safetensors not found: {aux_path}")
            aux_embed, aux_head = load_nextn_mtp_auxiliary_safetensors(aux_path)
            self._embed_weight = aux_embed.to(dev)
            self._shared_head_w = aux_head.to(dev)
        if self._embed_weight is None:
            raise ValueError(
                "Token embeddings missing: pass --embed-head-aux-safetensors with tensors accepted by "
                "load_nextn_mtp_auxiliary_safetensors, or build that file once via "
                "models/demos/speculative_deepseek_r1_broad/scripts/materialize_nextn_embed_head_aux_from_r1_shards.py "
                "(downloads only the two HF shards that contain embed + shared_head.head; see agent_plan.md)."
            )
        if self._shared_head_w is None:
            raise ValueError(
                "shared_head.head missing: pass embed_head_aux_safetensors or run "
                "materialize_nextn_embed_head_aux_from_r1_shards.py (see agent_plan.md)."
            )

        self._apply_config_embed_head_validate(cfg, dev)
        self.bound = True
        logger.info(
            "Bound NextN MTP draft: fusion=%s aux_embed_head=%s config_dir=%s",
            nextn_path,
            embed_head_aux_safetensors,
            cfg_dir,
        )

    def bind_from_nextn_hub(
        self,
        nextn_repo_id: str,
        *,
        embed_head_aux_safetensors: str | Path | None = None,
        weights_filename: str = "nextn_layer_parameters.safetensors",
    ) -> None:
        """Download NextN fusion from the Hub; optional local aux file for embed/head if not in fusion."""
        if self.bound:
            return
        if safe_open is None:
            raise ImportError(
                "NextNMTPHeadDraftAdapter requires the `safetensors` package "
                "(e.g. pip install safetensors)."
            )
        dev = self.device
        weight_path = Path(hf_hub_download(nextn_repo_id, weights_filename))
        cfg = read_snapshot_config(weight_path.parent)

        self._bind_nextn_fusion_from_file(weight_path, dev)
        if embed_head_aux_safetensors is not None:
            aux_path = Path(embed_head_aux_safetensors).expanduser().resolve()
            if not aux_path.is_file():
                raise FileNotFoundError(f"embed_head_aux_safetensors not found: {aux_path}")
            aux_embed, aux_head = load_nextn_mtp_auxiliary_safetensors(aux_path)
            self._embed_weight = aux_embed.to(dev)
            self._shared_head_w = aux_head.to(dev)
        if self._embed_weight is None:
            raise ValueError(
                "Token embeddings missing: pass embed_head_aux_safetensors or run "
                "materialize_nextn_embed_head_aux_from_r1_shards.py (see agent_plan.md)."
            )
        if self._shared_head_w is None:
            raise ValueError(
                "shared_head.head missing: pass embed_head_aux_safetensors or run "
                "materialize_nextn_embed_head_aux_from_r1_shards.py (see agent_plan.md)."
            )

        self._apply_config_embed_head_validate(cfg, dev)
        self.bound = True
        logger.info(
            "Bound NextN MTP draft from Hub repo %s (%s) + local aux %s",
            nextn_repo_id,
            weights_filename,
            embed_head_aux_safetensors,
        )


class SglangStyleNextNMTPDraftAdapter(NextNMTPHeadDraftAdapter):
    """Alias of :class:`NextNMTPHeadDraftAdapter` (fusion + ``shared_head`` only).

    Despite the name, this is **not** the full ``DeepseekModelNextN`` stack in SGLang (no MoE/MLA layer
    after fusion). See :class:`NextNSglangStructureDraftAdapter` and ``SGLANG_NEXTN_AND_CPU.md``.
    """

    pass
