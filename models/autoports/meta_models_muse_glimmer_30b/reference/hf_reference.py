# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace reference for the ``meta-models/Muse-Glimmer-30B`` text decoder layer.

Everything here runs on the host in float32 and is only used by tests to produce the
golden tensors the TTNN :class:`FunctionalDecoder` is compared against.

Why a hand-driven reference instead of ``MuseGlimmerTextDecoderLayer.forward``?
The advertised context is 131072 tokens, so the golden path has to be able to

* compute the layer output for one *chunk* of query positions against a K/V prefix
  (the full ``[1, 32, 131072, 131072]`` eager attention matrix does not fit anywhere),
* fill a K/V cache for a long prefix cheaply (K/V of position ``p`` only depends on the
  layer input at ``p``, never on attention), and
* run a single decode step against that cache.

:class:`ReferenceDecoderLayer` therefore re-drives the exact HF submodules
(``q_proj``/``k_proj``/``v_proj``/``qk_norm``/``gate_proj``/``o_proj``, the four
centered RMS norms, the SwiGLU MLP, ``apply_rotary_pos_emb``,
``eager_attention_forward``) in the same order as ``MuseGlimmerTextDecoderLayer.forward``
and ``MuseGlimmerTextAttention.forward``. ``test_reference_matches_hf`` asserts this
re-drive is numerically identical to the untouched HF layer forward (with HF's own
``create_causal_mask`` / ``create_sliding_window_causal_mask``) for both layer kinds,
including across the sliding-window boundary, so the chunked reference inherits HF
fidelity.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerTextDecoderLayer,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

HF_MODEL_ID = "meta-models/Muse-Glimmer-30B"
LAYER_PREFIX = "model.language_model.layers"
EMBED_KEY = "model.language_model.embed_tokens.weight"

# Layer-local names of every parameter of one MuseGlimmerTextDecoderLayer.
LAYER_PARAM_NAMES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.gate_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def load_hf_config(model_id: str = HF_MODEL_ID, local_files_only: bool = True):
    """Load the top-level ``MuseGlimmerConfig`` (``.text_config`` holds the decoder).

    Resolved through :func:`snapshot_dir` rather than the model id so config, tokenizer and
    weights always come from the *same* local revision (see that function: a metadata-only
    revision can become ``refs/main``).
    """
    source = str(snapshot_dir(model_id)) if local_files_only else model_id
    return AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=local_files_only)


def load_text_config(model_id: str = HF_MODEL_ID, local_files_only: bool = True):
    return load_hf_config(model_id, local_files_only).text_config


def snapshot_dir(model_id: str = HF_MODEL_ID) -> Path:
    """Local HF snapshot directory that actually holds the weights for ``model_id``.

    ``refs/main`` can point at a revision whose local snapshot only has ``config.json`` (that
    happened mid-stage: a metadata-only revision appeared in the cache and
    ``snapshot_download`` then resolved to a directory with no
    ``model.safetensors.index.json``). Prefer the newest snapshot that has the weight index,
    and fall back to whatever the hub resolves so the error message stays informative.
    """
    from huggingface_hub import snapshot_download

    resolved = Path(snapshot_download(model_id, local_files_only=True, allow_patterns=["*.json"]))
    if (resolved / "model.safetensors.index.json").is_file():
        return resolved
    snapshots = resolved.parent
    complete = [
        candidate for candidate in snapshots.iterdir() if (candidate / "model.safetensors.index.json").is_file()
    ]
    if complete:
        return max(complete, key=lambda path: path.stat().st_mtime)
    return resolved


def _safetensors_index(model_id: str = HF_MODEL_ID) -> tuple[Path, dict]:
    root = snapshot_dir(model_id)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    return root, index["weight_map"]


def load_real_layer_state_dict(layer_idx: int, model_id: str = HF_MODEL_ID) -> dict[str, torch.Tensor]:
    """Load one decoder layer's real checkpoint weights (layer-local keys, bf16 on disk).

    Only the ~12 tensors of the requested layer are read out of the shards, so this does
    not materialise the 60 GB checkpoint.
    """
    from safetensors import safe_open

    root, weight_map = _safetensors_index(model_id)
    wanted = {name: f"{LAYER_PREFIX}.{layer_idx}.{name}" for name in LAYER_PARAM_NAMES}
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for local, full in wanted.items():
        if full not in weight_map:
            raise KeyError(f"{full} not found in the checkpoint index")
        by_shard.setdefault(weight_map[full], []).append((local, full))

    out: dict[str, torch.Tensor] = {}
    for shard, entries in by_shard.items():
        with safe_open(str(root / shard), framework="pt") as f:
            for local, full in entries:
                out[local] = f.get_tensor(full)
    return out


def load_real_embedding_rows(token_ids: torch.Tensor, model_id: str = HF_MODEL_ID) -> torch.Tensor:
    """Read only the requested rows of the real embedding matrix (``[n, hidden]``, fp32)."""
    from safetensors import safe_open

    root, weight_map = _safetensors_index(model_id)
    shard = weight_map[EMBED_KEY]
    flat = token_ids.reshape(-1).tolist()
    with safe_open(str(root / shard), framework="pt") as f:
        slicer = f.get_slice(EMBED_KEY)
        rows = [slicer[int(tid) : int(tid) + 1, :] for tid in flat]
    return torch.cat(rows, dim=0).to(torch.float32).reshape(*token_ids.shape, -1)


def real_token_ids(seq_len: int, *, model_id: str = HF_MODEL_ID, offset: int = 0) -> torch.Tensor:
    """``[1, seq_len]`` real token ids from the model's own tokenizer over English prose.

    Real ids matter for the real-weight tests: the layer input is then the checkpoint's
    own embedding rows for tokens the model actually sees, not a synthetic distribution.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir(model_id)), local_files_only=True, trust_remote_code=True
    )
    text_path = Path(__file__).resolve().parents[3] / "common" / "readiness_check" / "autoregressive_prompt.txt"
    text = text_path.read_text()
    ids: list[int] = []
    while len(ids) < seq_len + offset:
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    return torch.tensor(ids[offset : offset + seq_len], dtype=torch.long).reshape(1, seq_len)


def weight_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Per-tensor name/shape/dtype/mean/std, used to generate synthetic weights."""
    stats = {}
    for name, tensor in sorted(state_dict.items()):
        t = tensor.to(torch.float32)
        stats[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "mean": float(t.mean()),
            "std": float(t.std()),
            "absmax": float(t.abs().max()),
        }
    return stats


def synthetic_layer_state_dict(stats: dict[str, dict], seed: int = 0) -> dict[str, torch.Tensor]:
    """Deterministically regenerate a layer state dict from :func:`weight_stats` output.

    Real shapes and real per-tensor mean/std, random values — so CI does not need the
    60 GB checkpoint but every tensor still has the production geometry and scale.
    """
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for idx, name in enumerate(sorted(stats)):
        spec = stats[name]
        shape = tuple(spec["shape"])
        per_tensor = torch.Generator().manual_seed(seed * 1000003 + idx)
        values = torch.randn(shape, generator=per_tensor, dtype=torch.float32)
        out[name] = values * spec["std"] + spec["mean"]
    del generator
    return out


def unit_rms_hidden_states(shape, seed: int = 0) -> torch.Tensor:
    """Synthetic layer input with the distribution the real model produces.

    ``MuseGlimmerTextNormedEmbedding`` applies a weight-less RMSNorm to the embedding
    output, so the activation entering layer 0 has exactly unit RMS per token. Deeper
    layers keep a similar scale because every residual branch ends in a norm.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=generator, dtype=torch.float32)
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


def rope_cos_sin(head_dim: int, theta: float, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``MuseGlimmerTextRotaryEmbedding`` cos/sin for ``positions`` (``[..., S]``)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def additive_mask(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    *,
    sliding_window: int | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """HF-equivalent additive attention mask for arbitrary q/kv position vectors.

    Matches ``transformers.masking_utils``: ``causal_mask_function`` is
    ``kv_idx <= q_idx`` and ``sliding_window_overlay`` adds
    ``kv_idx > q_idx - sliding_window`` (so a window of exactly ``sliding_window``
    tokens, current position included).
    """
    q = q_positions.reshape(-1, 1)
    kv = kv_positions.reshape(1, -1)
    allowed = kv <= q
    if sliding_window is not None:
        allowed = allowed & (kv > q - sliding_window)
    mask = torch.zeros(allowed.shape, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.reshape(1, 1, *allowed.shape)


def bool_mask(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    *,
    sliding_window: int | None,
) -> torch.Tensor:
    """Boolean (True = attend) form of :func:`additive_mask` for ``F.sdpa``."""
    q = q_positions.reshape(-1, 1)
    kv = kv_positions.reshape(1, -1)
    allowed = kv <= q
    if sliding_window is not None:
        allowed = allowed & (kv > q - sliding_window)
    return allowed.reshape(1, 1, *allowed.shape)


@dataclass
class LayerKind:
    layer_idx: int
    layer_type: str
    rope_theta: float | None
    sliding_window: int | None

    @property
    def kind_id(self) -> str:
        return "sliding_rope" if self.layer_type == "sliding_attention" else "full_nope"


def layer_kinds(text_config) -> dict[str, LayerKind]:
    """The distinct decoder-layer kinds of this model, keyed by ``kind_id``.

    Muse-Glimmer interleaves ``[sliding, sliding, sliding, full]``; the representative
    layer chosen for each kind is the first layer of that kind.
    """
    kinds: dict[str, LayerKind] = {}
    for idx, layer_type in enumerate(text_config.layer_types):
        theta = text_config.layer_rope_theta[idx]
        kind = LayerKind(
            layer_idx=idx,
            layer_type=layer_type,
            rope_theta=float(theta) if theta else None,
            sliding_window=text_config.sliding_window if layer_type == "sliding_attention" else None,
        )
        kinds.setdefault(kind.kind_id, kind)
    return kinds


class ReferenceDecoderLayer:
    """Host float32 reference for one Muse-Glimmer text decoder layer."""

    def __init__(self, text_config, layer_idx: int, state_dict: dict[str, torch.Tensor]):
        self.text_config = text_config
        self.layer_idx = layer_idx
        self.layer_type = text_config.layer_types[layer_idx]
        theta = text_config.layer_rope_theta[layer_idx]
        self.rope_theta = float(theta) if theta else None
        self.sliding_window = text_config.sliding_window if self.layer_type == "sliding_attention" else None
        self.head_dim = text_config.head_dim
        self.num_kv_heads = text_config.num_key_value_heads

        with torch.device("meta"):
            layer = MuseGlimmerTextDecoderLayer(text_config, layer_idx)
        missing, unexpected = layer.load_state_dict(
            {k: v.to(torch.float32) for k, v in state_dict.items()}, strict=True, assign=True
        )
        if missing or unexpected:  # pragma: no cover - strict=True already raises
            raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
        layer.eval()
        self.layer = layer

    # ------------------------------------------------------------------ parts

    def _rope(self, positions: torch.Tensor):
        """cos/sin for ``apply_rotary_pos_emb`` (``unsqueeze_dim=1``).

        ``positions`` is either ``[S]`` (shared across the batch) or ``[batch, S]``.
        """
        if self.rope_theta is None:
            return None, None
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        cos, sin = rope_cos_sin(self.head_dim, self.rope_theta, positions)
        return cos, sin  # [1 or batch, S, head_dim]

    @torch.no_grad()
    def input_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layer.input_layernorm(hidden)

    @torch.no_grad()
    def compute_kv(self, normed_hidden: torch.Tensor, positions: torch.Tensor):
        """K/V for the given positions from the *post-input_layernorm* activations."""
        attn = self.layer.self_attn
        shape = (*normed_hidden.shape[:-1], -1, self.head_dim)
        k = attn.k_proj(normed_hidden).view(shape).transpose(1, 2)
        v = attn.v_proj(normed_hidden).view(shape).transpose(1, 2)
        k = attn.qk_norm(k)
        if self.rope_theta is not None:
            cos, sin = self._rope(positions)
            # apply_rotary_pos_emb rotates q and k together; rotate k against itself.
            _, k = apply_rotary_pos_emb(k, k, cos, sin)
        return k, v

    @torch.no_grad()
    def compute_q(self, normed_hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        attn = self.layer.self_attn
        shape = (*normed_hidden.shape[:-1], -1, self.head_dim)
        q = attn.q_proj(normed_hidden).view(shape).transpose(1, 2)
        q = attn.qk_norm(q) * attn.qk_scale_factor
        if self.rope_theta is not None:
            cos, sin = self._rope(positions)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        return q

    @torch.no_grad()
    def attention_out(
        self,
        normed_hidden: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
        backend: str = "sdpa",
    ) -> torch.Tensor:
        """Attention output (post ``o_proj``) for one block of query positions."""
        attn = self.layer.self_attn
        if backend == "eager":
            mask = additive_mask(q_positions, kv_positions, sliding_window=self.sliding_window)
            out, _ = eager_attention_forward(attn, q, k, v, mask, scaling=attn.scaling)
        elif backend == "sdpa":
            from transformers.models.muse_glimmer.modeling_muse_glimmer import repeat_kv

            mask = bool_mask(q_positions, kv_positions, sliding_window=self.sliding_window)
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                repeat_kv(k, attn.num_key_value_groups),
                repeat_kv(v, attn.num_key_value_groups),
                attn_mask=mask,
                scale=attn.scaling,
            )
            out = out.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"unknown backend {backend!r}")
        out = out.reshape(*normed_hidden.shape[:-1], -1)
        out = out * torch.sigmoid(attn.gate_proj(normed_hidden))
        return attn.o_proj(out)

    @torch.no_grad()
    def layer_tail(self, hidden: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """Sandwich norms, residuals and MLP (HF ``MuseGlimmerTextDecoderLayer.forward``)."""
        layer = self.layer
        h = hidden + layer.post_attention_layernorm(attn_out)
        y = layer.pre_feedforward_layernorm(h)
        y = layer.mlp(y)
        y = layer.post_feedforward_layernorm(y)
        return h + y

    # --------------------------------------------------------------- full ops

    @torch.no_grad()
    def forward_hf(self, hidden: torch.Tensor, *, past_len: int = 0) -> torch.Tensor:
        """Untouched HF layer forward with HF-built masks (small sequences only)."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        seq = hidden.shape[1]
        position_ids = torch.arange(past_len, past_len + seq).unsqueeze(0)
        mask_fn = create_sliding_window_causal_mask if self.sliding_window else create_causal_mask
        previous_impl = self.text_config._attn_implementation
        self.text_config._attn_implementation = "eager"
        try:
            mask = mask_fn(
                config=self.text_config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids,
                allow_is_causal_skip=False,
            )
            position_embeddings = None
            if self.rope_theta is not None:
                position_embeddings = self._rope(position_ids.squeeze(0))
            return self.layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=position_ids,
            )
        finally:
            self.text_config._attn_implementation = previous_impl

    @torch.no_grad()
    def forward_hf_decode_step(self, hidden: torch.Tensor, new_token: torch.Tensor) -> torch.Tensor:
        """One decode step through the untouched HF layer, using HF's own cache and mask.

        Control for :meth:`decode`: prefills ``hidden`` into a real ``DynamicCache`` via
        ``layer.forward``, then runs ``new_token`` as a second call with HF's mask builder
        given that cache. Only valid while ``hidden`` is shorter than the sliding window (a
        longer prompt would let HF's own cache truncate, which changes the comparison).
        """
        from transformers import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        past_len = hidden.shape[1]
        if self.sliding_window is not None and past_len >= self.sliding_window:
            raise ValueError("use a prompt shorter than the sliding window for this control")
        mask_fn = create_sliding_window_causal_mask if self.sliding_window else create_causal_mask
        previous_impl = self.text_config._attn_implementation
        self.text_config._attn_implementation = "eager"
        try:
            cache = DynamicCache(config=self.text_config)
            for tokens, offset in ((hidden, 0), (new_token, past_len)):
                position_ids = torch.arange(offset, offset + tokens.shape[1]).unsqueeze(0)
                mask = mask_fn(
                    config=self.text_config,
                    inputs_embeds=tokens,
                    attention_mask=None,
                    past_key_values=cache,
                    position_ids=position_ids,
                    allow_is_causal_skip=False,
                )
                position_embeddings = self._rope(position_ids.squeeze(0)) if self.rope_theta is not None else None
                out = self.layer(
                    tokens,
                    position_embeddings=position_embeddings,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                )
            return out
        finally:
            self.text_config._attn_implementation = previous_impl

    @torch.no_grad()
    def prefill(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int = 0,
        q_chunk: int | None = None,
        backend: str = "sdpa",
        keep_kv: bool = True,
        q_chunk_filter=None,
        on_chunk=None,
    ):
        """Chunked prefill reference.

        Args:
            hidden: ``[batch, seq, hidden]`` float32 layer input.
            q_chunk: query-block size; chosen from the sequence length when omitted.
            q_chunk_filter: optional ``(start, end) -> bool``; when given, only those
                query blocks are computed (used to keep the 131072-token reference
                affordable). Skipped blocks are returned as ``None``.
            on_chunk: optional ``(start, end, out_chunk) -> None`` callback. When set,
                chunk outputs are streamed to it and not concatenated (bounded memory).

        Returns:
            ``(out, k, v)`` where ``out`` is ``[batch, seq, hidden]`` (or ``None`` when
            ``on_chunk`` is used) and ``k``/``v`` are ``[batch, kv_heads, seq, head_dim]``
            caches for positions ``[start_pos, start_pos + seq)``.
        """
        batch, seq, _ = hidden.shape
        positions = torch.arange(start_pos, start_pos + seq)
        normed = self.input_norm(hidden)

        kv_chunk = 8192
        k_parts, v_parts = [], []
        for s in range(0, seq, kv_chunk):
            e = min(s + kv_chunk, seq)
            k_part, v_part = self.compute_kv(normed[:, s:e], positions[s:e])
            k_parts.append(k_part)
            v_parts.append(v_part)
        k = torch.cat(k_parts, dim=2) if len(k_parts) > 1 else k_parts[0]
        v = torch.cat(v_parts, dim=2) if len(v_parts) > 1 else v_parts[0]
        del k_parts, v_parts

        if q_chunk is None:
            q_chunk = self._default_q_chunk(seq)

        out_parts = []
        for s in range(0, seq, q_chunk):
            e = min(s + q_chunk, seq)
            if q_chunk_filter is not None and not q_chunk_filter(s, e):
                out_parts.append(None)
                continue
            q_pos = positions[s:e]
            if self.sliding_window is not None:
                kv_lo = max(0, s - self.sliding_window + 1)
            else:
                kv_lo = 0
            kv_hi = e
            q = self.compute_q(normed[:, s:e], q_pos)
            attn_out = self.attention_out(
                normed[:, s:e],
                q,
                k[:, :, kv_lo:kv_hi],
                v[:, :, kv_lo:kv_hi],
                q_positions=q_pos,
                kv_positions=positions[kv_lo:kv_hi],
                backend=backend,
            )
            del q
            out_chunk = self.layer_tail(hidden[:, s:e], attn_out)
            del attn_out
            if on_chunk is not None:
                on_chunk(s, e, out_chunk)
                out_parts.append(None)
                del out_chunk
            else:
                out_parts.append(out_chunk)

        out = None
        if on_chunk is None and all(part is not None for part in out_parts):
            out = torch.cat(out_parts, dim=1)
        elif on_chunk is None:
            out = out_parts  # sparse coverage: caller inspects the parts it asked for
        if not keep_kv:
            return out, None, None
        return out, k, v

    def _default_q_chunk(self, seq: int) -> int:
        """Query-block size that keeps reference attention memory bounded."""
        if self.sliding_window is not None:
            return min(seq, 4096)
        # Full attention: the mask is q_chunk * seq booleans and SDPA works block-wise.
        target = max(128, int(2**28 / max(seq, 1)))
        target = min(target, 4096, seq)
        return max(128, (target // 128) * 128) if target >= 128 else seq

    @torch.no_grad()
    def decode(
        self,
        hidden: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        *,
        backend: str = "sdpa",
    ):
        """One decode step.

        Args:
            hidden: ``[batch, 1, hidden]`` layer input for the new token.
            k, v: ``[batch, kv_heads, cache_len, head_dim]`` caches. The new token's K/V
                are written at ``positions[b]`` (the caches are updated in place, so the
                buffers must be long enough).
            positions: ``[batch]`` absolute positions of the new token.

        Returns:
            ``[batch, 1, hidden]`` layer output.
        """
        batch = hidden.shape[0]
        normed = self.input_norm(hidden)
        k_new, v_new = self.compute_kv(normed, positions.reshape(batch, 1))
        for b in range(batch):
            k[b, :, positions[b]] = k_new[b, :, 0]
            v[b, :, positions[b]] = v_new[b, :, 0]

        q = self.compute_q(normed, positions.reshape(batch, 1))
        outs = []
        for b in range(batch):
            pos = int(positions[b])
            lo = max(0, pos - self.sliding_window + 1) if self.sliding_window is not None else 0
            kv_pos = torch.arange(lo, pos + 1)
            attn_out = self.attention_out(
                normed[b : b + 1],
                q[b : b + 1],
                k[b : b + 1, :, lo : pos + 1],
                v[b : b + 1, :, lo : pos + 1],
                q_positions=positions[b : b + 1],
                kv_positions=kv_pos,
                backend=backend,
            )
            outs.append(self.layer_tail(hidden[b : b + 1], attn_out))
        return torch.cat(outs, dim=0)


@torch.no_grad()
def stacked_layer_input(
    text_config,
    layer_idx: int,
    token_ids: torch.Tensor,
    *,
    model_id: str = HF_MODEL_ID,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Real activation entering ``layer_idx``: real embeddings run through layers ``0..idx``.

    Used by the real-weight tests so the input distribution is the model's own, not a
    synthetic approximation.
    """
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerRMSNorm

    hidden = load_real_embedding_rows(token_ids, model_id=model_id)
    embed_norm = MuseGlimmerRMSNorm(eps=text_config.rms_norm_eps, with_scale=False)
    hidden = embed_norm(hidden)
    for idx in range(layer_idx):
        layer = ReferenceDecoderLayer(text_config, idx, load_real_layer_state_dict(idx, model_id=model_id))
        hidden, _, _ = layer.prefill(hidden, backend=backend, keep_kv=False)
        del layer
    return hidden


class StreamingPCC:
    """Pearson correlation accumulated over blocks, so long-context PCC never needs the
    whole golden tensor in memory at once.

    Uses the same definition as ``models.common.utility_functions.comp_pcc``
    (``torch.corrcoef`` of the flattened pair); ``test_streaming_pcc_matches_comp_pcc``
    checks the two agree.
    """

    def __init__(self):
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0
        self.max_abs_err = 0.0

    def update(self, golden: torch.Tensor, calculated: torch.Tensor) -> None:
        x = golden.reshape(-1).to(torch.float64)
        y = calculated.reshape(-1).to(torch.float64)
        if x.numel() != y.numel():
            raise ValueError(f"shape mismatch: {golden.shape} vs {calculated.shape}")
        self.n += x.numel()
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())
        self.max_abs_err = max(self.max_abs_err, float((x - y).abs().max()))

    @property
    def pcc(self) -> float:
        n = self.n
        if n == 0:
            raise ValueError("no samples")
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        vx = self.sxx / n - (self.sx / n) ** 2
        vy = self.syy / n - (self.sy / n) ** 2
        if vx <= 0 or vy <= 0:
            return 1.0 if cov == 0 else 0.0
        return cov / math.sqrt(vx * vy)


def configure_host_threads() -> None:
    """Use every core for the (matmul-bound) reference."""
    torch.set_num_threads(int(os.environ.get("MUSE_GLIMMER_REF_THREADS", os.cpu_count() or 8)))
