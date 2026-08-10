# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the ACE-Step 1.5 condition encoder — **Block 2**, 608.37 M params.

Reference, mirrored op-for-op:
``python_env/lib/python3.12/site-packages/diffusers/pipelines/ace_step/modeling_ace_step.py``
  * ``AceStepConditionEncoder``  — ``text_projector`` + ``lyric_encoder`` + ``timbre_encoder`` + packing
  * ``AceStepLyricEncoder``      — Linear(1024->2048, **bias**) + 8 x encoder layer + RMSNorm
  * ``AceStepTimbreEncoder``     — Linear(64->2048, **bias**)   + 4 x encoder layer + RMSNorm + pool
  * ``AceStepEncoderLayer``      — pre-LN, **no adaLN, no gates**

Param counts verified against the live reference (2026-07-31):
``lyric 404.789248 M + timbre 201.481216 M + text_projector 2.097152 M = 608.369664 M``.

Encoder layer — the *only* structural difference from the DiT block is pre-LN instead of adaLN,
which is why the attention itself comes from Block 1 unchanged::

    x = x + self_attn(input_layernorm(x))       # GQA 16/8, head_dim 128, QK-RMSNorm, RoPE 1e6
    x = x + mlp(post_attention_layernorm(x))    # SwiGLU 2048 -> 6144 -> 2048, no bias
    ...                                         # then one final `norm` after the stack

⚠ **The encoder layers use Qwen-style norm names, not the DiT's.** They are ``input_layernorm``
and ``post_attention_layernorm`` — *not* the DiT block's ``self_attn_norm`` / ``mlp_norm``. Nothing
in the master doc says so. The child names in this file match the reference exactly; verified by
loading the live reference state dict with ``strict=True`` (0 missing, 0 unexpected keys over all
103 parameters / 142 reference tensors).

``layer_types[i] = "sliding_attention" if (i+1) % 2 else "full_attention"`` — even layers get the
symmetric ``|i-j| <= 128`` band, odd layers are unmasked. Identical to the DiT, so
``AceStepDiTConfig.window_for_layer`` is reused verbatim.

Output: ``encoder_hidden_states [1, enc_L, 2048]`` with ``enc_L = L_lyr + n_timbre + L_txt``
(``n_timbre == 1`` at batch 1; architectural worst case 2305).

⚠ **Measured ``enc_L`` is 102-103, not the 400-700 the master doc estimates.** Block 0's goldens
give ``L_lyr = 32``, ``L_txt = 70``, ``n_timbre = 1`` -> ``enc_L = 103`` at S=128, and 102 at the
durations whose caption prompt tokenises the duration string to 69 rather than 70 tokens. So
``enc_L`` moves with the *caption text*, not with the audio duration, and it varies by a single
token between otherwise identical runs. Consequence for Block 4: a fixed-``enc_L`` trace wants
**one padded bucket** (e.g. 128) with a key mask, not a menu of per-duration shapes — the shape
does not correlate with duration at all.

What is reused from Block 1
---------------------------
``ttnn_ace_step_attention.AceStepSelfAttention``  the whole self-attention (fused QKV +
    ``nlp_create_qkv_heads`` + QK-RMSNorm + HF RoPE + windowed SDPA + ``to_out``), including its
    ``to_q``/``to_k``/``to_v`` -> ``qkv`` state-dict fold.
``ttnn_ace_step_common``  ``AceStepDiTConfig`` (layer types, TRAP-1 window, widths),
    ``make_linear`` / ``make_rms_norm``, ``build_rope_tables``, ``linear_compute_config`` /
    ``norm_compute_config`` / ``sdpa_compute_config``, ``to_device`` / ``to_host``.
``models.tt_dit.layers``  ``Module`` / ``ModuleList`` (state-dict loading), ``Linear`` (with the
    fused-SwiGLU packing), ``RMSNorm``.
Nothing in Block 1's files is edited. The one API addition we would like is recorded in
``BLOCK1_REQUESTS`` below and in ``ACE_STEP_1_5_COND_ENCODER.md``.

Traps honoured
--------------
* **TRAP-1** — the SDPA window is the *total* width; we only ever pass
  ``AceStepDiTConfig.window_for_layer(i)``, which is ``2 * sliding_window`` (256) or ``None``.
* **TRAP-2** — SDPA expands GQA in-kernel; K/V are never repeated (inherited from Block 1).
* **TRAP-5** — every batch-1 shortcut is tagged ``# BATCH-1 ASSUMPTION``.

Batch-1 shortcuts (grep ``BATCH-1 ASSUMPTION``)
-----------------------------------------------
* ``_pack_sequences`` (stable descending argsort + gather) is the **identity** when the mask is
  all-ones, which it is at B=1 with ``padding="longest"``. Implemented as a plain concat — no
  device-side sort. Verified numerically against the reference.
* ``unpack_timbre_embeddings`` (bincount / argsort / cumsum / one_hot + ``one_hot.T @ x``) is an
  **identity reshape** when ``refer_audio_order_mask == arange(1)``. Verified numerically.
* ``n_refs == 1``, so the timbre branch contributes exactly one token.

Sequence padding
----------------
``L_lyr``, ``L_txt`` and ``timbre_fix_frame`` (750) are generally **not** tile-aligned. Two modes:

``pad_mode="logical"`` (default)
    Tensors keep their true logical length; TTNN tile-pads implicitly with zeros. The non-causal
    SDPA kernel then *generates its own padding mask* for the tile remainder — see
    ``sdpa_program_factory.cpp`` (``use_padded_mask``: "If no mask provided: writer generates a
    mask with 0 for valid K and -inf for padded K"). This composes with ``sliding_window_size``,
    so no dense mask tensor is ever materialised and Block 1's attention is used unchanged.
``pad_mode="dense_mask"``
    Escape hatch. Zero-pads to a tile multiple and bakes ``|i-j| <= sliding_window`` *and* the
    key-padding mask into one dense ``[1, 1, S, S]`` additive mask (mirroring
    ``_create_4d_mask``), because the kernel refuses ``attn_mask`` and ``sliding_window_size``
    together. Uses ``_self_attention_with_mask``, which drives Block 1's own submodules.

Dead weight
-----------
``timbre_encoder.special_token [1, 1, 2048]`` is a real checkpoint tensor that is **never used**:
the concat is commented out upstream and the pooled vector is ``hidden_states[:, 0, :]``, the
first *audio* frame. Reproduced, and the tensor is dropped at load — see
``AceStepTimbreEncoder._prepare_torch_state``. ``null_condition_emb`` and ``silence_latent`` also
live on ``condition_encoder`` but belong to Blocks 1/4; they are stashed on the host, not uploaded.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import torch

import ttnn
from models.tt_dit.layers.module import Module, ModuleList

from .ttnn_ace_step_attention import AceStepSelfAttention
from .ttnn_ace_step_dit import AceStepMLP
from .ttnn_ace_step_common import (
    AceStepDiTConfig,
    apply_rope,
    build_rope_tables,
    linear_compute_config,
    make_linear,
    make_rms_norm,
    norm_compute_config,
    sdpa_compute_config,
    to_device,
    to_host,
)

TILE = 32

#: API additions we would like in Block 1's files, reported to the orchestrator rather than
#: patched in (Block 1 owns those files). Each is currently worked around locally.
BLOCK1_REQUESTS = (
    "ttnn_ace_step_attention.AceStepSelfAttention.forward: accept `attn_mask=None` and pass it "
    "through to SDPA (mutually exclusive with `window`). Worked around by "
    "`_self_attention_with_mask`, which drives the same submodules. Needed only for "
    "pad_mode='dense_mask'; the default pad_mode='logical' needs no change.",
    "ttnn_ace_step_dit.AceStepMLP: accept a `dtype` kwarg (it currently hard-wires bfloat16 via "
    "tt_dit's FeedForward, which does not thread dtype to its Linears). Block 2 imports it as-is, "
    "so with ACE_STEP_COND_DTYPE=float32 the MLP weights stay bf16 while everything else is fp32 — "
    "usable as an activation-precision bisect, but not a true fp32 run. Also worth moving out of "
    "dit.py into common.py, since both blocks use it.",
    "ttnn_ace_step_common.AceStepDiTConfig: `num_hidden_layers` is DiT-specific. Block 2 builds "
    "one config per encoder stack (8 and 4 layers) purely to reuse `window_for_layer`; a "
    "`layer_types_for(n)` free function would express that better.",
)


# --------------------------------------------------------------------------------------------- #
#                                        configuration                                          #
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AceStepCondConfig:
    """Mirrors ``AceStepConditionEncoder.__init__`` defaults (the deployed 2 B turbo config)."""

    hidden_size: int = 2048
    intermediate_size: int = 6144
    text_hidden_dim: int = 1024
    timbre_hidden_dim: int = 64
    num_lyric_encoder_hidden_layers: int = 8
    num_timbre_encoder_hidden_layers: int = 4
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    sliding_window: int = 128
    #: Reference audio is always resampled/segmented to exactly 30 s at 25 Hz.
    timbre_fix_frame: int = 750

    def stack_config(self, num_layers: int) -> AceStepDiTConfig:
        """Block 1's ``AceStepDiTConfig`` for one encoder stack.

        Built only so that ``resolved_layer_types`` / ``window_for_layer`` / ``sdpa_window_size``
        (TRAP-1) / ``qkv_width`` come from the single place that defines them. The encoder shares
        every one of those fields with the DiT; only the layer *count* differs.
        """
        return AceStepDiTConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            attention_bias=self.attention_bias,
            rms_norm_eps=self.rms_norm_eps,
            sliding_window=self.sliding_window,
        )


def _pad_to_tile(n: int) -> int:
    return -(-n // TILE) * TILE


# --------------------------------------------------------------------------------------------- #
#                          dense mask + masked attention (escape hatch)                         #
# --------------------------------------------------------------------------------------------- #


def build_encoder_mask(
    mesh_device,
    seq_len: int,
    padded_len: int,
    *,
    sliding_window: int | None,
) -> ttnn.Tensor:
    """``[1, 1, padded_len, padded_len]`` additive bf16 mask (0.0 keep, ``finfo.min`` masked).

    Mirrors ``_create_4d_mask(..., is_causal=False)``: a symmetric ``|i-j| <= sliding_window`` band
    when a window is given, intersected with the key-padding mask. The reference's own padding mask
    is all-ones at B=1 (``padding="longest"``), so the padding masked here is purely *our* tile
    remainder. SDPA requires a BF16/BFP8/BFP4 mask, TILE layout, in DRAM.
    """
    idx = torch.arange(padded_len)
    valid = torch.ones((padded_len, padded_len), dtype=torch.bool)
    if sliding_window is not None:
        valid &= (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= sliding_window
    valid &= (idx < seq_len).unsqueeze(0)  # pad columns are never attended
    mask = torch.full((1, 1, padded_len, padded_len), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    mask.masked_fill_(valid.unsqueeze(0).unsqueeze(0), 0.0)
    return to_device(mask, mesh_device, dtype=ttnn.bfloat16)


def _self_attention_with_mask(attn: AceStepSelfAttention, x_11SC: ttnn.Tensor, rope, attn_mask: ttnn.Tensor):
    """``AceStepSelfAttention.forward`` with a dense ``attn_mask`` instead of ``window``.

    Drives Block 1's *own* submodules (``attn.qkv`` / ``norm_q`` / ``norm_k`` / ``to_out``) and its
    compute configs, so there is exactly one weight tree and one set of conventions. Delete this
    the moment ``AceStepSelfAttention.forward`` grows an ``attn_mask`` kwarg — see
    ``BLOCK1_REQUESTS``.
    """
    cfg = attn.config
    qkv = attn.qkv(x_11SC, compute_kernel_config=attn.mm_compute_config)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(qkv)
    q = attn.norm_q(q, compute_kernel_config=attn.norm_compute_config)
    k = attn.norm_k(k, compute_kernel_config=attn.norm_compute_config)
    cos, sin = rope
    q = apply_rope(q, cos, sin, composite=attn.rope_composite)
    k = apply_rope(k, cos, sin, composite=attn.rope_composite)
    # TRAP-2: no repeat_interleave — SDPA expands GQA in-kernel.
    out = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,  # band + key padding baked in; sliding_window_size must stay unset
        is_causal=False,
        scale=cfg.attention_scale,
        program_config=attn.sdpa_program_config,
        compute_kernel_config=attn.sdpa_compute_config,
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    out = ttnn.experimental.nlp_concat_heads(out)
    return attn.to_out(out, compute_kernel_config=attn.mm_compute_config)


# --------------------------------------------------------------------------------------------- #
#                                          modules                                              #
# --------------------------------------------------------------------------------------------- #


class AceStepEncoderLayer(Module):
    """``AceStepEncoderLayer`` — pre-LN, NO adaLN, NO gates. 4D ``[1, 1, S, hidden]`` in and out."""

    def __init__(
        self,
        cond_config: AceStepCondConfig,
        stack_config: AceStepDiTConfig,
        *,
        mesh_device,
        dtype: ttnn.DataType = ttnn.bfloat16,
        rope_composite: bool = False,
        sdpa_program_config=None,
    ) -> None:
        super().__init__()
        self.cond_config = cond_config
        # Block 1's self-attention, unmodified: fused QKV, QK-RMSNorm, HF RoPE, windowed SDPA.
        self.self_attn = AceStepSelfAttention(
            stack_config,
            mesh_device=mesh_device,
            dtype=dtype,
            rope_composite=rope_composite,
            sdpa_program_config=sdpa_program_config,
        )
        self.input_layernorm = make_rms_norm(
            cond_config.hidden_size, eps=cond_config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype
        )
        self.post_attention_layernorm = make_rms_norm(
            cond_config.hidden_size, eps=cond_config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype
        )
        # Block 1's SwiGLU MLP, unmodified: gate_proj|up_proj folded into one fused-SwiGLU Linear
        # (`cat([up, gate], dim=0)`, because prepare_for_fused_swiglu defaults to gate_is_first=False).
        # It is byte-identical to what the DiT block needs, so there is nothing to specialise.
        self.mlp = AceStepMLP(stack_config, mesh_device=mesh_device)

        self.mm_compute_config = linear_compute_config(mesh_device, dtype)
        self.norm_compute_config = norm_compute_config(mesh_device)

    def forward(self, x_11SC: ttnn.Tensor, *, rope, window: int | None, attn_mask=None) -> ttnn.Tensor:
        """⚠ **Consumes** ``x_11SC``: the input is deallocated once its residual add is done, so a
        12-layer stack peaks at two live activations rather than 25. Do not call twice on the same
        tensor, and do not pass a tensor the caller still needs."""
        assert attn_mask is None or window is None, "SDPA rejects attn_mask together with a sliding window"

        h = self.input_layernorm(x_11SC, compute_kernel_config=self.norm_compute_config)
        if attn_mask is None:
            h = self.self_attn(h, rope=rope, window=window)
        else:
            h = _self_attention_with_mask(self.self_attn, h, rope, attn_mask)
        x = ttnn.add(x_11SC, h)
        ttnn.deallocate(h)
        ttnn.deallocate(x_11SC)

        h = self.post_attention_layernorm(x, compute_kernel_config=self.norm_compute_config)
        h = self.mlp(h, compute_kernel_config=self.mm_compute_config)
        out = ttnn.add(x, h)
        ttnn.deallocate(h)
        ttnn.deallocate(x)
        return out


class AceStepEncoderStack(Module):
    """Shared body of ``AceStepLyricEncoder`` / ``AceStepTimbreEncoder``:
    ``embed_tokens`` (Linear **with** bias) -> N x pre-LN layer -> final ``norm``.
    """

    def __init__(
        self,
        cond_config: AceStepCondConfig,
        *,
        num_layers: int,
        in_features: int,
        mesh_device,
        dtype: ttnn.DataType = ttnn.bfloat16,
        pad_mode: str = "logical",
        rope_composite: bool | None = None,
        sdpa_program_config=None,
    ) -> None:
        super().__init__()
        assert pad_mode in ("logical", "dense_mask"), pad_mode
        rope_composite = (dtype == ttnn.float32) if rope_composite is None else rope_composite
        self.cond_config = cond_config
        self.stack_config = cond_config.stack_config(num_layers)
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.pad_mode = pad_mode

        self.embed_tokens = make_linear(
            in_features, cond_config.hidden_size, bias=True, mesh_device=mesh_device, dtype=dtype
        )
        layers = [
            AceStepEncoderLayer(
                cond_config,
                self.stack_config,
                mesh_device=mesh_device,
                dtype=dtype,
                rope_composite=rope_composite,
                sdpa_program_config=sdpa_program_config,
            )
            for _ in range(num_layers)
        ]
        self.layers = ModuleList(layers)
        self._ordered_layers = layers  # plain list: iteration order without relying on dict keys
        self.norm = make_rms_norm(
            cond_config.hidden_size, eps=cond_config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype
        )

        self.mm_compute_config = linear_compute_config(mesh_device, dtype)
        self.norm_compute_config = norm_compute_config(mesh_device)
        # sdpa_compute_config is owned by AceStepSelfAttention; touched here only so a reader can
        # see that the encoder does not override Block 1's choice.
        self.sdpa_compute_config = sdpa_compute_config(mesh_device)

    def forward(self, x_11SC: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """``x_11SC`` ``[1, 1, S, in_features]`` -> ``[1, 1, S, hidden]``.

        ``seq_len`` is the *true* length. In ``pad_mode="logical"`` it equals ``S``; in
        ``pad_mode="dense_mask"`` ``S`` is ``seq_len`` rounded up to a tile and the extra rows are
        masked out as keys (their own output rows are garbage the caller slices off).
        """
        padded_len = x_11SC.shape[-2]
        x = self.embed_tokens(x_11SC, compute_kernel_config=self.mm_compute_config)

        # rotary_embedding_hf requires bf16; in fp32 mode the layers use apply_rope(composite=True),
        # which works at any dtype, so the tables must follow the activation dtype.
        rope = build_rope_tables(
            self.mesh_device,
            padded_len,
            head_dim=self.cond_config.head_dim,
            theta=self.cond_config.rope_theta,
            dtype=self.dtype,
        )
        masks = self._build_masks(seq_len, padded_len)
        try:
            for i, layer in enumerate(self._ordered_layers):
                window = self.stack_config.window_for_layer(i)
                if self.pad_mode == "dense_mask":
                    x = layer(x, rope=rope, window=None, attn_mask=masks[window is not None])
                else:
                    x = layer(x, rope=rope, window=window)
        finally:
            for t in (*rope, *(m for m in masks.values() if m is not None)):
                ttnn.deallocate(t)

        out = self.norm(x, compute_kernel_config=self.norm_compute_config)
        ttnn.deallocate(x)
        return out

    def _build_masks(self, seq_len: int, padded_len: int) -> dict[bool, ttnn.Tensor]:
        """``{True: sliding mask, False: full mask}`` for ``pad_mode="dense_mask"``, else empty."""
        if self.pad_mode != "dense_mask":
            return {}
        return {
            True: build_encoder_mask(
                self.mesh_device, seq_len, padded_len, sliding_window=self.cond_config.sliding_window
            ),
            False: build_encoder_mask(self.mesh_device, seq_len, padded_len, sliding_window=None),
        }


class AceStepLyricEncoder(AceStepEncoderStack):
    """``AceStepLyricEncoder`` — 8 layers, 404.79 M params. Input is the Qwen3 ``embed_tokens``
    lookup ``[1, L_lyr, 1024]`` (the lyric branch never runs the text encoder's transformer)."""


class AceStepTimbreEncoder(AceStepEncoderStack):
    """``AceStepTimbreEncoder`` — 4 layers, 201.48 M params. Input is reference-audio VAE latents
    ``[1, 750, 64]`` (``silence_latent[:, :750, :]`` for text2music)."""

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # DEAD WEIGHT: `special_token` is a real checkpoint tensor that is never read — the concat
        # that would have prepended it is commented out upstream, and the pooled vector is
        # hidden_states[:, 0, :], i.e. the first *audio* frame. Drop it so `strict=True` loading
        # does not trip on an unexpected key.
        self.special_token_unused = state.pop("special_token", None)


class TTNNAceStepConditionEncoder(Module):
    """``AceStepConditionEncoder`` — 608.37 M params. Runs **once** per generation.

    ``forward`` takes host torch tensors (the Qwen3 text encoder stays on host, and the timbre
    latents come from the VAE or ``silence_latent``) and returns the device
    ``encoder_hidden_states [1, enc_L, 2048]`` plus the host ``encoder_attention_mask [1, enc_L]``.
    """

    def __init__(
        self,
        config: AceStepCondConfig | None = None,
        *,
        mesh_device,
        dtype: ttnn.DataType = ttnn.bfloat16,
        pad_mode: str = "logical",
        rope_composite: bool | None = None,
        sdpa_program_config=None,
    ) -> None:
        super().__init__()
        self.config = config = config or AceStepCondConfig()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.pad_mode = pad_mode
        #: populated from the checkpoint buffer at load time; owned by Block 4 (the pipeline),
        #: never uploaded here.
        self.silence_latent: torch.Tensor | None = None
        self.null_condition_emb: torch.Tensor | None = None

        stack_kwargs = dict(
            mesh_device=mesh_device,
            dtype=dtype,
            pad_mode=pad_mode,
            rope_composite=rope_composite,
            sdpa_program_config=sdpa_program_config,
        )
        self.text_projector = make_linear(
            config.text_hidden_dim,
            config.hidden_size,
            bias=False,  # the only projection in the model with no bias
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.lyric_encoder = AceStepLyricEncoder(
            config,
            num_layers=config.num_lyric_encoder_hidden_layers,
            in_features=config.text_hidden_dim,
            **stack_kwargs,
        )
        self.timbre_encoder = AceStepTimbreEncoder(
            config,
            num_layers=config.num_timbre_encoder_hidden_layers,
            in_features=config.timbre_hidden_dim,
            **stack_kwargs,
        )
        self.mm_compute_config = linear_compute_config(mesh_device, dtype)

    # ------------------------------------------------------------------------ state dict loading

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Accept either a bare condition_encoder state dict or a full-model one.
        for prefix in ("condition_encoder.", "encoder."):
            if any(k.startswith(prefix) for k in state) and not any(k.startswith("lyric_encoder.") for k in state):
                for k in [k for k in state if k.startswith(prefix)]:
                    state[k.removeprefix(prefix)] = state.pop(k)
                break
        # Buffers that live on condition_encoder but belong to the solver / pipeline.
        self.null_condition_emb = state.pop("null_condition_emb", None)
        self.silence_latent = state.pop("silence_latent", None)

    # ------------------------------------------------------------------------------- upload/pack

    def _upload(self, x: torch.Tensor) -> tuple[ttnn.Tensor, int]:
        """``[1, L, D]`` host torch -> ``[1, 1, S, D]`` device TILE. ``S == L`` in ``pad_mode
        ="logical"``; rounded up to a tile in ``pad_mode="dense_mask"``."""
        seq_len = x.shape[1]
        if self.pad_mode == "dense_mask":
            padded = _pad_to_tile(seq_len)
            if padded != seq_len:
                x = torch.nn.functional.pad(x, (0, 0, 0, padded - seq_len))
        x = x.reshape(1, 1, x.shape[1], x.shape[2]).contiguous().to(torch.float32)
        return to_device(x, self.mesh_device, dtype=self.dtype), seq_len

    @staticmethod
    def _first_rows(x: ttnn.Tensor, n: int) -> ttnn.Tensor:
        """First ``n`` sequence rows of ``[1, 1, S, C]``, in ROW_MAJOR so a non-tile-aligned ``n``
        (the common case) is a plain memcpy rather than a tilized gather."""
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.slice(rm, [0, 0, 0, 0], [1, 1, n, x.shape[-1]])

    # ------------------------------------------------------------------------------------ forward

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        *,
        text_attention_mask: torch.Tensor | None = None,
        lyric_attention_mask: torch.Tensor | None = None,
        refer_audio_order_mask: torch.Tensor | None = None,
        assemble_on_host: bool = False,
        return_parts: bool = False,
    ):
        """Args mirror ``AceStepConditionEncoder.forward``.

        * ``text_hidden_states``  ``[1, L_txt, 1024]``  — Qwen3-Embedding-0.6B output (host).
        * ``lyric_hidden_states`` ``[1, L_lyr, 1024]``  — Qwen3 ``embed_tokens`` lookup (host).
        * ``refer_audio_acoustic_hidden_states_packed`` ``[1, 750, 64]`` — VAE latents (host).

        Returns ``(encoder_hidden_states, encoder_attention_mask)``; with ``return_parts=True`` also
        the three un-concatenated device pieces (padded), for oracle PCCs and for Block 4's reuse.
        """
        self._check_batch1(
            text_hidden_states,
            lyric_hidden_states,
            refer_audio_acoustic_hidden_states_packed,
            text_attention_mask,
            lyric_attention_mask,
            refer_audio_order_mask,
        )
        l_txt = text_hidden_states.shape[1]
        l_lyr = lyric_hidden_states.shape[1]

        # 1. text_projector — Linear(1024 -> 2048, bias=False). No transformer on the text branch.
        text_in, _ = self._upload(text_hidden_states)
        text_out = self.text_projector(text_in, compute_kernel_config=self.mm_compute_config)
        ttnn.deallocate(text_in)

        # 2. lyric encoder — 8 pre-LN layers.
        lyric_in, lyric_len = self._upload(lyric_hidden_states)
        lyric_out = self.lyric_encoder(lyric_in, lyric_len)
        ttnn.deallocate(lyric_in)

        # 3. timbre encoder — 4 pre-LN layers, then CLS-like pooling on the FIRST AUDIO FRAME.
        #    `special_token` is never concatenated (commented out upstream), so `[:, 0, :]` is
        #    audio frame 0, not a learned CLS position.
        timbre_in, timbre_len = self._upload(refer_audio_acoustic_hidden_states_packed)
        timbre_out = self.timbre_encoder(timbre_in, timbre_len)
        ttnn.deallocate(timbre_in)

        # 4. packing. BATCH-1 ASSUMPTION: both `_pack_sequences` calls degenerate to a plain concat
        #    (a stable descending argsort of an all-ones mask is the identity permutation), and
        #    `unpack_timbre_embeddings` degenerates to `pooled.reshape(1, 1, hidden)` when
        #    refer_audio_order_mask == arange(1). Both verified numerically against the reference.
        #    Order is lyric, then timbre, then text: enc_L = L_lyr + n_timbre + L_txt.
        n_timbre = 1  # BATCH-1 ASSUMPTION: one reference clip -> exactly one timbre token
        enc_l = l_lyr + n_timbre + l_txt

        if assemble_on_host:
            packed = torch.cat(
                [
                    to_host(lyric_out)[:, :, :l_lyr, :],
                    to_host(timbre_out)[:, :, :n_timbre, :],  # == hidden_states[:, 0, :]
                    to_host(text_out)[:, :, :l_txt, :],
                ],
                dim=2,
            )
            encoder_hidden_states = to_device(packed, self.mesh_device, dtype=self.dtype)
        else:
            pieces = [
                self._first_rows(lyric_out, l_lyr),
                self._first_rows(timbre_out, n_timbre),
                self._first_rows(text_out, l_txt),
            ]
            packed_rm = ttnn.concat(pieces, dim=2)
            for p in pieces:
                ttnn.deallocate(p)
            encoder_hidden_states = ttnn.to_layout(packed_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(packed_rm)

        # BATCH-1 ASSUMPTION: `padding="longest"` at B=1 makes every per-branch mask all-ones, so
        # the packed mask is all-ones too. The DiT overwrites it with None anyway (master §3.3); it
        # is returned only to keep the reference's signature.
        encoder_attention_mask = torch.ones(1, enc_l, dtype=torch.bool)

        if return_parts:
            return encoder_hidden_states, encoder_attention_mask, (lyric_out, timbre_out, text_out)
        for t in (lyric_out, timbre_out, text_out):
            ttnn.deallocate(t)
        return encoder_hidden_states, encoder_attention_mask

    # --------------------------------------------------------------------------------- guardrails

    def _check_batch1(self, text, lyric, timbre, text_mask, lyric_mask, order_mask) -> None:
        # BATCH-1 ASSUMPTION: every check below becomes a real argsort/gather at B > 1. See TRAP-5.
        c = self.config
        assert text.shape[0] == 1, f"BATCH-1 ASSUMPTION: text batch must be 1, got {text.shape[0]}"
        assert lyric.shape[0] == 1, f"BATCH-1 ASSUMPTION: lyric batch must be 1, got {lyric.shape[0]}"
        assert timbre.shape[0] == 1, f"BATCH-1 ASSUMPTION: n_refs must be 1, got {timbre.shape[0]}"
        assert text.shape[-1] == c.text_hidden_dim, f"text_hidden_dim mismatch: {text.shape[-1]}"
        assert lyric.shape[-1] == c.text_hidden_dim, f"lyric hidden dim mismatch: {lyric.shape[-1]}"
        assert timbre.shape[-1] == c.timbre_hidden_dim, f"timbre_hidden_dim mismatch: {timbre.shape[-1]}"
        for name, mask in (("text_attention_mask", text_mask), ("lyric_attention_mask", lyric_mask)):
            if mask is not None:
                assert bool(mask.reshape(-1).bool().all()), (
                    f"BATCH-1 ASSUMPTION: {name} must be all-ones (padding='longest' at B=1); a "
                    "ragged mask needs the real _pack_sequences argsort+gather path"
                )
        if order_mask is not None:
            flat = order_mask.reshape(-1).long()
            assert flat.numel() == 1 and torch.equal(flat, torch.zeros(1, dtype=torch.long)), (
                "BATCH-1 ASSUMPTION: refer_audio_order_mask must be arange(1) for the identity "
                f"unpack_timbre_embeddings path, got {flat.tolist()}"
            )


# --------------------------------------------------------------------------------------------- #
#                                  weight / reference loading                                   #
# --------------------------------------------------------------------------------------------- #

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "cond")

#: Same default as ``reference/ace_step_ref.py::DEFAULT_PIPELINE_PATH`` — the converted
#: diffusers-format checkpoint Block 0 produced. ``condition_encoder/config.json`` there matches
#: :class:`AceStepCondConfig`'s defaults field for field (verified 2026-07-31).
DEFAULT_PIPELINE_PATH = os.getenv("ACE_STEP_PIPELINE", "/localdev/acicovic/ace_step_diffusers")


def load_cond_state(path: str | None = None) -> dict[str, torch.Tensor]:
    """Fetch the torch ``condition_encoder`` state dict, in **diffusers** naming.

    Resolution order (first hit wins):
      1. explicit ``path`` (``.pt``/``.pth`` via ``torch.load``, ``.safetensors`` via safetensors);
      2. ``$ACE_STEP_COND_STATE``;
      3. ``$ACE_STEP_PIPELINE/condition_encoder/diffusion_pytorch_model.safetensors``
         (:data:`DEFAULT_PIPELINE_PATH`) — the real 608 M weights, fp32;
      4. Block 0's ``tt/ttnn_ace_step_weights.py`` — any of ``load_condition_encoder_state`` /
         ``load_cond_state`` / ``load_condition_encoder``;
      5. ``golden/cond/state_dict.pt``.

    The upstream -> diffusers key remap (``q_proj``->``to_q``, ``q_norm``->``norm_q``, ...) is
    Block 0's job (``ttnn_ace_step_weights.remap_upstream_to_diffusers``); see master doc §2.
    """
    if path:
        return _load_state_file(path)
    env = os.environ.get("ACE_STEP_COND_STATE")
    if env:
        return _load_state_file(env)
    safetensors_path = os.path.join(DEFAULT_PIPELINE_PATH, "condition_encoder", "diffusion_pytorch_model.safetensors")
    if os.path.exists(safetensors_path):
        return _load_state_file(safetensors_path)
    try:
        weights = importlib.import_module("models.experimental.ace_step_v15.tt.ttnn_ace_step_weights")
        for fn_name in ("load_condition_encoder_state", "load_cond_state", "load_condition_encoder"):
            fn = getattr(weights, fn_name, None)
            if callable(fn):
                return fn()
    except ImportError:
        pass
    fallback = os.path.join(GOLDEN_DIR, "state_dict.pt")
    if os.path.exists(fallback):
        return _load_state_file(fallback)
    raise FileNotFoundError(
        "No condition_encoder weights found. Provide one of: ACE_STEP_COND_STATE=<path>, "
        f"a diffusers pipeline at ACE_STEP_PIPELINE (looked in {safetensors_path}), "
        "Block 0's tt/ttnn_ace_step_weights.py, or golden/cond/state_dict.pt."
    )


def _load_state_file(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def reference_condition_encoder(config: AceStepCondConfig | None = None, seed: int = 1234):
    """Random-init fp32 ``diffusers.AceStepConditionEncoder`` at a fixed seed.

    Used by the block test until Block 0's goldens land — matching the tt_dit convention that
    block-level tests gate against random-init reference weights, with real weights behind a flag.
    """
    from diffusers.pipelines.ace_step.modeling_ace_step import AceStepConditionEncoder

    config = config or AceStepCondConfig()
    torch.manual_seed(seed)
    model = AceStepConditionEncoder(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        text_hidden_dim=config.text_hidden_dim,
        timbre_hidden_dim=config.timbre_hidden_dim,
        num_lyric_encoder_hidden_layers=config.num_lyric_encoder_hidden_layers,
        num_timbre_encoder_hidden_layers=config.num_timbre_encoder_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rope_theta=config.rope_theta,
        attention_bias=config.attention_bias,
        rms_norm_eps=config.rms_norm_eps,
        sliding_window=config.sliding_window,
    )
    return model.eval().to(torch.float32)
