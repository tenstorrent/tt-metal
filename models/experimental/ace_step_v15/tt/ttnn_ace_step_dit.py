# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ACE-Step 1.5 DiT (24-layer diffusion transformer) in TTNN.

Mirrors ``diffusers.models.transformers.ace_step_transformer`` op for op and consumes its
state dict directly. Both classes are ``models.tt_dit.layers.module.Module`` subclasses, so
weights load via ``load_torch_state_dict`` and round-trip through ``save`` / ``load``.

Block op order (master doc §3.3)::

     0. shift, scale, gate, c_shift, c_scale, c_gate = chunk(scale_shift_table + temb, 6, dim=1)
     1-2. h = self_attn_norm(x) * (1 + scale) + shift
     3-7. o = self_attn(h)                              # GQA + QK-norm + RoPE + windowed SDPA
     8.   x = x + o * gate                              # BARE gate, not o * (1 + gate)
     9.   h = cross_attn_norm(x)                        # RMSNorm only: no adaLN, no gate
    10.   o = cross_attn(h, encoder_kv)                 # no RoPE, mask=None
    11.   x = x + o                                     # plain residual
    12.   h = mlp_norm(x) * (1 + c_scale) + c_shift
    13.   h = down_proj(silu(gate_proj(h)) * up_proj(h))
    14.   x = x + h * c_gate

Deviations from the reference, all deliberate and all constant folds (§3.7)
--------------------------------------------------------------------------
* ``time_embed_r`` never runs. At inference ``timestep_r == timestep``, so it always sees
  ``t - t_r == 0`` and its output is a constant, folded into every ``scale_shift_table`` at
  load time by :func:`ttnn_ace_step_common.fold_time_embed_r`.
* The ``+1`` of ``(1 + scale)`` is folded into the modulation constants (see
  :class:`ttnn_ace_step_common.Modulation`).
* ``proj_in_conv`` / ``proj_out_conv`` are ``Linear``s, not convolutions: ``kernel == stride ==
  patch_size == 2`` with no padding, so they are non-overlapping and reduce to a row-major
  reshape plus a matmul. **This makes the DiT completely convolution-free.**
* The 256-wide timestep sinusoid is computed on the host in fp32 (it *must* be — see
  :func:`ttnn_ace_step_common.timestep_sinusoid`).
* Cross-attention K/V for all 24 layers are computed once via :meth:`precompute_cross_kv` and
  reused across denoising steps (TRAP-5).

Shape conventions
-----------------
Everything on device is 4D. ``[1, 1, T, C]`` for latent-rate tensors, ``[1, 1, S, hidden]``
for the layer stack (``S = ceil(T / 2)``), ``[1, heads, S, head_dim]`` inside attention. The
golden contract puts ``proj_in_conv`` / ``proj_out_conv`` in **NCL** ``[B, C, T]``; callers
transpose on the host.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.layers.embeddings import TimestepEmbedding
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.utils.substate import pop_substate

from .ttnn_ace_step_attention import AceStepCrossAttention, AceStepSelfAttention, capture_tensor
from .ttnn_ace_step_common import (
    TILE,
    AceStepDiTConfig,
    # AceStepMLP lives in common.py (the condition encoder's layers use the same MLP);
    # re-exported here so `from ...ttnn_ace_step_dit import AceStepMLP` keeps working.
    AceStepMLP,
    Modulation,
    build_rope_tables,
    conv1d_patch_to_linear,
    conv_transpose1d_patch_to_linear,
    fold_time_embed_r,
    linear_compute_config,
    make_linear,
    make_rms_norm,
    norm_compute_config,
    reshape_last_two,
    rms_norm_modulated,
    timestep_sinusoid_tt,
)


class AceStepTransformerBlock(Module):
    """One of the 24 DiT layers: adaLN self-attn, plain cross-attn, adaLN MLP.

    Heavier than an LLM layer — **three** sub-blocks, not two.

    The 6-way adaLN table lives in :class:`Modulation` as six separate ``[1, 1, 1, hidden]``
    parameters with the ``time_embed_r`` constant and the ``(1 + scale)`` offset already folded
    in. When the block is loaded standalone (no enclosing model), set
    :attr:`timestep_proj_r_fold` **before** ``load_torch_state_dict`` to apply the fold;
    leaving it ``None`` reproduces a reference run in which ``time_embed_r`` output zero.
    """

    def __init__(
        self,
        config: AceStepDiTConfig,
        *,
        layer_index: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        use_cross_attention: bool = True,
        fuse_adaln_into_norm: bool = False,
        rope_composite: bool = False,
        sdpa_program_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.mesh_device = mesh_device
        self.use_cross_attention = use_cross_attention
        self.fuse_adaln_into_norm = fuse_adaln_into_norm
        #: Constant ``time_embed_r`` 6-way projection, ``[6, hidden]``; set by the parent model.
        self.timestep_proj_r_fold: torch.Tensor | None = None

        self.self_attn_norm = make_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype
        )
        self.self_attn = AceStepSelfAttention(
            config,
            mesh_device=mesh_device,
            dtype=dtype,
            rope_composite=rope_composite,
            sdpa_program_config=sdpa_program_config,
        )

        if use_cross_attention:
            self.cross_attn_norm = make_rms_norm(
                config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype
            )
            self.cross_attn = AceStepCrossAttention(
                config, mesh_device=mesh_device, dtype=dtype, sdpa_program_config=sdpa_program_config
            )

        self.mlp_norm = make_rms_norm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype)
        self.mlp = AceStepMLP(config, mesh_device=mesh_device, dtype=dtype)
        self.mod = Modulation(
            config.hidden_size, num_chunks=config.num_modulation_chunks, mesh_device=mesh_device, dtype=dtype
        )

        self.mm_compute_config = linear_compute_config(mesh_device, dtype)
        self.norm_compute_config = norm_compute_config(mesh_device)

    @property
    def window(self) -> int | None:
        """TRAP-1-safe SDPA window for this layer: 256 on even layers, ``None`` on odd ones."""
        return self.config.window_for_layer(self.layer_index)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        table = state.pop("scale_shift_table", None)
        if table is not None:
            folded = Modulation.fold_state(
                table, self.timestep_proj_r_fold, num_chunks=self.config.num_modulation_chunks
            )
            for name, value in folded.items():
                state[f"mod.{name}"] = value

    def forward(
        self,
        x_11SC: ttnn.Tensor,
        *,
        modulation_chunks: list[ttnn.Tensor],
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        cross_kv: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        capture: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
    ) -> ttnn.Tensor:
        """Args:
        x_11SC: ``[1, 1, S, hidden_size]`` hidden states.
        modulation_chunks: the 6 per-step ``timestep_proj`` chunks, each
            ``[1, 1, 1, hidden_size]``, from :meth:`AceStepTransformer1DModel.timestep_proj_chunks`.
        rope: ``(cos, sin)`` for this sequence length.
        cross_kv: this layer's cached ``(k, v)``; ``None`` skips cross-attention, matching
            the reference's ``encoder_hidden_states is None`` branch.
        """
        shift, one_plus_scale, gate, c_shift, one_plus_c_scale, c_gate = self.mod(modulation_chunks)

        # --- steps 1-8: self-attention ------------------------------------------------- #
        h = rms_norm_modulated(
            self.self_attn_norm,
            x_11SC,
            one_plus_scale,
            shift,
            compute_kernel_config=self.norm_compute_config,
            fused=self.fuse_adaln_into_norm,
        )
        capture_tensor(capture, prefix + "self_attn_norm_modulated", h)
        attn = self.self_attn(h, rope=rope, window=self.window, capture=capture, prefix=prefix + "self_attn.")
        ttnn.deallocate(h)
        # BARE multiplicative gate: x + o * gate, NOT o * (1 + gate).
        gated = ttnn.multiply(attn, gate)
        ttnn.deallocate(attn)
        x = ttnn.add(x_11SC, gated)
        ttnn.deallocate(gated)
        capture_tensor(capture, prefix + "after_self_attn", x)

        # --- steps 9-11: cross-attention (unmodulated, ungated) ------------------------ #
        if self.use_cross_attention and cross_kv is not None:
            h = self.cross_attn_norm(x, compute_kernel_config=self.norm_compute_config)
            capture_tensor(capture, prefix + "cross_attn_norm", h)
            cross = self.cross_attn(h, cross_kv, capture=capture, prefix=prefix + "cross_attn.")
            ttnn.deallocate(h)
            x_next = ttnn.add(x, cross)  # plain residual: no gate
            ttnn.deallocate(cross)
            ttnn.deallocate(x)
            x = x_next
            capture_tensor(capture, prefix + "after_cross_attn", x)

        # --- steps 12-14: SwiGLU MLP --------------------------------------------------- #
        h = rms_norm_modulated(
            self.mlp_norm,
            x,
            one_plus_c_scale,
            c_shift,
            compute_kernel_config=self.norm_compute_config,
            fused=self.fuse_adaln_into_norm,
        )
        capture_tensor(capture, prefix + "mlp_norm_modulated", h)
        ff = self.mlp(h, compute_kernel_config=self.mm_compute_config)
        ttnn.deallocate(h)
        capture_tensor(capture, prefix + "mlp_out", ff)
        gated = ttnn.multiply(ff, c_gate)
        ttnn.deallocate(ff)
        x_next = ttnn.add(x, gated)
        ttnn.deallocate(gated)
        ttnn.deallocate(x)
        capture_tensor(capture, prefix + "out", x_next)

        for t in (shift, one_plus_scale, gate, c_shift, one_plus_c_scale, c_gate):
            ttnn.deallocate(t)
        return x_next


class AceStepTransformer1DModel(Module):
    """The full 24-layer ACE-Step 1.5 DiT.

    Usage::

        model = AceStepTransformer1DModel(AceStepDiTConfig(), mesh_device=mesh)
        model.load_torch_state_dict(diffusers_dit_state_dict)
        model.prepare_rope(seq_len=S)                       # once per duration
        cross_kv = model.precompute_cross_kv(encoder_11LC)  # once per generation
        for t in timesteps:                                 # 8 turbo steps
            v = model(x_11TC, ctx_11TC, t, cross_kv=cross_kv)

    ``prepare_rope`` and ``precompute_cross_kv`` must be called **before** any trace capture:
    ``execute_trace`` corrupts tensors allocated after capture (master doc §5).
    """

    def __init__(
        self,
        config: AceStepDiTConfig | None = None,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        fuse_adaln_into_norm: bool = False,
        rope_composite: bool = False,
        sdpa_program_config=None,
    ) -> None:
        super().__init__()
        config = config or AceStepDiTConfig()
        self.config = config
        self.mesh_device = mesh_device
        self.dtype = dtype
        #: Constant ``time_embed_r`` 2048-wide output; folded into ``mod_out`` at load.
        self.temb_r_fold: torch.Tensor | None = None
        self._rope_cache: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

        # -- patchify / de-patchify: Linears, not convolutions ------------------------- #
        self.proj_in = make_linear(
            config.patch_size * config.in_channels,
            config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.proj_out = make_linear(
            config.hidden_size,
            config.patch_size * config.audio_acoustic_hidden_dim,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        # -- timestep conditioning (the r path is folded away, see module docstring) ---- #
        self.time_embed = TimestepEmbedding(
            config.time_embed_in_channels,
            config.hidden_size,
            act_fn="silu",
            dtype=dtype,
            mesh_device=mesh_device,
            tp_mesh_axis=None,
        )
        self.time_proj = make_linear(
            config.hidden_size,
            config.num_modulation_chunks * config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.condition_embedder = make_linear(
            config.cross_attention_input_dim,
            config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.layers = ModuleList(
            AceStepTransformerBlock(
                config,
                layer_index=i,
                mesh_device=mesh_device,
                dtype=dtype,
                fuse_adaln_into_norm=fuse_adaln_into_norm,
                rope_composite=rope_composite,
                sdpa_program_config=sdpa_program_config,
            )
            for i in range(config.num_hidden_layers)
        )

        self.norm_out = make_rms_norm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype)
        self.mod_out = Modulation(config.hidden_size, num_chunks=2, mesh_device=mesh_device, dtype=dtype)

        self.fuse_adaln_into_norm = fuse_adaln_into_norm
        self.mm_compute_config = linear_compute_config(mesh_device, dtype)
        self.norm_compute_config = norm_compute_config(mesh_device)

    # ------------------------------------------------------------------------------------ #
    #                                weight preparation                                     #
    # ------------------------------------------------------------------------------------ #

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        cfg = self.config
        p = cfg.patch_size

        # 1. Fold the constant `time_embed_r` path. Inference always passes
        #    timestep_r == timestep, so this MLP always sees 0 (§3.7).
        r_state = pop_substate(state, "time_embed_r")
        temb_r: torch.Tensor | None = None
        timestep_proj_r: torch.Tensor | None = None
        if r_state:
            temb_r, timestep_proj_r = fold_time_embed_r(
                r_state, num_channels=cfg.time_embed_in_channels, scale=cfg.time_embed_scale
            )
        self.temb_r_fold = temb_r
        for i in range(len(self.layers)):
            self.layers[i].timestep_proj_r_fold = timestep_proj_r

        # 2. The output 2-way adaLN table. Both chunks receive the same `temb`, matching the
        #    reference's `temb.unsqueeze(1)` broadcast against a [1, 2, H] table.
        table = state.pop("scale_shift_table", None)
        if table is not None:
            for name, value in Modulation.fold_state(table, temb_r, num_chunks=2).items():
                state[f"mod_out.{name}"] = value

        # 3. time_embed.time_proj is a sibling here (tt_dit's TimestepEmbedding is only
        #    linear_1 / silu / linear_2), so lift it out of the time_embed substate.
        for suffix in ("weight", "bias"):
            value = state.pop(f"time_embed.time_proj.{suffix}", None)
            if value is not None:
                state[f"time_proj.{suffix}"] = value

        # 4. proj_in_conv: Conv1d(192 -> 2048, k=s=2) == reshape + Linear(384 -> 2048).
        weight = state.pop("proj_in_conv.weight", None)
        if weight is not None:
            state["proj_in.weight"] = conv1d_patch_to_linear(weight, p)
        bias = state.pop("proj_in_conv.bias", None)
        if bias is not None:
            state["proj_in.bias"] = bias

        # 5. proj_out_conv: ConvTranspose1d(2048 -> 64, k=s=2) == Linear(2048 -> 128) + reshape.
        #    Note the [C_in, C_out, k] weight layout and the p-times-tiled bias.
        weight = state.pop("proj_out_conv.weight", None)
        bias = state.pop("proj_out_conv.bias", None)
        if weight is not None:
            lin_w, lin_b = conv_transpose1d_patch_to_linear(weight, bias, p)
            state["proj_out.weight"] = lin_w
            if lin_b is not None:
                state["proj_out.bias"] = lin_b
        elif bias is not None:  # bias without weight: still needs tiling to stay consistent
            state["proj_out.bias"] = bias.repeat(p)

    # ------------------------------------------------------------------------------------ #
    #                            per-generation precomputation                              #
    # ------------------------------------------------------------------------------------ #

    def prepare_rope(self, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Build (and cache) the HF half-split ``(cos, sin)`` tables for ``seq_len``.

        Call before trace capture. ``S`` maxes out at 7500 against
        ``max_position_embeddings`` 32768, so RoPE never needs extension.
        """
        cached = self._rope_cache.get(seq_len)
        if cached is None:
            cached = build_rope_tables(
                self.mesh_device,
                seq_len,
                head_dim=self.config.head_dim,
                theta=self.config.rope_theta,
                dtype=self.dtype,
            )
            self._rope_cache[seq_len] = cached
        return cached

    def project_encoder_hidden_states(self, encoder_11LC: ttnn.Tensor) -> ttnn.Tensor:
        """``condition_embedder``: ``[1, 1, enc_L, encoder_hidden_size] -> [.., hidden_size]``.

        Timestep-independent; the reference re-runs it every step, which is pure waste (§3.7).
        """
        return self.condition_embedder(encoder_11LC, compute_kernel_config=self.mm_compute_config)

    def precompute_cross_kv(
        self,
        encoder_11LC: ttnn.Tensor,
        *,
        already_projected: bool = False,
        capture: dict[str, torch.Tensor] | None = None,
    ) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        """All 24 layers' cross-attention ``(k, v)``, computed once per generation.

        48 GEMMs hoisted out of the 8-step loop. Upstream does the same with an
        ``EncoderDecoderCache``; at batch 1 plain persistent DRAM tensors are equivalent and
        simpler (TRAP-5).

        # BATCH-1 ASSUMPTION: one (k, v) pair per layer, no user axis.
        """
        ctx = encoder_11LC if already_projected else self.project_encoder_hidden_states(encoder_11LC)
        capture_tensor(capture, "condition_embedder", ctx)
        kv = [
            self.layers[i].cross_attn.compute_kv(ctx, capture=capture, prefix=f"layers.{i}.cross_attn.")
            for i in range(len(self.layers))
        ]
        if not already_projected:
            ttnn.deallocate(ctx)
        return kv

    def timestep_proj_chunks(
        self,
        timestep,
        *,
        capture: dict[str, torch.Tensor] | None = None,
    ) -> tuple[list[ttnn.Tensor], ttnn.Tensor]:
        """Run the ``t`` half of the dual timestep embedding.

        Args:
            timestep: a python float, a 1-D torch tensor of length B, or a pre-built
                ``[1, 1, B, 256]`` ttnn sinusoid (see
                :func:`ttnn_ace_step_common.timestep_sinusoid_tt`).

        Returns:
            ``(chunks, temb)`` where ``chunks`` is the 6-way split of ``time_proj``'s output
            (each ``[1, 1, B, hidden]``) and ``temb`` is ``linear_2``'s output, used by the
            ``norm_out`` modulation.
        """
        cfg = self.config
        if isinstance(timestep, ttnn.Tensor):
            t_freq = timestep
            owns_freq = False
        else:
            t_freq = timestep_sinusoid_tt(
                timestep,
                self.mesh_device,
                num_channels=cfg.time_embed_in_channels,
                scale=cfg.time_embed_scale,
                dtype=self.dtype,
            )
            owns_freq = True
        capture_tensor(capture, "time_sinusoid", t_freq)

        temb = self.time_embed(t_freq)
        if owns_freq:
            ttnn.deallocate(t_freq)
        capture_tensor(capture, "temb", temb)

        activated = ttnn.silu(temb)
        flat = self.time_proj(activated, compute_kernel_config=self.mm_compute_config)
        ttnn.deallocate(activated)
        capture_tensor(capture, "timestep_proj", flat)

        # The reference does `.unflatten(1, (6, -1)).chunk(6, dim=1)`, i.e. six contiguous
        # 2048-wide slices of the 12288-wide projection. Chunking the LAST dim is the same
        # values and keeps every slice boundary tile-aligned (2048 = 64 tiles).
        # BATCH-1 ASSUMPTION: at B > 1 these stay [1, 1, B, hidden] and no longer broadcast
        # against [1, 1, S, hidden]; a real unflatten to [B, 6, hidden] would be needed.
        chunks = list(ttnn.chunk(flat, cfg.num_modulation_chunks, dim=-1))
        ttnn.deallocate(flat)
        return chunks, temb

    # ------------------------------------------------------------------------------------ #
    #                                  patch (un)folding                                    #
    # ------------------------------------------------------------------------------------ #

    def patchify(self, x_11TC: ttnn.Tensor) -> tuple[ttnn.Tensor, int]:
        """``[1, 1, T, in_channels] -> [1, 1, S, hidden_size]``, plus the original ``T``.

        ``proj_in_conv`` has ``kernel == stride == patch_size`` and no padding, so it is a
        row-major reshape that folds ``patch_size`` frames into the feature axis followed by a
        matmul. The reference zero-pads an odd ``T`` up to the patch boundary and crops the
        output back, which is reachable (``T = 25 * duration`` is odd for odd durations).
        """
        cfg = self.config
        p = cfg.patch_size
        original_t = int(x_11TC.shape[-2])
        channels = int(x_11TC.shape[-1])
        assert channels == cfg.in_channels, f"expected {cfg.in_channels} input channels, got {channels}"

        pad = (-original_t) % p
        x = ttnn.to_layout(x_11TC, ttnn.ROW_MAJOR_LAYOUT)
        if pad:
            # Reference: F.pad(hidden_states, (0, 0, 0, pad_length)) after the context concat.
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
        seq_len = (original_t + pad) // p
        x = ttnn.reshape(x, [1, 1, seq_len, p * channels])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        out = self.proj_in(x, compute_kernel_config=self.mm_compute_config)
        ttnn.deallocate(x)
        return out, original_t

    def unpatchify(self, x_11SC: ttnn.Tensor, original_t: int) -> ttnn.Tensor:
        """``[1, 1, S, hidden_size] -> [1, 1, original_t, audio_acoustic_hidden_dim]``."""
        cfg = self.config
        p = cfg.patch_size
        seq_len = int(x_11SC.shape[-2])
        wide = self.proj_out(x_11SC, compute_kernel_config=self.mm_compute_config)
        out = reshape_last_two(wide, p * seq_len, cfg.audio_acoustic_hidden_dim)
        ttnn.deallocate(wide)
        if out.shape[-2] != original_t:
            cropped = ttnn.slice(out, [0, 0, 0, 0], [1, 1, original_t, cfg.audio_acoustic_hidden_dim])
            ttnn.deallocate(out)
            out = cropped
        return out

    # ------------------------------------------------------------------------------------ #
    #                                       forward                                         #
    # ------------------------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states_11TC: ttnn.Tensor,
        context_latents_11TC: ttnn.Tensor | None,
        timestep,
        encoder_hidden_states_11LC: ttnn.Tensor | None = None,
        *,
        cross_kv: list[tuple[ttnn.Tensor, ttnn.Tensor]] | None = None,
        capture: dict[str, torch.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """Predict the velocity field.

        Args:
            hidden_states_11TC: noisy latents ``x_t``, ``[1, 1, T, 64]``. If
                ``context_latents_11TC`` is ``None`` this is taken to be the pre-concatenated
                ``[1, 1, T, 192]`` patchify input instead (handy for golden replay).
            context_latents_11TC: ``cat([src_latents(64), chunk_masks(64)], -1)``,
                ``[1, 1, T, 128]``.
            timestep: ``t_curr``. ``timestep_r`` is not an argument: it always equals
                ``timestep`` at inference and its whole path is folded into the weights.
            encoder_hidden_states_11LC: packed condition-encoder output,
                ``[1, 1, enc_L, encoder_hidden_size]``. Ignored when ``cross_kv`` is given.
            cross_kv: output of :meth:`precompute_cross_kv`.

        Returns:
            ``[1, 1, T, 64]`` velocity.
        """
        cfg = self.config

        # --- timestep embedding -------------------------------------------------------- #
        modulation_chunks, temb = self.timestep_proj_chunks(timestep, capture=capture)

        # --- context concat + patchify ------------------------------------------------- #
        if context_latents_11TC is None:
            patch_input = hidden_states_11TC
            owns_patch_input = False
        else:
            # Reference order: cat([context_latents, hidden_states], dim=-1) -> 128 + 64 = 192.
            patch_input = ttnn.concat([context_latents_11TC, hidden_states_11TC], dim=-1)
            owns_patch_input = True
        capture_tensor(capture, "proj_in_input", patch_input)
        x, original_t = self.patchify(patch_input)
        if owns_patch_input:
            ttnn.deallocate(patch_input)
        capture_tensor(capture, "proj_in", x)
        seq_len = int(x.shape[-2])

        # --- cross-attention K/V ------------------------------------------------------- #
        if cross_kv is None:
            if encoder_hidden_states_11LC is None:
                msg = "either encoder_hidden_states_11LC or cross_kv must be provided"
                raise ValueError(msg)
            cross_kv = self.precompute_cross_kv(encoder_hidden_states_11LC, capture=capture)
            owns_cross_kv = True
        else:
            owns_cross_kv = False
        assert len(cross_kv) == len(self.layers)

        # --- RoPE ---------------------------------------------------------------------- #
        rope = self.prepare_rope(seq_len)

        # --- 24 layers ----------------------------------------------------------------- #
        # Each block allocates its own output and leaves its input alone (so a standalone
        # block test can reuse the input), so free the previous activation here instead.
        for i in range(len(self.layers)):
            previous = x
            x = self.layers[i](
                previous,
                modulation_chunks=modulation_chunks,
                rope=rope,
                cross_kv=cross_kv[i],
                capture=capture,
                prefix=f"layers.{i}.",
            )
            ttnn.deallocate(previous)

        for chunk in modulation_chunks:
            ttnn.deallocate(chunk)
        if owns_cross_kv:
            for k, v in cross_kv:
                ttnn.deallocate(k)
                ttnn.deallocate(v)

        # --- adaptive output norm + de-patchify ---------------------------------------- #
        shift, one_plus_scale = self.mod_out([temb, temb])
        ttnn.deallocate(temb)
        x_norm = rms_norm_modulated(
            self.norm_out,
            x,
            one_plus_scale,
            shift,
            compute_kernel_config=self.norm_compute_config,
            fused=self.fuse_adaln_into_norm,
        )
        ttnn.deallocate(x)
        ttnn.deallocate(shift)
        ttnn.deallocate(one_plus_scale)
        capture_tensor(capture, "norm_out", x_norm)

        out = self.unpatchify(x_norm, original_t)
        ttnn.deallocate(x_norm)
        capture_tensor(capture, "proj_out", out)
        assert int(out.shape[-2]) == original_t
        assert int(out.shape[-1]) == cfg.audio_acoustic_hidden_dim
        return out


__all__ = [
    "TILE",
    "AceStepDiTConfig",
    "AceStepMLP",
    "AceStepTransformer1DModel",
    "AceStepTransformerBlock",
]
