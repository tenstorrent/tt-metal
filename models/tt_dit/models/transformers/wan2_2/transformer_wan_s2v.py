# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""WAN 2.2 Speech-to-Video DiT.

Subclasses ``WanTransformer3DModel`` and adds every S2V-specific
conditioning module from the reference ``WanModel_S2V``
(``wan/modules/s2v/model_s2v.py``) as on-device tt_dit modules:

  * ``audio_encoder`` — :class:`CausalAudioEncoder`.
  * ``audio_injector`` — :class:`AudioInjector_WAN` cross-attention slots
    invoked at the inject layer indices via :meth:`after_transformer_block`.
  * ``frame_packer`` — :class:`FramePackMotionerWan`.
  * ``cond_encoder`` — :class:`WanPatchEmbed` for the pose video.
  * ``trainable_cond_mask`` — :class:`Parameter` shape ``[3, dim]``.

:meth:`prepare_cond_emb` builds the per-clip device-side caches (pose
embedding, ref+motion+mask const tokens, noisy-mask broadcast). The
``inner_step`` block loop runs on the extended ``Sq = noisy + ref + motion``
sequence with audio cross-attention masked to noisy positions, then slices
off ref+motion before the output head.

The production config uses ``enable_framepack=True`` and
``enable_motioner=False``; the constructor refuses any other combination.
"""

from __future__ import annotations

import time

import torch

import ttnn

from ....layers.embeddings import WanPatchEmbed
from ....layers.module import Parameter
from ....utils.mochi import get_rot_transformation_mat
from ....utils.padding import get_padded_vision_seq_len, pad_vision_seq_parallel
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, float32_tensor, from_torch, local_device_to_torch
from .audio_utils_wan import AudioInjector_WAN, CausalAudioEncoder
from .motioner_wan import FramePackMotionerWan
from .s2v_rope import rope_precompute
from .transformer_wan import WanTransformer3DModel


class WanS2VTransformer3DModel(WanTransformer3DModel):
    """Speech-to-video variant of the WAN 2.2 DiT.

    Constructor mirrors ``WanTransformer3DModel`` and adds S2V-specific
    arguments. Defaults mirror the reference WanModel_S2V config.
    """

    def __init__(
        self,
        *,
        # Audio defaults match the production Wan2.2-S2V-14B config
        # (wav2vec2-large-xlsr-53: hidden_size=1024, num_hidden_layers=24
        # → num_audio_layers = 24 + 1 = 25).
        audio_dim: int = 1024,
        num_audio_layers: int = 25,
        num_audio_token: int = 4,
        audio_inject_layers: tuple[int, ...] = (
            0,
            4,
            8,
            12,
            16,
            20,
            24,
            27,
            30,
            33,
            36,
            39,
        ),
        enable_adain: bool = False,
        cond_dim: int = 16,
        # Production-only: `enable_framepack=True`, `enable_motioner=False`.
        # We accept the kwargs to keep the call sites verbose but reject any
        # non-production combination at construction time.
        enable_motioner: bool = False,
        enable_framepack: bool = True,
        motion_token_num: int = 1024,
        motioner_dim: int = 2048,
        num_layers: int = 40,
        **kwargs,
    ) -> None:
        if enable_motioner or not enable_framepack:
            raise NotImplementedError(
                "Only the production FramePacker path is supported "
                f"(enable_motioner={enable_motioner}, enable_framepack={enable_framepack})"
            )
        kwargs.setdefault("model_type", "s2v")
        super().__init__(num_layers=num_layers, **kwargs)

        self.audio_dim = audio_dim
        self.num_audio_layers = num_audio_layers
        self.num_audio_token = num_audio_token
        self.audio_inject_layers = tuple(audio_inject_layers)
        self.enable_adain = enable_adain
        self.cond_dim = cond_dim
        self.motion_token_num = motion_token_num
        self.motioner_dim = motioner_dim

        self.audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            num_layers=num_audio_layers,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain,
            mesh_device=self.mesh_device,
        )
        self.audio_injector = AudioInjector_WAN(
            dim=self.dim,
            num_heads=kwargs.get("num_heads", 40),
            inject_layers=self.audio_inject_layers,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
        )

        self.cond_encoder = WanPatchEmbed(
            patch_size=self.patch_size,
            in_channels=cond_dim,
            embed_dim=self.dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )
        # 3-entry {noisy=0, ref=1, motion=2} table added per-token after
        # ref/motion concat. Replicated (not TP-sharded) — 3 * dim is tiny.
        self.trainable_cond_mask = Parameter(
            total_shape=[3, self.dim],
            device=self.mesh_device,
        )
        self.frame_packer = FramePackMotionerWan(
            in_channels=16,
            inner_dim=self.dim,
            num_heads=kwargs.get("num_heads", 40),
            zip_frame_buckets=(1, 2, 16),
            drop_mode="padd",
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )

        # Per-clip state, populated by ``prepare_audio_emb`` / ``prepare_cond_emb``.
        # ``merged_audio_emb_flat`` is ``[1, B, T*(N+1), dim]`` per-frame audio
        # K/V; the block-diagonal frame mask constrains each Q to its frame.
        self.merged_audio_emb_flat: ttnn.Tensor | None = None
        self.num_frames: int = 0
        self.original_seq_len: int = 0
        # Cumulative seconds inside ``after_transformer_block``'s audio cross-
        # attn during the current clip. Read by the perf test for a per-stage
        # breakdown; reset on each ``prepare_audio_emb``.
        self._audio_inject_sec_accum: float = 0.0
        # Spatial caches built by ``prepare_cond_emb``: pose embedding for the
        # noisy patches, concatenated ref+motion const tokens (mask-augmented),
        # and the broadcast noisy-mask token.
        self._cached_pose_emb_1BND: ttnn.Tensor | None = None
        self._cached_const_tokens_1BND: ttnn.Tensor | None = None
        self._cached_noisy_mask_emb_1BND: ttnn.Tensor | None = None
        self._cached_total_seq_len: int = 0
        self._cached_padded_N_noisy: int = 0
        self._cached_padded_const: int = 0
        # Segmented timestep modulation (``zero_timestep=True``):
        # ``mask_noisy`` is 1 over [0, original_seq_len) else 0 (incl pad);
        # ``mask_constant`` is the complement over valid positions only;
        # ``timestep_proj_zero`` is the zero-t modulation that combines with
        # the real-t projection per-block to build per-token shift/scale/gate.
        self._cached_mask_noisy: ttnn.Tensor | None = None
        self._cached_mask_constant: ttnn.Tensor | None = None
        self._cached_timestep_proj_zero: ttnn.Tensor | None = None
        self._frame_attn_mask_cache: dict[tuple[int, int], ttnn.Tensor] = {}
        self._adain_modulation_cache: dict[tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        # AdaIN per-frame audio embedding (token 0). Host copy is fed to the
        # host-side ``repeat_interleave`` in ``_build_adain_modulation_for_layer``;
        # device copy goes through ``injector_adain_layers[i].linear``.
        self.audio_emb_global_token0_torch: torch.Tensor | None = None
        self.audio_emb_global_token0_dev: ttnn.Tensor | None = None

    # ----------------------------------------------------------------------
    # Audio preparation. Called once per audio clip by the pipeline.
    # ----------------------------------------------------------------------

    def prepare_audio_emb(
        self,
        wav2vec2_layers_torch: torch.Tensor,
        *,
        motion_frames: tuple[int, int] = (17, 5),
        target_num_frames: int | None = None,
    ) -> None:
        """Run the on-device CausalAudioEncoder and cache the per-frame audio tokens.

        Args:
            wav2vec2_layers_torch: ``[B, num_audio_layers, audio_dim, T_audio]``
                stack of wav2vec2 hidden states already aligned to the target
                video frame rate (output of ``get_audio_embed_bucket_fps``).
            motion_frames: ``[encoded_frames, latent_frames]`` from the
                reference; the first ``motion_frames[1]`` frames of audio
                correspond to the motion-latent prefix and are sliced off the
                final per-clip audio embedding.
        """
        # Reference pre-pads on the left by ``motion_frames[0]`` first-frame repeats.
        pre = wav2vec2_layers_torch[..., :1].expand(-1, -1, -1, motion_frames[0])
        audio_input = torch.cat([pre, wav2vec2_layers_torch], dim=-1)

        audio_emb_out = self.audio_encoder(audio_input)
        if self.enable_adain:
            audio_global_emb, audio_local_emb = audio_emb_out
        else:
            audio_global_emb, audio_local_emb = None, audio_emb_out

        local_torch = local_device_to_torch(audio_local_emb)
        local_torch = local_torch[:, motion_frames[1] :, :, :]
        # Snap T to the latent frame count: motion_frames=[17,5] can leave the
        # encoder one frame off when num_frames isn't a multiple of 4, which
        # would break the per-frame block-diagonal mask.
        if target_num_frames is not None and target_num_frames != local_torch.shape[1]:
            T_have = local_torch.shape[1]
            if target_num_frames > T_have:
                pad = local_torch[:, -1:, :, :].expand(-1, target_num_frames - T_have, -1, -1)
                local_torch = torch.cat([local_torch, pad], dim=1)
            else:
                local_torch = local_torch[:, :target_num_frames, :, :]
        B, T_video, num_tok_p1, dim = local_torch.shape
        self.num_frames = T_video
        self.num_audio_tokens_per_frame = num_tok_p1
        self._audio_inject_sec_accum = 0.0
        # Flatten to ``[1, B, T*(N+1), dim]`` cross-attn K/V; audio is small
        # enough that we keep it SP-replicated.
        flat_torch = local_torch.reshape(B, T_video * num_tok_p1, dim).unsqueeze(0).contiguous()
        self.merged_audio_emb_flat = from_torch(
            flat_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # New clip → drop per-T mask cache + per-injector K/V cache.
        self._frame_attn_mask_cache = {}
        self.audio_injector.invalidate_audio_kv_cache()

        if audio_global_emb is not None:
            global_torch = local_device_to_torch(audio_global_emb)
            global_torch = global_torch[:, motion_frames[1] :, :, :]
            if target_num_frames is not None and target_num_frames != global_torch.shape[1]:
                T_have = global_torch.shape[1]
                if target_num_frames > T_have:
                    pad = global_torch[:, -1:, :, :].expand(-1, target_num_frames - T_have, -1, -1)
                    global_torch = torch.cat([global_torch, pad], dim=1)
                else:
                    global_torch = global_torch[:, :target_num_frames, :, :]
            # AdaIN reads token 0. Keep host + device copies so the per-layer
            # repeat_interleave + silu+Linear can each use their preferred side.
            token0_torch = global_torch[:, :, 0, :].contiguous().unsqueeze(0)
            self.audio_emb_global_token0_torch = token0_torch
            self.audio_emb_global_token0_dev = bf16_tensor(
                token0_torch, device=self.mesh_device, layout=ttnn.TILE_LAYOUT
            )
        else:
            self.audio_emb_global_token0_torch = None
            self.audio_emb_global_token0_dev = None
        self._adain_modulation_cache = {}

    # ----------------------------------------------------------------------
    # Audio injection hook. Invoked after each transformer block in
    # ``inner_step``; only does work at the configured layer indices.
    # ----------------------------------------------------------------------

    def _get_or_build_frame_attn_mask(self, N_total: int) -> ttnn.Tensor:
        """Block-diagonal cross-attention mask for per-frame audio injection.

        Matches the per-device ``spatial = concat([noisy, const])`` layout.
        Const rows are 0.0 (uniform attention, not -inf — that would NaN the
        softmax); the resulting audio residual is then zeroed at const tokens
        by ``_cached_mask_noisy`` in :meth:`after_transformer_block`, matching
        the reference's noisy-only audio rearrange.
        """
        if self.original_seq_len == 0:
            raise RuntimeError("prepare_cond_emb() must be called before audio injection fires.")

        T_video = self.num_frames
        K_per_frame = self.num_audio_tokens_per_frame
        Sk = T_video * K_per_frame
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        padded_N_noisy = self._cached_padded_N_noisy
        padded_const = self._cached_padded_const

        cache_key = (padded_N_noisy, padded_const, Sk)
        if cache_key in self._frame_attn_mask_cache:
            return self._frame_attn_mask_cache[cache_key]

        noisy_len = self.original_seq_len
        if noisy_len % T_video != 0:
            raise RuntimeError(f"noisy_len={noisy_len} not divisible by T_video={T_video}.")
        hw_per_frame = noisy_len // T_video

        # Noisy part: ``[1, 1, padded_N_noisy, Sk]``, frame-block-diagonal
        # entries 0.0 for valid noisy positions, -inf elsewhere.
        noisy_mask_torch = torch.full((1, 1, padded_N_noisy, Sk), float("-inf"), dtype=torch.float32)
        for t in range(T_video):
            noisy_mask_torch[
                ..., t * hw_per_frame : (t + 1) * hw_per_frame, t * K_per_frame : (t + 1) * K_per_frame
            ] = 0.0

        # TILE_LAYOUT pads Sk to the next TILE multiple. The noisy mask pads
        # with -inf so SDPA softmax ignores the zero-filled padded K columns;
        # the all-zero const mask pads with 0.0.
        upload_kwargs = dict(
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=[None, None, sp_axis, None],
        )
        noisy_mask_tt = from_torch(noisy_mask_torch.contiguous(), pad_value=float("-inf"), **upload_kwargs)
        if padded_const > 0:
            const_mask_torch = torch.zeros(1, 1, padded_const, Sk, dtype=torch.float32)
            const_mask_tt = from_torch(const_mask_torch.contiguous(), pad_value=0.0, **upload_kwargs)
            mask_tt = ttnn.concat([noisy_mask_tt, const_mask_tt], dim=-2)
        else:
            mask_tt = noisy_mask_tt

        self._frame_attn_mask_cache[cache_key] = mask_tt
        return mask_tt

    def prepare_rope_features(self, hidden_states):
        """Reference-faithful rope for the extended ``noisy + ref + motion`` Sq.

        Uses the reference's :func:`rope_precompute` on the same grid_sizes the
        production ``WanModel_S2V`` constructs:

          * Noisy: ``f_o=0`` … ``f=F`` over the full ``H × W`` patched grid.
          * Ref:   ``f_o=30``, ``f=31``, ``range=(1, H, W)`` — a single frame
            placed at temporal position 30.
          * Motion (FramePackMotioner): three buckets with negative ``f_o``
            offsets; ``rope_precompute`` conjugates the rope for those slots
            (sin negated) to mark them as preceding the noisy clip.

        The complex output is converted to tt_dit's real ``(cos, sin)`` pair
        with ``repeat_interleave(2)`` so each head_dim slot holds the same
        cos/sin value pair (matches Diffusers' ``repeat_interleave_real=True``
        layout). Permute → pad → SP-shard → upload.
        """
        if self._cached_total_seq_len <= self.original_seq_len:
            return super().prepare_rope_features(hidden_states)

        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        _, _, F, H, W = hidden_states.shape
        pT, pH, pW = self.patch_size
        ppf, pph, ppw = F // pT, H // pH, W // pW
        N_noisy = ppf * pph * ppw
        num_heads = self.frame_packer.num_heads  # base WanTransformer3DModel doesn't stash num_heads
        head_dim = self.dim // num_heads

        freqs_ref = self.frame_packer.freqs

        def _grid(start_xyz, end_xyz, range_xyz):
            return [
                torch.tensor(start_xyz, dtype=torch.long).unsqueeze(0),
                torch.tensor(end_xyz, dtype=torch.long).unsqueeze(0),
                torch.tensor(range_xyz, dtype=torch.long).unsqueeze(0),
            ]

        noisy_grid = _grid([0, 0, 0], [ppf, pph, ppw], [ppf, pph, ppw])
        # Ref at temporal slot 30, motion buckets at negative temporal offsets —
        # exactly what ``WanModel_S2V.forward`` constructs.
        ref_grid = _grid([30, 0, 0], [31, pph, ppw], [1, pph, ppw])
        grid_sizes = [noisy_grid, ref_grid]
        N_ref = pph * ppw
        if self._cached_total_seq_len > N_noisy + N_ref:
            # Motion-bucket spatial extents come from FramePackMotioner's
            # Conv3d strides (1,2,2 / 2,4,4 / 4,8,8): trailing motion tokens
            # get zero rope unless the grid matches the proj output sizes.
            zb = self.frame_packer.zip_frame_buckets
            motion_post = _grid([-zb[0], 0, 0], [0, pph, ppw], [zb[0], pph, ppw])
            motion_2x = _grid(
                [-(zb[0] + zb[1]), 0, 0],
                [-(zb[0] + zb[1]) + zb[1] // 2, pph // 2, ppw // 2],
                [zb[1], pph, ppw],
            )
            motion_4x = _grid(
                [-(zb[0] + zb[1] + zb[2]), 0, 0],
                [-(zb[0] + zb[1] + zb[2]) + zb[2] // 4, pph // 4, ppw // 4],
                [zb[2], pph, ppw],
            )
            grid_sizes += [motion_post, motion_2x, motion_4x]

        # rope_precompute only reads ``x.size(1)`` (total seq_len) from this.
        N_total = self._cached_total_seq_len
        N_const = N_total - N_noisy
        placeholder = torch.zeros(1, N_total, num_heads, head_dim, dtype=torch.float32)

        freqs_complex = rope_precompute(placeholder, grid_sizes, freqs_ref, start=None)
        # All heads carry the same rope; take head 0 and broadcast.
        cos_half = freqs_complex.real[:, :, 0:1, :].float()
        sin_half = freqs_complex.imag[:, :, 0:1, :].float()
        # repeat_interleave(2) on the last dim matches Diffusers'
        # ``repeat_interleave_real=True`` layout (each rope slot is a 2D pair).
        cos_global = cos_half.repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
        sin_global = sin_half.repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)

        # Spatial is built as ``concat([noisy, const])`` per-device — i.e.
        # noisy and const are independently SP-sharded, then concatenated.
        # A naive global pad+SP-shard of [noisy, ref, motion] rope wouldn't
        # match: rope per-segment, SP-shard each, then concat on device.
        cos_noisy = cos_global[:, :, :N_noisy, :]
        cos_const = cos_global[:, :, N_noisy:N_total, :]
        sin_noisy = sin_global[:, :, :N_noisy, :]
        sin_const = sin_global[:, :, N_noisy:N_total, :]
        cos_noisy = pad_vision_seq_parallel(cos_noisy, num_devices=sp_factor)
        sin_noisy = pad_vision_seq_parallel(sin_noisy, num_devices=sp_factor)
        if N_const > 0:
            cos_const = pad_vision_seq_parallel(cos_const, num_devices=sp_factor)
            sin_const = pad_vision_seq_parallel(sin_const, num_devices=sp_factor)

        rope_upload_kwargs = dict(device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
        cos_n_tt = from_torch(cos_noisy.contiguous(), **rope_upload_kwargs)
        sin_n_tt = from_torch(sin_noisy.contiguous(), **rope_upload_kwargs)
        if N_const > 0:
            cos_c_tt = from_torch(cos_const.contiguous(), **rope_upload_kwargs)
            sin_c_tt = from_torch(sin_const.contiguous(), **rope_upload_kwargs)
            cos_tt = ttnn.concat([cos_n_tt, cos_c_tt], dim=-2)
            sin_tt = ttnn.concat([sin_n_tt, sin_c_tt], dim=-2)
        else:
            cos_tt = cos_n_tt
            sin_tt = sin_n_tt
        trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
        return cos_tt, sin_tt, trans_mat

    def _build_adain_modulation_for_layer(
        self,
        audio_attn_id: int,
        N_total: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Build per-token ``(shift, scale_plus_one)`` for one inject layer.

        Mirrors the reference's ``AdaLayerNorm(chunk_dim=1)``: silu + Linear
        on the per-frame audio embedding, then ``repeat_interleave`` to
        per-token over the **noisy** portion of Sq (the first
        ``original_seq_len`` tokens). ref/motion tokens get identity
        modulation (scale=1, shift=0) so AdaIN leaves them untouched. The
        +1 on scale is folded into the cached tensor so the on-device op is
        a plain multiply+add.
        """
        if self.audio_emb_global_token0_dev is None:
            raise RuntimeError("AdaIN enabled but audio_emb_global_token0_dev is None.")
        if self.original_seq_len == 0:
            raise RuntimeError("prepare_cond_emb() must be called before AdaIN modulation fires.")

        sp_factor = self.parallel_config.sequence_parallel.factor
        padded_N = get_padded_vision_seq_len(N_total, sp_factor)
        cache_key = (audio_attn_id, padded_N)
        if cache_key in self._adain_modulation_cache:
            return self._adain_modulation_cache[cache_key]

        T_video = self.num_frames
        noisy_len = self.original_seq_len
        if noisy_len % T_video != 0:
            raise RuntimeError(f"noisy_len={noisy_len} not divisible by T_video={T_video}.")
        hw_per_frame = noisy_len // T_video

        adain_layer = self.audio_injector.injector_adain_layers[audio_attn_id]
        proj_dev = adain_layer.linear(ttnn.silu(self.audio_emb_global_token0_dev))
        with torch.no_grad():
            projected = local_device_to_torch(proj_dev).reshape(1, T_video, 2 * self.dim).to(torch.float32)
            shift_per_frame, scale_per_frame = projected.chunk(2, dim=-1)
            shift_noisy = shift_per_frame.repeat_interleave(hw_per_frame, dim=1)
            # Fold +1 into scale so the device op is a plain `x * scale + shift`.
            scale_noisy = scale_per_frame.repeat_interleave(hw_per_frame, dim=1) + 1.0
            B = shift_noisy.shape[0]
            if padded_N > noisy_len:
                # Identity modulation on ref/motion + pad slots so AdaIN leaves them untouched.
                shift_extra = torch.zeros(B, padded_N - noisy_len, self.dim, dtype=shift_noisy.dtype)
                scale_extra = torch.ones(B, padded_N - noisy_len, self.dim, dtype=scale_noisy.dtype)
                shift_per_token = torch.cat([shift_noisy, shift_extra], dim=1)
                scale_per_token = torch.cat([scale_noisy, scale_extra], dim=1)
            else:
                shift_per_token = shift_noisy
                scale_per_token = scale_noisy

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        # 2D shard matches spatial_1BND's per-device layout so binary_ng's
        # last-two-dim broadcast classifier picks NONE.
        shard_mapping = {sp_axis: 2, tp_axis: 3}
        shift_tt = bf16_tensor_2dshard(
            shift_per_token.unsqueeze(0).to(torch.float32).contiguous(),
            self.mesh_device,
            shard_mapping=shard_mapping,
            layout=ttnn.TILE_LAYOUT,
        )
        scale_tt = bf16_tensor_2dshard(
            scale_per_token.unsqueeze(0).to(torch.float32).contiguous(),
            self.mesh_device,
            shard_mapping=shard_mapping,
            layout=ttnn.TILE_LAYOUT,
        )
        self._adain_modulation_cache[cache_key] = (shift_tt, scale_tt)
        return shift_tt, scale_tt

    def after_transformer_block(
        self,
        block_idx: int,
        spatial_1BND: ttnn.Tensor,
        N: int | None = None,
    ) -> ttnn.Tensor:
        """Apply the audio cross-attention residual at the configured layers.

        Instead of the reference's ``"b (t n) c -> (b t) n c"`` rearrange, we
        keep spatial SP-fractured and feed the flattened audio K/V plus a
        block-diagonal frame mask — each Q only attends to its own frame's
        audio tokens. No CCL ops, no rearrange.
        """
        if block_idx not in self.audio_injector.injected_block_id:
            return spatial_1BND
        if self.merged_audio_emb_flat is None:
            raise RuntimeError("prepare_audio_emb() must be called before forward().")

        _inject_t0 = time.perf_counter()
        audio_attn_id = self.audio_injector.injected_block_id[block_idx]

        if N is None:
            if self.original_seq_len:
                N = self.original_seq_len
            else:
                sp_factor = self.parallel_config.sequence_parallel.factor
                N = int(spatial_1BND.shape[-2]) * sp_factor

        # ``block.norm1`` is the no-affine DistributedLayerNorm that
        # all-gathers stats across TP — same effect as the reference's full-dim
        # ``LayerNorm(elementwise_affine=False)`` over the spatial input.
        block = self.blocks[block_idx]
        if self.enable_adain and self.audio_emb_global_token0_dev is not None:
            shift_full, scale_plus_one_full = self._build_adain_modulation_for_layer(audio_attn_id, N)
            x_normed = block.norm1(spatial_1BND)
            normed = ttnn.addcmul(shift_full, x_normed, scale_plus_one_full)
        else:
            normed = block.norm1(spatial_1BND)

        mask = self._get_or_build_frame_attn_mask(N)
        cached_kv = self.audio_injector._audio_kv_cache.get(audio_attn_id)  # noqa: SLF001
        if cached_kv is None:
            residual, fresh_kv = self.audio_injector.injector[audio_attn_id](
                spatial_1BND=normed,
                prompt_1BLP=self.merged_audio_emb_flat,
                N=int(spatial_1BND.shape[-2]),
                cross_attn_mask=mask,
                return_fresh_kv=True,
            )
            self.audio_injector._audio_kv_cache[audio_attn_id] = fresh_kv  # noqa: SLF001
        else:
            residual = self.audio_injector.injector[audio_attn_id](
                spatial_1BND=normed,
                prompt_1BLP=None,
                N=int(spatial_1BND.shape[-2]),
                cross_attn_mask=mask,
                cached_kv_BHNE=cached_kv,
            )
        # Zero the residual at const + pad rows so only noisy tokens get an
        # audio contribution — const rows used uniform-0 attention to keep
        # softmax finite, which would otherwise leak into the output.
        if self._cached_mask_noisy is not None:
            residual = ttnn.multiply(residual, self._cached_mask_noisy)
        out = ttnn.add(spatial_1BND, residual)
        self._audio_inject_sec_accum += time.perf_counter() - _inject_t0
        return out

    # ----------------------------------------------------------------------
    # Conditioning prep. Called once per clip by the pipeline before the
    # denoise loop. Builds the on-device caches that ``inner_step`` will
    # add/concat into the noisy spatial sequence at every step.
    # ----------------------------------------------------------------------

    def prepare_cond_emb(
        self,
        noisy_latents_torch: torch.Tensor,
        *,
        ref_latent_torch: torch.Tensor,
        motion_latents_torch: torch.Tensor,
        cond_states_torch: torch.Tensor | None = None,
        drop_first_motion: bool = True,
    ) -> None:
        """Build per-clip caches for the S2V conditioning paths.

        Inputs are CPU tensors:
          * ``noisy_latents_torch`` ``[B=1, C=16, F, H, W]`` — used to derive ``N_noisy``.
          * ``ref_latent_torch``    ``[B=1, C=16, 1, H, W]`` — VAE-encoded ref image.
          * ``motion_latents_torch`` ``[B=1, C=16, T_motion, H, W]``.
          * ``cond_states_torch``   pose video, or ``None`` for the reference's ``-1.0`` filler.

        Populates ``_cached_pose_emb_1BND``, ``_cached_const_tokens_1BND``
        (mask-augmented patched ref+motion), ``_cached_noisy_mask_emb_1BND``,
        and the noisy / total sequence lengths consumed by ``inner_step`` and
        the audio attn mask.
        """
        B, C, F, H, W = noisy_latents_torch.shape
        pT, pH, pW = self.patch_size
        N_noisy = (F // pT) * (H // pH) * (W // pW)
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # --- Pose embedding (cond_encoder) ---
        if cond_states_torch is None:
            cond_states_torch = -torch.ones_like(noisy_latents_torch)
        cond_1BNI, _ = self.preprocess_spatial_input_host(cond_states_torch, pad=True)
        padded_N_noisy = get_padded_vision_seq_len(N_noisy, sp_factor)
        cond_dev = bf16_tensor(
            cond_1BNI, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2, layout=ttnn.TILE_LAYOUT
        )
        self._cached_pose_emb_1BND = self.cond_encoder(cond_dev)

        # --- Reference latent (shares patch_embedding with noisy; un-padded
        # so the on-device concat with the noisy sequence is exact).
        ref_1BNI, N_ref = self.preprocess_spatial_input_host(ref_latent_torch, pad=False)
        ref_dev = bf16_tensor(ref_1BNI, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        ref_emb_1BND = self.patch_embedding(ref_dev)

        # --- Motion tokens (frame_packer) ---
        # When drop_first_motion is set (reference's first-clip default), the
        # model slices motion tokens to length 0 — skip frame_packer entirely.
        if drop_first_motion:
            motion_emb_1BND = None
            N_motion = 0
        else:
            motion_emb_1BND, _motion_rope = self.frame_packer(motion_latents_torch)
            N_motion = int(motion_emb_1BND.shape[-2])

        # --- trainable_cond_mask additions ---
        # The 3 x dim table is tiny; pull to host, gather, upload directly —
        # simpler than fighting ttnn.embedding's SP/TP-sharded output layout.
        N_total = N_noisy + N_ref + N_motion
        padded_N_noisy = get_padded_vision_seq_len(N_noisy, sp_factor)
        padded_const = get_padded_vision_seq_len(N_ref + N_motion, sp_factor)
        with torch.no_grad():
            mask_table = local_device_to_torch(self.trainable_cond_mask.data).reshape(3, self.dim).to(torch.float32)
            # Noisy slot (mask index 0), broadcast over all noisy positions.
            noisy_mask_torch = mask_table[0:1].view(1, 1, 1, self.dim).expand(1, 1, padded_N_noisy, self.dim)
            # ref + motion: [N_ref] copies of row 1, [N_motion] copies of row 2.
            const_mask_torch = torch.cat(
                [
                    mask_table[1:2].expand(N_ref, self.dim),
                    mask_table[2:3].expand(N_motion, self.dim),
                    torch.zeros(padded_const - (N_ref + N_motion), self.dim),
                ],
                dim=0,
            ).view(1, 1, padded_const, self.dim)

        self._cached_noisy_mask_emb_1BND = bf16_tensor_2dshard(
            noisy_mask_torch.contiguous(),
            self.mesh_device,
            shard_mapping={sp_axis: 2, tp_axis: 3},
            layout=ttnn.TILE_LAYOUT,
        )

        # ref_emb / motion_emb are TP-fractured; gather before the host mask-add.
        if self.parallel_config.tensor_parallel.factor > 1:
            ref_emb_full = self.ccl_manager.all_gather_persistent_buffer(ref_emb_1BND, dim=3, mesh_axis=tp_axis)
        else:
            ref_emb_full = ref_emb_1BND
        ref_torch = local_device_to_torch(ref_emb_full).reshape(1, B, N_ref, self.dim).to(torch.float32)

        const_token_segments = [ref_torch]
        if N_motion > 0:
            if self.parallel_config.tensor_parallel.factor > 1:
                motion_emb_full = self.ccl_manager.all_gather_persistent_buffer(
                    motion_emb_1BND, dim=3, mesh_axis=tp_axis
                )
            else:
                motion_emb_full = motion_emb_1BND
            motion_torch = local_device_to_torch(motion_emb_full).reshape(1, B, N_motion, self.dim).to(torch.float32)
            const_token_segments.append(motion_torch)
        const_token_segments.append(torch.zeros(1, B, padded_const - (N_ref + N_motion), self.dim))
        const_tokens = torch.cat(const_token_segments, dim=2)
        const_torch = const_tokens + const_mask_torch
        self._cached_const_tokens_1BND = bf16_tensor_2dshard(
            const_torch.contiguous(),
            self.mesh_device,
            shard_mapping={sp_axis: 2, tp_axis: 3},
            layout=ttnn.TILE_LAYOUT,
        )

        self.original_seq_len = N_noisy
        self._cached_total_seq_len = N_total
        self._cached_padded_N_noisy = padded_N_noisy
        self._cached_padded_const = padded_const

        # --- Segmented timestep modulation masks (zero_timestep=True path) ---
        # Real-t modulation goes on the noisy slot; zero-t goes on ref+motion.
        # Per-segment build + on-device concat (same reason as the rope path):
        # a naive global SP-shard of ``[padded_N_total, 1]`` wouldn't match the
        # per-device ``[noisy_local | const_local]`` spatial layout.
        seg_upload_kwargs = dict(device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2, layout=ttnn.TILE_LAYOUT)

        mask_n_noisy_torch = torch.zeros(1, 1, padded_N_noisy, 1, dtype=torch.float32)
        mask_n_noisy_torch[:, :, :N_noisy, :] = 1.0
        mask_n_const_torch = torch.zeros(1, 1, padded_const, 1, dtype=torch.float32)
        self._cached_mask_noisy = ttnn.concat(
            [
                float32_tensor(mask_n_noisy_torch.contiguous(), **seg_upload_kwargs),
                float32_tensor(mask_n_const_torch.contiguous(), **seg_upload_kwargs),
            ],
            dim=-2,
        )

        mask_c_noisy_torch = torch.zeros(1, 1, padded_N_noisy, 1, dtype=torch.float32)
        mask_c_const_torch = torch.zeros(1, 1, padded_const, 1, dtype=torch.float32)
        mask_c_const_torch[:, :, : N_ref + N_motion, :] = 1.0
        self._cached_mask_constant = ttnn.concat(
            [
                float32_tensor(mask_c_noisy_torch.contiguous(), **seg_upload_kwargs),
                float32_tensor(mask_c_const_torch.contiguous(), **seg_upload_kwargs),
            ],
            dim=-2,
        )

        # Zero-timestep projection: same shape as the per-step real-t
        # projection (``[1, B, 6, dim/tp]`` after unflatten), but constant
        # across the diffusion loop. Cache once per clip.
        zero_t_torch = torch.zeros(B, 1, 1, 1, dtype=torch.float32)
        zero_t_tt = float32_tensor(zero_t_torch, device=self.mesh_device)
        _, self._cached_timestep_proj_zero = self.prepare_timestep_conditioning(zero_t_tt)

    def _s2v_segmented_block_forward(
        self,
        block,
        spatial_1BND: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        timestep_proj_real: ttnn.Tensor,
        timestep_proj_zero: ttnn.Tensor,
        mask_noisy: ttnn.Tensor,
        mask_constant: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """One block with segmented (real-t / zero-t) modulation.

        Math-equivalent of the reference ``WanS2VAttentionBlock.forward``: per-
        token modulation built as ``real * mask_noisy + zero * mask_constant``
        so SP fracturing on Sq lines up trivially. Pad slots have both masks
        zero → identity modulation + the gate zeros attn/ffn out, so pad
        passthrough is automatic.

        Uses the unfused norm + modulation path: the base block's fused
        norm1/norm3 modulation requires a TILE_HEIGHT=32 weight tile along Sq
        which is incompatible with per-token weights.
        """
        shifted_real = ttnn.add(block.scale_shift_table.data, timestep_proj_real)
        shifted_zero = ttnn.add(block.scale_shift_table.data, timestep_proj_zero)

        shift_r, scale_r, gate_r, c_shift_r, c_scale_r, c_gate_r = ttnn.chunk(shifted_real, 6, dim=2)
        shift_z, scale_z, gate_z, c_shift_z, c_scale_z, c_gate_z = ttnn.chunk(shifted_zero, 6, dim=2)

        # ``ttnn.addcmul`` requires all three operands to share the spatial's
        # bf16 dtype; gates are the only modulation tensors that go through it
        # below, so typecast just those (matches T2V's "addcmul less accurate
        # with fp32 gate" workaround in ``transformer_wan.py``).
        gate_r = ttnn.typecast(gate_r, dtype=ttnn.bfloat16)
        gate_z = ttnn.typecast(gate_z, dtype=ttnn.bfloat16)
        c_gate_r = ttnn.typecast(c_gate_r, dtype=ttnn.bfloat16)
        c_gate_z = ttnn.typecast(c_gate_z, dtype=ttnn.bfloat16)

        def _per_token(real_chunk, zero_chunk):
            # real/zero: ``[1, B, 1, D/tp]``; masks: ``[1, 1, padded_N/sp, 1]``.
            # binary_ng broadcasts to ``[1, B, padded_N/sp, D/tp]``.
            return ttnn.add(
                ttnn.multiply(real_chunk, mask_noisy),
                ttnn.multiply(zero_chunk, mask_constant),
            )

        shift_msa = _per_token(shift_r, shift_z)
        scale_msa = _per_token(scale_r, scale_z)
        gate_msa = _per_token(gate_r, gate_z)
        c_shift_msa = _per_token(c_shift_r, c_shift_z)
        c_scale_msa = _per_token(c_scale_r, c_scale_z)
        c_gate_msa = _per_token(c_gate_r, c_gate_z)

        # Self-attention. Modulation stays as add(multiply, ...) because
        # shift/scale are fp32 and spatial is bf16 — addcmul would force an
        # extra typecast that exceeds its savings here.
        spatial_normed = block.norm1(spatial_1BND)
        spatial_normed = ttnn.add(
            ttnn.multiply(spatial_normed, ttnn.add(scale_msa, 1.0)),
            shift_msa,
        )
        attn_out = block.attn1(
            spatial_1BND=spatial_normed,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
        )
        # Gate residual matches T2V's ``ttnn.addcmul(spatial, ff, c_gate)``.
        spatial_1BND = ttnn.addcmul(spatial_1BND, attn_out, gate_msa)

        # Cross-attention. norm2 keeps its learned affine — the reference
        # calls ``norm2(x).float()`` with no per-segment scale/shift here.
        spatial_normed = block.norm2(spatial_1BND)
        attn_out = block.attn2(
            spatial_1BND=spatial_normed,
            N=N,
            prompt_1BLP=prompt_1BLP,
        )
        spatial_1BND = ttnn.add(spatial_1BND, attn_out)

        # FFN
        spatial_normed = block.norm3(spatial_1BND)
        spatial_normed = ttnn.add(
            ttnn.multiply(spatial_normed, ttnn.add(c_scale_msa, 1.0)),
            c_shift_msa,
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        ffn_out = block.ffn(spatial_normed, compute_kernel_config=block.ff_compute_kernel_config)
        spatial_1BND = ttnn.addcmul(spatial_1BND, ffn_out, c_gate_msa)

        return spatial_1BND

    def inner_step(
        self,
        spatial_1BNI,
        prompt_1BLP,
        rope_cos_1HND,
        rope_sin_1HND,
        trans_mat,
        N,
        timestep,
        gather_output=False,
    ):
        """S2V variant of the base block stack: noisy + (ref+motion) on Sq, with
        audio cross-attn hooked after each block, and the const tokens sliced
        off before the output head.
        """
        temb_11BD, timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)

        spatial_1BND = self.patch_embedding(spatial_1BNI)

        has_cond = self._cached_const_tokens_1BND is not None
        if has_cond:
            if self._cached_pose_emb_1BND is not None:
                spatial_1BND = ttnn.add(spatial_1BND, self._cached_pose_emb_1BND)
            if self._cached_noisy_mask_emb_1BND is not None:
                spatial_1BND = ttnn.add(spatial_1BND, self._cached_noisy_mask_emb_1BND)
            spatial_1BND = ttnn.concat([spatial_1BND, self._cached_const_tokens_1BND], dim=-2)
            N_block = self._cached_total_seq_len
        else:
            N_block = N

        use_segmented = has_cond and self._cached_mask_noisy is not None and self._cached_timestep_proj_zero is not None

        for idx, block in enumerate(self.blocks):
            if use_segmented:
                spatial_1BND = self._s2v_segmented_block_forward(
                    block,
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    N=N_block,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                    timestep_proj_real=timestep_proj_1BTD,
                    timestep_proj_zero=self._cached_timestep_proj_zero,
                    mask_noisy=self._cached_mask_noisy,
                    mask_constant=self._cached_mask_constant,
                )
            else:
                spatial_1BND = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    temb_1BTD=timestep_proj_1BTD,
                    N=N_block,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                )
            spatial_1BND = self.after_transformer_block(idx, spatial_1BND, N=N_block)

        # Drop the const (ref + motion) tail so the head + scheduler see only
        # the noisy-noise prediction.
        if has_cond:
            sp_factor = self.parallel_config.sequence_parallel.factor
            padded_noisy_per_dev = get_padded_vision_seq_len(self.original_seq_len, sp_factor) // sp_factor
            ends = list(spatial_1BND.shape)
            ends[-2] = padded_noisy_per_dev
            spatial_1BND = ttnn.slice(spatial_1BND, [0, 0, 0, 0], ends)

        scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
        shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)

        spatial_norm_1BND = self.norm_out(
            spatial_1BND, dynamic_weight=(1 + scale_11BD), dynamic_bias=shift_11BD, dtype=ttnn.float32
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_1BNI = self.proj_out(
            spatial_norm_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=self.output_dtype
        )

        if gather_output:
            spatial_1BNI = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BNI, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )

        return spatial_1BNI
