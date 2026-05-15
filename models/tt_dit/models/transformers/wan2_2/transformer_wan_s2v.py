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

import torch

import ttnn

from ....layers.embeddings import WanPatchEmbed
from ....layers.module import Parameter
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

        # ------------------------------------------------------------------
        # Audio path (on device).
        # ------------------------------------------------------------------
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

        # Conditioning modules on device. ``cond_encoder`` mirrors the patch
        # embedding for the pose video; ``trainable_cond_mask`` is a 3-entry
        # table {noisy=0, ref=1, motion=2} added per-token after ref/motion
        # concat.
        self.cond_encoder = WanPatchEmbed(
            patch_size=self.patch_size,
            in_channels=cond_dim,
            embed_dim=self.dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )
        # Replicated (not TP-sharded): the table is tiny (3 * 5120 floats),
        # and replication keeps the host gather in ``prepare_cond_emb``
        # device-axis-agnostic.
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

        # Per-clip state populated by ``prepare_audio_emb``.
        # Flattened per-frame audio K/V used by the cross-attention injector.
        # Shape ``[1, B, T*(N+1), dim]``; the block-diagonal frame mask
        # constrains each Q token's attention to its own frame's tokens.
        self.merged_audio_emb_flat: ttnn.Tensor | None = None
        self.num_frames: int = 0
        self.original_seq_len: int = 0
        # Per-clip caches populated by ``prepare_cond_emb``: pose embedding to
        # add to the noisy patches, concatenated ref+motion constant tokens
        # (already mask-augmented), and the broadcast noisy-mask token.
        self._cached_pose_emb_1BND: ttnn.Tensor | None = None
        self._cached_const_tokens_1BND: ttnn.Tensor | None = None
        self._cached_noisy_mask_emb_1BND: ttnn.Tensor | None = None
        self._cached_total_seq_len: int = 0
        self._cached_padded_N_noisy: int = 0
        self._cached_padded_const: int = 0
        # Segmented timestep modulation (zero_timestep=True in production):
        #   * ``_cached_mask_noisy``: 1.0 over [0, original_seq_len), 0.0 else;
        #     shape ``[1, 1, padded_N_total, 1]`` SP-fractured on Sq.
        #   * ``_cached_mask_constant``: complement of mask_noisy over valid
        #     positions; 0.0 over pad slots. Same shape.
        #   * ``_cached_timestep_proj_zero``: zero-t projection
        #     ``[1, B, 6, dim/tp]``; combines with the real-t projection
        #     per-block to build per-token shift/scale/gate.
        self._cached_mask_noisy: ttnn.Tensor | None = None
        self._cached_mask_constant: ttnn.Tensor | None = None
        self._cached_timestep_proj_zero: ttnn.Tensor | None = None
        # Per-spatial-shape caches. Mask is keyed by (padded_N, Sk); AdaIN
        # modulation cache is keyed by (audio_attn_id, padded_N).
        self._frame_attn_mask_cache: dict[tuple[int, int], ttnn.Tensor] = {}
        self._adain_modulation_cache: dict[tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        # ``[1, B, T, dim]`` per-frame audio embedding at token 0. The host
        # copy lets ``_build_adain_modulation_for_layer`` finish the
        # host-side ``repeat_interleave``; the device copy is used for the
        # silu+Linear projection through ``injector_adain_layers[i].linear``
        # (whose weights are already loaded on device).
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
        # Pre-pad audio_input on the left by ``motion_frames[0]`` repeats of the
        # first frame, matching the reference (audio_input[..., 0:1].repeat
        # before passing to the encoder).
        pre = wav2vec2_layers_torch[..., :1].expand(-1, -1, -1, motion_frames[0])
        audio_input = torch.cat([pre, wav2vec2_layers_torch], dim=-1)

        # CausalAudioEncoder returns either a single local tensor or
        # ``(global, local)`` depending on ``enable_adain``.
        audio_emb_out = self.audio_encoder(audio_input)
        if self.enable_adain:
            audio_global_emb, audio_local_emb = audio_emb_out
        else:
            audio_global_emb, audio_local_emb = None, audio_emb_out

        # Slice off the motion-latent prefix on the time dim.
        local_torch = local_device_to_torch(audio_local_emb)
        local_torch = local_torch[:, motion_frames[1] :, :, :]
        # Optionally snap T_video to match the spatial latent frame count.
        # The reference's default motion_frames=[17,5] can produce an off-by-
        # one audio T versus num_latent_frames when num_frames is not a
        # multiple of 4. We pad-replicate or truncate so the per-frame mask
        # has a clean integer mapping.
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
        # Flatten ``[B, T, N+1, dim] -> [1, B, T*(N+1), dim]`` for cross-attn
        # K/V. The audio is small enough that we keep it replicated across SP.
        flat_torch = local_torch.reshape(B, T_video * num_tok_p1, dim).unsqueeze(0).contiguous()
        self.merged_audio_emb_flat = from_torch(
            flat_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # New clip → invalidate per-shape mask cache (T may have changed).
        self._frame_attn_mask_cache = {}

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
            # AdaIN reads token 0 of the per-frame global embedding. Cache
            # both the host torch (for the host-side ``repeat_interleave``)
            # and an on-device copy (for the per-layer silu+Linear which
            # runs through the device-side AdaLayerNormZero.linear).
            token0_torch = global_torch[:, :, 0, :].contiguous().unsqueeze(0)
            self.audio_emb_global_token0_torch = token0_torch
            self.audio_emb_global_token0_dev = bf16_tensor(
                token0_torch, device=self.mesh_device, layout=ttnn.TILE_LAYOUT
            )
        else:
            self.audio_emb_global_token0_torch = None
            self.audio_emb_global_token0_dev = None
        # New clip → invalidate per-shape modulation cache (per-frame audio changed).
        self._adain_modulation_cache = {}

    # ----------------------------------------------------------------------
    # Audio injection hook. Invoked after each transformer block in
    # ``inner_step``; only does work at the configured layer indices.
    # ----------------------------------------------------------------------

    def _get_or_build_frame_attn_mask(self, N_total: int) -> ttnn.Tensor:
        """Block-diagonal cross-attention mask for per-frame audio injection.

        Matches the per-device layout of ``spatial = concat([noisy, const])``:
        each device's local Sq is ``padded_N_noisy/sp + padded_const/sp``,
        where the first chunk is the noisy slice for that device and the
        second is the const slice. The mask is built in two pieces (noisy
        part with frame-block-diagonal entries, const part all-zero) and
        ``ttnn.concat``'d on device so its per-device row mapping matches
        the spatial. The const-region entries are 0.0 (uniform attention,
        not -inf which would NaN softmax); the audio residual is then gated
        by ``_cached_mask_noisy`` in :meth:`after_transformer_block` so
        const tokens get **no** audio contribution — matching the reference
        (``WanModel_S2V`` only injects audio into noisy tokens via
        ``rearrange "b (t n) c -> (b t) n c"``).
        """
        from ....utils.padding import get_padded_vision_seq_len

        T_video = self.num_frames
        K_per_frame = self.num_audio_tokens_per_frame
        Sk = T_video * K_per_frame
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        padded_N_noisy = self._cached_padded_N_noisy or get_padded_vision_seq_len(
            self.original_seq_len or N_total, sp_factor
        )
        padded_const = self._cached_padded_const

        cache_key = (padded_N_noisy, padded_const, Sk)
        if cache_key in self._frame_attn_mask_cache:
            return self._frame_attn_mask_cache[cache_key]

        noisy_len = self.original_seq_len or N_total
        if noisy_len % T_video != 0:
            msg = f"noisy_len={noisy_len} not divisible by T_video={T_video}."
            raise RuntimeError(msg)
        hw_per_frame = noisy_len // T_video

        # Noisy part: ``[1, 1, padded_N_noisy, Sk]``, frame-block-diagonal
        # entries 0.0 for valid noisy positions, -inf elsewhere.
        noisy_mask_torch = torch.full((1, 1, padded_N_noisy, Sk), float("-inf"), dtype=torch.float32)
        for t in range(T_video):
            noisy_mask_torch[
                ..., t * hw_per_frame : (t + 1) * hw_per_frame, t * K_per_frame : (t + 1) * K_per_frame
            ] = 0.0

        def _upload(t_BCsk: torch.Tensor) -> ttnn.Tensor:
            return from_torch(
                t_BCsk.contiguous(),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_axes=[None, None, sp_axis, None],
            )

        if padded_const > 0:
            const_mask_torch = torch.zeros(1, 1, padded_const, Sk, dtype=torch.float32)
            mask_tt = ttnn.concat([_upload(noisy_mask_torch), _upload(const_mask_torch)], dim=-2)
        else:
            mask_tt = _upload(noisy_mask_torch)

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

        from ....utils.mochi import get_rot_transformation_mat
        from ....utils.padding import pad_vision_seq_parallel
        from ....utils.tensor import bf16_tensor as _bf16_tensor

        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        _, _, F, H, W = hidden_states.shape
        pT, pH, pW = self.patch_size
        ppf, pph, ppw = F // pT, H // pH, W // pW
        N_noisy = ppf * pph * ppw
        num_heads = self.frame_packer.num_heads  # base WanTransformer3DModel doesn't stash num_heads
        head_dim = self.dim // num_heads

        # Reference's rope_precompute needs the model's freqs table. We import
        # from the bound frame_packer (which already builds ``self.freqs`` with
        # the same construction as WanModel_S2V).
        freqs_ref = self.frame_packer.freqs

        # Build the per-segment grid_sizes mirroring WanModel_S2V.forward.
        def _grid(start_xyz, end_xyz, range_xyz):
            return [
                torch.tensor(start_xyz, dtype=torch.long).unsqueeze(0),
                torch.tensor(end_xyz, dtype=torch.long).unsqueeze(0),
                torch.tensor(range_xyz, dtype=torch.long).unsqueeze(0),
            ]

        # 1. Noisy spatial grid.
        noisy_grid = _grid([0, 0, 0], [ppf, pph, ppw], [ppf, pph, ppw])
        # 2. Reference image at temporal slot 30 (matches WanModel_S2V.forward).
        ref_grid = _grid([30, 0, 0], [31, pph, ppw], [1, pph, ppw])
        grid_sizes = [noisy_grid, ref_grid]
        # 3-5. Motion buckets — only when ``drop_first_motion`` was False
        # (i.e. when the const-token sequence actually contains motion tokens).
        # ``_cached_total_seq_len > N_noisy + N_ref`` indicates motion is
        # present in the sequence.
        N_ref = pph * ppw
        if self._cached_total_seq_len > N_noisy + N_ref:
            # Motion-token spatial extents mirror FramePackMotioner's Conv3d
            # output sizes (kernel == stride == (1,2,2)/(2,4,4)/(4,8,8) on
            # latent input of size (T, 2*pph, 2*ppw)):
            #   * post: (1, pph,     ppw)       — proj output H = lat_h // 2 = pph
            #   * 2x:   (1, pph//2,  ppw//2)    — proj_2x output H = lat_h // 4 = pph // 2
            #   * 4x:   (4, pph//4,  ppw//4)    — proj_4x output H = lat_h // 8 = pph // 4
            # The rope grid_sizes must match these or trailing motion tokens
            # get zero rope (no spatial info → broken attention).
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

        # Allocate a placeholder of the right size; rope_precompute only reads
        # shape from it (specifically ``x.size(1)`` for the seq_len total).
        N_total = self._cached_total_seq_len
        placeholder = torch.zeros(1, N_total, num_heads, head_dim, dtype=torch.float32)

        freqs_complex = rope_precompute(placeholder, grid_sizes, freqs_ref, start=None)
        # freqs_complex: [1, N_total, num_heads, head_dim/2] complex
        # All heads carry the same rope; take a single head and broadcast later
        # via the kernel's accepted ``[1, 1, N, head_dim]`` layout.
        cos_half = freqs_complex.real[:, :, 0:1, :].float()  # [1, N_total, 1, head_dim/2]
        sin_half = freqs_complex.imag[:, :, 0:1, :].float()
        # Each rope slot covers a 2D rotation pair → duplicate so the last dim
        # is head_dim (matches Diffusers' ``repeat_interleave_real=True`` layout).
        cos_global = cos_half.repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)  # [1, 1, N_total, head_dim]
        sin_global = sin_half.repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
        cos_global = pad_vision_seq_parallel(cos_global, num_devices=sp_factor)
        sin_global = pad_vision_seq_parallel(sin_global, num_devices=sp_factor)

        cos_tt = from_torch(cos_global, device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
        sin_tt = from_torch(sin_global, device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
        trans_mat = _bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
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
        from ....utils.padding import get_padded_vision_seq_len

        sp_factor = self.parallel_config.sequence_parallel.factor
        padded_N = get_padded_vision_seq_len(N_total, sp_factor)
        cache_key = (audio_attn_id, padded_N)
        if cache_key in self._adain_modulation_cache:
            return self._adain_modulation_cache[cache_key]

        if self.audio_emb_global_token0_dev is None:
            raise RuntimeError("AdaIN enabled but audio_emb_global_token0_dev is None.")

        T_video = self.num_frames
        noisy_len = self.original_seq_len or N_total
        if noisy_len % T_video != 0:
            raise RuntimeError(f"noisy_len={noisy_len} not divisible by T_video={T_video}.")
        hw_per_frame = noisy_len // T_video

        # On-device per-frame projection: silu + Linear from the loaded
        # injector_adain_layers[audio_attn_id].linear weights. Input shape
        # [1, B, T, dim]; output [1, B, T, 2*dim].
        adain_layer = self.audio_injector.injector_adain_layers[audio_attn_id]
        proj_dev = adain_layer.linear(ttnn.silu(self.audio_emb_global_token0_dev))
        with torch.no_grad():
            projected = local_device_to_torch(proj_dev).reshape(1, T_video, 2 * self.dim).to(torch.float32)
            shift_per_frame, scale_per_frame = projected.chunk(2, dim=-1)
            shift_noisy = shift_per_frame.repeat_interleave(hw_per_frame, dim=1)
            # Fold +1 into the cached scale so the on-device op is `x * scale + shift`.
            scale_noisy = scale_per_frame.repeat_interleave(hw_per_frame, dim=1) + 1.0
            B = shift_noisy.shape[0]
            # Extend to padded_N with identity modulation for ref/motion + pad slots.
            if padded_N > noisy_len:
                shift_extra = torch.zeros(B, padded_N - noisy_len, self.dim, dtype=shift_noisy.dtype)
                scale_extra = torch.ones(B, padded_N - noisy_len, self.dim, dtype=scale_noisy.dtype)
                shift_per_token = torch.cat([shift_noisy, shift_extra], dim=1)
                scale_per_token = torch.cat([scale_noisy, scale_extra], dim=1)
            else:
                shift_per_token = shift_noisy
                scale_per_token = scale_noisy

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        # 2D shard matches spatial_1BND's [1, B, Sq/sp, D/tp] per-device layout
        # so binary_ng's last-two-dim broadcast classifier picks NONE.
        shard_mapping = {sp_axis: 2, tp_axis: 3}

        def _upload(t: torch.Tensor) -> ttnn.Tensor:
            return bf16_tensor_2dshard(
                t.unsqueeze(0).to(torch.float32).contiguous(),
                self.mesh_device,
                shard_mapping=shard_mapping,
                layout=ttnn.TILE_LAYOUT,
            )

        shift_tt = _upload(shift_per_token)
        scale_tt = _upload(scale_per_token)
        self._adain_modulation_cache[cache_key] = (shift_tt, scale_tt)
        return shift_tt, scale_tt

    def after_transformer_block(
        self,
        block_idx: int,
        spatial_1BND: ttnn.Tensor,
        N: int | None = None,
    ) -> ttnn.Tensor:
        """Apply the audio cross-attention residual at the configured layers.

        The reference rearranges spatial as ``"b (t n) c -> (b t) n c"`` and
        cross-attends each per-frame slice against per-frame audio K/V. We
        keep spatial SP-fractured and instead flatten audio K/V to
        ``[1, B, T*(N+1), dim]`` plus a block-diagonal frame mask — each Q
        token attends only to its own frame's audio tokens. No CCL ops, no
        rearrange. When ``enable_adain=True`` the pre-norm is the per-frame
        AdaIN modulation in :meth:`_apply_adain_pre_norm`; otherwise a plain
        LN-no-affine.
        """
        if block_idx not in self.audio_injector.injected_block_id:
            return spatial_1BND
        if self.merged_audio_emb_flat is None:
            raise RuntimeError("prepare_audio_emb() must be called before forward().")

        audio_attn_id = self.audio_injector.injected_block_id[block_idx]

        # Recover the unsharded Sq. The parent passes N; if not supplied, fall
        # back to original_seq_len (when ref/motion concat is wired) or infer
        # from the SP-sharded shape.
        if N is None:
            if self.original_seq_len:
                N = self.original_seq_len
            else:
                sp_factor = self.parallel_config.sequence_parallel.factor
                N = int(spatial_1BND.shape[-2]) * sp_factor

        # Pre-norm for the audio cross-attn. The reference uses a full-dim
        # ``nn.LayerNorm(elementwise_affine=False)`` over the spatial
        # representation. ``block.norm1`` is a no-affine
        # :class:`DistributedLayerNorm` that all-gathers stats across TP, so
        # it gives the correct full-D normalization here.
        block = self.blocks[block_idx]
        if self.enable_adain and self.audio_emb_global_token0_dev is not None:
            shift_full, scale_plus_one_full = self._build_adain_modulation_for_layer(audio_attn_id, N)
            x_normed = block.norm1(spatial_1BND)
            normed = ttnn.add(ttnn.multiply(x_normed, scale_plus_one_full), shift_full)
        else:
            normed = block.norm1(spatial_1BND)

        # Cross-attention with block-diagonal frame mask.
        mask = self._get_or_build_frame_attn_mask(N)
        residual = self.audio_injector.injector[audio_attn_id](
            spatial_1BND=normed,
            prompt_1BLP=self.merged_audio_emb_flat,
            N=int(spatial_1BND.shape[-2]),
            cross_attn_mask=mask,
        )
        # Zero the audio contribution at const (ref+motion) and pad positions
        # so only noisy tokens get the cross-attn residual — matches the
        # reference's noisy-only ``rearrange`` audio injection. Without this
        # gate, const tokens would absorb the uniform-attention output we
        # use to keep softmax finite over const rows.
        if self._cached_mask_noisy is not None:
            residual = ttnn.multiply(residual, self._cached_mask_noisy)
        return ttnn.add(spatial_1BND, residual)

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

        Inputs are CPU tensors of standard shape:
          * ``noisy_latents_torch``: ``[B=1, C=16, F, H, W]`` — the noisy latent
            shape (used only to derive ``N_noisy``).
          * ``ref_latent_torch``:    ``[B=1, C=16, 1, H, W]`` — reference image
            VAE-encoded latent.
          * ``motion_latents_torch``: ``[B=1, C=16, T_motion, H, W]`` — motion
            latents (typically all zeros in single-clip mode).
          * ``cond_states_torch``:   ``[B=1, C=16, F, H, W]`` pose video, or
            ``None`` to use the reference's default ``-1.0`` filler.

        Caches as on-device tensors:
          * ``_cached_pose_emb_1BND``       — pose contribution added to the
            patched noisy sequence (or None if pose path is inactive).
          * ``_cached_const_tokens_1BND``   — concat of patched ref + motion,
            each pre-summed with its ``trainable_cond_mask`` slot (1 / 2).
          * ``_cached_noisy_mask_emb_1BND`` — broadcast of
            ``trainable_cond_mask[0]`` to all noisy positions.

        Sets ``self.original_seq_len`` and ``self._cached_total_seq_len`` so
        ``inner_step`` and the audio-injection mask know the noisy / total
        sequence boundaries.
        """
        from ....utils.padding import get_padded_vision_seq_len

        B, C, F, H, W = noisy_latents_torch.shape
        pT, pH, pW = self.patch_size
        N_noisy = (F // pT) * (H // pH) * (W // pW)
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # --- Pose embedding (cond_encoder) ---
        # Upload the patchified pose SP-sharded on Sq so the matmul output
        # matches the spatial's 2D layout exactly (SP on Sq, TP on dim).
        if cond_states_torch is None:
            cond_states_torch = -torch.ones_like(noisy_latents_torch)
        cond_1BNI, _ = self._patchify_for_embed(cond_states_torch, self.patch_size)
        # Pad Sq to ``padded_N_noisy`` (tile*sp aligned) before sharding.
        padded_N_noisy = get_padded_vision_seq_len(N_noisy, sp_factor)
        if padded_N_noisy > N_noisy:
            cond_pad = torch.zeros(
                cond_1BNI.shape[0],
                cond_1BNI.shape[1],
                padded_N_noisy - N_noisy,
                cond_1BNI.shape[3],
                dtype=cond_1BNI.dtype,
            )
            cond_1BNI = torch.cat([cond_1BNI, cond_pad], dim=2)
        cond_dev = bf16_tensor(
            cond_1BNI, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2, layout=ttnn.TILE_LAYOUT
        )
        self._cached_pose_emb_1BND = self.cond_encoder(cond_dev)

        # --- Reference latent (patch_embedding shared with noisy) ---
        ref_1BNI, N_ref = self._patchify_for_embed(ref_latent_torch, self.patch_size)
        ref_dev = bf16_tensor(ref_1BNI, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        ref_emb_1BND = self.patch_embedding(ref_dev)  # [1, B, N_ref, dim]

        # --- Motion tokens (frame_packer) ---
        # Production config ``drop_first_motion=True`` → for the first clip
        # the model slices motion tokens to length 0 (see
        # ``wan/modules/s2v/model_s2v.py:485 process_motion`` with
        # ``drop_motion_frames=True``). Skip frame_packer entirely so the
        # const-token sequence is just ``ref`` (no spurious motion tokens
        # polluting cross-attn).
        if drop_first_motion:
            motion_emb_1BND = None
            N_motion = 0
        else:
            motion_emb_1BND, _motion_rope = self.frame_packer(motion_latents_torch)
            N_motion = int(motion_emb_1BND.shape[-2])

        # --- trainable_cond_mask additions ---
        # The table is tiny ([3, dim]). Pull to host once, gather there, and
        # upload per-token mask tensors directly — much simpler than chasing
        # ttnn.embedding's SP/TP-sharded output layout.
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

        # ref_emb_1BND (and motion_emb_1BND when present) are TP-fractured on
        # the last dim. All-gather along the TP mesh axis so the host sees
        # the full dim before mask-add.
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
        # Per-segment padded lengths, used by ``_get_or_build_frame_attn_mask``
        # so the audio attn mask is concat-per-device-aligned with the spatial
        # (which is built as ``concat([noisy_padded, const_padded], dim=-2)``).
        self._cached_padded_N_noisy = padded_N_noisy
        self._cached_padded_const = padded_const

        # --- Segmented timestep modulation masks (zero_timestep=True path) ---
        # Reference (wan/modules/s2v/model_s2v.py:WanModel_S2V.forward):
        #   if self.zero_timestep:
        #       t = torch.cat([t, torch.zeros(1)])
        #       e0_real, e0_zero = time_projection(time_embedding(t)).split(...)
        #       seg_idx = self.original_seq_len   # noisy / constant boundary
        # We follow the same idea with per-token masks: positions in the noisy
        # slot get real-t modulation, positions in the const (ref+motion) slot
        # get zero-t modulation.
        #
        # The spatial's per-device layout after ``ttnn.concat([spatial, const])``
        # is ``[noisy_local | const_local]`` per device, where each piece is
        # already SP-sharded (each device has padded_N_noisy/sp noisy slots
        # followed by padded_const/sp const slots). A naive global SP-shard
        # of a ``[padded_N_total, 1]`` mask would put global positions
        # ``[d*padded_N_total/sp, (d+1)*padded_N_total/sp)`` on device d —
        # that does **not** match the per-device concat layout when
        # padded_N_noisy/sp ≠ padded_N_total/sp. So we build each segment
        # separately and ``ttnn.concat`` on device, mirroring how the
        # spatial sequence was assembled above.
        def _upload_seg(t_BNF: torch.Tensor) -> ttnn.Tensor:
            return float32_tensor(
                t_BNF.contiguous(),
                device=self.mesh_device,
                mesh_axis=sp_axis,
                shard_dim=2,
                layout=ttnn.TILE_LAYOUT,
            )

        mask_n_noisy_torch = torch.zeros(1, 1, padded_N_noisy, 1, dtype=torch.float32)
        mask_n_noisy_torch[:, :, :N_noisy, :] = 1.0
        mask_n_const_torch = torch.zeros(1, 1, padded_const, 1, dtype=torch.float32)
        self._cached_mask_noisy = ttnn.concat(
            [_upload_seg(mask_n_noisy_torch), _upload_seg(mask_n_const_torch)], dim=-2
        )

        mask_c_noisy_torch = torch.zeros(1, 1, padded_N_noisy, 1, dtype=torch.float32)
        mask_c_const_torch = torch.zeros(1, 1, padded_const, 1, dtype=torch.float32)
        mask_c_const_torch[:, :, : N_ref + N_motion, :] = 1.0
        self._cached_mask_constant = ttnn.concat(
            [_upload_seg(mask_c_noisy_torch), _upload_seg(mask_c_const_torch)], dim=-2
        )

        # Zero-timestep projection: same shape as the per-step real-t
        # projection (``[1, B, 6, dim/tp]`` after unflatten), but constant
        # across the diffusion loop. Cache once per clip.
        zero_t_torch = torch.zeros(B, 1, 1, 1, dtype=torch.float32)
        zero_t_tt = float32_tensor(zero_t_torch, device=self.mesh_device)
        _, self._cached_timestep_proj_zero = self.prepare_timestep_conditioning(zero_t_tt)

    @staticmethod
    def _patchify_for_embed(x_BCTHW: torch.Tensor, patch_size: tuple[int, int, int]) -> tuple[torch.Tensor, int]:
        """Host-side unfold for the ``WanPatchEmbed``-style projections.

        Returns ``([1, B, N, pT*pH*pW*C], N)``.
        """
        from ....utils.padding import pad_vision_seq_parallel  # local import to avoid cycle

        pT, pH, pW = patch_size
        B, C, F, H, W = x_BCTHW.shape
        patch_F, patch_H, patch_W = F // pT, H // pH, W // pW
        N = patch_F * patch_H * patch_W
        x = x_BCTHW.reshape(B, C, patch_F, pT, patch_H, pH, patch_W, pW)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, B, N, pT * pH * pW * C)
        return x, N

    # ----------------------------------------------------------------------
    # Segmented timestep modulation. Mirrors
    # ``WanS2VAttentionBlock.forward`` (wan/modules/s2v/model_s2v.py): real-t
    # scale/shift/gate apply to the noisy slot, zero-t to ref+motion. The
    # reference does this by `cat` of per-segment slices; we do the math-
    # equivalent thing under SP fracturing via per-token masks (built once
    # per clip in :meth:`prepare_cond_emb`).
    #
    # Note: this is the unfused path. The base block's ``norm1`` /
    # ``norm3`` fuse the affine modulation into the DistributedLayerNorm
    # kernel via ``dynamic_weight`` / ``dynamic_bias``. That kernel requires
    # the weight tile to be height ``TILE_HEIGHT=32`` along Sq, so per-token
    # weights aren't a fit. Instead we call LN with no affine and do the
    # multiply/add per-token outside. Similarly, the fused
    # ``attn1(addcmul_residual=..., addcmul_gate=...)`` path takes a single
    # per-sample gate; per-token gating happens as an explicit
    # ``residual + attn_out * gate`` here.
    # ----------------------------------------------------------------------

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
        """Run one transformer block with per-segment (real-t/zero-t) modulation.

        Reference (``WanS2VAttentionBlock.forward`` in
        ``wan/modules/s2v/model_s2v.py``):

            e_real = (modulation + e_real).chunk(6)
            e_zero = (modulation + e_zero).chunk(6)
            norm_x = LN_no_affine(x)
            parts = [norm_x[:noisy] * (1+scale_real) + shift_real,
                     norm_x[noisy:] * (1+scale_zero) + shift_zero]
            ... self-attn, gate by [gate_real, gate_zero] segment-wise ...
            ... cross-attn ...
            norm2_x = LN_no_affine(x); same per-segment c_scale/c_shift dance
            ... ffn, gate by [c_gate_real, c_gate_zero] segment-wise ...

        We do the math-equivalent thing with masks: per-token tensors built
        as ``real * mask_noisy + zero * mask_constant`` so SP fracturing on
        Sq lines up trivially. Pad slots have both masks 0 → identity-zero
        modulation; the gate then zeroes the attn/ffn contribution, so pad
        slots passthrough unchanged.
        """
        shifted_real = ttnn.add(block.scale_shift_table.data, timestep_proj_real)
        shifted_zero = ttnn.add(block.scale_shift_table.data, timestep_proj_zero)

        shift_r, scale_r, gate_r, c_shift_r, c_scale_r, c_gate_r = ttnn.chunk(shifted_real, 6, dim=2)
        shift_z, scale_z, gate_z, c_shift_z, c_scale_z, c_gate_z = ttnn.chunk(shifted_zero, 6, dim=2)

        # Match the base block's fp32→bf16 cast for the gates (the addcmul
        # path uses bf16 because fp32 gate input was less accurate).
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

        # Self-attention: LN(no affine) + per-token (1+scale) * x + shift.
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
        spatial_1BND = ttnn.add(spatial_1BND, ttnn.multiply(attn_out, gate_msa))

        # Cross-attention: norm2 keeps its learned affine and ignores
        # segmented modulation (matches the reference, which calls
        # ``self.norm2(x).float()`` with no per-segment scale/shift here).
        spatial_normed = block.norm2(spatial_1BND)
        attn_out = block.attn2(
            spatial_1BND=spatial_normed,
            N=N,
            prompt_1BLP=prompt_1BLP,
        )
        spatial_1BND = ttnn.add(spatial_1BND, attn_out)

        # FFN: LN(no affine) + per-token (1+c_scale) * x + c_shift, then
        # explicit AG (TP > 1) → FFN → per-token c_gate residual.
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
        spatial_1BND = ttnn.add(spatial_1BND, ttnn.multiply(ffn_out, c_gate_msa))

        return spatial_1BND

    # ----------------------------------------------------------------------
    # inner_step override. Same shape contract as the base, but inserts the
    # audio-injector hook after every block.
    # ----------------------------------------------------------------------

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
        """S2V variant of ``WanTransformer3DModel.inner_step``.

        Extends the base block-stack with the S2V conditioning paths:

          1. Pose (``cond_encoder``) and noisy-slot ``trainable_cond_mask``
             tokens are added to the patched noisy sequence.
          2. The constant ref + motion tokens (each pre-summed with their
             ``trainable_cond_mask`` slot in :meth:`prepare_cond_emb`) are
             concatenated along Sq.
          3. The block loop runs on the extended ``Sq = noisy + ref + motion``
             sequence with audio cross-attention masked to noisy tokens only.
          4. Before the head, the ref + motion tokens are sliced off so the
             scheduler step operates only on the noisy-noise prediction.
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

        # Slice off the ref + motion constant tokens before the head so the
        # downstream scheduler step sees only the noisy-noise prediction.
        if has_cond:
            from ....utils.padding import get_padded_vision_seq_len

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
