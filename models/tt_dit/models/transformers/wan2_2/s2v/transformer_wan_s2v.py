# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""WAN 2.2 Speech-to-Video DiT."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn

from .....layers.embeddings import WanPatchEmbed
from .....layers.module import Parameter
from .....utils.mochi import get_rot_transformation_mat
from .....utils.padding import get_padded_vision_seq_len, pad_vision_seq_parallel
from .....utils.tensor import bf16_tensor, bf16_tensor_2dshard, float32_tensor, from_torch, local_device_to_torch
from ..transformer_wan import WanTransformer3DModel
from .audio_utils import AudioInjector_WAN, CausalAudioEncoder
from .motioner import FramePackMotionerWan
from .rope_s2v import rope_precompute


@dataclass
class _S2VCaches:
    """Build-once device caches, reused across clips for the lifetime of the model."""

    frame_attn_mask: dict = field(default_factory=dict)
    adain_modulation: dict = field(default_factory=dict)
    adain_shape: dict = field(default_factory=dict)  # (E, zero_row, one_row) per shape
    noisy_mask_emb: dict = field(default_factory=dict)
    pose_emb_zero: dict = field(default_factory=dict)
    mask_noisy: dict = field(default_factory=dict)
    mask_constant: dict = field(default_factory=dict)
    timestep_proj_zero: ttnn.Tensor | None = None
    mask_table_torch: torch.Tensor | None = None


@dataclass
class _S2VClipState:
    """Per-clip state, repopulated by prepare_audio_emb / prepare_cond_emb."""

    merged_audio_emb_flat: ttnn.Tensor | None = None
    audio_emb_global_token0_dev: ttnn.Tensor | None = None
    num_frames: int = 0
    num_audio_tokens_per_frame: int = 0
    original_seq_len: int = 0
    total_seq_len: int = 0
    padded_N_noisy: int = 0
    padded_const: int = 0
    pose_emb: ttnn.Tensor | None = None
    const_tokens: ttnn.Tensor | None = None
    noisy_mask_emb: ttnn.Tensor | None = None
    mask_noisy: ttnn.Tensor | None = None
    mask_constant: ttnn.Tensor | None = None


def _slice_and_adjust_T(
    x: ttnn.Tensor, B: int, T_have: int, K: int, D: int, mf_lat: int, target_T: int | None
) -> ttnn.Tensor:
    """Trim mf_lat from front; pad/trim to target_T (right-repeat last frame if short)."""
    sliced = ttnn.slice(x, [0, mf_lat, 0, 0], [B, T_have, K, D])
    T_post = T_have - mf_lat
    if target_T is None or target_T == T_post:
        return sliced
    if target_T < T_post:
        return ttnn.slice(sliced, [0, 0, 0, 0], [B, target_T, K, D])
    last_frame = ttnn.slice(sliced, [0, T_post - 1, 0, 0], [B, T_post, K, D])
    return ttnn.concat([sliced] + [last_frame] * (target_T - T_post), dim=1)


class WanS2VTransformer3DModel(WanTransformer3DModel):
    """Speech-to-video variant of the WAN 2.2 DiT."""

    def __init__(
        self,
        *,
        audio_dim: int = 1024,
        num_audio_layers: int = 25,
        num_audio_token: int = 4,
        audio_inject_layers: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39),
        enable_adain: bool = False,
        cond_dim: int = 16,
        # Production is FramePacker-only; reject other combos at construction time.
        enable_motioner: bool = False,
        enable_framepack: bool = True,
        num_layers: int = 40,
        **kwargs,
    ) -> None:
        if enable_motioner or not enable_framepack:
            raise NotImplementedError(
                f"Only enable_framepack=True is supported (got enable_motioner={enable_motioner}, "
                f"enable_framepack={enable_framepack})"
            )
        kwargs.setdefault("model_type", "s2v")
        super().__init__(num_layers=num_layers, **kwargs)

        self.audio_dim = audio_dim
        self.num_audio_layers = num_audio_layers
        self.num_audio_token = num_audio_token
        self.audio_inject_layers = tuple(audio_inject_layers)
        self.enable_adain = enable_adain
        self.cond_dim = cond_dim

        self.audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            num_layers=num_audio_layers,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain,
            mesh_device=self.mesh_device,
            tp_mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=self.ccl_manager,
        )
        self.audio_injector = AudioInjector_WAN(
            dim=self.dim,
            num_heads=self.num_heads,
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
        # [3, dim] table: rows 0/1/2 = noisy/ref/motion mask embedding.
        self.trainable_cond_mask = Parameter(
            total_shape=[3, self.dim],
            device=self.mesh_device,
        )
        self.frame_packer = FramePackMotionerWan(
            in_channels=16,
            inner_dim=self.dim,
            num_heads=self.num_heads,
            zip_frame_buckets=(1, 2, 16),
            drop_mode="padd",
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )

        self._caches = _S2VCaches()
        self._clip = _S2VClipState()

    def prepare_audio_emb(
        self,
        wav2vec2_layers_torch: torch.Tensor,
        *,
        motion_frames: tuple[int, int] = (17, 5),
        target_num_frames: int | None = None,
    ) -> None:
        """Run CausalAudioEncoder on wav2vec2 features and cache per-frame audio tokens."""
        # Reference pre-pads on the left by motion_frames[0] first-frame repeats.
        pre = wav2vec2_layers_torch[..., :1].expand(-1, -1, -1, motion_frames[0])
        audio_input = torch.cat([pre, wav2vec2_layers_torch], dim=-1)

        audio_emb_out = self.audio_encoder(audio_input)
        if self.enable_adain:
            audio_global_emb, audio_local_emb = audio_emb_out
        else:
            audio_global_emb, audio_local_emb = None, audio_emb_out

        mf_lat = motion_frames[1]
        B = int(audio_local_emb.shape[0])
        T_have = int(audio_local_emb.shape[1])
        num_tok_p1 = int(audio_local_emb.shape[2])
        dim = int(audio_local_emb.shape[3])

        local_dev = _slice_and_adjust_T(audio_local_emb, B, T_have, num_tok_p1, dim, mf_lat, target_num_frames)
        T_video = int(local_dev.shape[1])
        self._clip.num_frames = T_video
        self._clip.num_audio_tokens_per_frame = num_tok_p1

        if self._clip.merged_audio_emb_flat is not None:
            ttnn.deallocate(self._clip.merged_audio_emb_flat)
        self._clip.merged_audio_emb_flat = ttnn.reshape(local_dev, [1, B, T_video * num_tok_p1, dim])
        self.audio_injector.invalidate_audio_kv_cache()

        if self._clip.audio_emb_global_token0_dev is not None:
            ttnn.deallocate(self._clip.audio_emb_global_token0_dev)
            self._clip.audio_emb_global_token0_dev = None
        if audio_global_emb is not None:
            global_dev = _slice_and_adjust_T(audio_global_emb, B, T_have, 1, dim, mf_lat, target_num_frames)
            self._clip.audio_emb_global_token0_dev = ttnn.reshape(global_dev, [1, B, T_video, dim])
        for kv in self._caches.adain_modulation.values():
            for t in kv:
                ttnn.deallocate(t)
        self._caches.adain_modulation = {}

    def _get_or_build_frame_attn_mask(self, N_total: int) -> ttnn.Tensor:
        """Block-diagonal audio cross-attn mask. Const rows are 0.0 (uniform,
        not -inf to avoid softmax NaN); their residual is zeroed downstream."""
        T_video = self._clip.num_frames
        K_per_frame = self._clip.num_audio_tokens_per_frame
        Sk = T_video * K_per_frame
        padded_N_noisy = self._clip.padded_N_noisy
        padded_const = self._clip.padded_const
        cache_key = (padded_N_noisy, padded_const, Sk)
        if cache_key in self._caches.frame_attn_mask:
            return self._caches.frame_attn_mask[cache_key]

        noisy_len = self._clip.original_seq_len
        hw_per_frame = noisy_len // T_video
        noisy_mask_torch = torch.full((1, 1, padded_N_noisy, Sk), float("-inf"), dtype=torch.float32)
        for t in range(T_video):
            noisy_mask_torch[
                ..., t * hw_per_frame : (t + 1) * hw_per_frame, t * K_per_frame : (t + 1) * K_per_frame
            ] = 0.0

        # TILE_LAYOUT pads Sk; pad_value=-inf on the noisy mask is load-bearing
        # so SDPA softmax ignores the zero-filled padded K columns.
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        upload_kwargs = dict(
            device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_axes=[None, None, sp_axis, None]
        )
        noisy_mask_tt = from_torch(noisy_mask_torch.contiguous(), pad_value=float("-inf"), **upload_kwargs)
        if padded_const > 0:
            const_mask_tt = from_torch(
                torch.zeros(1, 1, padded_const, Sk, dtype=torch.float32).contiguous(),
                pad_value=0.0,
                **upload_kwargs,
            )
            mask_tt = ttnn.concat([noisy_mask_tt, const_mask_tt], dim=-2)
        else:
            mask_tt = noisy_mask_tt
        self._caches.frame_attn_mask[cache_key] = mask_tt
        return mask_tt

    def get_rope_features(self, hidden_states):
        # Override: include _clip.total_seq_len in the key since S2V rope size
        # grows with the const segment (clip 0 vs clip 1+).
        key = (tuple(hidden_states.shape), int(self._clip.total_seq_len))
        if key not in self.cached_rope_features:
            self.cached_rope_features[key] = self.prepare_rope_features(hidden_states)
        return self.cached_rope_features[key]

    def prepare_rope_features(self, hidden_states):
        """RoPE for the extended noisy + ref + motion Sq (rope_precompute
        on grid_sizes matching WanModel_S2V.forward)."""
        if self._clip.total_seq_len <= self._clip.original_seq_len:
            return super().prepare_rope_features(hidden_states)

        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        _, _, F, H, W = hidden_states.shape
        pT, pH, pW = self.patch_size
        ppf, pph, ppw = F // pT, H // pH, W // pW
        N_noisy = ppf * pph * ppw
        head_dim = self.dim // self.num_heads

        def _grid(start_xyz, end_xyz, range_xyz):
            return [
                torch.tensor(start_xyz, dtype=torch.long).unsqueeze(0),
                torch.tensor(end_xyz, dtype=torch.long).unsqueeze(0),
                torch.tensor(range_xyz, dtype=torch.long).unsqueeze(0),
            ]

        # Noisy + ref. Ref at temporal slot 30; motion buckets (if present) at
        # negative offsets so rope_precompute conjugates their sin.
        grid_sizes = [
            _grid([0, 0, 0], [ppf, pph, ppw], [ppf, pph, ppw]),
            _grid([30, 0, 0], [31, pph, ppw], [1, pph, ppw]),
        ]
        N_ref = pph * ppw
        if self._clip.total_seq_len > N_noisy + N_ref:
            zb = self.frame_packer.zip_frame_buckets
            grid_sizes += [
                _grid([-zb[0], 0, 0], [0, pph, ppw], [zb[0], pph, ppw]),
                _grid(
                    [-(zb[0] + zb[1]), 0, 0],
                    [-(zb[0] + zb[1]) + zb[1] // 2, pph // 2, ppw // 2],
                    [zb[1], pph, ppw],
                ),
                _grid(
                    [-(zb[0] + zb[1] + zb[2]), 0, 0],
                    [-(zb[0] + zb[1] + zb[2]) + zb[2] // 4, pph // 4, ppw // 4],
                    [zb[2], pph, ppw],
                ),
            ]

        N_total = self._clip.total_seq_len
        N_const = N_total - N_noisy
        placeholder = torch.zeros(1, N_total, self.num_heads, head_dim, dtype=torch.float32)
        freqs_complex = rope_precompute(placeholder, grid_sizes, self.frame_packer.freqs, start=None)
        # All heads carry the same rope; take head 0 and broadcast.
        # repeat_interleave(2) matches Diffusers' repeat_interleave_real=True layout.
        cos_global = freqs_complex.real[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
        sin_global = freqs_complex.imag[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)

        # Per-segment SP-shard then on-device concat — matches the per-device
        # spatial layout [noisy_local | const_local].
        upload_kwargs = dict(device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])

        def _seg(slice_obj):
            return from_torch(pad_vision_seq_parallel(slice_obj, num_devices=sp_factor).contiguous(), **upload_kwargs)

        cos_n_tt = _seg(cos_global[:, :, :N_noisy, :])
        sin_n_tt = _seg(sin_global[:, :, :N_noisy, :])
        if N_const > 0:
            cos_tt = ttnn.concat([cos_n_tt, _seg(cos_global[:, :, N_noisy:N_total, :])], dim=-2)
            sin_tt = ttnn.concat([sin_n_tt, _seg(sin_global[:, :, N_noisy:N_total, :])], dim=-2)
        else:
            cos_tt, sin_tt = cos_n_tt, sin_n_tt
        trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
        return cos_tt, sin_tt, trans_mat

    def _build_adain_modulation_for_layer(
        self,
        audio_attn_id: int,
        N_total: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Per-token (shift, scale+1) for one inject layer via sparse-selector E."""
        sp_factor = self.parallel_config.sequence_parallel.factor
        noisy_len = self._clip.original_seq_len
        const_len = N_total - noisy_len
        padded_noisy = get_padded_vision_seq_len(noisy_len, sp_factor)
        padded_const = get_padded_vision_seq_len(const_len, sp_factor) if const_len > 0 else 0
        cache_key = (audio_attn_id, padded_noisy, padded_const)
        if cache_key in self._caches.adain_modulation:
            return self._caches.adain_modulation[cache_key]

        T_video = self._clip.num_frames
        shape_key = (padded_noisy, padded_const, T_video)
        if shape_key not in self._caches.adain_shape:
            padded_N = padded_noisy + padded_const
            hw_per_frame = noisy_len // T_video
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            tp_axis = self.parallel_config.tensor_parallel.mesh_axis

            E_torch = torch.zeros(padded_N, T_video + 1, dtype=torch.float32)
            for i in range(noisy_len):
                E_torch[i, i // hw_per_frame] = 1.0
            for i in range(noisy_len, padded_N):
                E_torch[i, T_video] = 1.0
            E_tt = bf16_tensor(
                E_torch.unsqueeze(0).unsqueeze(0).contiguous(),
                device=self.mesh_device,
                mesh_axis=sp_axis,
                shard_dim=2,
                layout=ttnn.TILE_LAYOUT,
            )
            sentinel = dict(device=self.mesh_device, mesh_axis=tp_axis, shard_dim=3, layout=ttnn.TILE_LAYOUT)
            zero_row_tt = bf16_tensor(torch.zeros(1, 1, 1, self.dim, dtype=torch.float32), **sentinel)
            one_row_tt = bf16_tensor(torch.ones(1, 1, 1, self.dim, dtype=torch.float32), **sentinel)
            self._caches.adain_shape[shape_key] = (E_tt, zero_row_tt, one_row_tt)
        E_tt, zero_row_tt, one_row_tt = self._caches.adain_shape[shape_key]

        adain_layer = self.audio_injector.injector_adain_layers[audio_attn_id]
        proj_dev = adain_layer.linear(ttnn.silu(self._clip.audio_emb_global_token0_dev))
        shift_pf, scale_pf = ttnn.chunk(proj_dev, 2, dim=-1)
        scale_pf_p1 = ttnn.add(scale_pf, 1.0)
        shift_ext = ttnn.concat([shift_pf, zero_row_tt], dim=-2)
        scale_ext = ttnn.concat([scale_pf_p1, one_row_tt], dim=-2)
        shift_tt = ttnn.matmul(E_tt, shift_ext)
        scale_tt = ttnn.matmul(E_tt, scale_ext)

        self._caches.adain_modulation[cache_key] = (shift_tt, scale_tt)
        return shift_tt, scale_tt

    def after_transformer_block(
        self,
        block_idx: int,
        spatial_1BND: ttnn.Tensor,
        N: int,
    ) -> ttnn.Tensor:
        """Apply audio cross-attention residual at configured inject layers.

        Keeps spatial SP-fractured and feeds flattened audio K/V + a block-
        diagonal frame mask (no rearrange, no CCL).
        """
        if block_idx not in self.audio_injector.injected_block_id:
            return spatial_1BND

        audio_attn_id = self.audio_injector.injected_block_id[block_idx]
        block = self.blocks[block_idx]

        # AdaIN modulation: per-token shift/scale fused into norm1 via addcmul.
        # Can't use norm1's dynamic_weight hook (requires per-batch gamma).
        if self.enable_adain and self._clip.audio_emb_global_token0_dev is not None:
            shift_full, scale_plus_one_full = self._build_adain_modulation_for_layer(audio_attn_id, N)
            normed = ttnn.addcmul(shift_full, block.norm1(spatial_1BND), scale_plus_one_full)
        else:
            normed = block.norm1(spatial_1BND)

        mask = self._get_or_build_frame_attn_mask(N)
        cached_kv = self.audio_injector._audio_kv_cache.get(audio_attn_id)  # noqa: SLF001
        if cached_kv is None:
            residual, fresh_kv = self.audio_injector.injector[audio_attn_id](
                spatial_1BND=normed,
                prompt_1BLP=self._clip.merged_audio_emb_flat,
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
        # Zero the residual at const + pad rows (const used uniform-0 attention
        # for softmax stability; we don't want that bleeding into the output).
        if self._clip.mask_noisy is not None:
            return ttnn.addcmul(spatial_1BND, residual, self._clip.mask_noisy)
        return ttnn.add(spatial_1BND, residual)

    def prepare_cond_emb(
        self,
        noisy_latents_torch: torch.Tensor,
        *,
        ref_latent_torch: torch.Tensor,
        motion_latents_torch: torch.Tensor,
        cond_states_torch: torch.Tensor | None = None,
        drop_first_motion: bool = True,
    ) -> None:
        """Build per-clip caches for the S2V conditioning paths."""
        # Only ``_clip.const_tokens`` is input-dependent; everything else
        # is shape-keyed and reused across clips.
        if self._clip.const_tokens is not None:
            ttnn.deallocate(self._clip.const_tokens)
            self._clip.const_tokens = None

        B, C, F, H, W = noisy_latents_torch.shape
        pT, pH, pW = self.patch_size
        N_noisy = (F // pT) * (H // pH) * (W // pW)
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        padded_N_noisy = get_padded_vision_seq_len(N_noisy, sp_factor)

        # Pose embedding. No-pose path (cond_states=None) is shape-cached.
        if cond_states_torch is None:
            cache = self._caches.pose_emb_zero
            if padded_N_noisy not in cache:
                cond_zero = torch.zeros_like(noisy_latents_torch)
                cond_1BNI, _ = self.preprocess_spatial_input_host(cond_zero, pad=True)
                cond_dev = bf16_tensor(
                    cond_1BNI, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2, layout=ttnn.TILE_LAYOUT
                )
                cache[padded_N_noisy] = self.cond_encoder(cond_dev)
            self._clip.pose_emb = cache[padded_N_noisy]
        else:
            cond_1BNI, _ = self.preprocess_spatial_input_host(cond_states_torch, pad=True)
            cond_dev = bf16_tensor(
                cond_1BNI, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2, layout=ttnn.TILE_LAYOUT
            )
            self._clip.pose_emb = self.cond_encoder(cond_dev)

        # Reference latent: un-padded so the on-device concat with noisy is exact.
        ref_1BNI, N_ref = self.preprocess_spatial_input_host(ref_latent_torch, pad=False)
        ref_emb_1BND = self.patch_embedding(bf16_tensor(ref_1BNI, device=self.mesh_device, layout=ttnn.TILE_LAYOUT))

        if drop_first_motion:
            motion_emb_1BND, N_motion = None, 0
        else:
            motion_emb_1BND = self.frame_packer(motion_latents_torch)
            N_motion = int(motion_emb_1BND.shape[-2])

        N_total = N_noisy + N_ref + N_motion
        padded_const = get_padded_vision_seq_len(N_ref + N_motion, sp_factor)

        # trainable_cond_mask is a static [3, dim] Parameter; D2H it once.
        if self._caches.mask_table_torch is None:
            with torch.no_grad():
                self._caches.mask_table_torch = (
                    local_device_to_torch(self.trainable_cond_mask.data).reshape(3, self.dim).to(torch.float32)
                )
        mask_table = self._caches.mask_table_torch
        with torch.no_grad():
            const_mask_torch = torch.cat(
                [
                    mask_table[1:2].expand(N_ref, self.dim),
                    mask_table[2:3].expand(N_motion, self.dim),
                    torch.zeros(padded_const - (N_ref + N_motion), self.dim),
                ],
                dim=0,
            ).view(1, 1, padded_const, self.dim)

        # Noisy-mask embedding: shape-only, cache by padded_N_noisy.
        if padded_N_noisy not in self._caches.noisy_mask_emb:
            with torch.no_grad():
                noisy_mask_torch = mask_table[0:1].view(1, 1, 1, self.dim).expand(1, 1, padded_N_noisy, self.dim)
                self._caches.noisy_mask_emb[padded_N_noisy] = bf16_tensor_2dshard(
                    noisy_mask_torch.contiguous(),
                    self.mesh_device,
                    shard_mapping={sp_axis: 2, tp_axis: 3},
                    layout=ttnn.TILE_LAYOUT,
                )
        self._clip.noisy_mask_emb = self._caches.noisy_mask_emb[padded_N_noisy]

        # ref/motion are TP-fractured; gather before the host mask-add.
        def _gather_tp(t):
            if self.parallel_config.tensor_parallel.factor > 1:
                return self.ccl_manager.all_gather_persistent_buffer(t, dim=3, mesh_axis=tp_axis)
            return t

        ref_torch = local_device_to_torch(_gather_tp(ref_emb_1BND)).reshape(1, B, N_ref, self.dim).to(torch.float32)
        const_token_segments = [ref_torch]
        if N_motion > 0:
            motion_torch = (
                local_device_to_torch(_gather_tp(motion_emb_1BND)).reshape(1, B, N_motion, self.dim).to(torch.float32)
            )
            const_token_segments.append(motion_torch)
        const_token_segments.append(torch.zeros(1, B, padded_const - (N_ref + N_motion), self.dim))
        const_torch = torch.cat(const_token_segments, dim=2) + const_mask_torch
        self._clip.const_tokens = bf16_tensor_2dshard(
            const_torch.contiguous(),
            self.mesh_device,
            shard_mapping={sp_axis: 2, tp_axis: 3},
            layout=ttnn.TILE_LAYOUT,
        )

        self._clip.original_seq_len = N_noisy
        self._clip.total_seq_len = N_total
        self._clip.padded_N_noisy = padded_N_noisy
        self._clip.padded_const = padded_const

        # Segmented timestep modulation masks. bf16 represents 0/1 exactly so no
        # dtype promotion downstream when multiplied by spatial/residual.
        seg_upload = dict(device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2, layout=ttnn.TILE_LAYOUT)

        def _build_seg_mask(padded_n: int, padded_c: int, n_active_noisy: int, c_active_start: int, c_active_end: int):
            mn = torch.zeros(1, 1, padded_n, 1, dtype=torch.bfloat16)
            mn[:, :, :n_active_noisy, :] = 1.0
            mc = torch.zeros(1, 1, padded_c, 1, dtype=torch.bfloat16)
            mc[:, :, c_active_start:c_active_end, :] = 1.0
            return ttnn.concat(
                [bf16_tensor(mn.contiguous(), **seg_upload), bf16_tensor(mc.contiguous(), **seg_upload)], dim=-2
            )

        mn_key = (padded_N_noisy, padded_const, N_noisy)
        if mn_key not in self._caches.mask_noisy:
            self._caches.mask_noisy[mn_key] = _build_seg_mask(padded_N_noisy, padded_const, N_noisy, 0, 0)
        self._clip.mask_noisy = self._caches.mask_noisy[mn_key]

        mc_key = (padded_N_noisy, padded_const, N_ref, N_motion)
        if mc_key not in self._caches.mask_constant:
            self._caches.mask_constant[mc_key] = _build_seg_mask(padded_N_noisy, padded_const, 0, 0, N_ref + N_motion)
        self._clip.mask_constant = self._caches.mask_constant[mc_key]

        # Zero-timestep projection is constant across audio/clip — build once.
        if self._caches.timestep_proj_zero is None:
            zero_t_tt = float32_tensor(torch.zeros(B, 1, 1, 1, dtype=torch.float32), device=self.mesh_device)
            _, self._caches.timestep_proj_zero = self.prepare_timestep_conditioning(zero_t_tt)

        # Pre-build per-layer AdaIN modulation so step 0 doesn't pay it inline.
        if self.enable_adain and self._clip.audio_emb_global_token0_dev is not None:
            for audio_attn_id in range(len(self.audio_inject_layers)):
                self._build_adain_modulation_for_layer(audio_attn_id, self._clip.total_seq_len)

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
        """Block forward with per-token segmented modulation (real-t on noisy,
        zero-t on const). Pad slots have both masks zero so the gate zeros
        attn/ffn out — pad passthrough is automatic.

        Unfused norm + modulation: base block's fused path requires TILE-aligned
        weight along Sq which conflicts with per-token weights.
        """

        # Cast all 6 chunks to bf16 (matches T2V; fp32 gates lose accuracy and
        # downstream multiply with the bf16 mask would otherwise auto-promote).
        def _chunks_bf16(x):
            return [ttnn.typecast(c, dtype=ttnn.bfloat16) for c in ttnn.chunk(x, 6, dim=2)]

        shift_r, scale_r, gate_r, c_shift_r, c_scale_r, c_gate_r = _chunks_bf16(
            ttnn.add(block.scale_shift_table.data, timestep_proj_real)
        )
        shift_z, scale_z, gate_z, c_shift_z, c_scale_z, c_gate_z = _chunks_bf16(
            ttnn.add(block.scale_shift_table.data, timestep_proj_zero)
        )

        def _per_token(real_chunk, zero_chunk):
            # [1, B, 1, D/tp] × [1, 1, padded_N/sp, 1] → [1, B, padded_N/sp, D/tp].
            return ttnn.add(ttnn.multiply(real_chunk, mask_noisy), ttnn.multiply(zero_chunk, mask_constant))

        shift_msa = _per_token(shift_r, shift_z)
        scale_msa = _per_token(scale_r, scale_z)
        gate_msa = _per_token(gate_r, gate_z)
        c_shift_msa = _per_token(c_shift_r, c_shift_z)
        c_scale_msa = _per_token(c_scale_r, c_scale_z)
        c_gate_msa = _per_token(c_gate_r, c_gate_z)

        # Self-attention. NOTE: add(multiply, ...) instead of addcmul — addcmul's
        # binary_ng subtile-broadcast asserts on certain padded Sq lengths from
        # lat_target_frames=20 (e.g. 8224 tiles on (2, 4) BH 480p).
        spatial_normed = ttnn.add(ttnn.multiply(block.norm1(spatial_1BND), ttnn.add(scale_msa, 1.0)), shift_msa)
        attn_out = block.attn1(
            spatial_1BND=spatial_normed, N=N, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat
        )
        spatial_1BND = ttnn.add(spatial_1BND, ttnn.multiply(attn_out, gate_msa))

        # Cross-attention (no per-segment modulation; norm2 has learned affine).
        attn_out = block.attn2(spatial_1BND=block.norm2(spatial_1BND), N=N, prompt_1BLP=prompt_1BLP)
        spatial_1BND = ttnn.add(spatial_1BND, attn_out)

        # FFN
        spatial_normed = ttnn.add(ttnn.multiply(block.norm3(spatial_1BND), ttnn.add(c_scale_msa, 1.0)), c_shift_msa)
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        ffn_out = block.ffn(spatial_normed, compute_kernel_config=block.ff_compute_kernel_config)
        return ttnn.add(spatial_1BND, ttnn.multiply(ffn_out, c_gate_msa))

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
        """S2V denoising step: [noisy | ref + motion] on Sq with audio cross-attn
        hooked after each block; slice off const tokens before the head."""
        temb_11BD, timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)

        spatial_1BND = self.patch_embedding(spatial_1BNI)

        has_cond = self._clip.const_tokens is not None
        if has_cond:
            if self._clip.pose_emb is not None:
                spatial_1BND = ttnn.add(spatial_1BND, self._clip.pose_emb)
            if self._clip.noisy_mask_emb is not None:
                spatial_1BND = ttnn.add(spatial_1BND, self._clip.noisy_mask_emb)
            spatial_1BND = ttnn.concat([spatial_1BND, self._clip.const_tokens], dim=-2)
            N_block = self._clip.total_seq_len
        else:
            N_block = N

        use_segmented = has_cond and self._clip.mask_noisy is not None and self._caches.timestep_proj_zero is not None

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
                    timestep_proj_zero=self._caches.timestep_proj_zero,
                    mask_noisy=self._clip.mask_noisy,
                    mask_constant=self._clip.mask_constant,
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
            padded_noisy_per_dev = get_padded_vision_seq_len(self._clip.original_seq_len, sp_factor) // sp_factor
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
