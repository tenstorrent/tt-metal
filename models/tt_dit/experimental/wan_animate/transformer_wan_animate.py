# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TT-NN Wan2.2-Animate-14B transformer.

Builds on the tt_dit Wan 2.2 transformer (``WanTransformer3DModel``): the 40 blocks, RoPE,
time/text embedding, norm_out/proj_out and all parallelism are reused unchanged. Animate adds,
relative to Wan 2.2 I2V-A14B (see ``models/tt_dit/experimental/models/Wan2_2_Animate.md``):

  device  pose_patch_embedding   fused into the main patch-embed matmul (reference_math #1)
  device  attn2 image K/V         ``WanAttention(image_kv=True)``: CLIP context branch in every block
  device  face_adapter x8         ``WanAnimateFaceAdapter``: masked cross-attention after blocks 0,5,..,35
  device  face_encoder            ``WanAnimateFaceEncoder``: causal temporal convs as matmuls, once/segment
  host    motion_encoder          LIA-style CNN on 512x512 face crops (torch, once/segment)
  host    image_embedder          CLIP-feature MLP (torch, once/segment)

Weights load from the Diffusers checkpoint ``Wan-AI/Wan2.2-Animate-14B-Diffusers`` (subfolder
``transformer``) with ``strict=True``: every key in it has a home here.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_wan_animate import (
    WanAnimateMotionEncoder as TorchWanAnimateMotionEncoder,
)
from diffusers.models.transformers.transformer_wan_animate import WanImageEmbedding as TorchWanImageEmbedding
from loguru import logger

import ttnn

from models.tt_dit.layers.embeddings import WanPatchEmbed
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import DistributedLayerNorm, LayerNorm
from models.tt_dit.models.transformers.wan2_2.attention_wan import WanAttention
from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanTransformer3DModel
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.substate import pop_substate
from models.tt_dit.utils.tensor import bf16_tensor, from_torch
from models.tt_dit.experimental.wan_animate import reference_math as rm


class WanAnimateFaceAdapter(Module):
    """One Animate face block: x + to_out(attn(pre_norm(x) -> per-frame 5 motion tokens)).

    The per-latent-frame grouping is expressed as an additive block-diagonal mask over the flattened
    motion tokens (``reference_math.face_attention_mask``), so the queries keep the transformer's
    sequence-parallel layout and the attention itself is the existing ``WanAttention`` cross path.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.pre_norm_q = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        # Motion tokens are replicated (K <= a few hundred rows), so a plain LayerNorm suffices.
        self.pre_norm_kv = LayerNorm(
            dim, norm_eps=eps, norm_elementwise_affine=False, bias=False, mesh_device=mesh_device
        )
        self.attn = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=False,
            per_head_qk_norm=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Diffusers keys: to_q/to_k/to_v/to_out (no ".0"), norm_q/norm_k [head_dim].
        for name in list(state):
            if name.split(".")[0] in ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k"):
                state[f"attn.{name}"] = state.pop(name)

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        motion_1BKD: ttnn.Tensor,
        *,
        N: int | ttnn.Tensor,
        face_mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """spatial: fractured N on SP, D on TP. motion: replicated. face_mask: [1,1,N/sp,K] fractured on SP."""
        x_1BND = self.pre_norm_q(spatial_1BND)
        kv_1BKD = self.pre_norm_kv(motion_1BKD)
        out_1BND = self.attn(spatial_1BND=x_1BND, N=N, prompt_1BLP=kv_1BKD, cross_attn_mask=face_mask)
        return ttnn.add(spatial_1BND, out_1BND)


class WanAnimateFaceEncoder(Module):
    """Diffusers ``WanAnimateFaceEncoder`` on device, with each causal Conv1d expressed as
    ``Linear(concat_k(P_k @ X))`` (``reference_math.causal_conv_selectors``). Runs once per segment
    on ~77 x 512 inputs, so all weights are replicated (about 36 MB in bf16)."""

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        kernel_size: int = 3,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.mesh_device = mesh_device
        hifi4 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config = hifi4
        k = kernel_size
        self.conv1_local = Linear(
            k * in_dim, hidden_dim * num_heads, mesh_device=mesh_device, compute_kernel_config=hifi4
        )
        self.conv2 = Linear(k * hidden_dim, hidden_dim, mesh_device=mesh_device, compute_kernel_config=hifi4)
        self.conv3 = Linear(k * hidden_dim, hidden_dim, mesh_device=mesh_device, compute_kernel_config=hifi4)
        self.out_proj = Linear(hidden_dim, out_dim, mesh_device=mesh_device, compute_kernel_config=hifi4)
        self.norm = LayerNorm(
            hidden_dim, norm_eps=eps, norm_elementwise_affine=False, bias=False, mesh_device=mesh_device
        )
        self.padding_tokens: torch.Tensor | None = None  # host; appended after the device part
        self._selectors: dict[tuple, ttnn.Tensor] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("conv1_local", "conv2", "conv3"):
            w = state.pop(f"{name}.weight", None)
            if w is not None:
                # [C_out, C_in, k] -> [C_out, k * C_in], matching concat_k(P_k @ X) on the input side.
                state[f"{name}.weight"] = w.permute(0, 2, 1).reshape(w.shape[0], -1)
        if "padding_tokens" in state:
            self.padding_tokens = state.pop("padding_tokens").detach().float().clone()

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().save(directory, prefix=prefix)
        torch.save(self.padding_tokens, Path(directory) / f"{prefix}padding_tokens.pt")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)
        self.padding_tokens = torch.load(Path(directory) / f"{prefix}padding_tokens.pt")

    def _conv(self, x_11TC: ttnn.Tensor, linear: Linear, t_in: int, stride: int) -> tuple[ttnn.Tensor, int]:
        """Causal replicate-padded conv on a [1, 1, T_pad, C] tile tensor; returns (y, t_out)."""
        rows_in = x_11TC.shape[-2]
        t_out = rm.conv1d_out_len(t_in, self.kernel_size, stride)
        rows_out = rm.round_up(t_out, ttnn.TILE_SIZE)
        key = (t_in, stride, rows_in)
        if key not in self._selectors:
            sel = rm.causal_conv_selectors(t_in, kernel=self.kernel_size, stride=stride, rows=rows_out, cols=rows_in)
            self._selectors[key] = [
                bf16_tensor(sel[k][None, None], device=self.mesh_device) for k in range(sel.shape[0])
            ]
        taps = [ttnn.matmul(p, x_11TC, compute_kernel_config=self.compute_kernel_config) for p in self._selectors[key]]
        return linear(ttnn.concat(taps, dim=-1)), t_out

    def _ln_silu(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.silu(self.norm(x))

    def forward(self, motion_vec: torch.Tensor) -> torch.Tensor:
        """[B=1, T, in_dim] host motion vectors -> [1, T', heads + 1, out_dim] host tensor (Diffusers layout)."""
        b, t, c = motion_vec.shape
        assert b == 1, "batch size 1 only"
        rows = rm.round_up(t, ttnn.TILE_SIZE)
        x = torch.zeros(1, 1, rows, c)
        x[0, 0, :t] = motion_vec[0].float()
        x_tt = bf16_tensor(x, device=self.mesh_device)

        y, t1 = self._conv(x_tt, self.conv1_local, t, 1)
        heads = []
        for n in range(self.num_heads):
            h = ttnn.slice(y, [0, 0, 0, n * self.hidden_dim], [1, 1, y.shape[-2], (n + 1) * self.hidden_dim])
            h = self._ln_silu(h)
            h, t2 = self._conv(h, self.conv2, t1, 2)
            h = self._ln_silu(h)
            h, t3 = self._conv(h, self.conv3, t2, 2)
            h = self._ln_silu(h)
            h = self.out_proj(h)
            heads.append(ttnn.to_torch(ttnn.get_device_tensors(h)[0])[0, 0, :t3].float())
        out = torch.stack(heads, dim=1)[None]  # [1, T', heads, D]
        pad = self.padding_tokens.reshape(1, 1, 1, -1).expand(1, out.shape[1], 1, -1)
        return torch.cat([out, pad], dim=2)


class WanAnimateTransformer3DModel(WanTransformer3DModel):
    """Wan2.2-Animate transformer: the Wan 2.2 transformer plus the Animate modules."""

    POSE_CHANNELS = rm.POSE_EXTRA_CHANNELS

    def __init__(
        self,
        *,
        patch_size: tuple = (1, 2, 2),
        num_heads: int = 40,
        dim: int = 5120,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        image_dim: int = 1280,
        motion_encoder_size: int = 512,
        motion_style_dim: int = 512,
        motion_dim: int = 20,
        motion_encoder_dim: int = 512,
        motion_encoder_channel_sizes: dict | None = None,
        face_encoder_hidden_dim: int = 1024,
        face_encoder_num_heads: int = 4,
        inject_face_latents_blocks: int = 5,
        motion_encoder_batch_size: int = 8,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        assert in_channels == 36, "Animate uses the I2V input layout: 16 noise + 4 mask + 16 conditioning channels"
        super().__init__(
            patch_size=patch_size,
            num_heads=num_heads,
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            rope_max_seq_len=rope_max_seq_len,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            model_type="i2v",
            added_kv_proj_dim=dim,
        )
        self.num_heads = num_heads
        self.inject_face_latents_blocks = inject_face_latents_blocks
        self.motion_encoder_batch_size = motion_encoder_batch_size
        self.motion_encoder_size = motion_encoder_size

        # Replace the I2V patch embedding with the pose-fused one (36 + 17 input channels).
        self.patch_embedding = WanPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels + self.POSE_CHANNELS,
            embed_dim=dim,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.face_encoder = WanAnimateFaceEncoder(
            in_dim=motion_encoder_dim,
            out_dim=dim,
            hidden_dim=face_encoder_hidden_dim,
            num_heads=face_encoder_num_heads,
            eps=eps,
            mesh_device=mesh_device,
        )
        self.face_adapter = ModuleList(
            WanAnimateFaceAdapter(
                dim=dim,
                num_heads=num_heads,
                eps=eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
            )
            # ceil, not floor: equal to the checkpoint's 40 // 5 = 8, and still gives a truncated
            # (e.g. 1-block test) model the adapter that follows block 0.
            for _ in range(-(-num_layers // inject_face_latents_blocks))
        )

        # Host (torch) fallbacks: run once per segment, tiny relative to the 40 blocks x 20 steps.
        self.image_embedder = TorchWanImageEmbedding(image_dim, dim).eval()
        self.motion_encoder = TorchWanAnimateMotionEncoder(
            size=motion_encoder_size,
            style_dim=motion_style_dim,
            motion_dim=motion_dim,
            out_dim=motion_encoder_dim,
            channels=motion_encoder_channel_sizes,
        ).eval()
        self._mask_cache: dict[tuple, ttnn.Tensor] = {}

    # --------------------------------------------------------------------------------------------
    # Weights
    # --------------------------------------------------------------------------------------------
    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        super()._prepare_torch_state(state)

        image_state = pop_substate(state, "condition_embedder.image_embedder")
        if image_state:
            self.image_embedder.load_state_dict(image_state)
        motion_state = pop_substate(state, "motion_encoder")
        if motion_state:
            self.motion_encoder.load_state_dict(motion_state)

        if "pose_patch_embedding.weight" in state and "patch_embedding.weight" in state:
            weight, bias = rm.fuse_pose_patch_embedding(
                state["patch_embedding.weight"],
                state["patch_embedding.bias"],
                state.pop("pose_patch_embedding.weight"),
                state.pop("pose_patch_embedding.bias"),
            )
            state["patch_embedding.weight"] = weight
            state["patch_embedding.bias"] = bias

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().save(directory, prefix=prefix)
        directory = Path(directory)
        torch.save(self.image_embedder.state_dict(), directory / f"{prefix}image_embedder.pt")
        torch.save(self.motion_encoder.state_dict(), directory / f"{prefix}motion_encoder.pt")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)
        directory = Path(directory)
        self.image_embedder.load_state_dict(torch.load(directory / f"{prefix}image_embedder.pt"))
        self.motion_encoder.load_state_dict(torch.load(directory / f"{prefix}motion_encoder.pt"))

    # --------------------------------------------------------------------------------------------
    # Per-segment conditioning (host + device, computed once and reused for every step)
    # --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_motion(self, face_pixel_values: torch.Tensor) -> torch.Tensor:
        """[1, 3, T, S, S] face crops in [-1, 1] -> [1, T, motion_encoder_dim] (host, fp32)."""
        b, c, t, h, w = face_pixel_values.shape
        frames = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).float()
        out = [self.motion_encoder(batch) for batch in torch.split(frames, self.motion_encoder_batch_size)]
        return torch.cat(out).view(b, t, -1)

    @torch.no_grad()
    def prepare_face_conditioning(self, face_pixel_values: torch.Tensor) -> ttnn.Tensor:
        """Face crops -> replicated motion tokens [1, 1, K_pad, D] (zero frame for the reference latent)."""
        motion_vec = self.face_encoder(self.encode_motion(face_pixel_values))  # [1, T', 5, D]
        motion_vec = torch.cat([torch.zeros_like(motion_vec[:, :1]), motion_vec], dim=1)
        self._face_frames = motion_vec.shape[1]
        return bf16_tensor(rm.face_motion_tokens(motion_vec), device=self.mesh_device)

    @torch.no_grad()
    def prepare_image_conditioning(self, clip_hidden_states: torch.Tensor) -> ttnn.Tensor:
        """CLIP penultimate hidden states [1, 257, 1280] -> replicated image context [1, 1, 512, D]."""
        image_ctx = self.image_embedder(clip_hidden_states.float())
        return bf16_tensor(rm.pad_image_context(image_ctx), device=self.mesh_device)

    def get_masks(self, *, num_frames: int, tokens_per_frame: int, padded_n: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """(face_mask, image_mask), each [1, 1, padded_n, K] on device, fractured on SP along rows."""
        key = (num_frames, tokens_per_frame, padded_n)
        if key not in self._mask_cache:
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            face = rm.face_attention_mask(
                num_frames=num_frames, tokens_per_frame=tokens_per_frame, padded_seq_len=padded_n
            )
            image = rm.image_attention_mask(padded_seq_len=padded_n)
            self._mask_cache[key] = tuple(
                from_torch(m, device=self.mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, None, sp_axis, None])
                for m in (face, image)
            )
        return self._mask_cache[key]

    # --------------------------------------------------------------------------------------------
    # Denoising step
    # --------------------------------------------------------------------------------------------
    def animate_inner_step(
        self,
        *,
        spatial_1BNI: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        image_prompt_1BLP: ttnn.Tensor,
        motion_1BKD: ttnn.Tensor,
        face_mask: ttnn.Tensor,
        image_mask: ttnn.Tensor,
        rope_cos_1HND: ttnn.Tensor,
        rope_sin_1HND: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        N: int | ttnn.Tensor,
        timestep: ttnn.Tensor,
        gather_output: bool = True,
    ) -> ttnn.Tensor:
        """``WanTransformer3DModel.inner_step`` plus the Animate inputs. ``spatial_1BNI`` already holds
        the 36 I2V channels and the 17 pose channels per pixel (see ``reference_math.pose_conditioning``)."""
        temb_11BD, timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)
        spatial_1BND = self.patch_embedding(spatial_1BNI)

        for idx, block in enumerate(self.blocks):
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=timestep_proj_1BTD,
                N=N,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
                image_prompt_1BLP=image_prompt_1BLP,
                image_attn_mask=image_mask,
            )
            if idx % self.inject_face_latents_blocks == 0:
                adapter = self.face_adapter[idx // self.inject_face_latents_blocks]
                spatial_1BND = adapter(spatial_1BND, motion_1BKD, N=N, face_mask=face_mask)

        return self.cache_tail(
            spatial_1BND,
            prompt_1BLP,
            temb_11BD,
            timestep_proj_1BTD,
            N,
            rope_cos_1HND,
            rope_sin_1HND,
            trans_mat,
            num_bn_blocks=0,
            gather_output=gather_output,
        )

    def forward(
        self,
        spatial: torch.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        y: torch.Tensor | None = None,
        *,
        clip_hidden_states: torch.Tensor,
        pose_latents: torch.Tensor,
        face_pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Torch-in / torch-out forward with the Diffusers ``WanAnimateTransformer3DModel`` semantics.

        spatial: noisy latents [1, 16, T+1, H, W]; y: [1, 20, T+1, H, W] (mask + reference/temporal
        latents), so ``cat([spatial, y])`` is Diffusers' ``hidden_states``. prompt: [1, 1, L, text_dim]
        on device. pose_latents: [1, 16, T, H, W]. face_pixel_values: [1, 3, 4T-3, S, S].
        Returns the velocity [1, 16, T+1, H, W].
        """
        assert y is not None
        _, _, f, h, w = spatial.shape
        p_t, p_h, p_w = self.patch_size
        model_in = torch.cat([spatial, y, rm.pose_conditioning(pose_latents)], dim=1)

        rope_cos, rope_sin, trans_mat = self.get_rope_features(spatial)
        spatial_host, n = self.preprocess_spatial_input_host(model_in)
        padded_n = spatial_host.shape[2]
        spatial_1BNI = bf16_tensor(
            spatial_host,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        prompt_1BLP = self.prepare_text_conditioning(prompt)
        image_1BLP = self.prepare_image_conditioning(clip_hidden_states)
        motion_1BKD = self.prepare_face_conditioning(face_pixel_values)
        face_mask, image_mask = self.get_masks(
            num_frames=f // p_t, tokens_per_frame=(h // p_h) * (w // p_w), padded_n=padded_n
        )
        assert self._face_frames == f // p_t, f"face frames {self._face_frames} != latent frames {f // p_t}"

        out_1BNI = self.animate_inner_step(
            spatial_1BNI=spatial_1BNI,
            prompt_1BLP=prompt_1BLP,
            image_prompt_1BLP=image_1BLP,
            motion_1BKD=motion_1BKD,
            face_mask=face_mask,
            image_mask=image_mask,
            rope_cos_1HND=rope_cos,
            rope_sin_1HND=rope_sin,
            trans_mat=trans_mat,
            N=n,
            timestep=timestep,
            gather_output=True,
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(out_1BNI)[0])
        return self.postprocess_spatial_output_host(out, f, h, w, n)


def build_from_diffusers_config(
    config, *, mesh_device, ccl_manager: CCLManager, parallel_config: DiTParallelConfig, is_fsdp: bool, num_layers=None
) -> WanAnimateTransformer3DModel:
    """Construct the TT model from a Diffusers ``WanAnimateTransformer3DModel`` config (weights not loaded)."""
    c = config
    return WanAnimateTransformer3DModel(
        patch_size=tuple(c.patch_size),
        num_heads=c.num_attention_heads,
        dim=c.num_attention_heads * c.attention_head_dim,
        in_channels=c.in_channels,
        out_channels=c.out_channels,
        text_dim=c.text_dim,
        freq_dim=c.freq_dim,
        ffn_dim=c.ffn_dim,
        num_layers=num_layers if num_layers is not None else c.num_layers,
        cross_attn_norm=c.cross_attn_norm,
        eps=c.eps,
        rope_max_seq_len=c.rope_max_seq_len,
        image_dim=c.image_dim,
        motion_encoder_size=c.motion_encoder_size,
        motion_style_dim=c.motion_style_dim,
        motion_dim=c.motion_dim,
        motion_encoder_dim=c.motion_encoder_dim,
        motion_encoder_channel_sizes=c.motion_encoder_channel_sizes,
        face_encoder_hidden_dim=c.face_encoder_hidden_dim,
        face_encoder_num_heads=c.face_encoder_num_heads,
        inject_face_latents_blocks=c.inject_face_latents_blocks,
        motion_encoder_batch_size=c.motion_encoder_batch_size,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )


def log_param_budget(config) -> None:
    """Rough per-module parameter counts (for the design note / sanity on a new mesh)."""
    d = config.num_attention_heads * config.attention_head_dim
    n_ad = -(-config.num_layers // config.inject_face_latents_blocks)
    logger.info(f"image K/V in attn2: {2 * d * d * config.num_layers / 1e9:.2f} B params")
    logger.info(f"face adapters: {4 * d * d * n_ad / 1e9:.2f} B params")
    logger.info(f"pose patch embed: {16 * math.prod(config.patch_size) * d / 1e6:.1f} M params")
