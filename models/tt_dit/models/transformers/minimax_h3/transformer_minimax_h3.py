# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ....layers.embeddings import Timesteps
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from .token_refiner_minimax_h3 import MiniMaxH3TokenRefiner
from .transformer_block_minimax_h3 import MiniMaxH3TransformerBlock

# shift, scale -- the order `norm_out.linear` emits them in.
NUM_OUT_MODULATION_PARAMS = 2


class MiniMaxH3TimestepEmbedding(Module):
    """`linear_1 -> silu -> linear_2`, matching the checkpoint's `time_embedder` keys.

    Not `layers.embeddings.TimestepEmbedding`, whose `linear_2` is square; MiniMax-H3 narrows
    5376 -> 2688. Replicated (not TP-fractured) because `temb` is only a handful of rows and every
    AdaLN projection consumes it whole.

    Kept in float32, as the checkpoint declares (`_keep_in_fp32_modules`). The reference is explicit
    that this matters: all 50 blocks read this one `temb`, so rounding it early biases every block's
    modulation identically at every sampling step and accumulates over the denoising trajectory.
    """

    def __init__(self, *, in_channels: int, hidden_dim: int, out_dim: int, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.linear_1 = Linear(in_channels, hidden_dim, bias=True, mesh_device=mesh_device, dtype=ttnn.float32)
        self.linear_2 = Linear(hidden_dim, out_dim, bias=True, mesh_device=mesh_device, dtype=ttnn.float32)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.linear_2(ttnn.silu(self.linear_1(x)))


class MiniMaxH3AdaLayerNormOut(Module):
    """Final norm of the packed sequence, shift/scale modulated per row.

    Unlike the block's AdaLN, the table is indexed by `timestep_indices` alone -- one row per
    timestep, with no modality axis -- so no row folding is needed, only a per-parameter slice of the
    projection's output and a gather.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        time_embed_dim: int,
        eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        fsdp_mesh_axis: int | None = None,
    ) -> None:
        super().__init__()

        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.tp_factor = parallel_config.tensor_parallel.factor
        self.hidden_local = hidden_size // self.tp_factor

        self.norm = DistributedRMSNorm(
            embedding_dim=hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=True,
            mesh_axis=self.tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.linear = ColParallelLinear(
            time_embed_dim,
            NUM_OUT_MODULATION_PARAMS * hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Output dim is laid out [param][hidden]; reorder to [device][param][hidden_local] so the
        # column split gives each device both params restricted to its own hidden slice. Same trick as
        # the block's adaln_proj, minus the modality axis.
        def reorder(t: torch.Tensor) -> torch.Tensor:
            trailing = t.shape[1:]
            t = t.reshape(NUM_OUT_MODULATION_PARAMS, self.tp_factor, self.hidden_local, *trailing)
            t = t.permute(1, 0, 2, *range(3, t.ndim))
            return t.reshape(-1, *trailing)

        for key in ("weight", "bias"):
            value = state.get(f"linear.{key}")
            if value is None:
                continue
            if key == "bias":
                state[f"linear.{key}"] = reorder(value.unsqueeze(-1)).squeeze(-1)
            else:
                state[f"linear.{key}"] = reorder(value)

    def forward(self, hidden: ttnn.Tensor, temb: ttnn.Tensor, timestep_indices: ttnn.Tensor) -> ttnn.Tensor:
        activated = ttnn.silu(temb)
        if activated.dtype != ttnn.bfloat16:
            activated = ttnn.typecast(activated, ttnn.bfloat16)
        projected = self.linear(activated)  # [1, 1, num_timesteps, 2 * hidden_local]

        rows = projected.shape[2]
        tables = []
        for p in range(NUM_OUT_MODULATION_PARAMS):
            table = ttnn.slice(projected, [0, 0, 0, p * self.hidden_local], [1, 1, rows, (p + 1) * self.hidden_local])
            table = ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT)
            table = ttnn.reshape(table, (rows, self.hidden_local))
            tables.append(ttnn.to_layout(table, ttnn.TILE_LAYOUT))

        def gather(table: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.unsqueeze(ttnn.embedding(timestep_indices, table, layout=ttnn.TILE_LAYOUT), 0)

        shift, scale = gather(tables[0]), gather(tables[1])
        return ttnn.add(ttnn.mul(self.norm(hidden), ttnn.add(scale, 1.0)), shift)


class MiniMaxH3Transformer3DModel(Module):
    """MiniMax-H3 joint video + audio denoising transformer.

    One stack of blocks over a single packed 1-D sequence holding the text condition, the audio rows
    and the video rows. Full self-attention, no cross-attention, no per-modality block weights;
    modality-specific behaviour comes only from the two input patch projections, the per-row AdaLN
    modality tag and the two output heads.

    Packed-sequence layout
    ----------------------
    The reference builds the packed sequence with `index_copy` at caller-supplied row indices. A
    general scatter across an already-fractured sequence-parallel tensor would need cross-device
    movement, so the assembly happens *before* fracturing instead: each modality is projected while
    replicated on SP, the three streams are concatenated into the full packed sequence in natural
    order `[text | audio | video]`, the tail is zero-padded up to a multiple of `sp_factor * TILE`,
    and `ttnn.mesh_partition` then fractures that global sequence across the SP axis.

    Because the sequence is assembled globally, the caller's per-row metadata (`rope_cos`, `rope_sin`,
    `adaln_indices`, `timestep_indices`) is simply built for the padded global sequence in that same
    natural order and sharded contiguously on SP -- no device-major permutation to keep in step with
    the model. Outputs are gathered back on SP and sliced per modality, so each modality's rows come
    back in its own order, matching what the reference returns.

    The only remaining constraint is that each modality's row count be a multiple of `TILE_SIZE`, so
    that the device-side concat lands on tile boundaries. That is much weaker than requiring each to
    divide `sp_factor * TILE`. (Fully arbitrary per-modality lengths would need the concat done in
    ROW_MAJOR and converted back, which costs two layout passes over the whole packed sequence.)

    Padding
    -------
    Trailing zero rows only, and they need no attention mask: ring attention's `logical_n` masks the
    tail beyond the true sequence length internally. The reference's `-1`-tagged separate-document
    mask is therefore unnecessary. Interior padding would need a real mask, which is why each modality
    is tile-aligned rather than individually padded.

    Precision
    ---------
    The checkpoint is mixed: `proj_in`, `audio_proj_in`, `time_embedder`, `proj_out` and
    `audio_proj_out` are float32, everything else bfloat16. `time_embedder` is kept float32 here for
    the reason in its docstring. The four patch projections run bfloat16 for bringup -- they sit at
    the very start and very end of the network rather than inside the 50-block trajectory, so the
    coherent-bias argument does not apply to them. Revisit if end-to-end quality needs it.
    """

    def __init__(
        self,
        *,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        # NOTE: `rope_freq_dim` and `rope_theta` from the checkpoint config are deliberately absent.
        # The rotary embedding is computed by the caller and passed in as cos/sin, so accepting them
        # here would imply this module uses them.
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.freq_dim = freq_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.tp_factor = parallel_config.tensor_parallel.factor
        self.sp_mesh_axis = parallel_config.sequence_parallel.mesh_axis
        self.sp_factor = parallel_config.sequence_parallel.factor
        self.hidden_local = hidden_size // self.tp_factor

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        col_kwargs = {
            "bias": True,
            "mesh_device": mesh_device,
            "mesh_axis": self.tp_mesh_axis,
            "fsdp_mesh_axis": fsdp_mesh_axis,
            "ccl_manager": ccl_manager,
        }

        # 1. Per-modality input projections. Inputs are replicated on TP, outputs TP-fractured.
        self.proj_in = ColParallelLinear(video_patch_dim, hidden_size, **col_kwargs)
        self.audio_proj_in = ColParallelLinear(audio_in_channels, hidden_size, **col_kwargs)
        self.context_embedder = ColParallelLinear(text_dim, hidden_size, **col_kwargs)

        # 2. Timestep embedding, shared by every AdaLN projection.
        self.time_proj = Timesteps(
            num_channels=freq_dim,
            cos_first=True,  # == diffusers flip_sin_to_cos=True
            # The reference constructs `Timesteps` with diffusers' default max_period; this is
            # unrelated to rope_theta, which happens to have the same value.
            max_period=10000,
            downscale_freq_shift=0,
            scale=1,
            dtype=ttnn.float32,
            mesh_device=mesh_device,
        )
        self.time_embedder = MiniMaxH3TimestepEmbedding(
            in_channels=freq_dim,
            hidden_dim=time_embed_hidden_dim,
            out_dim=time_embed_dim,
            mesh_device=mesh_device,
        )

        # 3. Text stream refiner. It runs before the packed sequence is fractured, so its text stream
        # is replicated on SP and attention is local.
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
        )

        # 4. The block stack.
        self.transformer_blocks = ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    is_fsdp=is_fsdp,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Shared output norm and the two per-modality heads. The heads are replicated: their output
        # widths (96 and 32) are too narrow to fracture across TP at tile granularity.
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=hidden_size,
            time_embed_dim=time_embed_dim,
            eps=final_norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )
        self.proj_out = Linear(hidden_size, video_patch_dim, bias=True, mesh_device=mesh_device)
        self.audio_proj_out = Linear(hidden_size, audio_in_channels, bias=True, mesh_device=mesh_device)

    def forward(
        self,
        *,
        video_1BVC: ttnn.Tensor,
        audio_1BAC: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        timestep: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        timestep_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        video_1BVC: [1, 1, V, in_channels * prod(patch_size)], replicated on SP and TP
        audio_1BAC: [1, 1, A, audio_in_channels], replicated on SP and TP
        prompt_1BLP: [1, 1, L, text_dim], replicated on SP and TP
        timestep: [1, 1, num_timesteps, 1] float32, replicated. Unscaled, in [0, 1].
        adaln_indices: [1, 1, 1, S_padded_local] integers, `timestep_indices * 3 + token_tags`, built
            for the padded global sequence `[text | audio | video | pad]` and sharded on SP
        timestep_indices: [1, 1, 1, S_padded_local] integers, same order
        rope_cos/rope_sin: [1, 1, S_padded_local, rotary_dim] float32, same order, replicated on TP

        V, A and L must each be a multiple of `ttnn.TILE_SIZE`. The true packed length `L + A + V` is
        derived here, so the caller passes no lengths; the padded length is read off `rope_cos`.

        Returns `(video_velocity, audio_velocity)`, each replicated, in that modality's row order.
        """
        v_len = video_1BVC.shape[2]
        a_len = audio_1BAC.shape[2]
        l_len = prompt_1BLP.shape[2]
        seq_len = l_len + a_len + v_len
        tile = ttnn.TILE_SIZE
        for name, n in (("video", v_len), ("audio", a_len), ("text", l_len)):
            assert n % tile == 0, f"{name} rows ({n}) must be a multiple of TILE_SIZE ({tile})"

        # 1. Project each modality and refine the text stream, all still replicated on SP, then
        # assemble the full packed sequence in natural order.
        video_embeds = self.proj_in(video_1BVC)
        audio_embeds = self.audio_proj_in(audio_1BAC)
        text_embeds = self.token_refiner(self.context_embedder(prompt_1BLP))
        hidden = ttnn.concat([text_embeds, audio_embeds, video_embeds], dim=2)

        # 2. Zero-pad the tail up to a multiple of sp_factor * TILE, then fracture across SP.
        # Ring attention masks the pad rows via logical_n = seq_len, so no attention mask is needed.
        alignment = self.sp_factor * tile
        padded_len = ((seq_len + alignment - 1) // alignment) * alignment
        if padded_len != seq_len:
            pad = ttnn.zeros(
                [1, 1, padded_len - seq_len, self.hidden_local],
                dtype=hidden.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
            hidden = ttnn.concat([hidden, pad], dim=2)
        hidden = ttnn.mesh_partition(hidden, 2, cluster_axis=self.sp_mesh_axis)

        # 3. One timestep embedding per distinct noise level, shared by every AdaLN projection.
        temb = self.time_embedder(self.time_proj(timestep))

        # 4. Integer index tensors for the two gathers. ttnn.embedding wants [batch, seq] uint32.
        def as_indices(t: ttnn.Tensor) -> ttnn.Tensor:
            t = ttnn.reshape(t, (1, t.shape[-1]))
            return t if t.dtype == ttnn.uint32 else ttnn.typecast(t, ttnn.uint32)

        adaln_idx = as_indices(adaln_indices)
        timestep_idx = as_indices(timestep_indices)

        # 5. The block stack. `logical_n` is the *true* length, so ring attention ignores the pad tail.
        for block in self.transformer_blocks:
            hidden = block(
                hidden,
                N=seq_len,
                temb=temb,
                adaln_indices=adaln_idx,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        # 6. Output norm, then the two heads. Both heads are narrow (96 and 32), so projecting while
        # still SP-fractured and gathering afterwards moves far less data than gathering the 5376-wide
        # packed sequence would.
        hidden = self.norm_out(hidden, temb, timestep_idx)
        if self.tp_factor > 1:
            hidden = self.ccl_manager.all_gather_persistent_buffer(hidden, dim=3, mesh_axis=self.tp_mesh_axis)

        video_all = self.proj_out(hidden)
        audio_all = self.audio_proj_out(hidden)
        if self.sp_factor > 1:
            video_all = self.ccl_manager.all_gather_persistent_buffer(video_all, dim=2, mesh_axis=self.sp_mesh_axis)
            audio_all = self.ccl_manager.all_gather_persistent_buffer(audio_all, dim=2, mesh_axis=self.sp_mesh_axis)

        # 7. Select each modality's rows out of the reassembled global sequence. The reference runs both
        # heads over every row and selects afterwards, which is what this does.
        video_out = ttnn.slice(video_all, [0, 0, l_len + a_len, 0], [1, 1, seq_len, video_all.shape[-1]])
        audio_out = ttnn.slice(audio_all, [0, 0, l_len, 0], [1, 1, l_len + a_len, audio_all.shape[-1]])
        return video_out, audio_out
