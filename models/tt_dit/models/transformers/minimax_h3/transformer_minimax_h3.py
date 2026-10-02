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
from ....utils.tensor import from_torch, pad_single
from ....utils.tracing import StateTensor, traced_function
from .token_refiner_minimax_h3 import MiniMaxH3TokenRefiner
from .transformer_block_minimax_h3 import ADALN_TABLE_COPIES, MiniMaxH3TransformerBlock

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

    def forward(
        self,
        hidden: ttnn.Tensor,
        temb: ttnn.Tensor,
        timestep_indices: ttnn.Tensor,
    ) -> ttnn.Tensor:
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
    movement, so the assembly happens *before* fracturing instead, as a row gather over fixed-capacity
    streams so every program has a request-independent (traceable) shape: the projected streams form a
    source table `[text | condition video | condition audio | audio | video]`, `assembly_indices`
    gathers it into packed order, and `ttnn.mesh_partition` fractures it across SP.

    Padding
    -------
    Pad rows stay outside `[0, logical_n)`, which ring attention masks internally; interior padding is
    not allowed.

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
        # `rope_freq_dim` sets how many of each head's channels rotate: the reference builds cos/sin
        # of width 2 * 3 * rope_freq_dim and passes the rest of the head through. The tables
        # themselves are still the caller's job (see `prepare_rope_tables`), but attention needs the
        # width to relayout the Q/K rotary channels at weight-load time. `rope_theta` is absent
        # because only the caller's table construction uses it.
        rope_freq_dim: int = 16,
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
        self._temb_state = StateTensor()
        self._timestep_idx_state: dict[int, StateTensor] = {}
        # `position % ADALN_TABLE_COPIES` per padded length, built outside the traced block loop.
        self._adaln_spread: dict[int, ttnn.Tensor] = {}
        self._static_source_state = StateTensor()
        self.parallel_config = parallel_config
        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.tp_factor = parallel_config.tensor_parallel.factor
        self.sp_mesh_axis = parallel_config.sequence_parallel.mesh_axis
        self.sp_factor = parallel_config.sequence_parallel.factor
        self.hidden_local = hidden_size // self.tp_factor

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        # 3 rotary axes (t, h, w), each contributing rope_freq_dim frequencies, doubled by the
        # rotate-half convention.
        rotary_dim = 2 * 3 * rope_freq_dim
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
                    rotary_dim=rotary_dim,
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

    def prepare_static_sources(
        self,
        *,
        prompt_1BLP: ttnn.Tensor,
        prompt_windows: ttnn.Tensor | None = None,
        condition_video_1BKC: ttnn.Tensor | None = None,
        condition_audio_1BKC: ttnn.Tensor | None = None,
        prompt_cap: int,
        traced: bool = False,
    ) -> None:
        """Refine and project the step-invariant streams, once per request.

        Stores the `[text | condition video | condition audio]` source-table prefix that `forward` reads.
        """
        tile = ttnn.TILE_SIZE
        streams = {
            "prompt_1BLP": prompt_1BLP,
            "condition_video_1BKC": condition_video_1BKC,
            "condition_audio_1BKC": condition_audio_1BKC,
        }
        for name, stream in streams.items():
            if stream is not None and stream.shape[2] % tile:
                raise ValueError(f"{name} capacity {stream.shape[2]} must be a multiple of TILE ({tile})")

        refined = self.token_refiner(self.context_embedder(prompt_1BLP), cu_window_seqlens=prompt_windows)
        if refined.shape[2] < prompt_cap:
            refined = pad_single(refined, dim=2, back=prompt_cap - refined.shape[2])
        segments = [refined]
        if condition_video_1BKC is not None:
            segments.append(self.proj_in(condition_video_1BKC))
        if condition_audio_1BKC is not None:
            segments.append(self.audio_proj_in(condition_audio_1BKC))
        prefix = segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=2)
        self._static_source_state.update(prefix, traced=traced)

    def forward(
        self,
        *,
        video_1BVC: ttnn.Tensor,
        audio_1BAC: ttnn.Tensor,
        assembly_indices: ttnn.Tensor,
        video_out_indices: ttnn.Tensor,
        audio_out_indices: ttnn.Tensor,
        timestep: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        timestep_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        logical_n: ttnn.Tensor,
        pad_to: int,
        traced: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Every stream is a fixed-capacity buffer, true rows leading; `prepare_static_sources` must run first.

        video_1BVC: [1, 1, V_cap, in_channels * prod(patch_size)], replicated on SP and TP. Target rows only.
        audio_1BAC: [1, 1, A_cap, audio_in_channels], replicated on SP and TP. Target rows only.
        assembly_indices: [1, 1, 1, pad_to] integers, the source-table row of each packed row.
        video_out_indices / audio_out_indices: [1, 1, 1, V_cap] / [1, 1, 1, A_cap] integers, the
            packed row of each target row.
        timestep: [1, 1, num_slots, 1] float32, replicated. Unscaled, in [0, 1].
        adaln_indices: [1, 1, 1, S_padded_local] integers, `timestep_indices * 3 + token_tags`, built
            for the padded global sequence and sharded on SP
        timestep_indices: [1, 1, 1, S_padded_local] integers, same order
        rope_cos/rope_sin: [1, 1, S_padded_local, rotary_dim] float32, same order, replicated on TP
        logical_n: the true packed length `L + K + A + V` as a [1, 1, 1, 1] uint32 device tensor.
        pad_to: the padded packed length; keys the trace (one capture per `pad_to`).

        Returns `(video_velocity, audio_velocity)` as [1, 1, V_cap, .] / [1, 1, A_cap, .], target rows only.
        """
        tile = ttnn.TILE_SIZE
        alignment = self.sp_factor * tile
        if pad_to % alignment:
            raise ValueError(f"pad_to={pad_to} must be a multiple of sp_factor * TILE = {alignment}")
        if assembly_indices.shape[-1] != pad_to:
            raise ValueError(f"assembly_indices has {assembly_indices.shape[-1]} rows, pad_to is {pad_to}")
        if video_out_indices.shape[-1] != video_1BVC.shape[2]:
            raise ValueError("video_out_indices must match the video stream's capacity")
        if audio_out_indices.shape[-1] != audio_1BAC.shape[2]:
            raise ValueError("audio_out_indices must match the audio stream's capacity")
        for name, stream in (("audio_1BAC", audio_1BAC), ("video_1BVC", video_1BVC)):
            if stream.shape[2] % tile:
                raise ValueError(f"{name} capacity {stream.shape[2]} must be a multiple of TILE ({tile})")
        static_prefix = self._static_source_state.value
        if static_prefix is None:
            raise RuntimeError("prepare_static_sources must run before forward: the source-table prefix is unbound")

        def as_indices(t: ttnn.Tensor) -> ttnn.Tensor:
            t = ttnn.reshape(t, (1, t.shape[-1]))
            return t if t.dtype == ttnn.uint32 else ttnn.typecast(t, ttnn.uint32)

        source = ttnn.concat([static_prefix, self.audio_proj_in(audio_1BAC), self.proj_in(video_1BVC)], dim=2)
        source = ttnn.reshape(source, (source.shape[2], source.shape[3]))

        hidden = ttnn.embedding(as_indices(assembly_indices), source, layout=ttnn.TILE_LAYOUT)
        hidden = ttnn.unsqueeze(hidden, 0)
        hidden = ttnn.mesh_partition(hidden, 2, cluster_axis=self.sp_mesh_axis)

        self._temb_state.update(self.time_embedder(self.time_proj(timestep)), traced=traced)
        temb = self._temb_state.value

        adaln_idx = as_indices(adaln_indices)
        n_local = adaln_idx.shape[-1]
        if n_local not in self._adaln_spread:
            self._adaln_spread[n_local] = from_torch(
                (torch.arange(n_local) % ADALN_TABLE_COPIES).to(torch.int32).reshape(1, n_local),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_axes=[None, None],
            )
        adaln_spread = self._adaln_spread[n_local]
        ts_state = self._timestep_idx_state.setdefault(pad_to, StateTensor())
        ts_state.update(as_indices(timestep_indices), traced=traced)
        timestep_idx = ts_state.value

        hidden = self.run_blocks(
            hidden,
            logical_n,
            temb,
            adaln_idx,
            rope_cos,
            rope_sin,
            adaln_spread,
            traced=traced,
            tracer_trace_key=pad_to,
        )

        hidden = self.norm_out(
            hidden,
            temb,
            timestep_idx,
        )
        if self.tp_factor > 1:
            hidden = self.ccl_manager.all_gather(hidden, dim=3, mesh_axis=self.tp_mesh_axis, use_hyperparams=False)

        video_all = self.proj_out(hidden)
        audio_all = self.audio_proj_out(hidden)
        if self.sp_factor > 1:
            video_all = self.ccl_manager.all_gather(
                video_all, dim=2, mesh_axis=self.sp_mesh_axis, use_hyperparams=False
            )
            audio_all = self.ccl_manager.all_gather(
                audio_all, dim=2, mesh_axis=self.sp_mesh_axis, use_hyperparams=False
            )

        def select(all_rows: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
            table = ttnn.reshape(all_rows, (all_rows.shape[2], all_rows.shape[3]))
            return ttnn.unsqueeze(ttnn.embedding(as_indices(indices), table, layout=ttnn.TILE_LAYOUT), 0)

        return select(video_all, video_out_indices), select(audio_all, audio_out_indices)

    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=False)
    def run_blocks(
        self,
        hidden: ttnn.Tensor,
        logical_n: ttnn.Tensor,
        temb: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        adaln_spread: ttnn.Tensor,
    ) -> ttnn.Tensor:
        for block in self.transformer_blocks:
            hidden = block(
                hidden,
                logical_n,
                temb=temb,
                adaln_indices=adaln_indices,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                adaln_spread=adaln_spread,
            )
        return hidden

    def release_traces(self) -> None:
        """Release every captured `run_blocks` trace, across all `tracer_trace_key` buckets."""
        run_blocks = type(self).run_blocks
        tracers = [run_blocks._tracers.get(self), *run_blocks._tracers_keyed.get(self, {}).values()]
        for tracer in tracers:
            if tracer is not None:
                tracer.release_trace()
