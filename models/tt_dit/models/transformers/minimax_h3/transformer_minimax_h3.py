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
from ....utils.tracing import StateTensor, traced_function
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
    movement, so the assembly happens *before* fracturing instead -- and, so that every dispatched
    program has a request-independent shape (the property that lets one ttnn trace serve every
    request in a padded-length bucket), it is a **row gather over fixed-capacity streams** rather
    than a per-request concat:

    - Every input stream arrives at a fixed row capacity, a multiple of `TILE_SIZE`, with the true
      rows leading and a zero (or stale, equally ignored) tail: `prompt_1BLP` at `L_cap`,
      `condition_video_1BKC` / `condition_audio_1BKC` at their per-modality capacities,
      `audio_1BAC` at `A_cap`, `video_1BVC` at `V_cap`.
    - Each stream is projected at full capacity while replicated on SP (fixed-M matmuls; a per-row
      GEMM is row-independent, so the pad rows cost compute but change nothing), and the projections
      are concatenated -- fixed extents -- into one source table in the fixed segment order
      `[text | condition video | condition audio | audio | video]` (a condition segment exists only
      when its stream is passed, which is fixed per deployment). The text and condition segments are
      step-invariant, so `prepare_static_sources` refines and projects them **once per request**
      into a persistent prefix; `forward` projects only the two per-step streams (audio, video) and
      appends them.
    - `assembly_indices` ([1, 1, 1, pad_to] integers) then gathers the source rows into packed order
      `[text | condition blocks in packed order | audio | target video | pad]`. The packed
      interleave -- ref2va packs a reference's soundtrack rows immediately before its own video
      rows -- is index *content*, not shape, as is the true length of every stream. Pad indices
      point at row 0; pad rows are masked from attention by `logical_n` and their output is never
      selected, so their content is irrelevant.
    - `ttnn.mesh_partition` fractures the assembled `[1, 1, pad_to, hidden]` sequence across SP.

    The condition segment sits between text and audio in packed order, and that position is not a
    choice: `packing.build_packed_sequence` and `packing_ref2va.build_ref2va_packed_sequence` both
    put the conditioning rows there, and the caller's rope/AdaLN metadata is built in that layout's
    order. Two condition streams rather than one because the two modalities need different
    projections: audio rows go through `audio_proj_in` (32 wide) and video rows through `proj_in`
    (96 wide), so they cannot share a buffer at all.

    The load-bearing invariant is **one frame of reference for row indices**. `assembly_indices` is
    built by walking the condition blocks in packed order, which is the same walk that produced
    `layout.position_ids`, `token_tags`, `video_indices` and `audio_indices`. There is no second
    ordering to keep in step.

    Because the sequence is assembled globally, the caller's per-row metadata (`rope_cos`,
    `rope_sin`, `adaln_indices`, `timestep_indices`) is simply built for the padded global sequence
    in that same natural order and sharded contiguously on SP -- no device-major permutation to keep
    in step with the model. Outputs are gathered back on SP and selected per modality by
    `video_out_indices` / `audio_out_indices` -- again gathers with per-request content and fixed
    capacity shapes -- so each modality's rows come back in its own order at its stream's capacity,
    true rows leading.

    Padding
    -------
    Pad rows -- the stream tails beyond each true length, and the packed tail beyond `logical_n` --
    need no attention mask inside the packed sequence: ring attention's `logical_n` masks the tail
    beyond the true sequence length internally, and the assembly gather keeps every stream's pad
    rows *out* of `[0, logical_n)`, so no real row ever attends to one. Interior padding is what is
    *not* allowed -- a pad row between two modalities would sit inside `logical_n` and every real
    row would attend to it as a key and value -- and the gather is exactly what keeps the interior
    dense while the streams stay fixed-capacity. The one place padding is masked is the token
    refiner: it runs over the text stream *before* assembly, at `L_cap` and once per request
    (`prepare_static_sources`), using SDPA's windowed mode -- `prompt_windows = [0, true_len,
    L_cap]` fences the true tokens off from the pad tail, with the mask synthesized on device from
    the boundaries; the refined pad rows are then dropped by the assembly gather.

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
        self._timestep_idx_state = StateTensor()
        # The projected [text | condition] source-table prefix, step-invariant and read by the eager
        # shell after every trace replay -- same discipline as `_temb_state`: bound once before any
        # capture, refreshed by same-shape copy. Written by `prepare_static_sources`.
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
        traced: bool = False,
    ) -> None:
        """Refine and project the step-invariant streams, once per request.

        The text refinement and the condition projections depend only on the request, not on the
        denoise step -- running them inside `forward` would repeat them every step, and at the
        ref2va prompt capacity the refiner's O(L_cap^2) attention would dominate the step. The
        projected `[text | condition video | condition audio]` source-table prefix is stored in a
        persistent StateTensor that `forward` reads on every step: it is consumed by the eager
        shell after every trace replay, so it follows the `_temb_state` discipline -- bound once
        before any capture, refreshed by same-shape `ttnn.copy` when traced.

        prompt_1BLP: [1, 1, L_cap, text_dim], replicated on SP and TP, true rows leading.
        prompt_windows: `[0, true_len, L_cap]` window boundaries for the token refiner's windowed
            SDPA (1-D integer device tensor, replicated), fencing the true tokens off from the pad
            tail; None when the prompt exactly fills L_cap. Consumed here only, so it may be
            transient.
        condition_video_1BKC / condition_audio_1BKC: the conditioning rows of each modality, packed
            contiguously in packed-walk order at their own capacity; None when the deployment has no
            such stream (t2va/fl2va has no condition audio). Presence is fixed per deployment -- it
            sets the prefix shape -- so a request without conditioning passes a zero-filled buffer,
            not None.
        """
        tile = ttnn.TILE_SIZE
        streams = {
            "prompt_1BLP": prompt_1BLP,
            "condition_video_1BKC": condition_video_1BKC,
            "condition_audio_1BKC": condition_audio_1BKC,
        }
        for name, stream in streams.items():
            # The prefix joins a TILE-layout concat, which cuts on tile boundaries only.
            if stream is not None and stream.shape[2] % tile:
                raise ValueError(f"{name} capacity {stream.shape[2]} must be a multiple of TILE ({tile})")

        # Conditioning rows use the same weights as the target rows of their modality: a conditioning
        # row is a row of its own modality that happens to be pinned, and a per-row GEMM against a
        # shared weight is row-independent, so projecting them from a separate buffer is bit-identical
        # to projecting them in place.
        segments = [self.token_refiner(self.context_embedder(prompt_1BLP), cu_window_seqlens=prompt_windows)]
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
        Every stream is a fixed-capacity buffer, its true rows leading (see the class docstring).
        The step-invariant text and conditioning segments are not passed here: `prepare_static_sources`
        projects them once per request and this reads the stored prefix.

        video_1BVC: [1, 1, V_cap, in_channels * prod(patch_size)], replicated on SP and TP. The
            *target* rows only --- conditioning rows go through `prepare_static_sources`, not here.
        audio_1BAC: [1, 1, A_cap, audio_in_channels], replicated on SP and TP. Again target rows only.
        assembly_indices: [1, 1, 1, pad_to] integers, replicated -- source-table row of each packed
            row, in packed order `[text | condition blocks | audio | video | pad]`; pad rows point
            at row 0. Built against the segment order in the class docstring.
        video_out_indices / audio_out_indices: [1, 1, 1, V_cap] / [1, 1, 1, A_cap] integers,
            replicated -- padded-global-sequence row of each target row, entries past the true
            count pointing at any real row of that modality.
        timestep: [1, 1, num_slots, 1] float32, replicated. Unscaled, in [0, 1].
        adaln_indices: [1, 1, 1, S_padded_local] integers, `timestep_indices * 3 + token_tags`, built
            for the padded global sequence and sharded on SP
        timestep_indices: [1, 1, 1, S_padded_local] integers, same order
        rope_cos/rope_sin: [1, 1, S_padded_local, rotary_dim] float32, same order, replicated on TP
        logical_n: the true packed length `L + K + A + V` as a [1, 1, 1, 1] uint32 device tensor.
        pad_to: the padded packed length. Every per-device shape in the block stack is a function of
            it (and of num_slots), so it keys the trace: one capture per `pad_to`, selected via
            `tracer_trace_key`, replayed for any request whose true lengths -- all carried by the
            index tensors and `logical_n` -- fit inside it.

        Returns `(video_velocity, audio_velocity)` as [1, 1, V_cap, .] / [1, 1, A_cap, .],
        replicated, true target rows leading, in that modality's row order.

        Both hold the **target rows only** and not the conditioning rows the reference's
        `video_indices` / `audio_indices` would also cover. The caller discards conditioning-row
        velocity, because the loop re-imposes the anchors by only ever writing target rows. No
        detection power is lost: attention is full, so every target row attends to every conditioning
        row as a key and value, and a wrong conditioning rope, AdaLN tag or input projection shows up
        in this output. Entries past the true counts duplicate a real row's velocity; the caller's
        in-place Euler step advances the buffer tails with them, and nothing reads those tails back.
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
            # The source table is assembled with a TILE-layout concat, which cuts on tile boundaries
            # only -- and fixed capacities have no reason to be unaligned.
            if stream.shape[2] % tile:
                raise ValueError(f"{name} capacity {stream.shape[2]} must be a multiple of TILE ({tile})")
        static_prefix = self._static_source_state.value
        if static_prefix is None:
            raise RuntimeError("prepare_static_sources must run before forward: the source-table prefix is unbound")

        # Integer index tensors for the gathers. ttnn.embedding wants [batch, seq] uint32.
        def as_indices(t: ttnn.Tensor) -> ttnn.Tensor:
            t = ttnn.reshape(t, (1, t.shape[-1]))
            return t if t.dtype == ttnn.uint32 else ttnn.typecast(t, ttnn.uint32)

        # 1. Project the two per-step streams at full capacity, still replicated on SP, and append
        # them to the request's static prefix -- the source table in the fixed segment order.
        source = ttnn.concat([static_prefix, self.audio_proj_in(audio_1BAC), self.proj_in(video_1BVC)], dim=2)
        source = ttnn.reshape(source, (source.shape[2], source.shape[3]))

        # 2. Gather the source rows into packed order -- the layout is index content, so the shapes
        # here depend only on the capacities and pad_to -- then fracture across SP.
        hidden = ttnn.embedding(as_indices(assembly_indices), source, layout=ttnn.TILE_LAYOUT)
        hidden = ttnn.unsqueeze(hidden, 0)
        hidden = ttnn.mesh_partition(hidden, 2, cluster_axis=self.sp_mesh_axis)

        # 3. One timestep embedding per slot, shared by every AdaLN projection. Stabilized because it
        # is read again by `norm_out` after the trace replay -- see `_temb_state` in `__init__`.
        self._temb_state.update(self.time_embedder(self.time_proj(timestep)), traced=traced)
        temb = self._temb_state.value

        adaln_idx = as_indices(adaln_indices)
        # Only `norm_out`, after the replay, reads `timestep_idx` -- same hazard as `temb`.
        self._timestep_idx_state.update(as_indices(timestep_indices), traced=traced)
        timestep_idx = self._timestep_idx_state.value

        # 4. The traced block stack -- `pad_to` keys the capture, see its parameter doc above.
        hidden = self.run_blocks(
            hidden,
            logical_n,
            temb,
            adaln_idx,
            rope_cos,
            rope_sin,
            traced=traced,
            tracer_trace_key=pad_to,
        )

        # 5. Output norm, then the two heads. Both heads are narrow (96 and 32), so projecting while
        # still SP-fractured and gathering afterwards moves far less data than gathering the 5376-wide
        # packed sequence would.
        hidden = self.norm_out(
            hidden,
            temb,
            timestep_idx,
        )
        if self.tp_factor > 1:
            hidden = self.ccl_manager.all_gather_persistent_buffer(hidden, dim=3, mesh_axis=self.tp_mesh_axis)

        video_all = self.proj_out(hidden)
        audio_all = self.audio_proj_out(hidden)
        if self.sp_factor > 1:
            video_all = self.ccl_manager.all_gather_persistent_buffer(video_all, dim=2, mesh_axis=self.sp_mesh_axis)
            audio_all = self.ccl_manager.all_gather_persistent_buffer(audio_all, dim=2, mesh_axis=self.sp_mesh_axis)

        # 6. Select each modality's target rows out of the reassembled global sequence -- gathers
        # with per-request index content and capacity-fixed shapes, mirroring the assembly. The
        # reference runs both heads over every row and selects afterwards, which is what this does.
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
    ) -> ttnn.Tensor:
        for block in self.transformer_blocks:
            hidden = block(
                hidden,
                logical_n,
                temb=temb,
                adaln_indices=adaln_indices,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
        return hidden

    def release_traces(self) -> None:
        """Release every captured `run_blocks` trace, across all `tracer_trace_key` buckets.

        The Tracers themselves stay: a released tracer re-captures on its next traced call, so this
        costs each bucket one capture run, not a re-warm.
        """
        run_blocks = type(self).run_blocks
        tracers = [run_blocks._tracers.get(self), *run_blocks._tracers_keyed.get(self, {}).values()]
        for tracer in tracers:
            if tracer is not None:
                tracer.release_trace()
