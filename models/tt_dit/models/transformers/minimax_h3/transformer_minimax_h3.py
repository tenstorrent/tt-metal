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
from ....utils.tracing import traced_function
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
    movement, so the assembly happens *before* fracturing instead: each modality is projected while
    replicated on SP, the streams are concatenated into the full packed sequence in natural order
    `[text | condition | audio | target video]`, the tail is zero-padded up to a multiple of
    `sp_factor * TILE`, and `ttnn.mesh_partition` then fractures that global sequence across the SP
    axis.

    The condition stream sits between text and audio, and that position is not a choice:
    `packing.build_packed_sequence` and `packing_ref2va.build_ref2va_packed_sequence` both put the
    conditioning rows there, and the caller's rope/AdaLN metadata is built in that layout's order.

    It is a **list of typed blocks**, `[(tensor, modality), ...]`, in packed order:

    - `t2va` passes `None` and gets the three-stream `[text | audio | video]` assembly unchanged;
    - `fl2va` passes one `("video")` block, its keyframe anchors;
    - `ref2va` passes one block per reference *medium*, and the region is modality-interleaved --
      `[(ref1 audio, "audio"), (ref1 video, "video"), (ref2 video, "video"), ...]` -- because a video
      reference's soundtrack rows are packed immediately before its own video rows.

    A list rather than one tensor because the two modalities need different projections: audio rows go
    through `audio_proj_in` (32 wide) and video rows through `proj_in` (96 wide), so they cannot be
    concatenated before projection at all. Projecting each block separately is bit-identical to
    projecting a concatenation of same-modality blocks, because a per-row GEMM against a shared weight
    is row-independent -- which is what lets the blocks be placed apart here.

    The load-bearing invariant is **one frame of reference for row
    indices**. The block list is assembled by walking the reference list in packed order, which is the
    same walk that produced `layout.position_ids`, `token_tags`, `video_indices` and `audio_indices`.
    There is no second ordering to keep in step.

    Because the sequence is assembled globally, the caller's per-row metadata (`rope_cos`, `rope_sin`,
    `adaln_indices`, `timestep_indices`) is simply built for the padded global sequence in that same
    natural order and sharded contiguously on SP -- no device-major permutation to keep in step with
    the model. Outputs are gathered back on SP and sliced per modality, so each modality's rows come
    back in its own order, matching what the reference returns.

    Per-modality row counts are unconstrained. A concat in `TILE_LAYOUT` can only cut on a tile
    boundary, so when every stream's length is a multiple of `TILE_SIZE` the assembly happens there
    directly; otherwise the streams are converted to ROW_MAJOR, concatenated at row granularity, and
    the assembled sequence -- whose padded length *is* tile aligned -- is converted back once.
    Production t2va needs the second path: at 1344x768 / 124 frames the video rows are 37296
    (= 16 mod 32) and the audio rows 414 (= 30 mod 32). So does fl2va, whose condition rows are
    1008 per anchor (= 16 mod 32).

    Padding
    -------
    Trailing zero rows only, and they need no attention mask: ring attention's `logical_n` masks the
    tail beyond the true sequence length internally. The reference's `-1`-tagged separate-document
    mask is therefore unnecessary. Interior padding is what is *not* allowed -- a pad row between two
    modalities is inside `logical_n`, so every real row would attend to it as a key and value. That is
    why unaligned modalities are handled by changing the layout of the concat, not by padding each
    modality up to a tile.

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
        # Hold the sequence padding in one buffer instead of allocating it per forward. Only the
        # traced path needs it -- `ttnn.zeros` writes to device and a capture rejects writes -- so it
        # is off by default and the untraced path keeps its per-call allocation.
        cache_padding: bool = False,
    ) -> None:
        super().__init__()

        self.cache_padding = cache_padding
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self._pad_key: tuple[int, ttnn.DataType, ttnn.Layout] | None = None
        self._pad_buffer: ttnn.Tensor | None = None
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

    def forward(
        self,
        *,
        video_1BVC: ttnn.Tensor,
        audio_1BAC: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        condition_blocks: list[tuple[ttnn.Tensor, str]] | None = None,
        timestep: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        timestep_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        video_1BVC: [1, 1, V, in_channels * prod(patch_size)], replicated on SP and TP. The *target*
            rows only --- conditioning rows go in `condition_blocks`, not prepended here.
        audio_1BAC: [1, 1, A, audio_in_channels], replicated on SP and TP. Again the target rows only.
        prompt_1BLP: [1, 1, L, text_dim], replicated on SP and TP
        condition_blocks: `[(tensor, modality), ...]` in **packed order**, or None for t2va. A
            `"video"` block is [1, 1, k, in_channels * prod(patch_size)] and an `"audio"` block
            [1, 1, k, audio_in_channels]; per-block row counts are arbitrary. fl2va passes one
            `"video"` block; ref2va passes one per reference medium, interleaved.
        timestep: [1, 1, num_slots, 1] float32, replicated. Unscaled, in [0, 1].
        adaln_indices: [1, 1, 1, S_padded_local] integers, `timestep_indices * 3 + token_tags`, built
            for the padded global sequence `[text | condition | audio | video | pad]` and sharded on SP
        timestep_indices: [1, 1, 1, S_padded_local] integers, same order
        rope_cos/rope_sin: [1, 1, S_padded_local, rotary_dim] float32, same order, replicated on TP

        V, A, L and K are arbitrary; only their sum is padded, to a multiple of `sp_factor * TILE`. The
        true packed length `L + K + A + V` is derived here, so the caller passes no lengths.

        Returns `(video_velocity, audio_velocity)`, each replicated, in that modality's row order.

        Both hold the **target rows only** --- `V` and `A` of them --- and not the
        conditioning rows the reference's `video_indices` / `audio_indices` would also cover. The caller
        discards conditioning-row velocity, because the loop re-imposes the anchors by only ever writing
        rows from `num_condition_video_rows` / `num_condition_audio_rows` on. For ref2va the reference
        blocks are not even contiguous with each other, so returning them would cost a slice per block
        plus a concat every step for a value nobody reads.

        No detection power is lost: attention is full, so every target row attends to every conditioning
        row as a key and value, and a wrong conditioning rope, AdaLN tag or input projection shows up in
        this output. That is also why the target rows stay a single contiguous slice regardless of how
        many reference blocks precede them --- `audio_start = l_len + c_len` and
        `video_start = audio_start + a_len` hold for any block list.
        """
        v_len = video_1BVC.shape[2]
        a_len = audio_1BAC.shape[2]
        l_len = prompt_1BLP.shape[2]
        condition_blocks = condition_blocks or []
        for index, (block, modality) in enumerate(condition_blocks):
            if modality not in ("video", "audio"):
                raise ValueError(f"condition_blocks[{index}] has modality {modality!r}, not 'video' or 'audio'")
        condition_lengths = [block.shape[2] for block, _ in condition_blocks]
        c_len = sum(condition_lengths)
        seq_len = l_len + c_len + a_len + v_len
        tile = ttnn.TILE_SIZE
        # A tile-layout concat can only cut on a tile boundary, so it needs every modality's row
        # count to be a multiple of TILE. Production t2va satisfies none of that: at 1344x768 /
        # 124 frames the video rows are 37 * 1008 = 37296 (= 16 mod 32) and the audio rows
        # 207 * 2 = 414 (= 30 mod 32), and the text rows are whatever the prompt tokenizes to.
        # Assembling in ROW_MAJOR instead costs two layout passes over the packed sequence and
        # accepts any lengths. The tile path is kept for the aligned case because it is strictly
        # cheaper.
        # Every *block* is tested, not their sum: the concat cuts at each block boundary, so one
        # unaligned block forces the ROW_MAJOR path even when the totals are aligned.
        tile_aligned = not (
            v_len % tile or a_len % tile or l_len % tile or any(length % tile for length in condition_lengths)
        )

        # 1. Project each modality and refine the text stream, all still replicated on SP, then
        # assemble the full packed sequence in natural order.
        video_embeds = self.proj_in(video_1BVC)
        audio_embeds = self.audio_proj_in(audio_1BAC)
        text_embeds = self.token_refiner(self.context_embedder(prompt_1BLP))
        streams = [text_embeds]
        for block, modality in condition_blocks:
            # Same weights as the target rows of that modality: a conditioning row is a
            # row of its own modality that happens to be pinned, and the reference
            # projects it with the very same `proj_in` / `audio_proj_in`. A per-row GEMM
            # against a shared weight is row-independent, so projecting the blocks
            # separately is bit-identical to projecting them concatenated.
            streams.append(self.audio_proj_in(block) if modality == "audio" else self.proj_in(block))
        streams += [audio_embeds, video_embeds]

        # 2. Zero-pad the tail up to a multiple of sp_factor * TILE, then fracture across SP.
        # Ring attention masks the pad rows via logical_n = seq_len, so no attention mask is
        # needed. The padding must stay at the tail: interior pad rows would be attended to as
        # keys and values by every real row, which is why unaligned modalities are handled by
        # changing the layout of the concat rather than by padding each modality.
        alignment = self.sp_factor * tile
        padded_len = ((seq_len + alignment - 1) // alignment) * alignment
        pad_layout = ttnn.TILE_LAYOUT if tile_aligned else ttnn.ROW_MAJOR_LAYOUT
        if not tile_aligned:
            streams = [ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in streams]
        hidden = ttnn.concat(streams, dim=2)
        if padded_len != seq_len:
            hidden = ttnn.concat([hidden, self._padding_rows(padded_len - seq_len, hidden.dtype, pad_layout)], dim=2)
        if not tile_aligned:
            # `padded_len` is a multiple of sp_factor * TILE, so the assembled sequence is tile
            # aligned even though none of its parts was.
            hidden = ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)
        hidden = ttnn.mesh_partition(hidden, 2, cluster_axis=self.sp_mesh_axis)

        # 3. One timestep embedding per slot, shared by every AdaLN projection.
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

        # 7. Select each modality's rows out of the reassembled global sequence. The reference runs both
        # heads over every row and selects afterwards, which is what this does.
        audio_start = l_len + c_len
        video_start = audio_start + a_len
        video_out = ttnn.slice(video_all, [0, 0, video_start, 0], [1, 1, seq_len, video_all.shape[-1]])
        audio_out = ttnn.slice(audio_all, [0, 0, audio_start, 0], [1, 1, video_start, audio_all.shape[-1]])
        return video_out, audio_out

    def _padding_rows(self, rows: int, dtype: ttnn.DataType, layout: ttnn.Layout) -> ttnn.Tensor:
        """The zero rows that pad the packed sequence up to `padded_len`.

        Freshly allocated unless `cache_padding` is set, which the traced path needs: `ttnn.zeros`
        writes to device and a trace capture forbids writes ("Writes are not supported during trace
        capture"). `padded_len - seq_len` is fixed for a request and `concat` does not modify its
        operands, so one buffer serves every step.

        One slot, not a dict keyed by shape: `rows` stays below `sp_factor * TILE_SIZE`, so a dict
        would be bounded rather than unbounded -- but the bound is 1024 entries at SP=32, of zeros no
        later request reuses. The key is constant within a request, so one slot has the same hit rate.

        Replacing the reference rather than deallocating: a trace captured at the old shape still
        holds that buffer, and freeing it underneath the trace would be a use-after-free.
        """
        shape = [1, 1, rows, self.hidden_local]
        if not self.cache_padding:
            return ttnn.zeros(shape, dtype=dtype, layout=layout, device=self.mesh_device)

        key = (rows, dtype, layout)
        if self._pad_key != key:
            self._pad_buffer = ttnn.zeros(shape, dtype=dtype, layout=layout, device=self.mesh_device)
            self._pad_key = key
        return self._pad_buffer

    # `prep_run=False`, as in Wan: `_denoise` guarantees a complete untraced generation first. A single
    # prep forward is not enough -- it fills the program cache, but `CCLManager` still allocates its
    # persistent buffers lazily, which a capture rejects as a write.
    #
    # `clone_prep_inputs=False` because `forward` does not write to its inputs.
    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=False)
    def traced_step(
        self,
        *,
        video_1BVC: ttnn.Tensor,
        audio_1BAC: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        condition_blocks: list[tuple[ttnn.Tensor, str]] | None,
        timestep: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        timestep_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One denoising forward, shaped so `Tracer` can capture it.

        `Tracer` accepts only tensors and plain scalars, nested in tuples/lists/dicts, and validates
        that every input keeps its shape across calls. `timestep` is `[1, 1, num_slots, 1]`: the
        caller assigns each row a *fixed* slot by role rather than deduplicating levels per step, so
        `num_slots` is constant for the whole request and the tensor is a valid traced input -- only
        its values change, which the tracer copies in place.

        Everything else is fixed-shape for a given request: the row counts are set by the packed
        layout and the index tensors are built against `padded_len`.
        """
        return self.forward(
            video_1BVC=video_1BVC,
            audio_1BAC=audio_1BAC,
            prompt_1BLP=prompt_1BLP,
            condition_blocks=condition_blocks,
            timestep=timestep,
            adaln_indices=adaln_indices,
            timestep_indices=timestep_indices,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
