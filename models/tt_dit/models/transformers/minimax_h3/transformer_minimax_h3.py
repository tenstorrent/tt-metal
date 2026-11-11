# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from ....layers.embeddings import Timesteps
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.matmul import get_matmul_config, get_matmul_core_grid
from ....utils.tensor import from_torch, pad_single
from ....utils.tracing import StateTensor, traced_function
from .token_refiner_minimax_h3 import MiniMaxH3TokenRefiner
from .transformer_block_minimax_h3 import MODALITY_NUM, MiniMaxH3TransformerBlock, onehot_identity

# shift, scale -- the order `norm_out.linear` emits them in.
NUM_OUT_MODULATION_PARAMS = 2

# Any served per-device row count gives the output heads the same generic blocking (M_tiles >= 8).
_HEAD_CONFIG_ROWS = 4736

# all_gather_async workers per link on the dynamic-length path: the op's byte-based pick for every served
# length (the audio gather clears its 256 KB/link threshold by 1.2 %), pinned so the key cannot move.
_DYNAMIC_AG_WORKERS_PER_LINK = 4


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


@dataclass
class MiniMaxH3TwoTime:
    """Interval `(t, r)` conditioning: `temb = emb_t + gate * (emb_r - emb_t)`.

    Not a child module on purpose: a registered child would join the cached weight set and invalidate
    every existing cache.
    """

    embedder: MiniMaxH3TimestepEmbedding
    gate: float


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
        dynamic_length: bool = False,
        norm_row_capacity: int | None = None,
    ) -> None:
        super().__init__()

        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.tp_factor = parallel_config.tensor_parallel.factor
        self.hidden_local = hidden_size // self.tp_factor
        # Dynamic-length path: ROW_MAJOR gather tables (no untilize per gather) and a capacity-sized norm.
        self.dynamic_length = dynamic_length

        self.norm = DistributedRMSNorm(
            embedding_dim=hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=True,
            mesh_axis=self.tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            **({"row_capacity": norm_row_capacity} if dynamic_length else {}),
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
            tables.append(table if self.dynamic_length else ttnn.to_layout(table, ttnn.TILE_LAYOUT))

        def gather(table: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.unsqueeze(
                ttnn.embedding(timestep_indices, table, layout=ttnn.TILE_LAYOUT, runtime_rows=self.dynamic_length),
                0,
            )

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
    gathers it into packed order, and `ttnn.mesh_partition` fractures it across SP (or, with indices
    already sharded on SP, each device gathers only its own rows and no partition is needed).

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
        kv_gather_capacity: int | None = None,
        use_persistent_ccl_buffers: bool = True,
        dynamic_length: bool = False,
        audio_enabled: bool = True,
        num_output_heads: int = 1,
        pdd_head_fp32: bool = False,
    ) -> None:
        """`dynamic_length`: every length-keyed program choice is frozen at construction and the fused norms
        are sized for `kv_gather_capacity / sp` rows per device. Defaults to the legacy, length-keyed behaviour.

        `audio_enabled=False` is MiniMax-H3-Image: no audio projections and no audio rows in the packed sequence.
        `num_output_heads > 1` replaces the video head with that many per-step heads (PDD, Parallel Decoding
        Distillation); `forward(output_head=k)` picks head k, and the weights load from a `pdd_heads.{weight,bias}`
        stack of shape `[K, out, hidden]` / `[K, out]`."""
        super().__init__()

        self.audio_enabled = audio_enabled
        self.num_output_heads = num_output_heads
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.dynamic_length = dynamic_length
        # Per-device row capacity of the fused norms on the dynamic-length path (the K/V gather capacity / SP).
        norm_row_capacity = None
        if dynamic_length and kv_gather_capacity is not None:
            norm_row_capacity = kv_gather_capacity // parallel_config.sequence_parallel.factor
        self._temb_state = StateTensor()
        self._modulation_cache: dict = {}  # precompute_modulation: key -> per-block modulation tables
        self.refine_at_input_length = False  # see prepare_static_sources
        self._timestep_idx_state: dict[int, StateTensor] = {}  # traced path only, keyed by padded length
        self._eye_states: dict[int, StateTensor] = {}
        # The source table `forward` gathers rows from, `[text | cond video | cond audio | audio | video]`
        # at the arena caps, is one persistent ROW_MAJOR buffer written in place (`slice_write`), not rebuilt
        # by concat every step: a fragmented heap cannot supply a table-sized block each step. Allocated on
        # the first request, never freed (eviction skips non-weights). Static prefix written once per request.
        self._source_table: ttnn.Tensor | None = None
        self._static_prefix_rows = 0
        self._pending_prefix: list[ttnn.Tensor] | None = None
        self._prompt_windows_state = StateTensor()
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
        self.audio_proj_in = ColParallelLinear(audio_in_channels, hidden_size, **col_kwargs) if audio_enabled else None
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
        self.two_time: MiniMaxH3TwoTime | None = None

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
            use_persistent_ccl_buffers=use_persistent_ccl_buffers,
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
                    kv_gather_capacity=kv_gather_capacity,
                    use_persistent_ccl_buffers=use_persistent_ccl_buffers,
                    dynamic_length=dynamic_length,
                    norm_row_capacity=norm_row_capacity,
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
            dynamic_length=dynamic_length,
            norm_row_capacity=norm_row_capacity,
        )
        self.proj_out = Linear(hidden_size, video_patch_dim, bias=True, mesh_device=mesh_device)
        self.audio_proj_out = (
            Linear(hidden_size, audio_in_channels, bias=True, mesh_device=mesh_device) if audio_enabled else None
        )
        # PDD: one replicated head per denoise step. Eager or traced, switching heads only swaps which weight buffer
        # the (shape-identical) matmul reads, so it compiles nothing new.
        # The release runs these heads in fp32 (`F.linear` on fp32 weights after the final norm). `pdd_head_fp32`
        # (env H3_IMAGE_PDD_FP32=1) keeps their weights fp32 with an fp32-accumulating HiFi4 matmul; default bf16.
        head_kwargs = {}
        if pdd_head_fp32:
            head_kwargs = {
                "dtype": ttnn.float32,
                "compute_kernel_config": ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
                ),
            }
        self.proj_out_heads = (
            ModuleList(
                [
                    Linear(hidden_size, video_patch_dim, bias=True, mesh_device=mesh_device, **head_kwargs)
                    for _ in range(num_output_heads)
                ]
            )
            if num_output_heads > 1
            else None
        )
        # The heads' generic blocking is the same for every served length; resolve it once so the
        # dynamic-length path neither scans the tables nor takes the small-M 1D branch per call. The PDD heads share
        # the video head's shape and so its config.
        self._head_configs: dict[str, ttnn.MinimalMatmulConfig] = {}
        if dynamic_length:
            grid = get_matmul_core_grid(mesh_device)
            heads = (("video", video_patch_dim),) + ((("audio", audio_in_channels),) if audio_enabled else ())
            for name, width in heads:
                self._head_configs[name] = get_matmul_config(_HEAD_CONFIG_ROWS, hidden_size, width, grid)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Split a PDD head stack (`pdd_heads.weight` [K, out, hidden], `pdd_heads.bias` [K, out]) into per-step heads."""
        weight, bias = state.pop("pdd_heads.weight", None), state.pop("pdd_heads.bias", None)
        if weight is None:
            return
        if self.proj_out_heads is None or weight.shape[0] != self.num_output_heads:
            raise ValueError(f"PDD stack has {weight.shape[0]} heads; the model was built for {self.num_output_heads}")
        for k in range(weight.shape[0]):
            state[f"proj_out_heads.{k}.weight"] = weight[k]
            state[f"proj_out_heads.{k}.bias"] = bias[k]

    def prepare_static_sources(
        self,
        *,
        prompt_1BLP: ttnn.Tensor,
        prompt_len: int,
        condition_video_1BKC: ttnn.Tensor | None = None,
        condition_audio_1BKC: ttnn.Tensor | None = None,
        prompt_cap: int,
        traced: bool = False,
    ) -> None:
        """Refine and project the step-invariant streams, once per request.

        Stores the `[text | condition video | condition audio]` source-table prefix that `forward` reads.
        `prompt_len` is the true prompt length; the rows past it are padding.
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

        # Padding before the refiner, not after, keeps its programs independent of prompt length. A caller whose
        # prompts already arrive padded to a warmed rung (`refine_at_input_length`) refines at that rung and pads
        # after: the refiner then costs the rung, not the arena cap, and still compiles one program set per rung.
        refine_rows = prompt_1BLP.shape[2] if self.refine_at_input_length else prompt_cap
        if prompt_1BLP.shape[2] < refine_rows:
            prompt_1BLP = pad_single(prompt_1BLP, dim=2, back=refine_rows - prompt_1BLP.shape[2])
        prompt_windows = None
        if prompt_len < refine_rows:
            self._prompt_windows_state.update(
                from_torch(
                    torch.tensor([0, prompt_len, refine_rows], dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.Layout.ROW_MAJOR,
                    mesh_axes=[None],
                ),
                traced=traced,
            )
            prompt_windows = self._prompt_windows_state.value
        refined = self.token_refiner(self.context_embedder(prompt_1BLP), cu_window_seqlens=prompt_windows)
        if refined.shape[2] < prompt_cap:
            refined = pad_single(refined, dim=2, back=prompt_cap - refined.shape[2])
        segments = [refined]
        if condition_video_1BKC is not None:
            segments.append(self.proj_in(condition_video_1BKC))
        if condition_audio_1BKC is not None:
            segments.append(self.audio_proj_in(condition_audio_1BKC))
        # ROW_MAJOR is what `ttnn.embedding` reads (a TILE table would be untilized every step).
        # The segments are written into the persistent source table by the first `forward` of the
        # request, which is where the audio/video capacities -- and so the table size -- are known.
        if self._pending_prefix:
            for segment in self._pending_prefix:
                ttnn.deallocate(segment)
        self._pending_prefix = [ttnn.to_layout(segment, ttnn.ROW_MAJOR_LAYOUT) for segment in segments]
        self._static_prefix_rows = sum(segment.shape[2] for segment in self._pending_prefix)

    def _write_source_rows(self, segment_1BRC: ttnn.Tensor, row: int) -> int:
        """Write a ``[1, 1, rows, C]`` ROW_MAJOR segment into the source table at ``row``; return the next row."""
        rows, width = segment_1BRC.shape[2], segment_1BRC.shape[3]
        ttnn.experimental.slice_write(
            segment_1BRC, self._source_table, [0, 0, row, 0], [1, 1, row + rows, width], [1, 1, 1, 1]
        )
        return row + rows

    def _ensure_source_table(self, total_rows: int, width: int, dtype: ttnn.DataType) -> None:
        table = self._source_table
        if table is not None and (table.shape[2] < total_rows or table.shape[3] != width or table.dtype != dtype):
            ttnn.deallocate(table)
            table = None
        if table is None:
            self._source_table = ttnn.zeros(
                (1, 1, total_rows, width), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            )

    def forward(
        self,
        *,
        video_1BVC: ttnn.Tensor,
        audio_1BAC: ttnn.Tensor | None,
        assembly_indices: ttnn.Tensor,
        video_out_indices: ttnn.Tensor,
        audio_out_indices: ttnn.Tensor | None,
        timestep: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        timestep_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        logical_n: ttnn.Tensor,
        pad_to: int,
        endpoint_timestep: ttnn.Tensor | None = None,
        traced: bool = False,
        adaln_tile_map: ttnn.Tensor | None = None,
        adaln_expanded_indices: ttnn.Tensor | None = None,
        output_head: int = 0,
        modulation_key=None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        Audio-free (`audio_enabled=False`): pass `audio_1BAC=None, audio_out_indices=None`; the source table is
        `[text | condition video | video]` and the audio velocity returned is None. `output_head` picks the PDD head.

        Every stream is a fixed-capacity buffer, true rows leading; `prepare_static_sources` must run first.

        video_1BVC: [1, 1, V_cap, in_channels * prod(patch_size)], replicated on SP and TP. Target rows only.
        audio_1BAC: [1, 1, A_cap, audio_in_channels], replicated on SP and TP. Target rows only.
        assembly_indices: [1, 1, 1, pad_to] integers, the source-table row of each packed row.
        video_out_indices / audio_out_indices: [1, 1, 1, V_cap] / [1, 1, 1, A_cap] integers, the
            packed row of each target row.
        timestep: [1, 1, num_slots, 1] float32, replicated. Unscaled, in [0, 1].
        endpoint_timestep: like `timestep`, each slot's step endpoint `r`. Required when `two_time` is set.
        adaln_indices: [1, 1, 1, S_padded_local] integers, `timestep_indices * 3 + token_tags`, built
            for the padded global sequence and sharded on SP
        timestep_indices: [1, 1, 1, S_padded_local] integers, same order
        rope_cos/rope_sin: [1, 1, S_padded_local, rotary_dim] float32, same order, replicated on TP
        logical_n: the true packed length `L + K + A + V` as a [1, 1, 1, 1] uint32 device tensor.
        pad_to: the padded packed length; keys the trace together with the presence of the tile-row map (one
            capture per `(pad_to, map present)`).
        adaln_tile_map / adaln_expanded_indices: `adaln_tilerow.tilerow_remap` tables, sharded on SP; the norms read
            their modulation through the tile-row map. `None` (no map for this request) means the per-token gather.
        modulation_key: a key passed to `precompute_modulation` with this same `timestep`; the blocks then reuse
            those modulation tables instead of running their adaLN projections (every block's 2688 -> 96768
            linear, 13B params read per forward). The caller guarantees the timestep matches the key.

        Returns `(video_velocity, audio_velocity)` as [1, 1, V_cap, .] / [1, 1, A_cap, .], target rows only.
        """
        block_tables = None
        if modulation_key is not None:
            if modulation_key not in self._modulation_cache:
                raise KeyError(f"no precomputed modulation for key {modulation_key!r}; call precompute_modulation")
            block_tables = self._modulation_cache[modulation_key]
        tile = ttnn.TILE_SIZE
        alignment = self.sp_factor * tile
        if pad_to % alignment:
            raise ValueError(f"pad_to={pad_to} must be a multiple of sp_factor * TILE = {alignment}")
        if self.dynamic_length and (traced or ttnn.is_trace_capture_active(self.mesh_device)):
            raise RuntimeError(
                "dynamic_length is eager-only: the row count is a runtime argument, so a trace captured now would "
                "replay this length whatever the buffers hold; use the bucketed preset for a traced denoise"
            )
        # Indices already sharded on SP only need to gather the local rows, so the full-sequence
        # intermediate doesn't need to be materialised
        local_rows = pad_to // self.sp_factor
        indices_already_sharded = self.sp_factor > 1 and assembly_indices.shape[-1] == local_rows
        if assembly_indices.shape[-1] != pad_to and not indices_already_sharded:
            raise ValueError(
                f"assembly_indices has {assembly_indices.shape[-1]} rows; expected pad_to={pad_to} (replicated) "
                f"or pad_to / sp_factor = {local_rows} (sharded on SP)"
            )
        if video_out_indices.shape[-1] != video_1BVC.shape[2]:
            raise ValueError("video_out_indices must match the video stream's capacity")
        if (audio_1BAC is None) != (audio_out_indices is None) or (audio_1BAC is not None) != self.audio_enabled:
            raise ValueError(
                f"audio_1BAC / audio_out_indices must both be {'given' if self.audio_enabled else 'None'} "
                f"(audio_enabled={self.audio_enabled})"
            )
        if audio_1BAC is not None and audio_out_indices.shape[-1] != audio_1BAC.shape[2]:
            raise ValueError("audio_out_indices must match the audio stream's capacity")
        if self.proj_out_heads is not None and not 0 <= output_head < self.num_output_heads:
            raise ValueError(f"output_head {output_head} out of range for {self.num_output_heads} PDD heads")
        for name, stream in (("audio_1BAC", audio_1BAC), ("video_1BVC", video_1BVC)):
            if stream is not None and stream.shape[2] % tile:
                raise ValueError(f"{name} capacity {stream.shape[2]} must be a multiple of TILE ({tile})")
        if self._static_prefix_rows == 0:
            raise RuntimeError("prepare_static_sources must run before forward: the source-table prefix is unbound")

        def as_indices(t: ttnn.Tensor) -> ttnn.Tensor:
            t = ttnn.reshape(t, (1, t.shape[-1]))
            return t if t.dtype == ttnn.uint32 else ttnn.typecast(t, ttnn.uint32)

        audio_rows = (
            ttnn.to_layout(self.audio_proj_in(audio_1BAC), ttnn.ROW_MAJOR_LAYOUT) if audio_1BAC is not None else None
        )
        video_rows = ttnn.to_layout(self.proj_in(video_1BVC), ttnn.ROW_MAJOR_LAYOUT)
        audio_count = audio_rows.shape[2] if audio_rows is not None else 0
        total_rows = self._static_prefix_rows + audio_count + video_rows.shape[2]
        self._ensure_source_table(total_rows, video_rows.shape[3], video_rows.dtype)
        if self._pending_prefix is not None:
            row = 0
            for segment in self._pending_prefix:
                row = self._write_source_rows(segment, row)
                ttnn.deallocate(segment)
            self._pending_prefix = None
        row = self._static_prefix_rows
        if audio_rows is not None:
            row = self._write_source_rows(audio_rows, row)
            ttnn.deallocate(audio_rows)
        self._write_source_rows(video_rows, row)
        ttnn.deallocate(video_rows)

        # `ttnn.embedding` takes the `[1, 1, rows, C]` table as is; indices never reach past
        # `total_rows`, so rows beyond it (a smaller table layout than an earlier request) are inert.
        hidden = ttnn.embedding(
            as_indices(assembly_indices), self._source_table, layout=ttnn.TILE_LAYOUT, runtime_rows=self.dynamic_length
        )
        hidden = ttnn.unsqueeze(hidden, 0)
        if not indices_already_sharded:
            hidden = ttnn.mesh_partition(hidden, 2, cluster_axis=self.sp_mesh_axis)

        temb = self.time_embedder(self.time_proj(timestep))
        if self.two_time is not None:
            if endpoint_timestep is None:
                raise ValueError("two-time conditioning is set, so every forward needs endpoint_timestep")
            endpoint = self.two_time.embedder(self.time_proj(endpoint_timestep))
            temb = ttnn.add(temb, ttnn.multiply(ttnn.subtract(endpoint, temb), self.two_time.gate))
        self._temb_state.update(temb, traced=traced)
        temb = self._temb_state.value

        adaln_idx = as_indices(adaln_indices)
        if traced:
            # `norm_out` reads it eagerly after the traced block loop, but a replay may overwrite any buffer
            # allocated after its capture, so the traced path keeps one persistent copy per length.
            ts_state = self._timestep_idx_state.setdefault(pad_to, StateTensor())
            ts_state.update(as_indices(timestep_indices), traced=True)
            timestep_idx = ts_state.value
        else:
            # Eager (every dynamic-length forward): nothing replays, and a per-length copy would pin one
            # buffer per served length.
            timestep_idx = as_indices(timestep_indices)

        # Identity table of the one-hot gathers, refreshed from the host every forward like the rest of the
        # per-forward state: a trace captured earlier may overwrite long-lived device buffers on replay.
        rows = temb.shape[2] * MODALITY_NUM
        eye_state = self._eye_states.setdefault(rows, StateTensor())
        if self.dynamic_length:
            eye_state.update(onehot_identity(rows, self.mesh_device), traced=traced)
        else:
            eye_state.update(torch.eye(rows), traced=traced, dtype=ttnn.bfloat16, device=self.mesh_device)
        eye = eye_state.value

        hidden = self.run_blocks(
            hidden,
            logical_n,
            temb,
            adaln_idx,
            rope_cos,
            rope_sin,
            eye=eye,
            adaln_tile_map=adaln_tile_map,
            adaln_expanded_indices=as_indices(adaln_expanded_indices) if adaln_expanded_indices is not None else None,
            block_tables=block_tables,
            traced=traced,
            tracer_trace_key=(pad_to, adaln_tile_map is not None),
        )

        hidden = self.norm_out(
            hidden,
            temb,
            timestep_idx,
        )
        ag_kwargs = {"num_workers_per_link": _DYNAMIC_AG_WORKERS_PER_LINK} if self.dynamic_length else {}
        if self.tp_factor > 1:
            hidden = self.ccl_manager.all_gather(
                hidden, dim=3, mesh_axis=self.tp_mesh_axis, use_hyperparams=False, **ag_kwargs
            )

        video_head = self.proj_out_heads[output_head] if self.proj_out_heads is not None else self.proj_out
        video_all = video_head(hidden, matmul_config=self._head_configs.get("video"))
        audio_all = (
            self.audio_proj_out(hidden, matmul_config=self._head_configs.get("audio"))
            if audio_1BAC is not None
            else None
        )
        if self.sp_factor > 1:
            video_all = self.ccl_manager.all_gather(
                video_all, dim=2, mesh_axis=self.sp_mesh_axis, use_hyperparams=False, **ag_kwargs
            )
            if audio_all is not None:
                audio_all = self.ccl_manager.all_gather(
                    audio_all, dim=2, mesh_axis=self.sp_mesh_axis, use_hyperparams=False, **ag_kwargs
                )

        def select(all_rows: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
            table = ttnn.reshape(all_rows, (all_rows.shape[2], all_rows.shape[3]))
            # runtime_rows keeps the gather's and the TILE table's untilize's programs free of the request's length.
            gathered = ttnn.embedding(
                as_indices(indices), table, layout=ttnn.TILE_LAYOUT, runtime_rows=self.dynamic_length
            )
            return ttnn.unsqueeze(gathered, 0)

        audio_out = select(audio_all, audio_out_indices) if audio_all is not None else None
        return select(video_all, video_out_indices), audio_out

    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=False)
    def run_blocks(
        self,
        hidden: ttnn.Tensor,
        logical_n: ttnn.Tensor,
        temb: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        eye: ttnn.Tensor | None = None,
        adaln_tile_map: ttnn.Tensor | None = None,
        adaln_expanded_indices: ttnn.Tensor | None = None,
        block_tables: list[list[ttnn.Tensor]] | None = None,
    ) -> ttnn.Tensor:
        onehot = self.transformer_blocks[0].onehot_table(adaln_indices, temb.shape[2], eye)
        tilerow = self.transformer_blocks[0].tilerow_tables(adaln_tile_map, adaln_expanded_indices, temb.shape[2], eye)
        for i, block in enumerate(self.transformer_blocks):
            hidden = block(
                hidden,
                logical_n,
                temb=temb,
                adaln_indices=adaln_indices,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                onehot=onehot,
                tilerow=tilerow,
                tables=block_tables[i] if block_tables is not None else None,
            )
        ttnn.deallocate(onehot)
        if tilerow is not None:
            ttnn.deallocate(tilerow[1])
        return hidden

    def precompute_modulation(self, key, timestep: ttnn.Tensor) -> None:
        """Compute every block's modulation tables for `timestep` once and keep them under `key`.

        A sampler with a fixed schedule (H3-Image's 6 PDD steps) then passes `modulation_key=key` to `forward`,
        which skips the per-block adaLN projections: 13B params streamed from DRAM per forward for a handful of
        rows. Costs `6 tables x 32 rows x hidden_local` bf16 per block per key (about 0.5 MB at TP=4).

        A key already held is kept as is (the caller guarantees a key always means the same timestep), so calling
        this every step costs nothing after the first and never leaks the tables it would otherwise replace. The
        tables live until `clear_modulation_cache`; nothing tears this model down, so a long-lived owner that
        changes schedules should call it.
        """
        if key in self._modulation_cache:
            return
        if self.two_time is not None:
            raise NotImplementedError("precomputed modulation does not cover two-time conditioning")
        temb = self.time_embedder(self.time_proj(timestep))
        self._modulation_cache[key] = [block.modulation_tables(temb) for block in self.transformer_blocks]
        ttnn.deallocate(temb)

    def clear_modulation_cache(self) -> None:
        """Deallocate every `precompute_modulation` table and forget their keys."""
        for tables in self._modulation_cache.values():
            for block_tables in tables:
                for table in block_tables:
                    ttnn.deallocate(table)
        self._modulation_cache.clear()

    def release_traces(self) -> None:
        """Release every captured `run_blocks` trace, across all `tracer_trace_key` buckets."""
        run_blocks = type(self).run_blocks
        tracers = [run_blocks._tracers.get(self), *run_blocks._tracers_keyed.get(self, {}).values()]
        for tracer in tracers:
            if tracer is not None:
                tracer.release_trace()
