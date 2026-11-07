# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import ttnn

from ....layers.embeddings import TimestepEmbedding, Timesteps
from ....layers.linear import ColParallelLinear, Linear, prepare_chunked_linear_output
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedLayerNorm
from ....utils import tensor
from ....utils.matmul import register_matmul_configs
from ....utils.substate import rename_substate
from .transformer_block_flux3_klein import HIDDEN, Flux3KleinTransformerBlock

if TYPE_CHECKING:
    from ....parallel.config import DiTParallelConfig
    from ....parallel.manager import CCLManager

# Sequence order of the single stage; every mode has its own early stack.
MODES = ("image", "image_cond", "text")
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}
PAD_MODE_ID = 3
EARLY_DEPTH = 5
SINGLE_DEPTH = 28
TILE = 32

ROPE_THETA = 10000
ROPE_AXES_DIM = (32, 32, 32, 32)

TIME_CHANNELS = 256
# The 7 modulation heads of the checkpoint are merged into one linear of 20 chunks:
# early (image, image_cond, text) x (shift, scale, gate), single (same), final (shift, scale).
MOD_SOURCES = (
    ("early_stream_modulations.image.lin", 3),
    ("early_stream_modulations.image_cond.lin", 3),
    ("early_stream_modulations.txt.lin", 3),
    ("single_stream_modulations.image.lin", 3),
    ("single_stream_modulations.image_cond.lin", 3),
    ("single_stream_modulations.txt.lin", 3),
    ("final_layer.image.adaLN_modulation.1", 2),
)
MOD_CHUNKS = 20
MOD_SCALE_CHUNKS = (1, 4, 7, 10, 13, 16, 19)
MOD_FINAL_SHIFT_CHUNK, MOD_FINAL_SCALE_CHUNK = 18, 19
# The modulation input has 3 rows: the image timestep embedding, the conditioning timestep (zero) and a zero row
# that gives pad rows the identity modulation (0, 1, 0).
MOD_VEC_ROWS = 3
VEC_ROW_OF_MODE_ID = (0, 1, 1, 2)

STATE_PREFIXES = (
    ("emb_in.image", "emb_in_image"),
    ("emb_in.image_cond", "emb_in_image_cond"),
    ("time_in", "modulation.time_in"),
    ("ref_time_in", "modulation.ref_time_in"),
    ("early_stream_modulations", "modulation.early_stream_modulations"),
    ("single_stream_modulations", "modulation.single_stream_modulations"),
    ("final_layer.image.adaLN_modulation", "modulation.final_layer.image.adaLN_modulation"),
    ("content_mode_blocks", "early.content_mode_blocks"),
    ("txt_mode_blocks", "early.txt_mode_blocks"),
    ("final_layer.image.linear", "proj_out"),
)


# SP8 x TP4 MLP shapes (M, 3072, N) whose heuristic blocking overflows L1 on the 12x9 worker grid of a Galaxy
register_matmul_configs(
    {
        "12x9": {
            (2560, 3072, 4608): (4, 8, 16, (1, 4)),
            (3072, 3072, 4608): (4, 8, 16, (1, 4)),
            (8192, 3072, 2304): (11, 8, 8, (1, 4)),
            (8192, 3072, 4608): (6, 8, 16, (1, 4)),
            (8704, 3072, 2304): (8, 8, 8, (1, 4)),
            (8704, 3072, 4608): (6, 8, 16, (1, 4)),
            (9216, 3072, 2304): (8, 8, 8, (1, 4)),
            (9216, 3072, 4608): (6, 8, 16, (1, 4)),
        }
    }
)


def padded_length(length: int, sp_factor: int) -> int:
    alignment = sp_factor * TILE
    return -(-length // alignment) * alignment


@dataclass(frozen=True)
class Flux3KleinSequenceLayout:
    """Row layout of one sample on the SP axis.

    Each mode is a separate SP-sharded tensor of padded_length(L_m) rows. The single stage is
    [image | image_cond | text | pad] with all alignment rows at the end, reached with one all-gather of the per-device
    concat [img_d | cond_d | txt_d] and a row gather. Text rows past text_valid are masked as attention keys.
    """

    sp_factor: int
    lengths: dict[str, int]
    text_valid: int
    padded: dict[str, int] = field(init=False)
    rows: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        padded = {m: padded_length(self.lengths[m], self.sp_factor) for m in MODES}
        object.__setattr__(self, "padded", padded)
        object.__setattr__(self, "rows", {m: padded[m] // self.sp_factor for m in MODES})

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(m for m in MODES if self.lengths[m] > 0)

    @property
    def total(self) -> int:
        return sum(self.lengths.values())

    @property
    def total_padded(self) -> int:
        return sum(self.rows.values()) * self.sp_factor

    def attended(self, mode: str | None = None) -> int:
        masked_text = self.lengths["text"] - self.text_valid
        if mode is None:
            return self.total - masked_text
        return self.lengths[mode] - (masked_text if mode == "text" else 0)

    def _source_row(self, mode: str, i: int) -> int:
        device, local = divmod(i, self.rows[mode])
        offset = sum(self.rows[m] for m in MODES[: MODES.index(mode)])
        return device * sum(self.rows.values()) + offset + local

    def assembly_indices(self) -> torch.Tensor:
        indices = [self._source_row(m, i) for m in self.modes for i in range(self.lengths[m])]
        if self.total_padded > self.total:
            # pad rows copy an alignment row of the early stacks, which is exactly zero
            m = next(m for m in self.modes if self.padded[m] > self.lengths[m])
            indices += [self._source_row(m, self.lengths[m])] * (self.total_padded - self.total)
        return torch.tensor(indices, dtype=torch.int64)

    def image_out_indices(self) -> torch.Tensor:
        length = self.lengths["image"]
        return torch.tensor(list(range(length)) + [0] * (self.padded["image"] - length), dtype=torch.int64)

    def mode_ids(self, mode: str | None = None) -> torch.Tensor:
        if mode is None:
            ids = [MODES.index(m) for m in self.modes for _ in range(self.lengths[m])]
            return torch.tensor(ids + [PAD_MODE_ID] * (self.total_padded - self.total), dtype=torch.int64)
        pad = self.padded[mode] - self.lengths[mode]
        return torch.tensor([MODES.index(mode)] * self.lengths[mode] + [PAD_MODE_ID] * pad, dtype=torch.int64)


def rope_cos_sin(ids: torch.Tensor, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference rope_embed (float64 angles, fp32 entries) as interleaved cos / sin [1, 1, padded, 128];
    pad rows get the identity rotation."""
    pos = ids.to(torch.float64)
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(ROPE_AXES_DIM):
        scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        angle = torch.einsum("...n,d->...nd", pos[:, axis], 1.0 / (ROPE_THETA**scale))
        cos_parts.append(torch.cos(angle))
        sin_parts.append(torch.sin(angle))
    cos = torch.cat(cos_parts, dim=-1).float().repeat_interleave(2, dim=-1)
    sin = torch.cat(sin_parts, dim=-1).float().repeat_interleave(2, dim=-1)
    pad = padded - ids.shape[0]
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad), value=1.0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad), value=0.0)
    return cos.reshape(1, 1, padded, -1), sin.reshape(1, 1, padded, -1)


def modulation_indices(mode_ids: torch.Tensor, stage: str) -> tuple[torch.Tensor, ...]:
    """Rows of the [MOD_CHUNKS * MOD_VEC_ROWS, hidden] modulation table holding each row's (shift, scale, gate)."""
    stage_offset = {"early": 0, "single": 9}[stage]
    vec_row = torch.tensor(VEC_ROW_OF_MODE_ID)[mode_ids]
    out = []
    for param in range(3):
        # pad rows (mode id 3) read the image chunks at the zero vec row
        chunk_of_mode = torch.tensor([stage_offset + 3 * m + param for m in (0, 1, 2, 0)])
        out.append(MOD_VEC_ROWS * chunk_of_mode[mode_ids] + vec_row)
    return tuple(out)


def timestep_inputs(t_embed: float, t_ref_embed: float | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    t = torch.tensor([[t_embed], [0.0], [0.0]], dtype=torch.float32)
    t_ref = torch.tensor([[t_ref_embed]], dtype=torch.float32) if t_ref_embed is not None else None
    return t, t_ref


def gather_rows(table: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
    out = ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT)
    return ttnn.reshape(out, (1, out.shape[-2], out.shape[-1]))


def gather_rows_over_sp(x: ttnn.Tensor, indices: ttnn.Tensor, *, ccl_manager: CCLManager, sp_axis: int) -> ttnn.Tensor:
    gathered = ccl_manager.all_gather(x, dim=1, mesh_axis=sp_axis, use_hyperparams=False, use_persistent_buffer=True)
    return gather_rows(ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT), indices)


class Flux3KleinModulation(Module):
    def __init__(
        self, *, flow_map: bool, device: ttnn.MeshDevice, ccl_manager: CCLManager, parallel_config: DiTParallelConfig
    ) -> None:
        super().__init__()
        self.device = device
        self.parallel_config = parallel_config
        self.tp_factor = parallel_config.tensor_parallel.factor

        self.time_proj = Timesteps(
            num_channels=TIME_CHANNELS, cos_first=True, max_period=10000, dtype=ttnn.float32, mesh_device=device
        )
        embedding = lambda: TimestepEmbedding(  # noqa: E731
            TIME_CHANNELS, HIDDEN, act_fn="silu", dtype=ttnn.float32, bias=False, mesh_device=device
        )
        self.time_in = embedding()
        # the distilled flow map also embeds the target timestep
        self.ref_time_in = embedding() if flow_map else None
        self.lin = ColParallelLinear(
            HIDDEN,
            MOD_CHUNKS * HIDDEN,
            bias=True,
            mesh_device=device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            chunks=MOD_CHUNKS,
        )
        self._row_mask = None
        self.register_eviction_hook(self._release_row_mask)

    def _release_row_mask(self) -> None:
        if self._row_mask is not None:
            ttnn.deallocate(self._row_mask)
            self._row_mask = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for embedder in ("time_in", "ref_time_in") if self.ref_time_in is not None else ("time_in",):
            for src, dst in (("in_layer", "linear_1"), ("out_layer", "linear_2")):
                if f"{embedder}.{src}.weight" in state:
                    state[f"{embedder}.{dst}.weight"] = state.pop(f"{embedder}.{src}.weight")
        weights = [state.pop(f"{prefix}.weight", None) for prefix, _ in MOD_SOURCES]
        if all(w is not None for w in weights):
            state["lin.weight"] = torch.cat(weights, dim=0)
            # the reference applies (1 + scale); fold the 1 into a bias on the scale chunks
            state["lin.bias"] = torch.cat(
                [torch.full([HIDDEN], 1.0 if i in MOD_SCALE_CHUNKS else 0.0) for i in range(MOD_CHUNKS)]
            )
        prepare_chunked_linear_output(state, prefix="lin", device_count=self.tp_factor, chunks=MOD_CHUNKS)

    def _sinusoid(self, t: ttnn.Tensor) -> ttnn.Tensor:
        # the reference embeds the timestep in bf16
        return ttnn.typecast(ttnn.typecast(self.time_proj(t), ttnn.bfloat16), ttnn.float32)

    def forward(
        self, t_embed: ttnn.Tensor, t_ref_embed: ttnn.Tensor | None
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]:
        if self._row_mask is None:
            mask = torch.tensor([[1.0], [1.0], [0.0]]).expand(MOD_VEC_ROWS, HIDDEN).contiguous()
            self._row_mask = tensor.from_torch(
                mask[None], device=self.device, dtype=ttnn.float32, mesh_axes=[None, None, None]
            )
        vec = self.time_in(self._sinusoid(t_embed))
        if self.ref_time_in is not None:
            vec = ttnn.add(vec, self.ref_time_in(self._sinusoid(t_ref_embed)))
        vec = ttnn.reshape(vec, (1, vec.shape[-2], HIDDEN))
        act = ttnn.typecast(ttnn.multiply(ttnn.silu(vec), self._row_mask), ttnn.bfloat16)
        chunks = self.lin(act, parallel_config=self.parallel_config, dtype=ttnn.bfloat16)

        table = ttnn.concat([ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT) for c in chunks], dim=1)
        table = ttnn.reshape(table, (MOD_CHUNKS * MOD_VEC_ROWS, table.shape[-1]))

        def row0(chunk: ttnn.Tensor) -> ttnn.Tensor:
            c4 = ttnn.unsqueeze_to_4D(chunk)
            return ttnn.reshape(ttnn.slice(c4, [0, 0, 0, 0], [1, 1, 1, c4.shape[-1]]), (1, 1, c4.shape[-1]))

        return table, (row0(chunks[MOD_FINAL_SHIFT_CHUNK]), row0(chunks[MOD_FINAL_SCALE_CHUNK]))


class Flux3KleinEarlyStacks(Module):
    """The per-mode blocks that run before the modes are joined."""

    def __init__(
        self, *, device: ttnn.MeshDevice, ccl_manager: CCLManager, parallel_config: DiTParallelConfig, is_fsdp: bool
    ) -> None:
        super().__init__()
        self._ring = ccl_manager.topology == ttnn.Topology.Ring
        for mode in MODES:
            setattr(
                self,
                mode,
                ModuleList(
                    Flux3KleinTransformerBlock(
                        device=device, ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=is_fsdp
                    )
                    for _ in range(EARLY_DEPTH)
                ),
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for src, dst in (
            ("content_mode_blocks.image", "image"),
            ("content_mode_blocks.image_cond", "image_cond"),
            ("txt_mode_blocks", "text"),
        ):
            rename_substate(state, src, dst)

    def forward(self, mode: str, x: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        for block in getattr(self, mode):
            x = block(x, **kwargs)
        if self._ring:
            # the fused addcmul output aliases a persistent buffer the next stack reuses
            x = ttnn.clone(x)
        return x


@dataclass
class Flux3KleinPlan:
    """Device-side per-request constants: RoPE tables, modulation row indices and row gathers."""

    layout: Flux3KleinSequenceLayout
    early_rope: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]
    early_mod: dict[str, tuple[ttnn.Tensor, ...]]
    single_rope: tuple[ttnn.Tensor, ttnn.Tensor]
    single_mod: tuple[ttnn.Tensor, ...]
    assembly: ttnn.Tensor
    image_out: ttnn.Tensor

    def deallocate(self) -> None:
        for t in (
            *(t for pair in self.early_rope.values() for t in pair),
            *(t for idx in self.early_mod.values() for t in idx),
            *self.single_rope,
            *self.single_mod,
            self.assembly,
            self.image_out,
        ):
            ttnn.deallocate(t)


class Flux3KleinTransformer(Module):
    def __init__(
        self,
        *,
        flow_map: bool,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.flow_map = flow_map
        self.sp_factor = parallel_config.sequence_parallel.factor
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._tp_axis = tp_axis
        self._sp_axis = parallel_config.sequence_parallel.mesh_axis

        def embedding(mode: str) -> ColParallelLinear:
            return ColParallelLinear(
                IN_CHANNELS[mode], HIDDEN, bias=False, mesh_device=device, mesh_axis=tp_axis, ccl_manager=ccl_manager
            )

        self.txt_in = embedding("text")
        self.emb_in_image = embedding("image")
        self.emb_in_image_cond = embedding("image_cond")
        self.modulation = Flux3KleinModulation(
            flow_map=flow_map, device=device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )
        self.early = Flux3KleinEarlyStacks(
            device=device, ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=is_fsdp
        )
        self.single_blocks = ModuleList(
            Flux3KleinTransformerBlock(
                device=device, ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=is_fsdp
            )
            for _ in range(SINGLE_DEPTH)
        )
        self.norm_out = DistributedLayerNorm(
            HIDDEN,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )
        self.proj_out = Linear(HIDDEN, IN_CHANNELS["image"], bias=False, mesh_device=device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for src, dst in STATE_PREFIXES:
            rename_substate(state, src, dst)

    def prepare_plan(self, layout: Flux3KleinSequenceLayout, ids: dict[str, torch.Tensor]) -> Flux3KleinPlan:
        def rope(mode_ids: torch.Tensor, padded: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
            return tuple(
                tensor.from_torch(
                    t, device=self._device, dtype=ttnn.float32, mesh_axes=[None, None, self._sp_axis, None]
                )
                for t in rope_cos_sin(mode_ids, padded)
            )

        def indices(idx: torch.Tensor) -> ttnn.Tensor:
            return tensor.from_torch(
                idx.to(torch.int32)[None],
                device=self._device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_axes=[None, self._sp_axis],
            )

        modes = layout.modes
        return Flux3KleinPlan(
            layout=layout,
            early_rope={m: rope(ids[m][: layout.lengths[m]], layout.padded[m]) for m in modes},
            early_mod={m: tuple(indices(i) for i in modulation_indices(layout.mode_ids(m), "early")) for m in modes},
            single_rope=rope(torch.cat([ids[m][: layout.lengths[m]].float() for m in modes]), layout.total_padded),
            single_mod=tuple(indices(i) for i in modulation_indices(layout.mode_ids(), "single")),
            assembly=indices(layout.assembly_indices()),
            image_out=indices(layout.image_out_indices()),
        )

    def embed_text(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        return self.txt_in(tokens, parallel_config=self._parallel_config)

    def embed_image_cond(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        return self.emb_in_image_cond(tokens, parallel_config=self._parallel_config)

    def forward(
        self,
        *,
        image_tokens: ttnn.Tensor,
        text_embedded: ttnn.Tensor,
        image_cond_embedded: ttnn.Tensor | None,
        t_embed: ttnn.Tensor,
        t_ref_embed: ttnn.Tensor | None,
        plan: Flux3KleinPlan,
        cache=None,
    ) -> ttnn.Tensor:
        layout = plan.layout
        table, (final_shift, final_scale_plus_one) = self.modulation(t_embed, t_ref_embed)

        x = {"image": self.emb_in_image(image_tokens, parallel_config=self._parallel_config), "text": text_embedded}
        if image_cond_embedded is not None:
            x["image_cond"] = image_cond_embedded
        for mode in layout.modes:
            mod = tuple(gather_rows(table, idx) for idx in plan.early_mod[mode])
            x[mode] = self.early(
                mode, x[mode], rope=plan.early_rope[mode], mod=mod, logical_length=layout.attended(mode)
            )

        parts = [x[m] for m in layout.modes]
        local = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1)
        h = gather_rows_over_sp(local, plan.assembly, ccl_manager=self._ccl_manager, sp_axis=self._sp_axis)
        for part in parts:
            ttnn.deallocate(part)

        mod = tuple(gather_rows(table, idx) for idx in plan.single_mod)

        def run_block(block: Flux3KleinTransformerBlock, h: ttnn.Tensor) -> ttnn.Tensor:
            return block(h, rope=plan.single_rope, mod=mod, logical_length=layout.attended())

        if cache is not None:
            h = cache(self.single_blocks, h, run_block, layout=layout, sp_axis=self._sp_axis)
        else:
            for block in self.single_blocks:
                h = run_block(block, h)

        y = ttnn.squeeze(
            self.norm_out(ttnn.unsqueeze(h, 0), dynamic_weight=final_scale_plus_one, dynamic_bias=final_shift), 0
        )
        y = self._ccl_manager.all_gather(y, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=False)
        y = self.proj_out(y)
        return gather_rows_over_sp(y, plan.image_out, ccl_manager=self._ccl_manager, sp_axis=self._sp_axis)
