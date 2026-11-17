# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import ttnn

from ....layers.embeddings import TimestepEmbedding, Timesteps
from ....layers.linear import ColParallelLinear, Linear, prepare_chunked_linear_output
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedLayerNorm
from ....utils import tensor
from ....utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs, register_matmul_configs
from ....utils.substate import rename_substate
from .transformer_block_flux3_klein import HIDDEN, PRECISION_BF16, Flux3KleinPrecision, Flux3KleinTransformerBlock

if TYPE_CHECKING:
    from ....parallel.config import DiTParallelConfig
    from ....parallel.manager import CCLManager

# Sequence order of the single stage; every mode has its own early stack.
MODES = ("image", "image_cond", "text")
# the early stacks whose attended length changes per request (prompt length; reference size within its bucket)
VARIABLE_MODES = ("image_cond", "text")
# host RoPE rows kept for the image and text ids: the served set (5 image shapes, the text buckets) is ~160 MB
# (4096^2: 65536 rows x 128 fp32 x cos / sin = 64 MB); 16 entries bound it to ~1 GB per process
ROPE_ROWS_CACHE_SIZE = 16
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


# proj_out (fused matmul + strided reduce-scatter + addcmul) at the 4096 per-device rows: N block 8 = the core's whole
# 8 N tiles, K block 8, 1x4 subblock, 3-block L1 window
register_fused_mmrs_configs(
    {
        ttnn.CoreCoord(12, 10): {
            (m, 3072, 3072): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 6, 8, 8, 1, 4, None, 1, None, 3)
            for m in (8192, 8320, 8960)
        }
    }
)


# SP8 x TP4 MLP shapes (M, 3072, N) whose heuristic blocking overflows L1 on the 12x9 worker grid of a Galaxy
register_matmul_configs(
    {
        "12x9": {
            (2560, 3072, 4608): (4, 8, 16, (1, 4)),
            # per device at 1024x1024 / 2048x2048 with a 4 MP reference: 512 / 2048 image + 2048 reference + 128-512
            # text rows
            **{
                (m, 3072, n): (4, 8, n_block, (1, 4))
                for m in (2688, 2816, 4224, 4352)
                for n, n_block in ((2304, 8), (4608, 16))
            },
            (3072, 3072, 4608): (4, 8, 16, (1, 4)),
            # per device at 4096x2304 / 2304x4096: 4608 image + 0 / 128 / 256 / 512 text + 0 / 512 / 2048 reference rows
            **{
                (m, 3072, n): (4, 8, n_block, (1, 4))
                for m in (4608, 4736, 4864, 5120, 5248, 5376, 5632, 6784, 6912, 7168)
                for n, n_block in ((2304, 8), (4608, 16))
            },
            (8192, 3072, 2304): (11, 8, 8, (1, 4)),
            (8192, 3072, 4608): (6, 8, 16, (1, 4)),
            (8704, 3072, 2304): (8, 8, 8, (1, 4)),
            (8704, 3072, 4608): (6, 8, 16, (1, 4)),
            (9216, 3072, 2304): (8, 8, 8, (1, 4)),
            (9216, 3072, 4608): (6, 8, 16, (1, 4)),
            # per device at 4096x4096 with a 4 MP reference: 8192 image + 2048 reference + 512 text rows
            (10752, 3072, 2304): (8, 8, 8, (1, 4)),
            (10752, 3072, 4608): (6, 8, 16, (1, 4)),
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
    [image | image_cond valid | text | image_cond pad | pad] with all alignment rows at the end, reached with one
    all-gather of the per-device concat [img_d | cond_d | txt_d] and a row gather. Text rows past text_valid and
    reference rows past cond_valid (a reference bucket's padding) are masked as attention keys: every masked row sits
    after every attended one.
    """

    sp_factor: int
    lengths: dict[str, int]
    text_valid: int
    cond_valid: int | None = None
    padded: dict[str, int] = field(init=False)
    rows: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.cond_valid is None:
            object.__setattr__(self, "cond_valid", self.lengths["image_cond"])
        assert 0 <= self.cond_valid <= self.lengths["image_cond"]
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
        masked = {
            "text": self.lengths["text"] - self.text_valid,
            "image_cond": self.lengths["image_cond"] - self.cond_valid,
        }
        if mode is None:
            return self.total - sum(masked.values())
        return self.lengths[mode] - masked.get(mode, 0)

    def single_ranges(self) -> list[tuple[str, int, int]]:
        """(mode, start, stop) of the real single-stage rows in order: the attended rows first, then the masked ones."""
        ranges = [(m, 0, self.cond_valid if m == "image_cond" else self.lengths[m]) for m in self.modes]
        ranges.append(("image_cond", self.cond_valid, self.lengths["image_cond"]))
        return [r for r in ranges if r[2] > r[1]]

    def _source_row(self, mode: str, i: int | torch.Tensor) -> int | torch.Tensor:
        device, local = i // self.rows[mode], i % self.rows[mode]
        offset = sum(self.rows[m] for m in MODES[: MODES.index(mode)])
        return device * sum(self.rows.values()) + offset + local

    def assembly_indices(self) -> torch.Tensor:
        indices = [self._source_row(m, torch.arange(start, stop)) for m, start, stop in self.single_ranges()]
        if self.total_padded > self.total:
            # pad rows copy an alignment row of the early stacks, which is exactly zero
            m = next(m for m in self.modes if self.padded[m] > self.lengths[m])
            indices.append(torch.full((self.total_padded - self.total,), self._source_row(m, self.lengths[m])))
        return torch.cat(indices)

    def image_out_indices(self) -> torch.Tensor:
        length = self.lengths["image"]
        return torch.tensor(list(range(length)) + [0] * (self.padded["image"] - length), dtype=torch.int64)

    def mode_ids(self, mode: str | None = None) -> torch.Tensor:
        if mode is None:
            ids = [torch.full((stop - start,), MODES.index(m)) for m, start, stop in self.single_ranges()]
            return torch.cat([*ids, torch.full((self.total_padded - self.total,), PAD_MODE_ID)])
        pad = self.padded[mode] - self.lengths[mode]
        return torch.tensor([MODES.index(mode)] * self.lengths[mode] + [PAD_MODE_ID] * pad, dtype=torch.int64)


def ids_digest(ids: torch.Tensor) -> tuple:
    """Shape and content key of token ids."""
    return tuple(ids.shape), hashlib.sha1(ids.contiguous().numpy().tobytes()).hexdigest()


def rope_rows(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference rope_embed (float64 angles, fp32 entries) as interleaved cos / sin [rows, 128]."""
    pos = ids.to(torch.float64)
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(ROPE_AXES_DIM):
        scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        angle = torch.einsum("...n,d->...nd", pos[:, axis], 1.0 / (ROPE_THETA**scale))
        cos_parts.append(torch.cos(angle))
        sin_parts.append(torch.sin(angle))
    cos = torch.cat(cos_parts, dim=-1).float().repeat_interleave(2, dim=-1)
    sin = torch.cat(sin_parts, dim=-1).float().repeat_interleave(2, dim=-1)
    return cos, sin


def rope_cos_sin(ids: torch.Tensor, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    """rope_rows as [1, 1, padded, 128]; pad rows get the identity rotation."""
    return pad_rope(*rope_rows(ids), padded)


def pad_rope(cos: torch.Tensor, sin: torch.Tensor, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    pad = padded - cos.shape[0]
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
        self,
        *,
        flow_map: bool,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        precision: Flux3KleinPrecision,
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
            dtype=precision.weight_dtype,
            bias_dtype=ttnn.bfloat16,
            mesh_device=device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            chunks=MOD_CHUNKS,
            compute_kernel_config=precision.matmul_config(device.arch()),
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
        self,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool,
        precision: Flux3KleinPrecision,
    ) -> None:
        super().__init__()
        self._ring = ccl_manager.topology == ttnn.Topology.Ring
        for mode in MODES:
            setattr(
                self,
                mode,
                ModuleList(
                    Flux3KleinTransformerBlock(
                        device=device,
                        ccl_manager=ccl_manager,
                        parallel_config=parallel_config,
                        is_fsdp=is_fsdp,
                        precision=precision,
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
class Flux3KleinPlanParts:
    """The plan constants that follow the request: the reference's RoPE, the single stage's RoPE, modulation rows and
    row gather, and the attended lengths (device tensors on a ring: ring SDPA compiles once per bucket, not per prompt
    or reference size). Built as host tensors (plan_parts) off the request path, then uploaded (upload_plan_parts)."""

    layout: Flux3KleinSequenceLayout
    cond_rope: tuple[ttnn.Tensor, ...] = ()
    single_rope: tuple[ttnn.Tensor, ...] = ()
    single_mod: tuple[ttnn.Tensor, ...] = ()
    assembly: ttnn.Tensor | None = None
    early_logical_n: dict[str, int | ttnn.Tensor] = field(default_factory=dict)
    single_logical_n: int | ttnn.Tensor = 0

    def tensors(self) -> list[ttnn.Tensor]:
        lengths = (n for n in (*self.early_logical_n.values(), self.single_logical_n) if isinstance(n, ttnn.Tensor))
        return [*self.cond_rope, *self.single_rope, *self.single_mod, self.assembly, *lengths]

    def deallocate(self) -> None:
        for t in self.tensors():
            ttnn.deallocate(t)


@dataclass
class Flux3KleinPlan:
    """Device-side constants of one set of mode lengths and image / text ids: the image and text RoPE, the early
    modulation rows and the image output gather. ``parts`` holds the request's (Flux3KleinPlanParts)."""

    layout: Flux3KleinSequenceLayout
    early_rope: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]
    early_mod: dict[str, tuple[ttnn.Tensor, ...]]
    image_out: ttnn.Tensor
    parts: Flux3KleinPlanParts | None = None

    def deallocate(self) -> None:
        """The plan's own tensors; its parts belong to the caller's parts cache."""
        for t in (
            *(t for pair in self.early_rope.values() for t in pair),
            *(t for idx in self.early_mod.values() for t in idx),
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
        precision: Flux3KleinPrecision = PRECISION_BF16,
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
        self._rope_rows_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._rope_rows_lock = threading.Lock()

        def embedding(mode: str) -> ColParallelLinear:
            return ColParallelLinear(
                IN_CHANNELS[mode], HIDDEN, bias=False, mesh_device=device, mesh_axis=tp_axis, ccl_manager=ccl_manager
            )

        self.txt_in = embedding("text")
        self.emb_in_image = embedding("image")
        self.emb_in_image_cond = embedding("image_cond")
        self.modulation = Flux3KleinModulation(
            flow_map=flow_map,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            precision=precision,
        )
        self.early = Flux3KleinEarlyStacks(
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            precision=precision,
        )
        self.single_blocks = ModuleList(
            Flux3KleinTransformerBlock(
                device=device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                precision=precision,
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

    def _rope(self, cos: torch.Tensor, sin: torch.Tensor, padded: int, *, on_host: bool = False) -> tuple:
        return tuple(
            tensor.from_torch(
                t, device=self._device, dtype=ttnn.float32, mesh_axes=[None, None, self._sp_axis, None], on_host=on_host
            )
            for t in pad_rope(cos, sin, padded)
        )

    def _indices(self, idx: torch.Tensor, *, on_host: bool = False) -> ttnn.Tensor:
        return tensor.from_torch(
            idx.to(torch.int32)[None],
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_axes=[None, self._sp_axis],
            on_host=on_host,
        )

    def _logical_n(self, n: int, *, on_host: bool = False) -> int | ttnn.Tensor:
        if self.sp_factor == 1:
            return n
        return tensor.from_torch(
            torch.tensor([n], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self._device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None, None, None, None],
            on_host=on_host,
        )

    def _rope_rows(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # the image and text ids repeat across requests: their host rows are kept (a few shapes and text buckets)
        key = ids_digest(ids)
        with self._rope_rows_lock:
            rows = self._rope_rows_cache.pop(key, None)
        rows = rows or rope_rows(ids)
        with self._rope_rows_lock:
            self._rope_rows_cache[key] = rows
            while len(self._rope_rows_cache) > ROPE_ROWS_CACHE_SIZE:
                self._rope_rows_cache.pop(next(iter(self._rope_rows_cache)))
        return rows

    def prepare_plan(self, layout: Flux3KleinSequenceLayout, ids: dict[str, torch.Tensor]) -> Flux3KleinPlan:
        """The constants of ``layout``'s mode lengths and ``ids``' image / text ids; set ``parts`` before forward."""
        modes = layout.modes
        return Flux3KleinPlan(
            layout=layout,
            early_rope={
                m: self._rope(*self._rope_rows(ids[m][: layout.lengths[m]]), layout.padded[m])
                for m in modes
                if m != "image_cond"
            },
            early_mod={m: tuple(map(self._indices, modulation_indices(layout.mode_ids(m), "early"))) for m in modes},
            image_out=self._indices(layout.image_out_indices()),
        )

    def plan_parts(self, layout: Flux3KleinSequenceLayout, ids: dict[str, torch.Tensor]) -> Flux3KleinPlanParts:
        """The request's plan parts as host tensors: no device access, so it runs on a helper thread while the
        encoders run (vectorised; the image and text RoPE rows come from the host cache)."""
        rows = {m: self._rope_rows(ids[m][: layout.lengths[m]]) for m in layout.modes if m != "image_cond"}
        parts = Flux3KleinPlanParts(layout=layout)
        if "image_cond" in layout.modes:
            rows["image_cond"] = rope_rows(ids["image_cond"][: layout.lengths["image_cond"]])
            parts.cond_rope = self._rope(*rows["image_cond"], layout.padded["image_cond"], on_host=True)
        single = (torch.cat([rows[m][k][start:stop] for m, start, stop in layout.single_ranges()]) for k in (0, 1))
        parts.single_rope = self._rope(*single, layout.total_padded, on_host=True)
        parts.single_mod = tuple(
            self._indices(i, on_host=True) for i in modulation_indices(layout.mode_ids(), "single")
        )
        parts.assembly = self._indices(layout.assembly_indices(), on_host=True)
        parts.early_logical_n = {
            m: self._logical_n(layout.attended(m), on_host=True) for m in layout.modes if m in VARIABLE_MODES
        }
        parts.single_logical_n = self._logical_n(layout.attended(), on_host=True)
        return parts

    def upload_plan_parts(self, parts: Flux3KleinPlanParts) -> Flux3KleinPlanParts:
        def up(t):
            if not isinstance(t, ttnn.Tensor):
                return t
            return ttnn.to_device(t, self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return Flux3KleinPlanParts(
            layout=parts.layout,
            cond_rope=tuple(map(up, parts.cond_rope)),
            single_rope=tuple(map(up, parts.single_rope)),
            single_mod=tuple(map(up, parts.single_mod)),
            assembly=up(parts.assembly),
            early_logical_n={m: up(n) for m, n in parts.early_logical_n.items()},
            single_logical_n=up(parts.single_logical_n),
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
        parts = plan.parts
        layout = parts.layout
        table, (final_shift, final_scale_plus_one) = self.modulation(t_embed, t_ref_embed)

        x = {"image": self.emb_in_image(image_tokens, parallel_config=self._parallel_config), "text": text_embedded}
        if image_cond_embedded is not None:
            x["image_cond"] = image_cond_embedded
        for mode in layout.modes:
            mod = tuple(gather_rows(table, idx) for idx in plan.early_mod[mode])
            # the image stack's length is fixed per size; the text and reference stacks' follow the request
            length = parts.early_logical_n.get(mode, layout.attended(mode))
            rope = parts.cond_rope if mode == "image_cond" else plan.early_rope[mode]
            x[mode] = self.early(mode, x[mode], rope=rope, mod=mod, logical_length=length)

        stacks = [x[m] for m in layout.modes]
        local = stacks[0] if len(stacks) == 1 else ttnn.concat(stacks, dim=1)
        h = gather_rows_over_sp(local, parts.assembly, ccl_manager=self._ccl_manager, sp_axis=self._sp_axis)
        for stack in stacks:
            ttnn.deallocate(stack)

        mod = tuple(gather_rows(table, idx) for idx in parts.single_mod)

        def run_block(block: Flux3KleinTransformerBlock, h: ttnn.Tensor) -> ttnn.Tensor:
            return block(h, rope=parts.single_rope, mod=mod, logical_length=parts.single_logical_n)

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
