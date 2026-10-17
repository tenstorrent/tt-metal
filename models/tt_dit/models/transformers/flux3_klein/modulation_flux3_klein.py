# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein modulation heads: the time path, the merged modulation linear and the per-token table.

Reference (``reference/flux3_klein/dit.py``): ``vec[m] = time_in(timestep_embedding(t_m))`` with the image rows at
the step's timestep and the text / image_cond rows at 0 (``:199-200``), ``+ ref_time_in(timestep_embedding(t_next))``
on every row for the flow-map (distilled) model (``:201-206``); then six ``ModeModulation`` heads (``:124``, chunk
order shift, scale, gate) and the final layer's ``adaLN_modulation`` (``:134``, shift, scale), all ``Linear(silu(vec))``
without bias; the consumers apply ``(1 + scale) * LN(x) + shift`` and ``x + gate * (...)``.

Device form (design record ``docs/M2_M3_DESIGN.md`` decisions 6 and 7):

* Time path. The host supplies ``embedding_t = bf16(1000 * bf16(t))`` exactly as the reference computes it in the
  bf16 timestep rows (``pipelines/flux3_klein/scheduler.py`` ``Flux3KleinStep.embedding_t`` / ``embedding_t_next``),
  as an fp32 ``[3, 1]`` column ``[[embedding_t], [0], [0]]`` (:func:`timestep_inputs`). ``Timesteps`` (fp32, cos first,
  256 channels, max period 10000) computes the sinusoid, ``ttnn.typecast`` to bf16 reproduces the reference's
  ``.to(t)`` rounding bit for bit (measured on every schedule point of both models), a second typecast returns it to
  fp32 and ``TimestepEmbedding`` (fp32 operands, HiFi4, no bias) is ``MLPEmbedder``. ``vec`` is fp32 ``[1, 3, 3072]``:
  row 0 the image row at ``t``, row 1 the text / image_cond row at 0, row 2 a third row whose ACTIVATION is masked
  to exactly zero below (so every head emits its bias only).
* Merged heads. The seven reference heads are stacked into one ``ColParallelLinear(3072, 20 * 3072, chunks=20)``
  in :data:`FLUX3_KLEIN_MOD_SOURCES` order with a synthesized bias of 1.0 on every scale chunk
  (:data:`FLUX3_KLEIN_MOD_SCALE_CHUNKS`), the Flux2 convention (``transformer_flux2.py`` ``time_mod``). The input is
  ``typecast(silu(vec) * [[1], [1], [0]], bf16)``; with fp32 accumulation a zero input row yields the bias exactly,
  so the third row of every chunk is 0 (shift, gate) or 1 (scale): the modulation of a PAD row.
* Table. The 20 chunks ``[1, 3, 3072 / TP]`` bf16 become one ROW_MAJOR table ``[60, 3072 / TP]`` with row
  ``3 * chunk + vec_row`` (:meth:`Flux3KleinModulation.table`), replicated over SP and fractured over TP exactly like
  the activations' last dimension. Per-token ``(shift, scale_plus_one, gate)`` for the early stacks and the single
  stage are ``ttnn.embedding`` gathers from it (:func:`gather_modulation`) with constant per-shape uint32 index
  tensors built by :func:`modulation_table_indices` from a per-row mode id (0 image, 1 image_cond, 2 text, 3 pad);
  the final layer's broadcast rows are ``ttnn.slice`` of chunks 18 / 19 at row 0 (:meth:`final_rows`).

Measured (base ``time_in`` and heads, the 2x2 submesh and the 4x8 ring agree): device bf16 sinusoid == reference on
all 100 + 32 schedule points; ``vec`` vs the reference bf16 ``vec`` pcc 0.999997 / rel_rmse 0.0032; the 40 real
table rows vs the torch heads 0.999998 / 0.0021 (base) and 0.999997 / 0.0026 (distilled); pad rows exactly 0 / 1 / 0;
every gather an exact copy of the table rows. The test file ``tests/models/flux3_klein/test_modulation_flux3_klein.py``
carries the gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ....layers.embeddings import TimestepEmbedding, Timesteps
from ....layers.linear import ColParallelLinear, prepare_chunked_linear_output
from ....layers.module import Module
from ....utils import tensor

if TYPE_CHECKING:
    from ....parallel.config import DiTParallelConfig
    from ....parallel.manager import CCLManager
    from ....pipelines.flux3_klein.scheduler import Flux3KleinStep

# dit.py:13 and :42-47, :183-184.
FLUX3_KLEIN_HIDDEN = 3072
FLUX3_KLEIN_TIME_CHANNELS = 256
FLUX3_KLEIN_TIME_MAX_PERIOD = 10000

# (reference key prefix as seen inside THIS module's substate, chunk count, offsets of the scale chunks within the
# head). Row order is the chunk order of the merged weight (dit.py:124 shift, scale, gate; :134 shift, scale).
FLUX3_KLEIN_MOD_SOURCES = (
    ("early_stream_modulations.image.lin", 3, (1,)),
    ("early_stream_modulations.image_cond.lin", 3, (1,)),
    ("early_stream_modulations.txt.lin", 3, (1,)),
    ("single_stream_modulations.image.lin", 3, (1,)),
    ("single_stream_modulations.image_cond.lin", 3, (1,)),
    ("single_stream_modulations.txt.lin", 3, (1,)),
    ("final_layer.image.adaLN_modulation.1", 2, (1,)),
)
FLUX3_KLEIN_MOD_CHUNKS = 20
# The chunks that carry the consumers' `+1` in the synthesized bias.
FLUX3_KLEIN_MOD_SCALE_CHUNKS = (1, 4, 7, 10, 13, 16, 19)
# vec rows: 0 image at t, 1 image_cond / text at 0, 2 the masked zero-activation row (the pad rows' modulation).
FLUX3_KLEIN_MOD_VEC_ROWS = 3
FLUX3_KLEIN_MOD_TABLE_ROWS = FLUX3_KLEIN_MOD_CHUNKS * FLUX3_KLEIN_MOD_VEC_ROWS  # 60; table row = 3 * chunk + vec_row
# Mode ids of modulation_table_indices: 0 image, 1 image_cond, 2 text, 3 pad row (equals
# layout_flux3_klein.FLUX3_KLEIN_PAD_MODE_ID; kept separate because the two files are independent).
FLUX3_KLEIN_MOD_PAD_MODE_ID = 3
FLUX3_KLEIN_MOD_FINAL_SHIFT_CHUNK = 18
FLUX3_KLEIN_MOD_FINAL_SCALE_CHUNK = 19

_STAGE_OFFSET = {"early": 0, "single": 9}
_MODE_OFFSET = {"image": 0, "image_cond": 3, "text": 6}
_PARAM_OFFSET = {"shift": 0, "scale": 1, "gate": 2}
_MODE_NAMES = ("image", "image_cond", "text")
# vec row per mode id (image, image_cond, text, pad).
_VEC_ROW_OF_MODE_ID = (0, 1, 1, 2)

assert sum(count for _, count, _ in FLUX3_KLEIN_MOD_SOURCES) == FLUX3_KLEIN_MOD_CHUNKS
assert (
    tuple(
        offset + i
        for offset, (_, _, scale_offsets) in zip(
            [sum(c for _, c, _ in FLUX3_KLEIN_MOD_SOURCES[:h]) for h in range(len(FLUX3_KLEIN_MOD_SOURCES))],
            FLUX3_KLEIN_MOD_SOURCES,
        )
        for i in scale_offsets
    )
    == FLUX3_KLEIN_MOD_SCALE_CHUNKS
)


def mod_chunk(stage: str, mode: str, param: str) -> int:
    """Chunk of the merged modulation linear holding ``param`` of ``mode``'s head of ``stage``: early.image 0, 1, 2;
    early.image_cond 3, 4, 5; early.text 6, 7, 8; single.image 9 ...; single.text 15, 16, 17 (final shift 18 and
    scale 19 are :data:`FLUX3_KLEIN_MOD_FINAL_SHIFT_CHUNK` / ``_SCALE_CHUNK``). ``text`` maps to the reference's
    ``txt`` head."""
    if stage not in _STAGE_OFFSET:
        raise ValueError(f"stage must be one of {tuple(_STAGE_OFFSET)}, got {stage!r}")
    if mode not in _MODE_OFFSET:
        raise ValueError(f"mode must be one of {tuple(_MODE_OFFSET)}, got {mode!r}")
    if param not in _PARAM_OFFSET:
        raise ValueError(f"param must be one of {tuple(_PARAM_OFFSET)}, got {param!r}")
    return _STAGE_OFFSET[stage] + _MODE_OFFSET[mode] + _PARAM_OFFSET[param]


def modulation_table_indices(
    mode_ids: torch.Tensor, stage: str
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Table rows of ``(shift, scale_plus_one, gate)`` for every token of a sequence given its per-row mode id
    (``[N]`` in {0 image, 1 image_cond, 2 text, 3 pad}): ``3 * mod_chunk(stage, mode, param) + vec_row`` with
    ``vec_row`` 0 for image, 1 for image_cond / text and 2 (the bias-only row) for a pad row, which reads the image
    head's chunk of ``stage`` so it lands on 0 / 1 / 0."""
    if stage not in _STAGE_OFFSET:
        raise ValueError(f"stage must be one of {tuple(_STAGE_OFFSET)}, got {stage!r}")
    mode_ids = torch.as_tensor(mode_ids).to(torch.long).reshape(-1)
    if mode_ids.numel() and (mode_ids.min() < 0 or mode_ids.max() > FLUX3_KLEIN_MOD_PAD_MODE_ID):
        raise ValueError(f"mode ids must lie in 0..{FLUX3_KLEIN_MOD_PAD_MODE_ID}")
    vec_row = torch.tensor(_VEC_ROW_OF_MODE_ID, dtype=torch.long)[mode_ids]
    out = []
    for param in ("shift", "scale", "gate"):
        chunk_of_mode = torch.tensor(
            [mod_chunk(stage, m, param) for m in _MODE_NAMES] + [mod_chunk(stage, "image", param)], dtype=torch.long
        )
        out.append(FLUX3_KLEIN_MOD_VEC_ROWS * chunk_of_mode[mode_ids] + vec_row)
    return tuple(out)


def timestep_inputs(step: Flux3KleinStep, flow_map: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Host columns for :meth:`Flux3KleinModulation.vec`: fp32 ``[3, 1]`` ``[[embedding_t], [0], [0]]`` (image row
    at the step, text / image_cond row at 0, the pad row) and, for the flow map, fp32 ``[1, 1]``
    ``[[embedding_t_next]]`` (``dit.py:202``: the target rounded to the bf16 timestep dtype, then ``1000 * t``)."""
    t_embed = torch.tensor([[step.embedding_t], [0.0], [0.0]], dtype=torch.float32)
    t_ref = torch.tensor([[step.embedding_t_next]], dtype=torch.float32) if flow_map else None
    return t_embed, t_ref


def timesteps_to_device(
    t_embed: torch.Tensor, t_ref_embed: torch.Tensor | None, *, device: ttnn.MeshDevice
) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
    """Upload :func:`timestep_inputs` as replicated fp32 tensors."""
    upload = lambda t: tensor.from_torch(t, device=device, dtype=ttnn.float32, mesh_axes=[None, None])  # noqa: E731
    return upload(t_embed), (upload(t_ref_embed) if t_ref_embed is not None else None)


class Flux3KleinModulation(Module):
    """Time embedders + the merged modulation heads of one Klein DiT (see the module docstring).

    Parameters (after :meth:`_prepare_torch_state`): ``time_in.linear_1.weight``, ``time_in.linear_2.weight``,
    ``ref_time_in.*`` (flow map only), ``lin.weight`` ``[3072, 61440]``-shaped on device (TP-fractured, chunk-major per
    device) and ``lin.bias``. The reference keys of this part load strictly: ``time_in.in_layer.weight``,
    ``time_in.out_layer.weight``, ``ref_time_in.in_layer.weight``, ``ref_time_in.out_layer.weight`` and the seven
    ``<prefix>.weight`` of :data:`FLUX3_KLEIN_MOD_SOURCES`.
    """

    def __init__(
        self,
        *,
        flow_map: bool,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.flow_map = flow_map
        self.device = device
        self.parallel_config = parallel_config
        self.tp_axis = tp_axis
        self.tp_factor = parallel_config.tensor_parallel.factor
        self.hidden_local = FLUX3_KLEIN_HIDDEN // self.tp_factor

        self.time_proj = Timesteps(
            num_channels=FLUX3_KLEIN_TIME_CHANNELS,
            cos_first=True,
            max_period=FLUX3_KLEIN_TIME_MAX_PERIOD,
            dtype=ttnn.float32,
            mesh_device=device,
        )
        self.time_in = TimestepEmbedding(
            FLUX3_KLEIN_TIME_CHANNELS,
            FLUX3_KLEIN_HIDDEN,
            act_fn="silu",
            dtype=ttnn.float32,
            bias=False,
            mesh_device=device,
        )
        self.ref_time_in = (
            TimestepEmbedding(
                FLUX3_KLEIN_TIME_CHANNELS,
                FLUX3_KLEIN_HIDDEN,
                act_fn="silu",
                dtype=ttnn.float32,
                bias=False,
                mesh_device=device,
            )
            if flow_map
            else None
        )
        self.lin = ColParallelLinear(
            FLUX3_KLEIN_HIDDEN,
            FLUX3_KLEIN_MOD_CHUNKS * FLUX3_KLEIN_HIDDEN,
            bias=True,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            chunks=FLUX3_KLEIN_MOD_CHUNKS,
        )
        # Row mask of the heads' activation (a device constant outside the parameters, so it is built lazily by
        # `row_mask` and freed with the weights through the eviction hook, the Module convention for such state).
        self._row_mask: ttnn.Tensor | None = None
        self.register_eviction_hook(self._release_row_mask)

    def row_mask(self) -> ttnn.Tensor:
        """Replicated fp32 ``[1, 3, 3072]``: rows 0 and 1 pass, row 2 is exactly zero so every chunk's third row is
        its bias alone (0 / 1 / 0). Full width rather than a broadcast column: 36 KB, and ``x * 1.0`` / ``x * 0.0``
        are exact in fp32 either way. Built on first use, released by :meth:`deallocate_weights`."""
        if self._row_mask is None:
            mask = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32).expand(
                FLUX3_KLEIN_MOD_VEC_ROWS, FLUX3_KLEIN_HIDDEN
            )
            self._row_mask = tensor.from_torch(
                mask.contiguous()[None], device=self.device, dtype=ttnn.float32, mesh_axes=[None, None, None]
            )
        return self._row_mask

    def _release_row_mask(self) -> None:
        if self._row_mask is not None:
            ttnn.deallocate(self._row_mask)
            self._row_mask = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # MLPEmbedder in_layer / out_layer -> TimestepEmbedding linear_1 / linear_2 (pop-and-rename when present; the
        # ref_time_in keys are renamed only when the module has that child, so a flow-map state fed to a base model is
        # reported under the reference's own names).
        embedders = ("time_in", "ref_time_in") if self.ref_time_in is not None else ("time_in",)
        for embedder in embedders:
            for src, dst in (("in_layer", "linear_1"), ("out_layer", "linear_2")):
                key = f"{embedder}.{src}.weight"
                if key in state:
                    state[f"{embedder}.{dst}.weight"] = state.pop(key)
        # The seven heads -> one chunked weight + the synthesized bias; never a partial merge (a missing head leaves
        # lin.weight / lin.bias absent so the strict loader reports them).
        weights = [state.pop(f"{prefix}.weight", None) for prefix, _, _ in FLUX3_KLEIN_MOD_SOURCES]
        if all(w is not None for w in weights):
            state["lin.weight"] = torch.cat(weights, dim=0)
            state["lin.bias"] = torch.cat(
                [
                    torch.full([FLUX3_KLEIN_HIDDEN], 1.0 if i in scale_offsets else 0.0)
                    for _, count, scale_offsets in FLUX3_KLEIN_MOD_SOURCES
                    for i in range(count)
                ]
            )
        prepare_chunked_linear_output(state, prefix="lin", device_count=self.tp_factor, chunks=FLUX3_KLEIN_MOD_CHUNKS)

    # --- time path --------------------------------------------------------------------------------------------------

    def sinusoid_bf16(self, t_embed: ttnn.Tensor) -> ttnn.Tensor:
        """``timestep_embedding`` of the reference as it leaves ``.to(t)``: bf16 ``[..., N, 256]`` for an fp32
        ``[N, 1]`` column of ``embedding_t`` values (cos first). Bit-equal to the reference on every schedule point."""
        return ttnn.typecast(self.time_proj(t_embed), ttnn.bfloat16)

    def sinusoid(self, t_embed: ttnn.Tensor) -> ttnn.Tensor:
        """The bf16 sinusoid widened back to fp32: the embedders' input."""
        return ttnn.typecast(self.sinusoid_bf16(t_embed), ttnn.float32)

    def vec(self, t_embed: ttnn.Tensor, t_ref_embed: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """``vec`` rows fp32 ``[1, N, 3072]`` for an fp32 ``[N, 1]`` replicated column of ``embedding_t`` values
        (``N = 3`` in the model: image at ``t``, 0, 0); the flow map adds ``ref_time_in`` of the fp32 ``[1, 1]``
        ``embedding_t_next`` to every row (``dit.py:201-206``). ``t_ref_embed`` is required by, and only by, a
        flow-map instance (``dit.py:189-190``)."""
        if (t_ref_embed is not None) != (self.ref_time_in is not None):
            raise ValueError("t_ref_embed is required by, and only by, a flow-map Flux3KleinModulation")
        v = self.time_in(self.sinusoid(t_embed))
        if self.ref_time_in is not None:
            v = ttnn.add(v, self.ref_time_in(self.sinusoid(t_ref_embed)))
        return ttnn.reshape(v, (1, v.shape[-2], FLUX3_KLEIN_HIDDEN))

    # --- heads ------------------------------------------------------------------------------------------------------

    def chunks(self, vec: ttnn.Tensor) -> list[ttnn.Tensor]:
        """The 20 head chunks ``[1, 3, 3072 / TP]`` bf16 TILE (TP-fractured) of ``vec`` ``[1, 3, 3072]`` fp32; the
        third row of every chunk is the bias alone."""
        assert tuple(vec.shape) == (1, FLUX3_KLEIN_MOD_VEC_ROWS, FLUX3_KLEIN_HIDDEN), tuple(vec.shape)
        act = ttnn.multiply(ttnn.silu(vec), self.row_mask())
        act = ttnn.typecast(act, ttnn.bfloat16)
        return self.lin(act, parallel_config=self.parallel_config, dtype=ttnn.bfloat16)

    def table(self, chunks: list[ttnn.Tensor]) -> ttnn.Tensor:
        """ROW_MAJOR bf16 ``[60, 3072 / TP]`` (row ``3 * chunk + vec_row``), replicated over SP, fractured over TP:
        the source of every :func:`gather_modulation`."""
        rows = [ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT) for c in chunks]
        t = ttnn.concat(rows, dim=1)
        return ttnn.reshape(t, (FLUX3_KLEIN_MOD_TABLE_ROWS, t.shape[-1]))

    def final_rows(self, chunks: list[ttnn.Tensor]) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(shift, scale_plus_one)`` of the final layer: row 0 (the image row) of chunks 18 and 19 as
        ``[1, 1, 3072 / TP]`` bf16 TILE (``mesh_axes=[None, None, tp]``), the dynamic bias / weight of the final
        ``DistributedLayerNorm``."""

        def row0(chunk: ttnn.Tensor) -> ttnn.Tensor:
            """Row 0 of a ``[1, 3, 3072 / TP]`` chunk as ``[1, 1, 3072 / TP]``."""
            c4 = ttnn.unsqueeze_to_4D(chunk)
            row = ttnn.slice(c4, [0, 0, 0, 0], [1, 1, 1, c4.shape[-1]])
            return ttnn.reshape(row, (1, 1, c4.shape[-1]))

        return row0(chunks[FLUX3_KLEIN_MOD_FINAL_SHIFT_CHUNK]), row0(chunks[FLUX3_KLEIN_MOD_FINAL_SCALE_CHUNK])

    def forward(
        self, t_embed: ttnn.Tensor, t_ref_embed: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]:
        """``(table, (final_shift, final_scale_plus_one))`` for one step (see :meth:`vec` for the inputs)."""
        chunks = self.chunks(self.vec(t_embed, t_ref_embed))
        return self.table(chunks), self.final_rows(chunks)


def gather_modulation(table: ttnn.Tensor, indices: ttnn.Tensor) -> ttnn.Tensor:
    """Per-token modulation rows ``[1, N, 3072 / TP]`` bf16 TILE from the ROW_MAJOR ``table`` and uint32 ROW_MAJOR
    ``indices`` ``[1, N]`` (per device its own ``N`` rows, ``N % 32 == 0`` for the tilized output). Uploaded as
    ``tensor.from_torch(idx.to(torch.int32)[None], layout=ROW_MAJOR, dtype=ttnn.uint32, mesh_axes=[None, sp_axis])``
    the result carries the activations' ``[None, sp, tp]`` distribution."""
    out = ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT)
    return ttnn.reshape(out, (1, out.shape[-2], out.shape[-1]))
