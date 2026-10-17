# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein early (per-mode) stacks and the single-stage assembly on a tt_dit mesh.

``Flux3KleinEarlyStacks`` holds the reference's ``content_mode_blocks.image``, ``content_mode_blocks.image_cond``
and ``txt_mode_blocks`` (``reference/flux3_klein/dit.py:172-176, 211-214``) as three ``ModuleList``s of
``Flux3KleinTransformerBlock`` named ``image``, ``image_cond``, ``text``. Each mode runs on its own SP-sharded
tensor with private attention (the reference's per-mode ``bounds``); nothing mixes modes before the single stage.

Tensor contract of :meth:`Flux3KleinEarlyStacks.forward` (``mesh_axes`` as ``utils.tensor.from_torch`` takes them):

* ``x``: ``[1, L_m_pad, 3072]`` bf16, ``mesh_axes=[None, sp, tp]``, ``L_m_pad = packing.padded_length(L_m, SP)``,
  rows ``[L_m, L_m_pad)`` zero. Output: same shape and distribution; its pad rows are exactly zero.
* ``mod = (shift, scale_plus_one, gate)``: PER TOKEN ``[1, L_m_pad, 3072]`` bf16, ``mesh_axes=[None, sp, tp]``: the
  mode's early modulation row on the real rows and ``(0, 1, 0)`` on the pad rows. A zero pad row then stays
  exactly zero through every block (``LN(0) * 1 + 0 = 0`` -> finite q/k/v, ``gate = 0`` -> the residual adds 0),
  which the assembly below relies on (``docs/M2_M3_DESIGN.md`` decision 4; measured: tails exactly 0 after each
  block and real rows bit-equal to the broadcast-modulation run). Broadcast modulation is NOT accepted here: the
  block's docstring explains why its pad rows would drift.
* ``rope = (cos, sin)``: fp32 ``[1, 1, L_m_pad, 128]``, ``mesh_axes=[None, None, sp, None]``, pad rows cos 1 / sin 0
  (the block's contract).
* ``logical_length = L_m``: the real row count, forwarded to every block (ring SDPA key mask). For text this is the
  encoder's 80-multiple: all of those rows are real tokens.
* The returned tensor is the CALLER's (deallocate or keep it freely). A single block's direct output on Ring is NOT:
  it aliases a CCL ping-pong buffer that the next same-shape block call reuses (see ``forward``).

An EMPTY mode is not run at all: the reference gives it one zero phantom token (``dit.py:192-198``) that is never
attended by any real row and never enters the single stage, so the real rows do not depend on it (decision 5).

Single-stage assembly (decision 2; ``layout_flux3_klein.py`` holds the index arithmetic): ``assemble_single_stage``
concatenates the non-empty modes' per-device shards (``ttnn.concat`` on dim 1, tile-aligned rows), all-gathers the
local concat over SP into the ring-ordered buffer ``P`` (replicated over SP, ``[1, L_pad, 3072 / TP]``), converts it
to ROW_MAJOR and lets every device gather ITS rows of the global order ``G = [image | image_cond | text | pad]``
with ``ttnn.embedding`` from a constant uint32 index tensor (``indices_to_device``). ``ttnn.embedding`` needs a
ROW_MAJOR bf16 table, uint32 indices and, for a tilized output, a multiple of 32 indices per device
(``ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:33-52``); every per-device row count
here is a tile multiple by construction of ``padded_length``. The CCLManager's all-gather reshapes rank 3 to 4
and back itself (``parallel/manager.py:884-929``). The same primitive (``gather_rows_over_sp``) maps the final
layer's ``[1, R, 128]`` output back into the image stack's own SP layout with ``layout.image_out_indices()``.
Measured exact (``torch.equal``) on the (2,2) submesh and the (4,8) ring, and a block on the assembled tensor
matches torch at the M1 floor (``tests/models/flux3_klein/test_stacks_flux3_klein.py``, the S3 assembly row).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ....layers.module import Module, ModuleList
from ....utils import tensor
from ....utils.substate import rename_substate
from .transformer_block_flux3_klein import Flux3KleinTransformerBlock

if TYPE_CHECKING:
    from ....parallel.config import DiTParallelConfig
    from ....parallel.manager import CCLManager

FLUX3_KLEIN_EARLY_DEPTH = 5  # dit.py:21 DEPTH

# Reference state prefix -> child name (dit.py:172-176); MODES order.
FLUX3_KLEIN_EARLY_STACK_PREFIXES = (
    ("content_mode_blocks.image", "image"),
    ("content_mode_blocks.image_cond", "image_cond"),
    ("txt_mode_blocks", "text"),
)


class Flux3KleinEarlyStacks(Module):
    """The three per-mode early stacks; see the module docstring for the tensor contract."""

    def __init__(
        self,
        *,
        depth: int = FLUX3_KLEIN_EARLY_DEPTH,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self._ring = ccl_manager.topology == ttnn.Topology.Ring
        for _, name in FLUX3_KLEIN_EARLY_STACK_PREFIXES:
            setattr(
                self,
                name,
                ModuleList(
                    Flux3KleinTransformerBlock(
                        device=device, ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=is_fsdp
                    )
                    for _ in range(depth)
                ),
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """``content_mode_blocks.image.{i}.* -> image.{i}.*``, ``content_mode_blocks.image_cond.{i}.* -> image_cond.{i}.*``,
        ``txt_mode_blocks.{i}.* -> text.{i}.*``. Only keys that are present move (a strict load reports the rest);
        each block's own ``_prepare_torch_state`` then fuses its 8 reference keys."""
        for src, dst in FLUX3_KLEIN_EARLY_STACK_PREFIXES:
            rename_substate(state, src, dst)

    def stack(self, mode: str) -> ModuleList:
        """The ``ModuleList`` of blocks of ``mode`` (``image``, ``image_cond`` or ``text``)."""
        if mode not in ("image", "image_cond", "text"):
            raise ValueError(f"unknown mode {mode!r}")
        return getattr(self, mode)

    def forward(
        self,
        mode: str,
        x: ttnn.Tensor,
        *,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        mod: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
        logical_length: int,
    ) -> ttnn.Tensor:
        """Run ``mode``'s blocks in order on ``x`` ``[1, L_m_pad / SP, 3072 / TP]`` (per device) with the per-token
        ``mod`` and ``rope`` of that mode; ``logical_length = L_m``. Output: the input's shape and distribution."""
        for block in self.stack(mode):
            x = block(x, rope=rope, mod=mod, logical_length=logical_length)
        if self._ring and len(self.stack(mode)) > 0:
            # On Ring the block's fused matmul + reduce-scatter + addcmul returns one of the CCLManager's TWO
            # alternating reduce-scatter ping-pong buffers for that shard shape (layers/linear.py
            # forward_fused_addcmul -> parallel/manager.py get_rs_ping_pong_buffer), which the next block call of the
            # same shape overwrites. Two modes of equal padded length (image 4096 + one 1024x1024 reference, or
            # cond 80 + text 80) would clobber each other's early output before the assembly, so the stack hands
            # back a private copy. (Linear returns a fresh addcmul output.) Measured on the 4x8 ring: with the direct
            # output deallocated after its download, the third same-shape block call failed with TT_FATAL "Input
            # Tensor is not allocated" (the buffer alternates between two), and a RETAINED direct output was overwritten
            # after two more same-shape block calls (same buffer address) while the clone stayed intact.
            x = ttnn.clone(x)
        return x


# --- single-stage assembly ---------------------------------------------------------------------------------------


def indices_to_device(indices: torch.Tensor, *, device: ttnn.MeshDevice, sp_axis: int) -> ttnn.Tensor:
    """``[N]`` integer row indices -> ``[1, N]`` uint32 ROW_MAJOR, ``mesh_axes=[None, sp_axis]``: device ``d`` holds its
    own ``N / SP`` values (its rows of the gathered output). ``N`` must be a multiple of ``32 * SP`` so each device
    gathers a tile-aligned row count."""
    sp = tuple(device.shape)[sp_axis]
    n = indices.shape[-1]
    if indices.dim() != 1 or n % (ttnn.TILE_SIZE * sp) != 0:
        raise ValueError(
            f"indices must be a [N] vector with N % {ttnn.TILE_SIZE * sp} == 0, got {tuple(indices.shape)}"
        )
    return tensor.from_torch(
        indices.to(torch.int32)[None],
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_axes=[None, sp_axis],
    )


def gather_rows_over_sp(x: ttnn.Tensor, indices: ttnn.Tensor, *, ccl_manager: CCLManager, sp_axis: int) -> ttnn.Tensor:
    """``x`` ``[1, R_in, C]`` per device (SP-sharded rows, bf16 TILE; ``C`` TP-fractured such as ``3072 / TP`` or
    replicated such as 128) -> ``[1, R_out, C]`` per device whose row ``r`` is row ``indices_d[r]`` of the SP
    all-gather of ``x`` (``[1, R_in * SP, C]`` in device order). ``indices`` comes from :func:`indices_to_device`
    (``R_out`` values per device, a multiple of 32). An exact copy; ``C`` keeps its distribution."""
    gathered = ccl_manager.all_gather(x, dim=1, mesh_axis=sp_axis, use_hyperparams=False, use_persistent_buffer=True)
    gathered = ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.embedding(indices, gathered, layout=ttnn.TILE_LAYOUT)
    return ttnn.reshape(out, (1, out.shape[-2], out.shape[-1]))


def assemble_single_stage(
    parts: list[ttnn.Tensor], indices: ttnn.Tensor, *, ccl_manager: CCLManager, sp_axis: int
) -> ttnn.Tensor:
    """The single-stage sequence ``G = [image | image_cond | text | pad]`` ``[1, L_pad, 3072]`` (``mesh_axes
    [None, sp, tp]``) from ``parts``, the non-empty modes' early-stack outputs in ``MODES`` order, and
    ``indices = indices_to_device(layout.assembly_indices())``. Every pad row of every part must be exactly zero:
    the pad rows of ``G`` are copies of one of them."""
    if not parts:
        raise ValueError("assemble_single_stage needs at least the image part")
    local = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1)
    return gather_rows_over_sp(local, indices, ccl_manager=ccl_manager, sp_axis=sp_axis)
