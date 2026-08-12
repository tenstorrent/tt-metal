# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-resident AdaLN modulation tables, precomputed on host for a fixed sampling schedule.

The AdaLN projections never reach the device. `adaln_proj` across 50 blocks
is 26 GB of the checkpoint (6.50 GB per device at TP=4); the table that replaces it is 1.416 GB
(0.354 GB per device). It is built on host by
`pipelines/minimax_h3/adaln_precompute.precompute_adaln_table`, which reads the safetensors directly
and never uploads the weights, so both the read and the residency disappear. Relative to on-device
projection it also saves one matmul per block per step (measured 0.58 ms, 3.4% of the 5s block).

**A table is valid for exactly one schedule.** Its rows are `(step, timestep level, modality)`
triples derived from both schedulers' sigmas, the per-modality shifts and the keyframe noise-aug
floor. Reusing one across a different step count or shift silently modulates every block slightly
wrong at every step, in the same direction -- which is why the pipeline keys its disk cache on all of
those and why `num_steps` is asserted against the caller's schedule here.

Row addressing needs no per-step slicing. `MiniMaxH3AdalnTable.adaln_indices(step, ...)` returns
*absolute* rows into the whole concatenated table, so a block gathers from the resident tensor with
exactly the `ttnn.embedding` call it already used for the per-step tables.

Two host-side conventions are baked in at build time rather than paid per block:

* **The `1 +` on the scales.** AdaLN applies `(1 + scale)`; folding it into the table costs nothing
  here and saves an elementwise op over the whole packed sequence per scale per block.
* **The TP hidden split.** Each device holds only its own `hidden_local` columns, so there is no
  reorder-for-TP dance at load time -- the shard is the natural column split of the last axis.
"""

from __future__ import annotations

import torch

import ttnn

from ....parallel.config import DiTParallelConfig
from ....utils.tensor import bf16_tensor
from .transformer_block_minimax_h3 import _SCALE_MLP, _SCALE_MSA, NUM_MODULATION_PARAMS

# The host-side table type lives in `pipelines/minimax_h3/adaln_precompute` and is *not* imported
# here: a model importing from pipelines inverts the layering and risks an import cycle. The
# table is consumed structurally (`block_params`, `final_shift`, `final_scale`, `step_offsets`,
# `num_layers`, `hidden_size`, `num_steps`), so any object with that surface works -- which is also
# what lets a test drive this with a hand-built table.

# The two parameters AdaLN applies as `(1 + scale)`, taken from the block itself so the two cannot
# drift apart.
_SCALE_PARAMS = (_SCALE_MSA, _SCALE_MLP)


class MiniMaxH3AdalnCache:
    """Device-resident modulation tables for every (step, modality, block) of one schedule.

    `block_tables(layer)` returns the six `[rows, hidden_local]` tables that block expects, in
    parameter order, ready for `ttnn.embedding`. `final_tables()` returns the `norm_out` pair.
    """

    def __init__(
        self,
        table,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        num_layers: int,
        hidden_size: int,
    ) -> None:
        if table.num_layers != num_layers:
            raise ValueError(f"table holds {table.num_layers} layers, model has {num_layers}")
        if table.hidden_size != hidden_size:
            raise ValueError(f"table hidden_size {table.hidden_size} != model {hidden_size}")

        tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        tp_factor = parallel_config.tensor_parallel.factor
        if hidden_size % tp_factor:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by TP factor {tp_factor}")

        self.num_steps = table.num_steps
        self.num_layers = num_layers
        self.hidden_local = hidden_size // tp_factor
        self._step_offsets = table.step_offsets

        # block_params is [layers, rows * MODALITY_NUM, params, hidden]. Split the parameter axis into
        # the six tables each block gathers from, folding the `1 +` into the scales, and shard the
        # hidden axis across TP. Row order (`row * MODALITY_NUM + modality`) is already what
        # `adaln_indices` addresses, so rows pass through untouched.
        self._block_tables: list[list[ttnn.Tensor]] = []
        for layer in range(num_layers):
            per_param = []
            for param in range(NUM_MODULATION_PARAMS):
                rows = table.block_params[layer, :, param, :].to(torch.float32)
                if param in _SCALE_PARAMS:
                    rows = rows + 1.0
                per_param.append(
                    bf16_tensor(rows, device=mesh_device, mesh_axis=tp_mesh_axis, shard_dim=-1),
                )
            self._block_tables.append(per_param)

        # `norm_out` is indexed by timestep alone -- no modality axis -- so these carry `rows` rows
        # rather than `rows * MODALITY_NUM`.
        self._final_shift = bf16_tensor(
            table.final_shift.to(torch.float32), device=mesh_device, mesh_axis=tp_mesh_axis, shard_dim=-1
        )
        self._final_scale = bf16_tensor(
            table.final_scale.to(torch.float32) + 1.0, device=mesh_device, mesh_axis=tp_mesh_axis, shard_dim=-1
        )

    def block_tables(self, layer: int) -> list[ttnn.Tensor]:
        return self._block_tables[layer]

    def final_tables(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """`(shift, scale)`, with the `1 +` already folded into scale."""
        return self._final_shift, self._final_scale

    def step_offset(self, step: int) -> int:
        """First absolute row of `step`, for callers building their own indices."""
        if not 0 <= step < self.num_steps:
            raise ValueError(f"step {step} outside the table's {self.num_steps} steps")
        return int(self._step_offsets[step])

    def assert_covers(self, num_inference_steps: int) -> None:
        """Fail loudly when a table is reused against a schedule it was not built for."""
        if num_inference_steps != self.num_steps:
            raise ValueError(
                f"AdaLN cache was built for {self.num_steps} forwards but this run does "
                f"{num_inference_steps}; rebuild it for this schedule"
            )
