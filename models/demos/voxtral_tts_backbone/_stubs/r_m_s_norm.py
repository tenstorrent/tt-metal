# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`r_m_s_norm` (`MistralRMSNorm`) for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

This is the plan's REUSE target used as-is: the canonical
`models/common/rmsnorm.py::RMSNorm`. Only the construction needed adapting —
the scaffold called it with `mesh_device=`/`args=`/`layer_num=`, but the canonical
signature is `RMSNorm(device, dim, state_dict, weight_key, ...)` and it looks the
gain up as `f"{weight_key}.weight"`, so the HF module's `weight` is re-keyed here.

The library module owns the compute (`ttnn.rms_norm`, ROW_MAJOR gain reshaped to
`[1, 1, dim/32, 32]`, HiFi2 + fp32 accumulate) and normalizes over the last dim,
which is what HF's RMSNorm does. `_stubs/decoder_layer.py` builds its two norms
through this same adapter.

`build` touches torch only to re-key/cast the checkpoint gain; the forward is the
library's ttnn dispatch — `models/common/native_probe.py` sees it as native.
"""
from __future__ import annotations

import torch
import ttnn

from models.common.rmsnorm import RMSNorm

TILE = 32
#: Core count the tt_transformers sharded-norm helper is tuned around. The real
#: count is the divisor of the tile-width nearest this that fits the grid.
_NORM_CORE_TARGET = 32


def _decode_shard_plan(device, dim: int):
    """Width-shard plan for a ONE-ROW (decode) norm, or None if it can't be made.

    A decode row is `[1, 1, dim]`: the interleaved norm kernel parallelizes over
    the ROW axis, so with a single row it has nothing to spread and lands on one
    core. Splitting the EMBEDDING axis instead is what fills the grid, and that
    means sharding the tensor -- a program config alone would be inert.
    """
    n_tiles = dim // TILE
    grid = device.compute_with_storage_grid_size()
    candidates = [c for c in range(1, grid.x * grid.y + 1) if n_tiles % c == 0]
    candidates.sort(key=lambda c: abs(c - _NORM_CORE_TARGET))
    for cores in candidates:
        for rows in range(1, grid.y + 1):
            if cores % rows or cores // rows > grid.x:
                continue
            core_grid = ttnn.CoreGrid(y=rows, x=cores // rows)
            block_w = n_tiles // cores
            subblock_w = next(s for s in (4, 3, 2, 1) if block_w % s == 0)
            return (
                ttnn.create_sharded_memory_config(
                    shape=(1, 1, TILE, dim),
                    core_grid=core_grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                ),
                ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(core_grid.x, core_grid.y),
                    subblock_w=subblock_w,
                    block_h=1,  # one tile row -- the decode shape
                    block_w=block_w,
                    inplace=False,
                ),
            )
    return None


class TtRMSNorm:
    """Adapter around the canonical `models/common/rmsnorm.py::RMSNorm`."""

    _WEIGHT_KEY = "norm"

    def __init__(self, canonical_instance, decode_plan=None) -> None:
        self._impl = canonical_instance
        self._decode_plan = decode_plan

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("r_m_s_norm build needs the HF MistralRMSNorm module to read its gain from")
        weight = torch_module.weight.detach().to(torch.float32)
        dim = int(weight.shape[-1])
        eps = getattr(torch_module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(torch_module, "eps", 1e-5)
        canonical = RMSNorm(
            device=device,
            dim=dim,
            state_dict={f"{cls._WEIGHT_KEY}.weight": weight},
            weight_key=cls._WEIGHT_KEY,
            eps=float(eps),
        )
        return cls(canonical, _decode_shard_plan(device, dim))

    def _is_decode_row(self, x) -> bool:
        """One tile row of activation -- the decode step's `[1, 1, dim]`."""
        shape = list(x.padded_shape)
        if int(shape[-2]) != TILE:
            return False
        batch = 1
        for dim in shape[:-2]:
            batch *= int(dim)
        return batch == 1

    def __call__(self, x, *_args, **_ignored):
        # Prefill has many rows, so the interleaved kernel already spreads over
        # the row axis. Decode has ONE row and would otherwise run on a single
        # core, so it goes through the width-sharded path instead.
        if self._decode_plan is not None and self._is_decode_row(x):
            input_memcfg, program_config = self._decode_plan
            return self._impl(
                ttnn.to_memory_config(x, input_memcfg),
                mode="decode",
                in_sharded=True,
                norm_config={"sharded_program_config": program_config},
            )
        return self._impl(x, mode="prefill")


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)


# Module-level shim with the component's lowercase slug name. Kept for
# backward compatibility with legacy SMOKE/PCC tests that import the
# slug directly.
def r_m_s_norm(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)
