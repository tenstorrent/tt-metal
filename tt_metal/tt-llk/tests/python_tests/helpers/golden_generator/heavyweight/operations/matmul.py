# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Matmul — the MVMUL family."""

import torch

from .chain import Chain, Registers
from .golden import Golden, OpConfig

#: Datums along one edge of a tile.
TILE_DIM = 32


class MatmulGolden(Golden):
    """L1 -> SrcA, L1 -> SrcB -> product -> Dest -> L1.

    Fidelity phases and accumulation rounding order are **not** modelled; this
    is the exact product at Dest precision, close to HiFi4 and wrong at LoFi.
    """

    op_name = "matmul"

    def build_chain(self, cfg: OpConfig) -> Chain:
        return Chain(
            [
                self.l1_to_srcA(cfg, source="in0"),
                self.l1_to_srcB(cfg, source="in1"),
                self.src_to_dest(self.apply, reads=("srcA", "srcB")),
                self.dest_to_l1(cfg, into="out"),
            ]
        )

    @staticmethod
    def _as_tile(values: torch.Tensor) -> torch.Tensor:
        return values.float().reshape(TILE_DIM, TILE_DIM)

    def apply(self, regs: Registers) -> torch.Tensor:
        return (self._as_tile(regs["srcA"]) @ self._as_tile(regs["srcB"])).reshape(-1)
