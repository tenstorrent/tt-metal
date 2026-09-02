# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Datacopy."""

from .chain import Chain
from .golden import Golden, OpConfig


class DataCopyGolden(Golden):
    """A copy is the dataflow with nothing in the middle.

    Because dest_to_l1 really packs, a format-converting copy is requantized
    onto the output lattice, which a golden that only casts dtypes is not.
    """

    op_name = "datacopy"

    def build_chain(self, cfg: OpConfig) -> Chain:
        return Chain(
            [
                self.l1_to_srcA(cfg, source="in0"),
                self.src_to_dest(lambda regs: regs["srcA"], reads=("srcA",)),
                self.dest_to_l1(cfg, into="out"),
            ]
        )
