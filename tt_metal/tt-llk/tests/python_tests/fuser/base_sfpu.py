# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .block_data import BlockData
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation
    from .sfpu_node import SfpuNode

from .indexing import InvocationGranularity


class Sfpu:
    """Base class for fused test SFPU code generators.

    Subclasses represent specific SFPU operations (e.g. UnarySfpu, BinarySfpu)
    and override methods to emit the C++ LLK calls that configure and drive the
    SFPU Unit.

    Unlike Fpu, SFPU operates on dest register data that was already computed
    by a prior FPU stage or loaded via datacopy. It has no unpacker — SfpuNode
    has no unpacker field at all.

    The lifecycle called by the pipeline is:
        init() -> planned calls to calculate() -> uninit()

    Entirely skipped during UNPACK_ISOLATE, PACK_ISOLATE, and L1_CONGESTION perf runs.

    To create a new SFPU:
        1. Subclass Sfpu
        2. Override get_headers() with the required LLK header files
        3. Override init(), calculate(), uninit() to emit the C++ LLK calls
        4. Bind the corresponding callable from fuser.golden.sfpu
    """

    granularity = InvocationGranularity.NONE
    input_count = 1

    def init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "SfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the SFPU before calculation."""
        return ""

    def calculate(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "SfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that performs one planned SFPU call."""
        return ""

    def uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "SfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the SFPU after calculation."""
        return ""

    def get_headers(self) -> List[str]:
        """Return headers that declare this SFPU's generated calls."""
        return []

    def __str__(self) -> str:
        return f"{self.__name__}"
