# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The architectures an operation can run on, and the per-op factories.

An operation does not have a subclass per architecture. What differs between
architectures is **data**, namely which blocks to use, so each architecture states
that once here and every operation takes one as an argument. Everything an
operation needs to know about the hardware itself it asks the blocks: the FPU
carries its own multiplier width, the packer its own edge-mask polarity.

The factories at the bottom are the package's public surface, so a test writes
``get_quasar_matmul_golden(MathFidelity.HiFi4)`` and never touches the blocks.
They are named as functions rather than as classes because that is what they
are, which leaves the class-shaped name free for a pipeline class that needs it.

A factory name says which architecture the golden runs on and nothing more. It
does not promise that the pipeline is shared, nor that it is specialised: that
is read off the first argument of its ``_factory`` call, so an architecture that
grows its own chain changes one row here and no caller. See `README.md`, "When
an architecture differs".
"""

import re
from dataclasses import dataclass
from functools import partial
from typing import Callable

from ..data_transfer_blocks import (
    BlackholeDataTransferBlocks,
    DataTransferBlocks,
    QuasarDataTransferBlocks,
    WormholeDataTransferBlocks,
)
from .pipelines.datacopy import DataCopyGolden
from .pipelines.eltwise import EltwiseBinaryGolden
from .pipelines.matmul import MatmulGolden
from .pipelines.reuse_dest import EltwiseBinaryReuseDestGolden


@dataclass(frozen=True)
class ArchProfile:
    """What an operation needs to know about the architecture it runs on."""

    #: Named in refusals, so they say which device rather than naming the op.
    name: str

    #: The data-transfer blocks for this architecture. A callable, because the
    #: blocks are constructed per golden rather than shared.
    #:
    #: This and a name are all that is left: everything else an operation used
    #: to be told about the architecture turned out to belong to a unit behind
    #: this field. The FPU's per-phase mantissa split was the last of them.
    #: An operation asks for it through the blocks' ``fpu_*`` methods rather
    #: than holding the unit, so the units stay private to that package.
    blocks: Callable[[], DataTransferBlocks]


QUASAR = ArchProfile(name="Quasar", blocks=QuasarDataTransferBlocks)
WORMHOLE = ArchProfile(name="Wormhole", blocks=WormholeDataTransferBlocks)
BLACKHOLE = ArchProfile(name="Blackhole", blocks=BlackholeDataTransferBlocks)


# ----------------------------------------------------------------------
# Per-architecture factories.
#
# ``partial`` rather than a subclass: there is nothing for a subclass to add
# beyond binding the architecture, and a class would be somewhere to override
# a chain. ``partial`` also survives being a class attribute, where a plain
# function would bind as a method and pass ``self`` as the first argument.
# ----------------------------------------------------------------------


def _factory(golden_class, arch: ArchProfile):
    """`golden_class` bound to `arch`, under the name a caller imports.

    ``partial`` carries no ``__name__``, which breaks anything that introspects
    it: a ``pytest.mark.parametrize`` id, a repr in a failure, a log line. Set
    it here, and from the same rule the bindings below are named by, so the two
    cannot drift apart.
    """

    def snake(name: str) -> str:
        """``EltwiseBinaryReuseDestGolden`` to ``eltwise_binary_reuse_dest_golden``."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    f = partial(golden_class, arch=arch)
    f.__name__ = f"get_{arch.name.lower()}_{snake(golden_class.__name__)}"
    f.__doc__ = f"{golden_class.__name__} on {arch.name}. See {arch.name} above."
    return f


get_quasar_data_copy_golden = _factory(DataCopyGolden, QUASAR)
get_quasar_eltwise_binary_golden = _factory(EltwiseBinaryGolden, QUASAR)
get_quasar_matmul_golden = _factory(MatmulGolden, QUASAR)
get_quasar_eltwise_binary_reuse_dest_golden = _factory(
    EltwiseBinaryReuseDestGolden, QUASAR
)

get_wormhole_data_copy_golden = _factory(DataCopyGolden, WORMHOLE)
get_wormhole_eltwise_binary_golden = _factory(EltwiseBinaryGolden, WORMHOLE)
get_wormhole_matmul_golden = _factory(MatmulGolden, WORMHOLE)

get_blackhole_data_copy_golden = _factory(DataCopyGolden, BLACKHOLE)
get_blackhole_eltwise_binary_golden = _factory(EltwiseBinaryGolden, BLACKHOLE)
get_blackhole_matmul_golden = _factory(MatmulGolden, BLACKHOLE)

__all__ = [
    "BLACKHOLE",
    "QUASAR",
    "WORMHOLE",
    "ArchProfile",
    "get_blackhole_data_copy_golden",
    "get_blackhole_eltwise_binary_golden",
    "get_blackhole_matmul_golden",
    "get_quasar_data_copy_golden",
    "get_quasar_eltwise_binary_golden",
    "get_quasar_eltwise_binary_reuse_dest_golden",
    "get_quasar_matmul_golden",
    "get_wormhole_data_copy_golden",
    "get_wormhole_eltwise_binary_golden",
    "get_wormhole_matmul_golden",
]
