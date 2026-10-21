# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The architectures an operation can run on, and the per-op factories.

An operation does not have a subclass per architecture. What differs between
architectures is **data** -- which blocks to use, and how the FPU splits a
mantissa per fidelity phase -- so each architecture states those once here and
every operation takes one as an argument.

Before this, each fact was restated once per operation: `blocks` appeared in
five files and the Quasar mantissa split in three. The file count grew as
``ops x architectures`` for information that is ``ops + architectures``, so a
new operation meant three near-identical subclasses and a new architecture
meant four.

The familiar names are kept as factories at the bottom, so
``QuasarMatmulGolden(MathFidelity.HiFi4)`` still works.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

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
    blocks: Callable[[], DataTransferBlocks]

    #: Explicit mantissa bits the multiplier takes from each operand per
    #: fidelity phase, as (srcA, srcB). ``None`` means this architecture's
    #: split is not modelled, so a multiply is computed exactly and
    #: ``math_fidelity`` is ignored -- which warns rather than passing quietly.
    #:
    #: Quasar multiplies 7x7 bits per phase, so a src datum's 10 explicit bits
    #: split 7 high / 3 low on **both** operands. That symmetry is a Quasar
    #: property: Wormhole and Blackhole split 4/6, and their SrcA additionally
    #: drops its least significant bit at every fidelity, so even HiFi4 is not
    #: the exact product there.
    mantissa_split: Optional[Tuple[int, int]] = None


QUASAR = ArchProfile(
    name="Quasar",
    blocks=QuasarDataTransferBlocks,
    mantissa_split=(7, 7),
)

#: Wormhole and Blackhole leave the split unmodelled -- see the field above.
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
    """`golden_class` bound to `arch`, named so it reads like the class it replaces.

    ``partial`` carries no ``__name__``, which breaks anything that introspects
    it -- a ``pytest.mark.parametrize`` id, a repr in a failure, a log line. Set
    it here so the factories are drop-in for the classes they replaced.
    """
    f = partial(golden_class, arch=arch)
    f.__name__ = f"{arch.name}{golden_class.__name__}"
    f.__doc__ = f"{golden_class.__name__} on {arch.name}. See {arch.name} above."
    return f


QuasarDataCopyGolden = _factory(DataCopyGolden, QUASAR)
QuasarEltwiseBinaryGolden = _factory(EltwiseBinaryGolden, QUASAR)
QuasarMatmulGolden = _factory(MatmulGolden, QUASAR)
QuasarEltwiseBinaryReuseDestGolden = _factory(EltwiseBinaryReuseDestGolden, QUASAR)

WormholeDataCopyGolden = _factory(DataCopyGolden, WORMHOLE)
WormholeEltwiseBinaryGolden = _factory(EltwiseBinaryGolden, WORMHOLE)
WormholeMatmulGolden = _factory(MatmulGolden, WORMHOLE)

BlackholeDataCopyGolden = _factory(DataCopyGolden, BLACKHOLE)
BlackholeEltwiseBinaryGolden = _factory(EltwiseBinaryGolden, BLACKHOLE)
BlackholeMatmulGolden = _factory(MatmulGolden, BLACKHOLE)

__all__ = [
    "BLACKHOLE",
    "QUASAR",
    "WORMHOLE",
    "ArchProfile",
    "BlackholeDataCopyGolden",
    "BlackholeEltwiseBinaryGolden",
    "BlackholeMatmulGolden",
    "QuasarDataCopyGolden",
    "QuasarEltwiseBinaryGolden",
    "QuasarEltwiseBinaryReuseDestGolden",
    "QuasarMatmulGolden",
    "WormholeDataCopyGolden",
    "WormholeEltwiseBinaryGolden",
    "WormholeMatmulGolden",
]
