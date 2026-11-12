# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The architectures an operation can run on, and the factory.

An operation has no subclass per architecture. What differs is data, namely
which blocks to use, so each architecture states that once here and every
operation takes one as an argument. Anything else about the hardware an
operation asks the blocks.

See `README.md`, "Architecture bindings".
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Mapping, Tuple

from ..data_transfer_blocks import (
    BlackholeDataTransferBlocks,
    DataTransferBlocks,
    QuasarDataTransferBlocks,
    WormholeDataTransferBlocks,
)
from .golden import Golden
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
# The factory.
#
# One function for every operation, not one per operation and not one per
# operation and architecture pair. Both of those grow a name per combination
# for information that is really two independent lists, so the operation and
# the architecture both travel as arguments and a test that resolves either at
# run time passes it straight through.
#
# What varies per operation lives in the registries below rather than in
# branches inside ``get_golden``: an if-chain there would grow a case per
# operation, which is the shape this replaced.
# ----------------------------------------------------------------------


class Op(Enum):
    """The operations this package models.

    An enum rather than the pipeline class itself, because a test that
    cross-validates imports both goldens and the lightweight one already
    exports ``MatmulGolden``, ``DataCopyGolden`` and ``EltwiseBinaryGolden``.
    An ``Op`` member cannot collide with those.
    """

    DataCopy = auto()
    EltwiseBinary = auto()
    Matmul = auto()
    EltwiseBinaryReuseDest = auto()


#: The pipeline each operation runs. Adding an operation is one member above
#: and one row here.
_PIPELINES: Mapping["Op", type] = {
    Op.DataCopy: DataCopyGolden,
    Op.EltwiseBinary: EltwiseBinaryGolden,
    Op.Matmul: MatmulGolden,
    Op.EltwiseBinaryReuseDest: EltwiseBinaryReuseDestGolden,
}

#: Architectures an operation has actually been checked on. Absent means all
#: of them. This is a statement about validation, not about what the pipeline
#: is capable of running: every pipeline will run anywhere and hand back
#: numbers, which is exactly why an unchecked pair has to be refused here.
_VALIDATED_ON: Mapping["Op", Tuple[ArchProfile, ...]] = {
    Op.EltwiseBinaryReuseDest: (QUASAR,),
}

#: Where an architecture runs a different pipeline for the same operation,
#: keyed by ``(op, arch)``. Empty today. A chain that diverges goes here rather
#: than into a branch inside an operation, so the difference stays visible to
#: anyone reading how goldens are made.
_ARCH_PIPELINES: Mapping[Tuple["Op", ArchProfile], type] = {}


def get_golden(op: "Op", arch: ArchProfile, *args, **kwargs) -> Golden:
    """The golden for `op` on `arch`, with the rest forwarded to the pipeline.

    Every argument after `arch` belongs to the operation, so a matmul takes
    ``math_fidelity`` and an element-wise binary takes ``operation`` and
    ``math_fidelity``. They are forwarded rather than restated, so each
    pipeline keeps one definition of its own arguments and their defaults, and
    a wrong call is reported by that pipeline naming the parameter it wanted.
    """
    if op not in _PIPELINES:
        raise ValueError(
            f"{op!r} is not an operation this package models. It has "
            f"{', '.join(member.name for member in Op)}."
        )
    archs = _VALIDATED_ON.get(op)
    if archs is not None and arch not in archs:
        raise ValueError(
            f"{op.name} has only been modelled and checked on "
            f"{', '.join(a.name for a in archs)}, not on {arch.name}. The "
            f"pipeline would run there and hand back numbers nobody has "
            f"compared against a device, which is why this refuses instead."
        )
    pipeline = _ARCH_PIPELINES.get((op, arch), _PIPELINES[op])
    return pipeline(*args, arch=arch, **kwargs)


__all__ = [
    "BLACKHOLE",
    "QUASAR",
    "WORMHOLE",
    "ArchProfile",
    "Op",
    "get_golden",
]
