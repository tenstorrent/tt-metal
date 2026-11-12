# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The architectures an operation can run on, and the per-operation factories.

An operation does not have a subclass per architecture. What differs between
architectures is **data**, namely which blocks to use, so each architecture states
that once here and every operation takes one as an argument. Everything an
operation needs to know about the hardware itself it asks the blocks: the FPU
carries its own multiplier width, the packer its own edge-mask polarity.

The factories at the bottom are the package's public surface, so a test writes
``get_matmul_golden(QUASAR, MathFidelity.HiFi4)`` and never touches the blocks.
They are named as functions because that is what they are, which leaves the
class-shaped name free for a pipeline class that wants it.

There is one per operation rather than one per operation and architecture pair.
The architecture travels as an argument, so a test that reads its own
architecture at run time passes it straight through, and a fourth architecture
adds no names here. What an architecture needs that is specific to it goes in
the body of the factory, which is why these are functions rather than bound
partials. See `README.md`, "When an architecture differs".
"""

from dataclasses import dataclass
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
# The factories.
#
# One per operation rather than one per operation and architecture pair. The
# architecture is a value, so it belongs in an argument rather than in ten
# names, and a test that resolves its architecture at run time can pass it
# straight through.
#
# Plain functions rather than ``partial``, because a function has a body, and
# the body is where an architecture that needs a different pipeline, or that
# has no pipeline at all, is dealt with. Each takes `arch` first and forwards
# the rest to the pipeline, so the pipeline keeps one definition of its own
# arguments and their defaults.
# ----------------------------------------------------------------------


def get_data_copy_golden(arch: ArchProfile, *args, **kwargs) -> DataCopyGolden:
    """A datacopy golden on `arch`. Takes no arguments of its own."""
    return DataCopyGolden(*args, arch=arch, **kwargs)


def get_eltwise_binary_golden(
    arch: ArchProfile, *args, **kwargs
) -> EltwiseBinaryGolden:
    """An element-wise binary golden on `arch`.

    Forwards ``operation`` and ``math_fidelity``.
    """
    return EltwiseBinaryGolden(*args, arch=arch, **kwargs)


def get_matmul_golden(arch: ArchProfile, *args, **kwargs) -> MatmulGolden:
    """A matmul golden on `arch`. Forwards ``math_fidelity``."""
    return MatmulGolden(*args, arch=arch, **kwargs)


def get_eltwise_binary_reuse_dest_golden(
    arch: ArchProfile, *args, **kwargs
) -> EltwiseBinaryReuseDestGolden:
    """A reuse-dest golden on `arch`. Forwards ``operation``, ``math_fidelity``
    and ``reuse_dest_type``.

    Quasar only, and that is enforced rather than merely documented: the
    pipeline would run happily on another architecture and hand back numbers
    nobody has checked against a device.
    """
    if arch is not QUASAR:
        raise ValueError(
            f"reuse_dest has only been modelled and checked on Quasar, not on "
            f"{arch.name}. Nothing about the pipeline refuses to run there, "
            f"which is why this does: the result would look plausible and be "
            f"unverified."
        )
    return EltwiseBinaryReuseDestGolden(*args, arch=arch, **kwargs)


__all__ = [
    "BLACKHOLE",
    "QUASAR",
    "WORMHOLE",
    "ArchProfile",
    "get_data_copy_golden",
    "get_eltwise_binary_golden",
    "get_eltwise_binary_reuse_dest_golden",
    "get_matmul_golden",
]
