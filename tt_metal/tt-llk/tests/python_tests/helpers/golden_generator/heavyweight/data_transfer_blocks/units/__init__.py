# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""One module per hardware unit the data path runs through.

A unit owns its transfers and the format tables that go with them, so an
architecture is assembled from units rather than subclassed. When a future
generation changes one unit and reuses the rest, only that unit needs a second
implementation -- see :mod:`..arch`.

Not every module here is a unit. :mod:`.fidelity` is the FPU multiplier's
decomposition -- the phase tables, the mantissa split, the pre-carry flush --
and :mod:`.pack_effects` is the packer's ReLU, edge masking and stochastic
rounding. Each is applied by one unit and touched by no other, so each sits
beside the unit whose behaviour it describes. A module belongs at the package
root instead when more than one unit needs it, which is why
:mod:`..l1_support` and :mod:`..storage` are up there.

There is no SFPU unit yet: nothing in this golden models the SFPU, so the class
would be empty and every architecture would have to supply a no-op. It gets
added with the first real SFPU block; the shape here does not change when it
does.
"""

from .fpu import Fpu
from .pack import Pack
from .unpack import Unpack

__all__ = ["Fpu", "Pack", "Unpack"]
