# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""One module per hardware unit the data path runs through.

A unit owns its transfers and the format tables that go with them, so an
architecture is assembled from units rather than subclassed. See :mod:`..arch`.

Not everything here is a unit. :mod:`.fidelity` and :mod:`.pack_effects` each
describe one unit's behaviour and are touched by no other, so they sit beside
it. A module needed by more than one unit lives at the package root instead.

There is no SFPU unit: nothing in this golden models the SFPU.
"""

from .fpu import Fpu
from .pack import Pack
from .unpack import Unpack

__all__ = ["Fpu", "Pack", "Unpack"]
