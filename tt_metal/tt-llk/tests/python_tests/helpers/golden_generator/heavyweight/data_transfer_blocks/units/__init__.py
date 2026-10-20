# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""One module per hardware unit the data path runs through.

A unit owns its transfers and the format tables that go with them, so an
architecture is assembled from units rather than subclassed. When a future
generation changes one unit and reuses the rest, only that unit needs a second
implementation -- see :mod:`..arch`.

There is no SFPU unit yet: nothing in this golden models the SFPU, so the class
would be empty and every architecture would have to supply a no-op. It gets
added with the first real SFPU block; the shape here does not change when it
does.
"""

from .fpu import Fpu
from .pack import Pack
from .unpack import Unpack

__all__ = ["Fpu", "Pack", "Unpack"]
