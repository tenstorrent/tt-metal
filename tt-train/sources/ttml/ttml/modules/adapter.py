# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ForwardInvocation:
    """Snapshot of a single forward call, passed to adapters.

    Attributes:
        module: The module that produced the output.
        args: Positional arguments passed to the module's __call__.
        kwargs: Keyword arguments passed to the module's __call__.
        output: The value returned by the module's forward().
    """

    module: Any
    args: Sequence[Any]
    kwargs: Mapping[str, Any]
    output: Any


class Adapter:
    """Base class for post-forward output adaptation.

    Subclasses override __call__ to modify or augment the forward output,
    and override parameters() to expose any trainable Parameters.
    """

    __slots__ = ()

    def __call__(self, fwd: ForwardInvocation) -> Any:
        """Return the forward output unchanged."""
        return fwd.output

    def parameters(self) -> dict:
        """Return trainable parameters keyed by name. Override in subclasses."""
        return {}


class IdentityAdapter(Adapter):
    """No-op adapter."""

    __slots__ = ()
