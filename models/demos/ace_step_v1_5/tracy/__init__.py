"""
Minimal stub `tracy` package for TTNN initialization.

TTNN's nanobind extension imports `tracy.ttnn_profiler_wrapper.callable_decorator`
when built with tracy support. The full `tt-metal/tools/tracy` package has heavy
optional dependencies; for bring-up we provide a tiny no-op implementation.
"""

from __future__ import annotations

import contextlib
from typing import Iterator


@contextlib.contextmanager
def signpost(_name: str, *_args, **_kwargs) -> Iterator[None]:
    yield
