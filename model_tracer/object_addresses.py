# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Strip per-process memory addresses from object reprs.

Kept in its own module, with no imports beyond ``re``, so test conftests can use it without pulling in
the tracer's dependencies.
"""

import re

OBJECT_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def strip_object_addresses(value):
    """Strip Python object memory addresses from a string value.

    Converts e.g. '<ttnn._ttnn.global_semaphore.global_semaphore object at 0x782ac28d15f0>'
    to '<ttnn._ttnn.global_semaphore.global_semaphore object>' so that
    runtime pointer values don't affect deduplication, hashing or test IDs.
    """
    if isinstance(value, str):
        return OBJECT_ADDR_RE.sub("", value)
    return value
