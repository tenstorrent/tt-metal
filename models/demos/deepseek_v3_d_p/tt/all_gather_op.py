# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The persistent-output all-gather every prefill call site uses.

``ttnn.experimental.fabric_all_gather`` and ``ttnn.experimental.high_bw_all_gather`` share one contract (same
arguments, same output), so the model calls this and picks the implementation once, from ``DS_ALL_GATHER_OP``:
``fabric`` (default) or ``high_bw``.
"""

import os

import ttnn

ALL_GATHER_OP = os.environ.get("DS_ALL_GATHER_OP", "fabric")
if ALL_GATHER_OP not in ("fabric", "high_bw"):
    raise ValueError(f"DS_ALL_GATHER_OP must be 'fabric' or 'high_bw', got {ALL_GATHER_OP!r}")


def persistent_all_gather(*args, **kwargs):
    if ALL_GATHER_OP == "fabric":
        return ttnn.experimental.fabric_all_gather(*args, **kwargs)
    return ttnn.experimental.high_bw_all_gather(*args, **kwargs)
