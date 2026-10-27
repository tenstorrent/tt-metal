# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Switches of the bounded prefill replay (cross-layer KV sharing: layers 21-39 only need to run on the last R tokens of a prompt).

``DSV41_PF_REPLAY`` = R (default 2560, the exact variant: bit-identical to running every layer on every chunk; 0 = off). Active only where it is supported: paged prefill read mode
(``DSV41_PF_PAGED=read``) with the per-user state mask (``DSV41_PF_UMASK=1``), which is the server build and the demo recipes; anywhere else the switch is ignored (the single chunk trace, as before).
At boot a build with R > 0 captures the chunk trace as two halves (A: embedding .. layer 20, B: layers 21..39 + head). ``DSV41_PF_REPLAY_FILE`` names a file holding R that overrides the value live (0: every chunk
runs both halves, numerically the single trace) for A/B runs of a server.
"""

import os

DEFAULT_R = 2560


def _env_r():
    return int(os.environ.get("DSV41_PF_REPLAY", str(DEFAULT_R)))


def supported():
    return os.environ.get("DSV41_PF_PAGED", "0") == "read" and os.environ.get("DSV41_PF_UMASK", "0") == "1"


def split_enabled():
    """boot-time: capture the chunk trace as two halves and allocate the first-replay-chunk tensors"""
    return _env_r() > 0 and supported()


def replay_tokens():
    """R in force now (live file switch, else the environment); 0 = every chunk runs every layer"""
    if not split_enabled():
        return 0
    f = os.environ.get("DSV41_PF_REPLAY_FILE")
    if f:
        try:
            return int(open(f).read().strip())
        except (OSError, ValueError):
            pass
    return _env_r()
