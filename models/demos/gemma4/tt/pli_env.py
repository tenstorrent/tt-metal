# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Gemma4 per-layer-input mechanism."""

import os

PLI_ENV = "GEMMA4_PLI"


def pli_on_device(default):
    """Return True for device PLI: ``GEMMA4_PLI`` when set, otherwise ``default``.

    Plain decode defaults to host PLI. Speculative decoding defaults to device
    PLI, which its fused bodies require. An empty value counts as unset.
    """
    value = os.environ.get(PLI_ENV, "").strip().lower()
    if not value:
        return default
    if value not in ("device", "host"):
        raise ValueError(f"{PLI_ENV} must be 'device' or 'host', got {value!r}")
    return value == "device"
