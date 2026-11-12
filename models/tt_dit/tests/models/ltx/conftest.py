# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""LTX test hooks (serving line).

``LTX_QUALITY=high|medium|fast`` (or legacy ``LTX_FAST=1``) expands to the served bundle at collection
time -- the pipeline reads the sigma schedule at import. ``apply_quality_env`` is the shared definition
(see ``utils/ltx.py``) so the pytest bundle cannot drift from the ltx_server worker's; with no
``LTX_QUALITY`` set it falls back to ``apply_fast_env``.

The host CPU pinning re-exec that lived here went away with ``utils/host_affinity.py`` on main
(#60670): the pipelines cap torch threads to half the CPU mask instead.
"""

from __future__ import annotations

from models.tt_dit.utils.ltx import apply_quality_env

apply_quality_env()
