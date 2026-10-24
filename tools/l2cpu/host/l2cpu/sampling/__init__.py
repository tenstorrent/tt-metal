# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""l2cpu.sampling: host side of the on-chip sampling application (tools/l2cpu/sampling).

    from l2cpu.sampling import boot, local_desc, layout as S
    fw, region, info = boot(device)            # fresh chip reset, ttnn device open; keep `region` alive
"""
from . import layout
from .fw import SamplingError, SamplingFw, allocate_region, boot, local_desc, pack_desc

__all__ = ["layout", "SamplingError", "SamplingFw", "allocate_region", "boot", "local_desc", "pack_desc"]
