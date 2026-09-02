# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.wormhole.fpu.common import (  # noqa: F401
    configure_math,
    hw_configure_math,
    math_dest_section_done,
    math_pack_sync_init,
    math_wait_for_dest,
)


def math_dest_remap_config(required: bool) -> str:
    if not required:
        return ""
    return "_llk_math_reconfig_remap_(true);\n"
