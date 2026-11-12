# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 CPU reference helpers."""

from models.demos.deepseek_v3_d_p.reference.glm_5_3.mtp import (
    fused_mtp_reference,
    glm_mtp_module_reference,
)

__all__ = ["fused_mtp_reference", "glm_mtp_module_reference"]
