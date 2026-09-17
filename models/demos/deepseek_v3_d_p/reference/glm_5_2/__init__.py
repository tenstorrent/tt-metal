# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 CPU reference helpers.

Only the multi-token-prediction reference lives here; the decoder layer, DSA MLA and MoE are reused
from ``reference.glm_5_1``. GLM-5.2 ships MTP weights but no MTP code, so this reference is composed.
"""

from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import (
    fused_mtp_reference,
    glm_mtp_module_reference,
)

__all__ = ["fused_mtp_reference", "glm_mtp_module_reference"]
