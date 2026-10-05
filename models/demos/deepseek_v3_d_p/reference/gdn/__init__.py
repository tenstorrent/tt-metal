# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Torch FP32 reference of the Qwen Gated DeltaNet (GDN) linear-attention layer."""

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.layer import GDNReferenceState, gdn_forward_reference

__all__ = ["GDNConfig", "GDNReferenceState", "gdn_forward_reference"]
