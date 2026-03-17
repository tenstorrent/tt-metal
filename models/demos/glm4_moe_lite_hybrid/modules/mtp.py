# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-Token Prediction (MTP) for GLM-4.7-Flash hybrid.

Re-exports the agentic MTP forward pass which runs decoder layer 47 eagerly
after the main model decode to produce one draft token for speculative decoding.

Flow: embed main tokens -> enorm + hnorm -> concat + project -> decoder layer 47
      -> shared head norm + LM head -> argmax -> draft token IDs
"""

from models.demos.glm4_moe_lite.tt.mtp_forward import mtp_forward_eager

__all__ = ["mtp_forward_eager"]
