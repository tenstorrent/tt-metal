# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MoEBlock is an alias for the original MoE implementation.
Since we're using all the original components, we can just import and alias the original.
"""

from models.demos.deepseek_v3.tt.moe import MoE

# Create alias for backward compatibility
MoEBlock = MoE
