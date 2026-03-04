# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Re-export layer_norm so stage test files can do:
#   from .layer_norm import layer_norm

from ttnn.operations.layer_norm import layer_norm

__all__ = ["layer_norm"]
