# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim for test imports."""

from ttnn.operations.row_mean_rm import row_mean_rm

__all__ = ["row_mean_rm"]
