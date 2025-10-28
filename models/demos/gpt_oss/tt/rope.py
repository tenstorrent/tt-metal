# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Import get_rot_transformation_mat from tt_transformers to avoid code duplication
from models.tt_transformers.tt.common import get_rot_transformation_mat

__all__ = ["get_rot_transformation_mat"]
