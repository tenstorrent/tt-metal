# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# autoflake: skip_file

# generation_utils.py has been moved to models/common. This is a temporary redirecting file.
# Update your imports accordingly as soon as possible.
# Use: models.common.generation_utils

from models.common.generation_utils import (
    _get_logits_processor,
    _merge_criteria_processor_list,
    get_logits_processor,
    pad_input_32,
    run_generate,
)
