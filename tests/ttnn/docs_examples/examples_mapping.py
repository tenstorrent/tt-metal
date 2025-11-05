# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from .test_gather import test_gather_example

EXAMPLES_DICT = {"ttnn.gather": test_gather_example}
