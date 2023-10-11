# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.t5.tt.t5_for_conditional_generation import (
    t5_base_for_conditional_generation,
)
from models.experimental.t5.demo.demo_utils import run_demo_t5


def test_demo_t5_base():
    run_demo_t5(t5_base_for_conditional_generation)
