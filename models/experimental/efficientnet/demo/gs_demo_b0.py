# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.efficientnet.demo.demo_utils import run_gs_demo
from models.experimental.efficientnet.tt.efficientnet_model import efficientnet_b0


def test_gs_demo_b0():
    run_gs_demo(efficientnet_b0)
