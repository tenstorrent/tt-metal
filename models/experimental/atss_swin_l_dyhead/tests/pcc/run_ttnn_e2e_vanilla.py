#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the strict ATSS E2E PCC test with vanilla Swin-L precision.

The checked-in ATSS default enables stage-2 MLP precision. This wrapper
disables all high-precision backbone stages for this process and invokes the
same ``test_ttnn_e2e.py`` test. Additional command-line arguments are forwarded
to pytest.

The custom boat checkpoint is expected to reproduce the original baseline
failure at head level-0 centerness (approximately 0.959824 versus the strict
0.96 threshold). That failure is the comparison this wrapper is intended to
preserve.
"""

import sys
from pathlib import Path

import pytest

import models.experimental.atss_swin_l_dyhead.tt.tt_atss_model as tt_atss_model
from models.experimental.atss_swin_l_dyhead.tt.tt_swin_backbone import build_atss_backbone


def build_vanilla_atss_backbone(checkpoint_path, device, input_h=None, input_w=None):
    """Build ATSS's Swin-L backbone with all precision promotions disabled."""
    return build_atss_backbone(
        checkpoint_path,
        device,
        input_h=input_h,
        input_w=input_w,
        high_precision_mlp_stages=(),
        high_precision_attn_stages=(),
    )


def main():
    # TtATSSModel.from_checkpoint resolves this module-level symbol during test
    # execution. The override is process-local and leaves source defaults intact.
    tt_atss_model.build_atss_backbone = build_vanilla_atss_backbone

    test_path = Path(__file__).resolve().with_name("test_ttnn_e2e.py")
    pytest_args = [str(test_path), "-v", "-s", *sys.argv[1:]]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
