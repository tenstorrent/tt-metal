# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# `assert_with_pcc` now lives in models.common.utility_functions, alongside the comp_pcc
# it wraps. It was moved because the bge_large_en performant runners -- which are
# serving code, not test code -- imported it from here, and that edge from a shipped
# module into a test package makes the runners unimportable from the packaged
# `tt-metal-models` distribution, which does not include test directories.
#
# Re-exported here so the pcc tests in this directory keep working unchanged.
from models.common.utility_functions import assert_with_pcc

__all__ = ["assert_with_pcc", "construct_pcc_assert_message"]


def construct_pcc_assert_message(message, expected_pytorch_result, actual_pytorch_result):
    messages = []
    messages.append(message)
    messages = [str(m) for m in messages]
    return "\n".join(messages)
