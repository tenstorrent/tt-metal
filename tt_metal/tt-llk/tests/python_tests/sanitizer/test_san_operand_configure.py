# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from helpers.test_config import TestConfig


def test_san_operand_configure():
    TestConfig("sources/sanitizer/san_operand_configure_test.cpp").run()
