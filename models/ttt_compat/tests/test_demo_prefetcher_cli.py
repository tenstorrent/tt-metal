# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from _pytest.config.argparsing import Parser

from models.ttt_compat.demo.conftest import pytest_addoption


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("flag_first", [False, True])
def test_prefetcher_flag_preserves_test_path(enabled, flag_first):
    parser = Parser(_ispytest=True)
    pytest_addoption(parser)
    test_path = "models/ttt_compat/demo/simple_text_demo.py"
    flag = ["--use_prefetcher"] if enabled else []
    args = flag + [test_path] if flag_first else [test_path] + flag
    options = parser.parse(args + ["--repeat_batches", "2"])
    assert options.use_prefetcher is enabled
    assert options.file_or_dir == [test_path]
    assert options.repeat_batches == 2
