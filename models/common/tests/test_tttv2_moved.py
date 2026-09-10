# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "models.common.modules",
        "models.common.llm_runtime",
        "models.common.models",
        "models.common.modules.mlp.mlp_1d",
        "models.common.models.llama3_8b.generator",
    ],
)
def test_tttv2_old_path_raises(name, expect_error):
    with expect_error(ImportError, "tt_transformers"):
        __import__(name)
