# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(scope="session")
def hunyuan_tokenizer():
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_pretrained()
