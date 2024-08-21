# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.mamba.demo.demo import run_demo
from models.experimental.mamba.reference.decode_model import MambaPretrainedModelName
from models.utility_functions import skip_for_grayskull


@pytest.mark.skip(reason="Test failing, see issue #7551")
@skip_for_grayskull("Not supported on Grayskull")
@pytest.mark.parametrize(
    "model_version, batch, genlen",
    (
        ("state-spaces/mamba-370m", 1, 4),
        ("state-spaces/mamba-370m", 2, 4),
        ("state-spaces/mamba-370m", 3, 4),
    ),
)
def test_demo(
    model_version: MambaPretrainedModelName,
    batch: int,
    genlen: int,
):
    prompt = ["Hello" for _ in range(batch)]

    res1 = run_demo(
        prompt, "cpu", generated_sequence_length=genlen, model_version=model_version, display=False, use_cache=False
    )
    assert len(res1) == genlen + 1

    res2 = run_demo(
        prompt, "wh", generated_sequence_length=genlen, model_version=model_version, display=False, use_cache=False
    )
    assert len(res2) == genlen + 1

    assert res1 == res2, "Model outputs should match"
