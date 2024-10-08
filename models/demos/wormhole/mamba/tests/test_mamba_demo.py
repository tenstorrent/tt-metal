# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import List

from models.demos.wormhole.mamba.demo.demo import run_mamba_demo

from difflib import SequenceMatcher


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "user_input, expected_output, model_version, max_gen_len, prefill_chunk_size",
    (
        (
            [
                "Climate change refers to long-term shifts in temperatures and weather patterns. Such shifts can be natural due to changes in the sun's activity or volcanic eruptions."
            ],
            ["They can also be caused by human activity, such"],
            "state-spaces/mamba-2.8b-slimpj",
            10,
            32,
        ),
        (
            [
                "The city of Sarnia is located on the eastern shore of Lake Huron at its extreme southern point where it flows into the St. Clair River . Most of the surrounding area is flat , and the elevation ranges from 169 metres ( 554 ft ) and 281 metres ( 922 ft ) above sea level . The soil mostly comprises clay . Despite this high percentage of clay , the soil is remarkably rich for cultivation . Prior to the Ice Age , glaciers covered most of the area , as can be seen not only by the existence of the Great Lakes themselves but also of alluvial sand deposits, terminal moraines, and rich oil reserves."
            ],
            ["The Great Lakes are the largest freshwater system in"],
            "state-spaces/mamba-2.8b-slimpj",
            10,
            128,
        ),
    ),
)
def test_demo(
    user_input: List[str],
    expected_output: List[str],
    model_version,
    device,
    use_program_cache,
    get_tt_cache_path,
    max_gen_len,
    prefill_chunk_size,
):
    assert len(user_input) == len(expected_output)

    demo_result = run_mamba_demo(
        prompts=user_input,
        model_version=model_version,
        device=device,
        generated_sequence_length=max_gen_len,
        display=True,
        cache_dir=get_tt_cache_path(model_version),
        prefill_chunk_size=prefill_chunk_size,
    )

    expected = user_input[0] + expected_output[0]
    actual = demo_result.generated_text[0]

    def similarity(x, y) -> float:
        return SequenceMatcher(None, x, y).ratio()

    assert similarity(actual, expected) > 0.99, "Expected demo output to match provided value"
