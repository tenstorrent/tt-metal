# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.wormhole.mamba.demo.demo import run_mamba_demo, run_mamba_prefill_decode_demo
import pytest


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "user_input, model_version, max_gen_len",
    ((["Hello World"], "state-spaces/mamba-2.8b-slimpj", 2),),
)
def test_demo(user_input, model_version, device, use_program_cache, get_tt_cache_path, max_gen_len):
    return run_mamba_demo(
        prompts=user_input,
        model_version=model_version,
        device=device,
        generated_sequence_length=max_gen_len,
        display=False,
        cache_dir=get_tt_cache_path(model_version),
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "user_input, model_version, max_gen_len",
    (
        (
            [
                "Climate change refers to long-term shifts in temperatures and weather patterns. Such shifts can be natural due to changes in the sun's activity or volcanic eruptions."
            ],
            "state-spaces/mamba-2.8b-slimpj",
            4,
        ),
        (
            [
                "The city of Sarnia is located on the eastern shore of Lake Huron at its extreme southern point where it flows into the St. Clair River . Most of the surrounding area is flat , and the elevation ranges from 169 metres ( 554 ft ) and 281 metres ( 922 ft ) above sea level . The soil mostly comprises clay . Despite this high percentage of clay , the soil is remarkably rich for cultivation . Prior to the Ice Age , glaciers covered most of the area , as can be seen not only by the existence of the Great Lakes themselves but also of alluvial sand deposits, terminal moraines, and rich oil reserves."
            ],
            "state-spaces/mamba-2.8b-slimpj",
            4,
        ),
    ),
)
def test_prefill_decode_demo(user_input, model_version, device, use_program_cache, get_tt_cache_path, max_gen_len):
    return run_mamba_prefill_decode_demo(
        prompts=user_input,
        model_version=model_version,
        device=device,
        generated_sequence_length=max_gen_len,
        display=False,
        cache_dir=get_tt_cache_path(model_version),
    )
