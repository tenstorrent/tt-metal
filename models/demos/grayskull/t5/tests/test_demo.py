# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.grayskull.t5.demo.question_answering_demo import test_functional_t5_demo as qa_demo
from models.demos.grayskull.t5.demo.question_answering_demo import test_functional_t5_demo_squadv2 as qa_demo_dataset
from models.demos.grayskull.t5.demo.conditional_generation_demo import test_t5_demo_for_summarize as summarize_demo
from models.demos.grayskull.t5.demo.conditional_generation_demo import (
    test_t5_demo_for_summarize_dataset as summarize_dataset_demo,
)
import pytest
from models.demos.grayskull.t5.tt import ttnn_functional_t5, ttnn_optimized_functional_t5


@pytest.mark.parametrize(
    "input_path",
    (("models/demos/grayskull/t5/demo/input_data_qa.json"),),
)
@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name", "use_optimized_version"),
    ((8, 128, 5, "t5-small", True),),
)
def test_qa_batch_8(
    device, use_program_cache, batch_size, sequence_length, max_tokens, model_name, input_path, use_optimized_version
):
    expected_answers = {
        0: "archaeology",
        1: "The Duchy of Norman",
        2: "males",
        3: "married outside their immediate French",
        4: "biostratigraph",
        5: "chemotaxis",
        6: "1992",
        7: "statocys",
    }
    _, predicted_answers = qa_demo(
        device,
        use_program_cache,
        batch_size,
        sequence_length,
        max_tokens,
        model_name,
        input_path,
        use_optimized_version,
    )
    for i in range(batch_size):
        assert expected_answers[i] == predicted_answers[i]


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name", "use_optimized_version"),
    ((8, 128, 5, "t5-small", True),),
)
def test_qa_squadv2(
    device, use_program_cache, batch_size, sequence_length, max_tokens, model_name, use_optimized_version
):
    qa_demo_dataset(
        device, use_program_cache, batch_size, sequence_length, max_tokens, model_name, use_optimized_version
    )


@pytest.mark.parametrize(
    "input_path",
    (("models/demos/grayskull/t5/demo/input_data_cg.json"),),
)
@pytest.mark.parametrize(("batch_size", "sequence_length", "max_tokens", "model_name"), ((8, 128, 64, "t5-small"),))
def test_summarize(input_path, device, use_program_cache, batch_size, sequence_length, max_tokens, model_name):
    expected_answers = {
        0: "",
        1: "climate change is one of the most pressing issues facing our planet today. burning of fossil fuels, deforestation, and industrial activities have led to a rise in greenhouse gas emissions. the impacts of climate change are already being felt, with rising sea levels, melting glaciers, and more frequent heat",
        2: "space exploration has captivated humanity for centuries. from the first moon landing to the exploration of Mars, humans have always been drawn to the mysteries of the cosmos. :: id=2=2=2=2=2=2=2=2=2=",
        3: "mental health is a fundamental aspect of overall well-being, yet it remains a neglected and stigmatized issue in many societies. mental illnesses, such as depression, anxiety, and schizophrenia, affect millions of people worldwide. access to mental health care and support services remains limited, particularly in low- and",
        4: "",
        5: "",
        6: "plastic waste poses a significant threat to marine life. pollution, overfishing, and habitat destruction threaten marine ecosystems. a chimio....... ",
        7: "sustainable agriculture is essential for ensuring food security, preserving natural resources, and mitigating the environmental impacts of agriculture. sustainable agriculture aims to meet the needs of the present without compromising the ability of future generations to meet their own needs....",
    }
    _, answers = summarize_demo(
        input_path, device, use_program_cache, batch_size, sequence_length, max_tokens, model_name
    )
    for i in range(batch_size):
        assert expected_answers[i] == answers[i]


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name"),
    ((8, 128, 64, "t5-small"),),
)
def test_summarize_dataset(device, use_program_cache, batch_size, sequence_length, max_tokens, model_name):
    summarize_dataset_demo(device, use_program_cache, batch_size, sequence_length, max_tokens, model_name)
