# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
from models.demos.resnet.demo.demo import run_resnet_inference

@pytest.mark.parametrize(
    "batch_size, input_loc",
    (
        (8, "models/demos/resnet/demo/images/"),
    ),
)
def test_demo_sample(
    use_program_cache,
    batch_size,
    input_loc,
    imagenet_label_dict,
    device,
):
    expected_prediction = ['ruffed grouse, partridge, Bonasa umbellus', 'hay', 'lab coat, laboratory coat', 'Tibetan mastiff', 'Border terrier', 'quail', 'brown bear, bruin, Ursus arctos', 'space heater']
    measurements, predictions = run_resnet_inference(
        batch_size,
        input_loc,
        imagenet_label_dict,
        device,
    )

    for predicted, expected in zip(predictions, expected_prediction):
        assert predicted == expected, "Some predictions are not what we expected!"
