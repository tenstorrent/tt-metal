# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.resnet.demo.demo import run_resnet_inference, run_resnet_imagenet_inference


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((20, "models/demos/resnet/demo/images/"),),
)
def test_demo_sample(device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator):
    expected_prediction = [
        "ruffed grouse, partridge, Bonasa umbellus",
        "hay",
        "quail",
        "Tibetan mastiff",
        "Border terrier",
        "brown bear, bruin, Ursus arctos",
        "soap dispenser",
        "vestment",
        "gorilla, Gorilla gorilla",
        "sports car, sport car",
        "toilet tissue, toilet paper, bathroom tissue",
        "velvet",
        "lakeside, lakeshore",
        "pirate, pirate ship",
        "bee eater",
        "Border collie",
        "turnstile",
        "cardoon",
        "Pembroke, Pembroke Welsh corgi",
        "Christmas stocking",
    ]
    measurements, predictions = run_resnet_inference(
        batch_size, input_loc, imagenet_label_dict, device, model_location_generator
    )

    for predicted, expected in zip(predictions, expected_prediction):
        assert predicted == expected, "Some predictions are not what we expected!"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((20, 250),),
)
def test_demo_imagenet(batch_size, iterations, imagenet_label_dict, model_location_generator, device):
    run_resnet_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, device)
