import pytest
from pathlib import Path
import ast


def model_location_generator_(rel_path):
    internal_weka_path = Path("/mnt/MLPerf")
    has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

    if has_internal_weka:
        return Path("/mnt/MLPerf") / rel_path
    else:
        return Path("/opt/tt-metal-models") / rel_path


@pytest.fixture(scope="session")
def model_location_generator():
    return model_location_generator_


@pytest.fixture
def imagenet_label_dict(model_location_generator):
    imagenet_class_labels_path = "tt_dnn-models/samples/imagenet_class_labels.txt"
    path = model_location_generator(imagenet_class_labels_path)
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels
