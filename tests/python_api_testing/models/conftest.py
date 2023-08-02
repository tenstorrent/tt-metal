import pytest
from PIL import Image
import torchvision.transforms as transforms
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
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@pytest.fixture
def imagenet_sample_input(model_location_generator):
    path = "models/sample_data/ILSVRC2012_val_00048736.JPEG"

    im = Image.open(path)
    im = im.resize((224, 224))
    return transforms.ToTensor()(im).unsqueeze(0)


@pytest.fixture
def mnist_sample_input(model_location_generator):
    path = "models/sample_data/torchvision_mnist_digit_7.jpg"
    im = Image.open(path)
    return im


@pytest.fixture
def iam_ocr_sample_input(model_location_generator):
    path = "models/sample_data/iam_ocr_image.jpg"
    im = Image.open(path)
    return im


@pytest.fixture
def hf_cat_image_sample_input(model_location_generator):
    path = "models/sample_data/huggingface_cat_image.jpg"
    im = Image.open(path)
    return im
