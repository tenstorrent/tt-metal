"""
Model pytests for yolov5
"""
import pytest
import os
from PIL import Image
from yolov5_320 import startup_pybuda, clear_pybuda, YoloV5Handler

MLDATA = "/mnt/mldata"


@pytest.mark.skipif(
    not os.path.isdir(MLDATA),
    reason="Skipping test as we are not in a TT devtools environment.",
)
def test_model():
    startup_pybuda()
    model = YoloV5Handler()
    model.initialize()
    response = model.handle(Image.open("puppy.jpg"))
    print("the response is: ", response)
    assert response["labels"][0] == "dog"
    print("test_model PASSED")


if __name__ == "__main__":
    test_model()
