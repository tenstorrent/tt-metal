from pathlib import Path
import sys
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from python_api_testing.models.yolov7.reference.models.experimental import attempt_load


def get_yolov7_fused_cpu_model(model_location_generator):
    # Get model weights
    model_path = model_location_generator("models", model_subdir = "Yolo")
    weights = str(model_path / "yolov7.pt")

    # Load model
    model = attempt_load(weights, map_location="cpu")  # load FP32 model
    model = model.fuse().eval()

    return model
