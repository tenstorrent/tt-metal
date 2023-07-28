import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import tt_lib
import torch
from torch import nn
from loguru import logger
from python_api_testing.models.yolov3.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov3.tt.yolov3_detection_model import TtDetectionModel
from python_api_testing.models.yolov3.reference.models.common import autopad
from python_api_testing.models.yolov3.reference.utils.dataloaders import LoadImages
from python_api_testing.models.yolov3.reference.utils.general import check_img_size
from utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, profiler
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report

BATCH_SIZE = 1


def test_perf(use_program_cache, model_location_generator):
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "yolov3-fused"

    # Initialize the device
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # Load yolo
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(
        weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False
    )
    state_dict = reference_model.state_dict()
    reference_model = reference_model.model

    tt_module = TtDetectionModel(
        cfg=model_config_path,
        state_dict=state_dict,
        base_address="model.model",
        device=device,
    )

    with torch.no_grad():
        tt_module.eval()
        reference_model.eval()

        # Load data
        stride = max(int(max(reference_model.stride)), 32)  # model stride
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

        path, im, _, _, _ = next(iter(dataset))
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        tt_im = torch2tt_tensor(im, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

        profiler.start(cpu_key)
        pt_out = reference_model(im)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_out = tt_module(tt_im)
        profiler.end(first_key)

        profiler.start(second_key)
        tt_out = tt_module(tt_im)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        "yolov3", BATCH_SIZE, first_iter_time, second_iter_time, comments, cpu_time
    )
    tt_lib.device.CloseDevice(device)
