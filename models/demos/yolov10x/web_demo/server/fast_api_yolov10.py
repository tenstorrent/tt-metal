# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import ttnn
from models.demos.yolov10x.runner.performant_runner import YOLOv10PerformantRunner

# from models.experimental.yolo_common.yolo_web_demo.yolo_evaluation_utils import postprocess

app = FastAPI(
    title="YOLOv10 object detection",
    description="Inference engine to detect objects in image.",
    version="0.0",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)


def get_dispatch_core_config():
    # TODO: 11059 move dispatch_core_type to device_params when all tests are updated to not use WH_ARCH_YAML env flag
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)

    return dispatch_core_config


@app.on_event("startup")
async def startup():
    global model
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        print("WH_ARCH_YAML:", os.environ.get("WH_ARCH_YAML"))
        device_id = 0
        device = ttnn.CreateDevice(
            device_id,
            dispatch_core_config=get_dispatch_core_config(),
            l1_small_size=10 * 1024,
            trace_region_size=3211264,
            num_command_queues=2,
        )
        device.enable_program_cache()
        model = YOLOv10PerformantRunner(
            device,
            1,
            act_dtype=ttnn.bfloat8_b,
            weight_dtype=ttnn.bfloat8_b,
        )
    else:
        device_id = 0
        device = ttnn.CreateDevice(device_id, l1_small_size=10 * 1024, trace_region_size=3211264, num_command_queues=2)
        device.enable_program_cache()
        model = YOLOv10PerformantRunner(
            device,
            1,
            act_dtype=ttnn.bfloat8_b,
            weight_dtype=ttnn.bfloat8_b,
        )


@app.on_event("shutdown")
async def shutdown():
    model.release()


def process_output(output):
    outs = []
    output = output
    cnt = 0
    for item in output:
        cnt = cnt + 1
        output_i = [element.item() for element in item]
        outs.append(output_i)
    return outs


def postprocess(preds, img, orig_img):
    nc = 80
    max_det = 300
    args = {"conf": 0.5, "iou": 0.7, "agnostic_nms": False, "max_det": 300, "classes": None}
    preds = preds.permute(0, 2, 1)
    batch_size, anchors, _ = preds.shape
    boxes, scores = preds.split([4, nc], dim=-1)
    index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
    boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
    scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
    scores, index = scores.flatten(1).topk(min(max_det, anchors))
    i = torch.arange(batch_size)[..., None]
    preds = torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
    preds = non_max_suppression(
        preds,
        args["conf"],
        args["iou"],
        agnostic=args["agnostic_nms"],
        max_det=args["max_det"],
        classes=args["classes"],
    )

    results = []
    for pred in preds:
        # pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append({"boxes": {"xyxy": pred[:, :4], "conf": pred[:, -2], "cls": pred[:, -1]}})

    return results


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]

        boxes = x[:, :4] + c
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logger.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    return output


@app.post("/objdetection_v2")
async def objdetection_v2(file: UploadFile = File(...)):
    contents = await file.read()
    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image1 = np.array(image)
    if type(image1) == np.ndarray and len(image1.shape) == 3:  # cv2 image
        image = torch.from_numpy(image1).float().div(255.0).unsqueeze(0)
    elif type(image1) == np.ndarray and len(image1.shape) == 4:
        image = torch.from_numpy(image1).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()
    image = torch.permute(image, (0, 3, 1, 2))
    response = model.run(image)
    # print("response", response)
    r = ttnn.to_torch(response)
    print(r.shape)
    # names = load_coco_class_names()
    results = postprocess(r, image, image1)[0]
    t2 = time.time()
    logging.info("The inference on the sever side took: %.3f seconds", t2 - t1)
    conf_thresh = 0.6
    nms_thresh = 0.5

    output = []
    # print(results["boxes"]["xyxy"])
    for i in range(len(results["boxes"]["xyxy"])):
        output.append(
            torch.concat(
                (
                    results["boxes"]["xyxy"][i] / 640,
                    results["boxes"]["conf"][i].unsqueeze(0),
                    results["boxes"]["conf"][i].unsqueeze(0),
                    results["boxes"]["cls"][i].unsqueeze(0),
                ),
                dim=0,
            )
            .numpy()
            .tolist()
        )
    t3 = time.time()
    logging.info("The post-processing to get the boxes took: %.3f seconds", t3 - t2)

    return output
