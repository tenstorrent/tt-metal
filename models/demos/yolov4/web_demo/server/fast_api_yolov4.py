# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import json
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from models.demos.yolov4.tests.yolov4_perfomant_webdemo import Yolov4Trace2CQ
import ttnn

import cv2
import numpy as np
import torch
import time

app = FastAPI(
    title="YOLOv4 object detection",
    description="Inference engine to detect objects in image.",
    version="0.0",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.on_event("startup")
async def startup():
    device_id = 0
    device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3096576, num_command_queues=2)
    ttnn.enable_program_cache(device)
    global model
    model = Yolov4Trace2CQ()
    model.initialize_yolov4_trace_2cqs_inference(device)


@app.on_event("shutdown")
async def shutdown():
    model.release_yolov4_trace_2cqs_inference()


def process_output(output):
    outs = []
    output = output
    cnt = 0
    for item in output:
        cnt = cnt + 1
        print("cnt: ", cnt)
        print("item is: ", item)
        output_i = [element.item() for element in item]
        # output_i = item[0]
        print("output_i as passed into process_output: ", output_i)
        # output_i = [element.item() for element in output_i]
        outs.append(output_i)
    print("\n\n\nouts before return is: ", outs)
    return outs
    # return [output]


def post_processing(img, conf_thresh, nms_thresh, output):
    print("output before post_processing: ", output)
    print()
    print("the length of output before post processing is: ", len(output))
    box_array = output[0]
    confs = output[1]
    print("confs: ", confs)
    t1 = time.time()

    box_array = np.array(box_array.to(torch.float32))
    confs = np.array(confs.to(torch.float32))

    num_classes = confs.shape[2]
    print("num_classes: ", num_classes)

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [
                            ll_box_array[k, 0],
                            ll_box_array[k, 1],
                            ll_box_array[k, 2],
                            ll_box_array[k, 3],
                            ll_max_conf[k],
                            ll_max_conf[k],
                            ll_max_id[k],
                        ]
                    )

        bboxes_batch.append(bboxes)

    t3 = time.time()

    print("-----------------------------------")
    print("       max and argmax : %f" % (t2 - t1))
    print("                  nms : %f" % (t3 - t2))
    print("Post processing total : %f" % (t3 - t1))
    print("-----------------------------------")

    print("bboxes_batch: ", bboxes_batch)
    return bboxes_batch


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


@app.post("/objdetection_v2")
async def objdetection_v2(file: UploadFile = File(...)):
    contents = await file.read()
    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)
    if type(image) == np.ndarray and len(image.shape) == 3:  # cv2 image
        image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    elif type(image) == np.ndarray and len(image.shape) == 4:
        image = torch.from_numpy(image).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()
    response = model.run_traced_inference(image)
    t2 = time.time()
    print("the inference on the sever side took: ", t2 - t1)
    # conf_thresh = 0.6
    conf_thresh = 0.1
    nms_thresh = 0.5

    boxes = post_processing(image, conf_thresh, nms_thresh, response)
    print("\n\n\nboxes after post processing: ", boxes)
    """
bboxes_batch:  [[[0.01953125, 0.2890625, 1.015625, 0.984375, 0.97265625, 0.97265625, 0], [0.41210938, 0.66015625, 0.67578125, 0.99609375, 0.30273438, 0.30273438, 41]]]



boxes after post processing:  [[[0.01953125, 0.2890625, 1.015625, 0.984375, 0.97265625, 0.97265625, 0], [0.41210938, 0.66015625, 0.67578125, 0.99609375, 0.30273438, 0.30273438, 41]]]
cnt:  1
output_i as passed into process_output:  [0.01953125, 0.2890625, 1.015625, 0.984375, 0.97265625, 0.97265625, 0]




    """
    output = boxes[0]
    # output = boxes
    try:
        output = process_output(output)
    except Exception as E:
        print("the Exception is: ", E)
        print("No objects detected!")
        return []
    t3 = time.time()
    print("the post processing to get the boexes took: ", t3 - t2)
    return output
