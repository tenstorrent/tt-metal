# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time

import numpy as np
import torch

# from fastapi import FastAPI, File, UploadFile
from PIL import Image

import ttnn

# from models.demos.yolov9c.reference import yolov9c
from models.demos.yolov8s.tests.yolov8s_e2e_performant import Yolov8sTrace2CQ
from models.demos.yolov9c.demo.demo_utils import load_coco_class_names

# from models.demos.yolov9c.tt import ttnn_yolov9c
from models.experimental.yolo_evaluation.yolo_evaluation_utils import postprocess

model = None


def get_dispatch_core_config():
    # TODO: 11059 move dispatch_core_type to device_params when all tests are updated to not use WH_ARCH_YAML env flag
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)

    return dispatch_core_config


def startup():
    global model
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        print("WH_ARCH_YAML:", os.environ.get("WH_ARCH_YAML"))
        device_id = 0
        device = ttnn.CreateDevice(
            device_id,
            dispatch_core_config=get_dispatch_core_config(),
            l1_small_size=24576,
            trace_region_size=3211264,
            num_command_queues=2,
        )
        device.enable_program_cache()
        model = Yolov8sTrace2CQ()
    else:
        device_id = 0
        device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3211264, num_command_queues=2)
        device.enable_program_cache()
        model = Yolov8sTrace2CQ()
    model.initialize_yolov8s_trace_2cqs_inference(device, 1)


# def shutdown():
#    model.release()


def process_output(output):
    outs = []
    output = output
    cnt = 0
    for item in output:
        cnt = cnt + 1
        output_i = [element.item() for element in item]
        outs.append(output_i)
    return outs


def post_processing(img, conf_thresh, nms_thresh, output):
    box_array = output[0]
    confs = output[1]

    box_array = np.array(box_array.to(torch.float32))
    confs = np.array(confs.to(torch.float32))

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

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


def objdetection_v2():
    # contents = await file.read()
    # Load and convert the image to RGB
    global model
    image = Image.open("/home/ttuser/ssinghal/tt-metal/models/sample_data/huggingface_cat_image.jpg").convert("RGB")
    padding_size = (640, 640)

    # Create a new image with the desired size and black background
    new_img = Image.new("RGB", padding_size, "black")

    # Calculate position to paste the original image
    position = ((padding_size[0] - image.width) // 2, (padding_size[1] - image.height) // 2)

    # Paste the original image onto the new image
    new_img.paste(image, position)

    # new_img.save(output_path)
    image1 = np.array(new_img)
    print(image1.shape)
    if type(image1) == np.ndarray and len(image1.shape) == 3:  # cv2 image
        image = torch.from_numpy(image1).float().div(255.0).unsqueeze(0)
    elif type(image1) == np.ndarray and len(image1.shape) == 4:
        image = torch.from_numpy(image1).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    image = torch.permute(image, (0, 3, 1, 2))
    t1 = time.time()
    # ttnn_im = ttnn.from_torch(image, dtype=ttnn.bfloat16)
    response = model.run(image)
    response = ttnn.to_torch(response)
    names = load_coco_class_names()
    results = postprocess(response, image, image1, names=names)[0]
    print(results)
    t2 = time.time()
    logging.info("The inference on the sever side took: %.3f seconds", t2 - t1)
    conf_thresh = 0.8
    nms_thresh = 0.6

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
    print(output)
    # boxes = post_processing(image, conf_thresh, nms_thresh, response)
    # output = boxes[0]
    # output = boxes
    # try:
    #    #output = process_output(output)
    # except Exception as E:
    #    print("the Exception is: ", E)
    #    print("No objects detected!")
    #    return []
    t3 = time.time()
    logging.info("The post-processing to get the boxes took: %.3f seconds", t3 - t2)

    return output


startup()
objdetection_v2()
shutdown()
