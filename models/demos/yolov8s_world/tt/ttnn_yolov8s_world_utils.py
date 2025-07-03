# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from ttnn.model_preprocessing import (
    ParameterDict,
    ParameterList,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    preprocess_linear_weight,
)

import ttnn
from models.demos.yolov8s_world.reference.yolov8s_world import (
    SPPF,
    C2f,
    C2fAttn,
    Conv,
    ImagePoolingAttn,
    WorldDetect,
    WorldModel,
)


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["projections"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_download(file, repo="ultralytics/assets"):
    tests = Path(__file__).parent.parent
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolov8s-world.pt"
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"
        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"
        except Exception as e:
            logger.info(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                logger.info(f"ERROR: Download failure for {msg}")
            else:
                logger.info(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        logger.info(f"Loading weights from: {weight_path}")
        ckpt = torch.load(weight_path, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=64) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int = 8, grid_cols: int = 8):
    rows = num_cores // grid_cols
    assert rows <= grid_rows, "Not enough cores for specified core grid"
    ranges = []
    if rows != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, rows - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, rows),
                ttnn.CoreCoord(remainder - 1, rows),
            )
        )
    return ttnn.CoreRangeSet({*ranges})


def to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT):
    if x.get_layout() != layout:
        x = ttnn.to_layout(x, layout)
    return x


def sharded_concat(
    input_tensors, num_cores=64, dim=3, skip_s2i=False
):  # expected input tensors to be in fp16, RM, same (h*w)
    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_configs = []

    for i in range(len(input_tensors)):
        input_sharded_memory_config = ttnn.create_sharded_memory_config(
            (shard_height, input_tensors[i].shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_configs.append(input_sharded_memory_config)

    sharded_inputs = [
        ttnn.to_memory_config(tensor, config) for tensor, config in zip(input_tensors, input_sharded_memory_configs)
    ]

    total_width = sum(tensor.shape[-1] for tensor in input_tensors)
    out_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, total_width),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    output = ttnn.concat(sharded_inputs, dim, memory_config=out_sharded_memory_config)
    if not skip_s2i:
        output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


def concat(tensors, dim=-1, use_sharded_concat=True, skip_s2i=False):
    if use_sharded_concat:
        processed_tensors = [
            ttnn.to_dtype(to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT), ttnn.bfloat16) for tensor in tensors
        ]
        return sharded_concat(processed_tensors, dim=dim, skip_s2i=skip_s2i)
    else:
        return ttnn.concat([*tensors], dim=dim, memory_config=ttnn.L1_MEMORY_CONFIG)


def ttnn_decode_bboxes(device, distance, anchor_points, xywh=True, dim=1):
    distance = ttnn.to_layout(distance, ttnn.ROW_MAJOR_LAYOUT)
    # lt, rb = ttnn.split(distance, 2, 1, memory_config=ttnn.L1_MEMORY_CONFIG)  # if done in tile : tt-metal issue #17017

    lt = distance[:, : distance.shape[1] // 2, :]
    rb = distance[:, distance.shape[1] // 2 :, :]

    lt = ttnn.to_layout(lt, ttnn.TILE_LAYOUT)
    rb = ttnn.to_layout(rb, ttnn.TILE_LAYOUT)

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = x1y1 + x2y2
        c_xy = ttnn.div(c_xy, 2)
        wh = x2y2 - x1y1
        return ttnn.concat([c_xy, wh], 1, memory_config=ttnn.L1_MEMORY_CONFIG)


def make_anchors(device, feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i]
        sx = torch.arange(end=w) + grid_cell_offset
        sy = torch.arange(end=h) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))

    a = torch.cat(anchor_points).transpose(0, 1).unsqueeze(0)
    b = torch.cat(stride_tensor).transpose(0, 1)

    return (
        ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
        ttnn.from_torch(
            b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
    )


def fold_batch_norm2d_into_conv2d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")

    weight = conv.weight
    bias = conv.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    return weight, bias


def fold_batch_norm2d_into_conv2d_split(conv, bn):
    weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
    bias = bias.reshape(1, 1, 1, -1)
    chunk_size = bias.shape[-1] // 2
    return (
        weight[:chunk_size, :, :, :],
        bias[:, :, :, :chunk_size],
        weight[chunk_size:, :, :, :],
        bias[:, :, :, chunk_size:],
    )


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, WorldModel):
            parameters["model"] = {}
            for index, child in enumerate(model.model):
                parameters["model"][index] = {}
                if isinstance(child, Conv):
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.conv, child.bn)
                    parameters["model"][index]["conv"] = {}
                    parameters["model"][index]["conv"]["weight"] = ttnn.from_torch(conv_weight)
                    parameters["model"][index]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )
                elif isinstance(child, C2f):
                    parameters["model"][index]["cv1_a"] = {}
                    conv_weight_a, conv_bias_a, conv_weight_b, conv_bias_b = fold_batch_norm2d_into_conv2d_split(
                        child.cv1.conv, child.cv1.bn
                    )
                    parameters["model"][index]["cv1_a"]["conv"] = {}
                    parameters["model"][index]["cv1_a"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight_a,
                    )
                    parameters["model"][index]["cv1_a"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias_a,
                    )
                    parameters["model"][index]["cv1_b"] = {}
                    parameters["model"][index]["cv1_b"]["conv"] = {}
                    parameters["model"][index]["cv1_b"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight_b,
                    )
                    parameters["model"][index]["cv1_b"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias_b,
                    )

                    parameters["model"][index]["cv2"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv2.conv, child.cv2.bn)
                    parameters["model"][index]["cv2"]["conv"] = {}
                    parameters["model"][index]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight,
                    )
                    parameters["model"][index]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )

                    parameters["model"][index]["m"] = {}
                    for index_2, child_2 in enumerate(child.m):
                        parameters["model"][index]["m"][index_2] = {}

                        parameters["model"][index]["m"][index_2]["cv1"] = {}
                        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.cv1.conv, child_2.cv1.bn)
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"] = {}
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"]["weight"] = ttnn.from_torch(
                            conv_weight,
                        )
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"]["bias"] = ttnn.from_torch(
                            conv_bias.reshape(1, 1, 1, -1),
                        )

                        parameters["model"][index]["m"][index_2]["cv2"] = {}
                        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.cv2.conv, child_2.cv2.bn)
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"] = {}
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                            conv_weight,
                        )
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                            conv_bias.reshape(1, 1, 1, -1),
                        )
                elif isinstance(child, SPPF):
                    parameters["model"][index]["cv1"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv1.conv, child.cv1.bn)
                    parameters["model"][index]["cv1"]["conv"] = {}
                    parameters["model"][index]["cv1"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight,
                    )
                    parameters["model"][index]["cv1"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )

                    parameters["model"][index]["cv2"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv2.conv, child.cv2.bn)
                    parameters["model"][index]["cv2"]["conv"] = {}
                    parameters["model"][index]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight,
                    )
                    parameters["model"][index]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )
                elif isinstance(child, C2fAttn):
                    parameters["model"][index]["cv1_a"] = {}
                    conv_weight_a, conv_bias_a, conv_weight_b, conv_bias_b = fold_batch_norm2d_into_conv2d_split(
                        child.cv1.conv, child.cv1.bn
                    )
                    parameters["model"][index]["cv1_a"]["conv"] = {}
                    parameters["model"][index]["cv1_a"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight_a,
                    )
                    parameters["model"][index]["cv1_a"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias_a,
                    )
                    parameters["model"][index]["cv1_b"] = {}
                    parameters["model"][index]["cv1_b"]["conv"] = {}
                    parameters["model"][index]["cv1_b"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight_b,
                    )
                    parameters["model"][index]["cv1_b"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias_b,
                    )

                    parameters["model"][index]["cv2"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv2.conv, child.cv2.bn)
                    parameters["model"][index]["cv2"]["conv"] = {}
                    parameters["model"][index]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight,
                    )
                    parameters["model"][index]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )

                    parameters["model"][index]["m"] = {}
                    for index_2, child_2 in enumerate(child.m):
                        parameters["model"][index]["m"][index_2] = {}

                        parameters["model"][index]["m"][index_2]["cv1"] = {}
                        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.cv1.conv, child_2.cv1.bn)
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"] = {}
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"]["weight"] = ttnn.from_torch(
                            conv_weight,
                        )
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"]["bias"] = ttnn.from_torch(
                            conv_bias.reshape(1, 1, 1, -1),
                        )

                        parameters["model"][index]["m"][index_2]["cv2"] = {}
                        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.cv2.conv, child_2.cv2.bn)
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"] = {}
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                            conv_weight,
                        )
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                            conv_bias.reshape(1, 1, 1, -1),
                        )
                    parameters["model"][index]["attn"] = {}
                    if child.attn.ec == None:
                        parameters["model"][index]["attn"]["ec"] = {}
                    else:
                        assert False, "give support for Ec"
                    parameters["model"][index]["attn"]["bias"] = ttnn.from_torch(
                        child.attn.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        device=device,
                    )
                    parameters["model"][index]["attn"]["gl"] = {}
                    parameters["model"][index]["attn"]["gl"]["weight"] = preprocess_linear_weight(
                        child.attn.gl.weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["attn"]["gl"]["bias"] = preprocess_linear_bias(
                        child.attn.gl.bias, dtype=ttnn.bfloat8_b
                    )

                    parameters["model"][index]["attn"]["proj_conv"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                        child.attn.proj_conv.conv, child.attn.proj_conv.bn
                    )
                    parameters["model"][index]["attn"]["proj_conv"]["conv"] = {}
                    parameters["model"][index]["attn"]["proj_conv"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight,
                    )
                    parameters["model"][index]["attn"]["proj_conv"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )
                elif isinstance(child, ImagePoolingAttn):
                    ##query
                    parameters["model"][index]["query"] = {}
                    # layernorm
                    parameters["model"][index]["query"][0] = {}
                    parameters["model"][index]["query"][0]["weight"] = preprocess_layernorm_parameter(
                        child.query[0].weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["query"][0]["bias"] = preprocess_layernorm_parameter(
                        child.query[0].bias, dtype=ttnn.bfloat8_b
                    )

                    # linear
                    parameters["model"][index]["query"][1] = {}
                    parameters["model"][index]["query"][1]["weight"] = preprocess_linear_weight(
                        child.query[1].weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["query"][1]["bias"] = preprocess_linear_bias(
                        child.query[1].bias, dtype=ttnn.bfloat8_b
                    )

                    ##key
                    parameters["model"][index]["key"] = {}
                    # layernorm
                    parameters["model"][index]["key"][0] = {}
                    parameters["model"][index]["key"][0]["weight"] = preprocess_layernorm_parameter(
                        child.key[0].weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["key"][0]["bias"] = preprocess_layernorm_parameter(
                        child.key[0].bias, dtype=ttnn.bfloat8_b
                    )

                    # linear
                    parameters["model"][index]["key"][1] = {}
                    parameters["model"][index]["key"][1]["weight"] = preprocess_linear_weight(
                        child.key[1].weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["key"][1]["bias"] = preprocess_linear_bias(
                        child.key[1].bias, dtype=ttnn.bfloat8_b
                    )

                    ##value
                    parameters["model"][index]["value"] = {}
                    # layernorm
                    parameters["model"][index]["value"][0] = {}
                    parameters["model"][index]["value"][0]["weight"] = preprocess_layernorm_parameter(
                        child.value[0].weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["value"][0]["bias"] = preprocess_layernorm_parameter(
                        child.value[0].bias, dtype=ttnn.bfloat8_b
                    )

                    # linear
                    parameters["model"][index]["value"][1] = {}
                    parameters["model"][index]["value"][1]["weight"] = preprocess_linear_weight(
                        child.value[1].weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["value"][1]["bias"] = preprocess_linear_bias(
                        child.value[1].bias, dtype=ttnn.bfloat8_b
                    )

                    # proj
                    parameters["model"][index]["proj"] = {}
                    parameters["model"][index]["proj"]["weight"] = preprocess_linear_weight(
                        child.proj.weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["model"][index]["proj"]["bias"] = preprocess_linear_bias(
                        child.proj.bias, dtype=ttnn.bfloat8_b
                    )

                    # projections
                    parameters["model"][index]["projections"] = {}

                    parameters["model"][index]["projections"][0] = {}
                    parameters["model"][index]["projections"][0]["weight"] = ttnn.from_torch(
                        child.projections[0].weight,
                    )
                    parameters["model"][index]["projections"][0]["bias"] = ttnn.from_torch(
                        child.projections[0].bias.reshape(1, 1, 1, -1),
                    )

                    parameters["model"][index]["projections"][1] = {}
                    parameters["model"][index]["projections"][1]["weight"] = ttnn.from_torch(
                        child.projections[1].weight,
                    )
                    parameters["model"][index]["projections"][1]["bias"] = ttnn.from_torch(
                        child.projections[1].bias.reshape(1, 1, 1, -1),
                    )

                    parameters["model"][index]["projections"][2] = {}
                    parameters["model"][index]["projections"][2]["weight"] = ttnn.from_torch(
                        child.projections[2].weight,
                    )
                    parameters["model"][index]["projections"][2]["bias"] = ttnn.from_torch(
                        child.projections[2].bias.reshape(1, 1, 1, -1),
                    )

                elif isinstance(child, WorldDetect):
                    parameters["model"][index]["cv2"] = {}
                    for i_1, child_1 in enumerate(child.cv2):
                        parameters["model"][index]["cv2"][i_1] = {}
                        for i_2, child_2 in enumerate(child_1):
                            parameters["model"][index]["cv2"][i_1][i_2] = {}
                            if i_2 == 2:
                                parameters["model"][index]["cv2"][i_1][i_2]["weight"] = ttnn.from_torch(
                                    child_2.weight,
                                )
                                parameters["model"][index]["cv2"][i_1][i_2]["bias"] = ttnn.from_torch(
                                    child_2.bias.reshape(1, 1, 1, -1),
                                )
                            else:
                                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.conv, child_2.bn)
                                parameters["model"][index]["cv2"][i_1][i_2]["conv"] = {}
                                parameters["model"][index]["cv2"][i_1][i_2]["conv"]["weight"] = ttnn.from_torch(
                                    conv_weight,
                                )
                                parameters["model"][index]["cv2"][i_1][i_2]["conv"]["bias"] = ttnn.from_torch(
                                    conv_bias.reshape(1, 1, 1, -1),
                                )

                    parameters["model"][index]["cv3"] = {}
                    for i_1, child_1 in enumerate(child.cv3):
                        parameters["model"][index]["cv3"][i_1] = {}
                        for i_2, child_2 in enumerate(child_1):
                            parameters["model"][index]["cv3"][i_1][i_2] = {}
                            if i_2 == 2:
                                parameters["model"][index]["cv3"][i_1][i_2]["weight"] = ttnn.from_torch(
                                    child_2.weight,
                                )
                                parameters["model"][index]["cv3"][i_1][i_2]["bias"] = ttnn.from_torch(
                                    child_2.bias.reshape(1, 1, 1, -1),
                                )
                            else:
                                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.conv, child_2.bn)
                                parameters["model"][index]["cv3"][i_1][i_2]["conv"] = {}
                                parameters["model"][index]["cv3"][i_1][i_2]["conv"]["weight"] = ttnn.from_torch(
                                    conv_weight,
                                )
                                parameters["model"][index]["cv3"][i_1][i_2]["conv"]["bias"] = ttnn.from_torch(
                                    conv_bias.reshape(1, 1, 1, -1),
                                )
                    parameters["model"][index]["dfl"] = {}
                    parameters["model"][index]["dfl"]["conv"] = {}
                    parameters["model"][index]["dfl"]["conv"]["weight"] = ttnn.from_torch(
                        child.dfl.conv.weight,
                    )
                    if child.dfl.conv.bias == None:
                        parameters["model"][index]["dfl"]["conv"]["bias"] = None

                    parameters["model"][index]["cv4"] = {}
                    for i_1, child_1 in enumerate(child.cv4):
                        parameters["model"][index]["cv4"][i_1] = {}
                        parameters["model"][index]["cv4"][i_1]["bias"] = ttnn.from_torch(
                            child_1.bias,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                            device=device,
                        )
                        parameters["model"][index]["cv4"][i_1]["logit_scale"] = ttnn.from_torch(
                            child_1.logit_scale,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                            device=device,
                        )

                    strides = [8, 16, 32]

                    feats = [
                        (640 // 8, 640 // 8),
                        (640 // 16, 640 // 16),
                        (640 // 32, 640 // 32),
                    ]  # value depend on res

                    anchors, strides = make_anchors(
                        device, feats, strides
                    )  # Optimization: Processing make anchors outside model run

                    parameters["model"][index]["anchors"] = anchors
                    parameters["model"][index]["strides"] = strides

            parameters["txt_feats"] = ttnn.from_torch(
                model.txt_feats,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                device=device,  # keeping the dtype as bfloat16 instead of float32 affects demo result
            )

        return parameters

    return custom_preprocessor


def tt_adaptive_to_max_pool2d(input_shape, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    input_height, input_width = input_shape[1], input_shape[2]
    output_height, output_width = output_size

    # Check if dimensions are valid
    if input_height < output_height or input_width < output_width:
        raise ValueError("Output size cannot be larger than input size for max pooling")

    # Calculate stride (might be floating point)
    stride_h_float = input_height / output_height
    stride_w_float = input_width / output_width

    # Round down stride to integer
    stride_h = math.floor(stride_h_float)
    stride_w = math.floor(stride_w_float)

    # Ensure stride is at least 1
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    # Calculate kernel size
    kernel_h = input_height - (output_height - 1) * stride_h
    kernel_w = input_width - (output_width - 1) * stride_w

    # Handle case where kernel size might be too large
    if kernel_h > input_height:
        kernel_h = input_height
    if kernel_w > input_width:
        kernel_w = input_width

    # Calculate if this is an exact conversion
    is_exact = (
        stride_h_float == stride_h
        and stride_w_float == stride_w
        and input_height == (output_height - 1) * stride_h + kernel_h
        and input_width == (output_width - 1) * stride_w + kernel_w
    )

    message = ""
    if not is_exact:
        message = (
            "Note: This is an approximation. For non-integer stride ratios, "
            "AdaptiveMaxPool2d uses a more complex logic with varying kernel sizes."
        )

    return kernel_w, stride_h, 0, message


def ttnn_custom_normalize(x, dim, device):
    # Convert input to tiled layout
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Square the tensor using multiply
    x_squared = ttnn.multiply(x, x)

    # Sum along the specified dimension
    if dim == 1:
        sum_squared = ttnn.sum(x_squared, dim=1, keepdim=True)
    else:
        sum_squared = ttnn.sum(x_squared, dim=-1, keepdim=True)

    # Add small epsilon and calculate square root
    sum_squared = ttnn.add(sum_squared, 1e-12)
    norm = ttnn.sqrt(sum_squared)

    # Create a tensor of ones with the same shape as x
    ones = ttnn.ones_like(
        tensor=x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Multiply norm by ones to match input shape
    norm_expanded = ttnn.multiply(norm, ones)

    # Divide input by expanded norm
    normalized = ttnn.divide(x, norm_expanded)

    return normalized
