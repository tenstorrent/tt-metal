# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import torch
import torchvision.models as models

import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from models.common.utility_functions import comp_pcc


def _conv2d(x_tt, w, device, bs, h, w_in, in_c, out_c, ksize, stride, pad):
    if isinstance(w, torch.Tensor):
        w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    out = ttnn.conv2d(
        input_tensor=x_tt,
        weight_tensor=w,
        in_channels=in_c,
        out_channels=out_c,
        device=device,
        batch_size=bs,
        input_height=h,
        input_width=w_in,
        kernel_size=ksize,
        stride=stride,
        padding=pad,
        dilation=(1, 1),
        groups=1,
    )

    out = ttnn.to_torch(out)
    out_h = (h + 2 * pad[0] - ksize[0]) // stride[0] + 1
    out_w = (w_in + 2 * pad[1] - ksize[1]) // stride[1] + 1
    return out.reshape(bs, out_h, out_w, out_c).permute(0, 3, 1, 2).contiguous()


def _bn(x, w, b, mean, var, relu=True, eps=1e-5):
    x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + eps)
    x = x * w.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
    return torch.relu(x) if relu else x


def _to_tt(x_torch, device):
    return ttnn.from_torch(
        x_torch.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _block_weights(block):
    """pull conv+bn weights out of a torchvision bottleneck block"""
    ds = block.downsample
    return dict(
        c1w=block.conv1.weight,
        b1=(block.bn1.weight, block.bn1.bias, block.bn1.running_mean, block.bn1.running_var),
        c2w=block.conv2.weight,
        b2=(block.bn2.weight, block.bn2.bias, block.bn2.running_mean, block.bn2.running_var),
        c3w=block.conv3.weight,
        b3=(block.bn3.weight, block.bn3.bias, block.bn3.running_mean, block.bn3.running_var),
        ds_cw=ds[0].weight if ds else None,
        ds_bn=(ds[1].weight, ds[1].bias, ds[1].running_mean, ds[1].running_var) if ds else None,
    )


def bottleneck(x, bw, device, bs, in_c, mid_c, out_c, stride):
    h, w = x.shape[2], x.shape[3]
    identity = x

    out = _conv2d(_to_tt(x, device), bw["c1w"], device, bs, h, w, in_c, mid_c, (1, 1), (1, 1), (0, 0))
    out = _bn(out, *bw["b1"])

    out = _conv2d(
        _to_tt(out, device),
        bw["c2w"],
        device,
        bs,
        out.shape[2],
        out.shape[3],
        mid_c,
        mid_c,
        (3, 3),
        (stride, stride),
        (1, 1),
    )
    out = _bn(out, *bw["b2"])

    out = _conv2d(
        _to_tt(out, device), bw["c3w"], device, bs, out.shape[2], out.shape[3], mid_c, out_c, (1, 1), (1, 1), (0, 0)
    )
    out = _bn(out, *bw["b3"], relu=False)

    if bw["ds_cw"] is not None:
        identity = _conv2d(
            _to_tt(identity, device), bw["ds_cw"], device, bs, h, w, in_c, out_c, (1, 1), (stride, stride), (0, 0)
        )
        identity = _bn(identity, *bw["ds_bn"], relu=False)

    return torch.relu(out + identity)


def resnet50_backbone(x, model, device, bs=1):
    # stem
    out = _conv2d(_to_tt(x, device), model.conv1.weight, device, bs, 224, 224, 3, 64, (7, 7), (2, 2), (3, 3))
    out = _bn(out, model.bn1.weight, model.bn1.bias, model.bn1.running_mean, model.bn1.running_var)
    out = torch.nn.functional.max_pool2d(out, 3, stride=2, padding=1)

    # layer configs: (layer, in_c per block, mid_c, out_c, first_stride)
    layers = [
        (model.layer1, [64, 256, 256], 64, 256, 1),
        (model.layer2, [256, 512, 512], 128, 512, 2),
        (model.layer3, [512, 1024, 1024], 256, 1024, 2),
        (model.layer4, [1024, 2048, 2048], 512, 2048, 2),
    ]

    features = []
    for layer, in_cs, mid_c, out_c, first_stride in layers:
        for i, block in enumerate(layer):
            stride = first_stride if i == 0 else 1
            out = bottleneck(
                out, _block_weights(block), device, bs, in_cs[min(i, len(in_cs) - 1)], mid_c, out_c, stride
            )
        features.append(out)

    return features  # [c2, c3, c4, c5]


def test_full_resnet50(device):
    torch.manual_seed(0)

    resnet = models.resnet50(pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        stem = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(x))))
        refs = [resnet.layer1(stem)]
        refs.append(resnet.layer2(refs[-1]))
        refs.append(resnet.layer3(refs[-1]))
        refs.append(resnet.layer4(refs[-1]))

    feats = resnet50_backbone(x, resnet, device)

    names = ["c2", "c3", "c4", "c5"]
    for name, ref, feat in zip(names, refs, feats):
        pcc = comp_pcc(ref, feat)
        pcc = pcc[1] if isinstance(pcc, tuple) else pcc
        print(f"{name} pcc: {pcc:.4f}")
        assert pcc >= 0.98, f"{name} pcc too low: {pcc:.4f}"

    print("resnet50 backbone ok")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        test_full_resnet50(device)
    finally:
        ttnn.close_device(device)
