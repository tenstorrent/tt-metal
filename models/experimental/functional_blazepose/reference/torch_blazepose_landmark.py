# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2
from models.experimental.functional_blazepose.reference.torch_blazepose import blazeblock


def basepose_land_mark(x, parameters):
    batch = x.shape[0]
    if batch == 0:
        return (
            torch.zeros((0,)),
            torch.zeros((0, 31, 4)),
            torch.zeros((0, 128, 128)),
        )

    x = F.pad(x, (0, 1, 0, 1), "constant", 0)

    conv = nn.Conv2d(
        in_channels=3,
        out_channels=24,
        kernel_size=3,
        stride=2,
        padding=0,
    )
    conv.weight = parameters.backbone1[0].weight
    conv.bias = parameters.backbone1[0].bias
    x = conv(x)
    relu = nn.ReLU(inplace=True)
    x = relu(x)

    for i in range(2, 4):
        x = blazeblock(x, 24, 24, 3, 1, 1, False, parameters.backbone1, i)

    for i in range(0, 4):
        if i == 0:
            y = blazeblock(x, 24, 48, 3, 2, 0, False, parameters.backbone2, i)
        else:
            y = blazeblock(y, 48, 48, 3, 1, 1, False, parameters.backbone2, i)

    for i in range(0, 5):
        if i == 0:
            z = blazeblock(y, 48, 96, 3, 2, 0, False, parameters.backbone3, i)
        else:
            z = blazeblock(z, 96, 96, 3, 1, 1, False, parameters.backbone3, i)

    for i in range(0, 6):
        if i == 0:
            w = blazeblock(z, 96, 192, 3, 2, 0, False, parameters.backbone4, i)
        else:
            w = blazeblock(w, 192, 192, 3, 1, 1, False, parameters.backbone4, i)

    for i in range(0, 7):
        if i == 0:
            v = blazeblock(w, 192, 288, 3, 2, 0, False, parameters.backbone5, i)
        else:
            v = blazeblock(v, 288, 288, 3, 1, 1, False, parameters.backbone5, i)

    # No problem above
    up_conv = nn.Conv2d(288, 288, 3, 1, 1, groups=288)
    up_conv.weight = parameters.up1[0].weight
    up_conv.bias = parameters.up1[0].bias
    v1 = up_conv(v)

    up_conv = nn.Conv2d(288, 48, 1)
    up_conv.weight = parameters.up1[1].weight
    up_conv.bias = parameters.up1[1].bias
    v1 = up_conv(v1)
    act = nn.ReLU(inplace=True)
    v1 = act(v1)

    up2_conv = nn.Conv2d(192, 192, 3, 1, 1, groups=192)
    up2_conv.weight = parameters.up2[0].weight
    up2_conv.bias = parameters.up2[0].bias
    w1 = up2_conv(w)

    up2_conv = nn.Conv2d(192, 48, 1)
    up2_conv.weight = parameters.up2[1].weight
    up2_conv.bias = parameters.up2[1].bias
    w1 = up2_conv(w1)
    act = nn.ReLU(inplace=True)
    w1 = act(w1)

    w1 = w1 + F.interpolate(v1, scale_factor=2, mode="bilinear")

    up3_conv = nn.Conv2d(96, 96, 3, 1, 1, groups=96)
    up3_conv.weight = parameters.up3[0].weight
    up3_conv.bias = parameters.up3[0].bias
    z1 = up3_conv(z)

    up3_conv = nn.Conv2d(96, 48, 1)
    up3_conv.weight = parameters.up3[1].weight
    up3_conv.bias = parameters.up3[1].bias
    z1 = up3_conv(z1)
    act = nn.ReLU(inplace=True)
    z1 = act(z1)
    z1 = z1 + F.interpolate(w1, scale_factor=2, mode="bilinear")

    up4_conv = nn.Conv2d(48, 48, 3, 1, 1, groups=48)
    up4_conv.weight = parameters.up4[0].weight
    up4_conv.bias = parameters.up4[0].bias
    y1 = up4_conv(y)

    up4_conv = nn.Conv2d(48, 48, 1)
    up4_conv.weight = parameters.up4[1].weight
    up4_conv.bias = parameters.up4[1].bias
    y1 = up4_conv(y1)
    act = nn.ReLU(inplace=True)
    y1 = act(y1)

    y1 = y1 + F.interpolate(z1, scale_factor=2, mode="bilinear")

    up9_conv = nn.Conv2d(24, 24, 3, 1, 1, groups=24)
    up9_conv.weight = parameters.up9[0].weight
    up9_conv.bias = parameters.up9[0].bias
    x1 = up9_conv(x)

    up9_conv = nn.Conv2d(24, 8, 1)
    up9_conv.weight = parameters.up9[1].weight
    up9_conv.bias = parameters.up9[1].bias
    x1 = up9_conv(x1)
    act = nn.ReLU(inplace=True)
    x1 = act(x1)

    up8_conv = nn.Conv2d(48, 48, 3, 1, 1, groups=48)
    up8_conv.weight = parameters.up8[0].weight
    up8_conv.bias = parameters.up8[0].bias
    conv8 = up8_conv(y1)

    up8_conv = nn.Conv2d(48, 8, 1)
    up8_conv.weight = parameters.up8[1].weight
    up8_conv.bias = parameters.up8[1].bias
    conv8 = up8_conv(conv8)
    act = nn.ReLU(inplace=True)
    conv8 = act(conv8)
    seg = x1 + F.interpolate(conv8, scale_factor=2, mode="bilinear")

    block6_conv = nn.Conv2d(8, 8, 3, 1, 1, groups=8)
    block6_conv.weight = parameters.block6[0].weight
    block6_conv.bias = parameters.block6[0].bias
    block6 = block6_conv(seg)

    block6_conv = nn.Conv2d(8, 8, 1)
    block6_conv.weight = parameters.block6[1].weight
    block6_conv.bias = parameters.block6[1].bias
    block6 = block6_conv(block6)
    act = nn.ReLU(inplace=True)
    seg = act(block6)

    segmentation = nn.Conv2d(8, 1, 3, padding=1)
    segmentation.weight = parameters.segmentation.weight
    segmentation.bias = parameters.segmentation.bias
    seg = segmentation(seg).squeeze(1)

    up5_conv = nn.Conv2d(96, 96, 3, 1, 1, groups=96)
    up5_conv.weight = parameters.up5[0].weight
    up5_conv.bias = parameters.up5[0].bias
    up5 = up5_conv(z)

    up5_conv = nn.Conv2d(96, 96, 1)
    up5_conv.weight = parameters.up5[1].weight
    up5_conv.bias = parameters.up5[1].bias
    up5 = up5_conv(up5)
    act = nn.ReLU(inplace=True)
    up5 = act(up5)

    for i in range(0, 5):
        if i == 0:
            out = blazeblock(y1, 48, 96, 3, 2, 0, False, parameters.block1, i)
        else:
            out = blazeblock(out, 96, 96, 3, 1, 1, False, parameters.block1, i)

    out = out + up5

    up6_conv = nn.Conv2d(192, 192, 3, 1, 1, groups=192)
    up6_conv.weight = parameters.up6[0].weight
    up6_conv.bias = parameters.up6[0].bias
    up6 = up6_conv(w)

    up6_conv = nn.Conv2d(192, 192, 1)
    up6_conv.weight = parameters.up6[1].weight
    up6_conv.bias = parameters.up6[1].bias
    up6 = up6_conv(up6)
    act = nn.ReLU(inplace=True)
    up6 = act(up6)

    for i in range(0, 6):
        if i == 0:
            out = blazeblock(out, 96, 192, 3, 2, 0, False, parameters.block2, i)
        else:
            out = blazeblock(out, 192, 192, 3, 1, 1, False, parameters.block2, i)

    out = out + up6

    up7_conv = nn.Conv2d(288, 288, 3, 1, 1, groups=288)
    up7_conv.weight = parameters.up7[0].weight
    up7_conv.bias = parameters.up7[0].bias
    up7 = up7_conv(v)

    up7_conv = nn.Conv2d(288, 288, 1)
    up7_conv.weight = parameters.up7[1].weight
    up7_conv.bias = parameters.up7[1].bias
    up7 = up7_conv(up7)
    act = nn.ReLU(inplace=True)
    up7 = act(up7)

    for i in range(0, 7):
        if i == 0:
            out = blazeblock(out, 192, 288, 3, 2, 0, False, parameters.block3, i)
        else:
            out = blazeblock(out, 288, 288, 3, 1, 1, False, parameters.block3, i)

    out = out + up7

    for i in range(0, 15):
        if i == 0 or i == 8:
            out = blazeblock(out, 288, 288, 3, 2, 0, False, parameters.block4, i)
        else:
            out = blazeblock(out, 288, 288, 3, 1, 1, False, parameters.block4, i)

    temp = out
    out = F.pad(out, (0, 0, 0, 0, 0, 0), "constant", 0)
    conv1 = nn.Conv2d(
        in_channels=288,
        out_channels=288,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=288,
    )
    conv1.weight = parameters.block5.convs[0].weight
    conv1.bias = parameters.block5.convs[0].bias

    temp = conv1(temp)
    conv2 = nn.Conv2d(
        in_channels=288,
        out_channels=288,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    conv2.weight = parameters.block5.convs[1].weight
    conv2.bias = parameters.block5.convs[1].bias

    temp = conv2(temp)
    act = nn.ReLU(inplace=True)
    out = act(out + temp)

    flag = nn.Conv2d(288, 1, 2)
    flag.weight = parameters.flag.weight
    flag.bias = parameters.flag.bias

    flag_out = flag(out).view(-1).sigmoid()

    landmarks = nn.Conv2d(288, 124, 2)
    landmarks.weight = parameters.landmarks.weight
    landmarks.bias = parameters.landmarks.bias

    landmark = landmarks(out).view(batch, 31, 4) / 256

    return flag_out, landmark, seg


def extract_roi(frame, xc, yc, theta, scale):
    # take points on unit square and transform them according to the roi
    resolution = 256
    points = torch.tensor([[-1, -1, 1, 1], [-1, 1, -1, 1]], device=scale.device).view(1, 2, 4)
    points = points * scale.view(-1, 1, 1) / 2
    theta = theta.view(-1, 1, 1)
    R = torch.cat(
        (
            torch.cat((torch.cos(theta), -torch.sin(theta)), 2),
            torch.cat((torch.sin(theta), torch.cos(theta)), 2),
        ),
        1,
    )
    center = torch.cat((xc.view(-1, 1, 1), yc.view(-1, 1, 1)), 1)
    points = R @ points + center

    # use the points to compute the affine transform that maps
    # these points back to the output square
    res = resolution
    points1 = np.array([[0, 0, res - 1], [0, res - 1, 0]], dtype=np.float32).T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].cpu().numpy().T
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res, res))  # , borderValue=127.5)
        img = torch.tensor(img, device=scale.device)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype("float32")
        affine = torch.tensor(affine, device=scale.device)
        affines.append(affine)
    if imgs:
        imgs = torch.stack(imgs).permute(0, 3, 1, 2).float() / 255.0  # / 127.5 - 1.0
        affines = torch.stack(affines)
    else:
        imgs = torch.zeros((0, 3, res, res), device=scale.device)
        affines = torch.zeros((0, 2, 3), device=scale.device)

    return imgs, affines, points


def denormalize_landmarks(landmarks, affines):
    resolution = 256
    landmarks[:, :, :2] *= resolution
    for i in range(len(landmarks)):
        landmark, affine = landmarks[i], affines[i]
        landmark = (affine[:, :2] @ landmark[:, :2].T + affine[:, 2:]).T
        landmarks[i, :, :2] = landmark
    return landmarks
