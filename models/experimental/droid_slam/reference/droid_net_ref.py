# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reference DROID-SLAM neural network.

This is a faithful torch port of the upstream princeton-vl/DROID-SLAM
neural components, with two adjustments that let it run without the
unavailable native wheels:

  * torch_scatter.scatter_mean is replaced with a pure-torch helper.
  * GradientClip is dropped (it is an inference no-op).

The non-neural parts of DROID-SLAM (Bundle Adjustment, lietorch SE3,
CUDA correlation lookup) are kept outside this file — the tt-nn port
focuses on the CNN encoders and the UpdateModule, which dominate wall
time of the SLAM front-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_mean(src, index, dim):
    out = torch.zeros_like(src)
    counts = torch.zeros_like(src)
    idx_shape = [1] * src.dim()
    idx_shape[dim] = index.numel()
    idx = index.view(idx_shape).expand_as(src)
    out.scatter_add_(dim, idx, src)
    counts.scatter_add_(dim, idx, torch.ones_like(src))
    return out / counts.clamp(min=1.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="instance", stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = _make_norm(norm_fn, planes)
        self.norm2 = _make_norm(norm_fn, planes)
        if stride != 1:
            self.norm3 = _make_norm(norm_fn, planes)
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride), self.norm3)
        else:
            self.downsample = None

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


def _make_norm(norm_fn, planes):
    if norm_fn == "instance":
        return nn.InstanceNorm2d(planes)
    if norm_fn == "none":
        return nn.Sequential()
    raise ValueError(norm_fn)


DIM = 32


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="instance"):
        super().__init__()
        self.norm_fn = norm_fn
        self.norm1 = _make_norm(norm_fn, DIM)
        self.conv1 = nn.Conv2d(3, DIM, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM, stride=1)
        self.layer2 = self._make_layer(2 * DIM, stride=2)
        self.layer3 = self._make_layer(4 * DIM, stride=2)
        self.conv2 = nn.Conv2d(4 * DIM, output_dim, kernel_size=1)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b * n, c1, h1, w1)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128 + 128 + 64):
        super().__init__()
        self.convz = nn.Conv2d(h_planes + i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes + i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes + i_planes, h_planes, 3, padding=1)
        self.w = nn.Conv2d(h_planes, h_planes, 1)
        self.convz_glo = nn.Conv2d(h_planes, h_planes, 1)
        self.convr_glo = nn.Conv2d(h_planes, h_planes, 1)
        self.convq_glo = nn.Conv2d(h_planes, h_planes, 1)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)
        b, c, h, w = net.shape
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h * w).mean(-1).view(b, c, 1, 1)
        z = torch.sigmoid(self.convz(net_inp) + self.convz_glo(glo))
        r = torch.sigmoid(self.convr(net_inp) + self.convr_glo(glo))
        q = torch.tanh(self.convq(torch.cat([r * net, inp], dim=1)) + self.convq_glo(glo))
        return (1 - z) * net + z * q


class GraphAgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.eta = nn.Sequential(nn.Conv2d(128, 1, 3, padding=1), nn.Softplus())
        self.upmask = nn.Sequential(nn.Conv2d(128, 8 * 8 * 9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch * num, ch, ht, wd)
        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))
        # scatter_mean(net, ix, dim=1) is an identity when `ix` is the
        # range [0, num) in order — this is the common case in the SLAM
        # front-end where `ii` already arrives as distinct keyframe
        # sources per edge (no duplicates, no reordering). Skip the
        # scatter in that case so torch.compile doesn't graph-break on
        # scatter_add_.
        if ix.numel() == num and bool(torch.equal(ix, torch.arange(num, device=ix.device))):
            net = net.view(batch, num, 128, ht, wd).view(-1, 128, ht, wd)
        else:
            net = net.view(batch, num, 128, ht, wd)
            net = _scatter_mean(net, ix, dim=1)
            net = net.view(-1, 128, ht, wd)
        net = self.relu(self.conv2(net))
        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8 * 8 * 9, ht, wd)
        return 0.01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super().__init__()
        cor_planes = 4 * (2 * 3 + 1) ** 2
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # NOTE: upstream checkpoint stores 3 output channels here even
        # though only the first two are consumed (the `[..., :2]` slice
        # below). Keep the full 3-channel conv so pretrained weights
        # load cleanly.
        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
        )
        self.gru = ConvGRU(128, 128 + 128 + 64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch * num, -1, ht, wd)
        inp = inp.view(batch * num, -1, ht, wd)
        corr = corr.view(batch * num, -1, ht, wd)
        flow = flow.view(batch * num, -1, ht, wd)
        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)
        delta = self.delta(net).view(batch, num, -1, ht, wd)
        weight = self.weight(net).view(batch, num, -1, ht, wd)
        delta = delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        weight = weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        net = net.view(batch, num, -1, ht, wd)
        eta, upmask = self.agg(net, ii.to(net.device))
        return net, delta, weight, eta, upmask


class DroidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn="instance")
        self.cnet = BasicEncoder(output_dim=256, norm_fn="none")
        self.update = UpdateModule()

    def _normalize(self, images):
        # ImageNet normalization — BGR→RGB channel swap baked in via the
        # reversed mean/std so the permutation can disappear into the
        # fused prologue.
        mean = torch.as_tensor(
            [0.406, 0.456, 0.485], device=images.device, dtype=images.dtype
        )
        std = torch.as_tensor(
            [0.225, 0.224, 0.229], device=images.device, dtype=images.dtype
        )
        return (images / 255.0 - mean[:, None, None]) / std[:, None, None]

    def extract_features(self, images):
        # images arrive in BGR channel order (cv2-style). The reversed
        # mean/std above does the implicit channel swap.
        images = self._normalize(images)
        fmaps = self.fnet(images)
        net_hidden = self.cnet(images)
        net, inp = net_hidden.split([128, 128], dim=2)
        return fmaps, torch.tanh(net), torch.relu(inp)
