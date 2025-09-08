# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as f


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(
        self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channel_out)
        self.activation = (
            self.default_act
            if activation is True
            else aactivationct
            if isinstance(activation, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class DsConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=None, dilation=1, bias=False):
        super().__init__()
        if padding is None:
            p = (dilation * (kernel_size - 1)) // 2
        self.dw = nn.Conv2d(
            channel_in,
            channel_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=p,
            dilation=dilation,
            groups=channel_in,
            bias=bias,
        )

        self.pw = nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(channel_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class Bottleneck(nn.Module):
    def __init__(self, channel_in, channel_out, shortcut=True, groups=1, kernel=(3, 3), expasion_ration=0.5):
        super().__init__()
        hidden_channel = int(c2 * expasion_ration)  # hidden channels
        self.cv1 = Conv(channel_in, hidden_channel, k[0], 1)
        self.cv2 = Conv(hidden_channel, channel_out, k[1], 1, groups=groups)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# CSP Bottleneck with 3 convolutions
class C3(nn.Module):
    def __init__(self, channel_in, channel_out, n=1, shortcut=True, groups=1, expansion_ratio=0.5):
        super().__init__()
        hidden_channel = int(c2 * expansion_ratio)  # hidden channels
        self.cv1 = Conv(channel_in, hidden_channel, 1, 1)
        self.cv2 = Conv(channel_in, hidden_channel, 1, 1)
        self.cv3 = Conv(2 * hidden_channel, channel_out, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(
                Bottleneck(hidden_channel, hidden_channel, shortcut, groups, kernels=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class DSBottleneck(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        shortcut=True,
        expasion_ration=0.5,
        kernel_size_dsconv_1=3,
        kernel_size_dsconv_2=5,
        dilatation_dsconv_2=1,
    ):
        super().__init__()
        hidden_channel = int(channel_out * expasion_ration)
        self.cv1 = DSConv(channel_in, hidden_channel, kernel_size_dsconv_1, stride=1, padding=None, dilatation=1)
        self.cv2 = DSConv(
            hidden_channel, channel_out, kernel_size_dsconv_2, stride=1, padding=None, dilatation=dilatation_dsconv_2
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    def __init__(
        self,
        channel_in,
        channel_out,
        n=1,
        shortcut=True,
        groups=1,
        expansion_ratio=0.5,
        kernel_size_dsconv_1=3,
        kernel_size_dsconv_2=5,
        dilatation_dsconv_2=1,
    ):
        super().__init__(channel_in, channel_out, n, shortcut, groups, expansion_ratio)
        hidden_channel = int(channel_out * expasion_ration)

        self.m = nn.Sequential(
            *(
                DSBottleneck(
                    hidden_channel,
                    hidden_channel,
                    shortcut=shortcut,
                    expasion_ration=1.0,
                    kernel_size_dsconv_1=kernel_size_dsconv_1,
                    kernel_size_dsconv_2=kernel_size_dsconv_2,
                    dilatation_dsconv_2=dilatation_dsconv_2,
                )
                for _ in range(n)
            )
        )


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, channel_in, channel_out, n=1, shortcut=False, groups=1, expasion_ratio=0.5):
        super().__init__()
        self.hidden_channel = int(channel_out * expasion_ratio)  # hidden channels
        self.cv1 = Conv(channel_in, 2 * self.hidden_channel, 1, 1)
        self.cv2 = Conv((2 + n) * self.hidden_channel, channel_out, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(
                self.hidden_channel, self.hidden_channel, shortcut, groups, kernels=((3, 3), (3, 3)), expasion_ratio=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DSC3k2(C2f):
    def __init__(
        self,
        channel_in,
        channel_out,
        n=1,
        dsc3k=False,
        expasion_ratio=0.5,
        groups=1,
        shortcut=True,
        kernel_size_dsconv_1=3,
        kernel_size_dsconv_2=7,
        dilatation_dsconv_2=1,
    ):
        super().__init__(channel_in, channel_out, n, shortcut, groups, expasion_ratio)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.hidden_channel,
                    self.hidden_channel,
                    n=2,
                    shortcut=shortcut,
                    groups=groups,
                    e=1.0,
                    kernel_size_dsconv_1=kernel_size_dsconv_1,
                    kernel_size_dsconv_2=kernel_size_dsconv_2,
                    dilatation_dsconv_2=dilatation_dsconv_2,
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(
                    self.hidden_channel,
                    self.hidden_channel,
                    shortcut=shortcut,
                    expasion_ratio=1.0,
                    kernel_size_dsconv_1=kernel_size_dsconv_1,
                    kernel_size_dsconv_2=kernel_size_dsconv_2,
                    dilatation_dsconv_2=dilatation_dsconv_2,
                )
                for _ in range(n)
            )


class AAttn(nn.Module):
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, activation=False)
        self.v = Conv(dim, all_head_dim, 1, activation=False)
        self.proj = Conv(all_head_dim, dim, 1, activation=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, groups=dim, activation=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(q.contiguous().half(), k.contiguous().half(), v.contiguous().half()).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = v @ attn.transpose(-2, -1)

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class ABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        n=1,
        area_attention_2=True,
        area=1,
        residual=False,
        mlp_ratio=2.0,
        expansion_ratio=0.5,
        groups=1,
        shortcut=True,
    ):
        super().__init__()
        hidden_channel = int(channel_out * expansion_ratio)  # hidden channels
        assert hidden_channel % 32 == 0, "Dimension of ABlock be a multiple of 32."

        num_heads = hidden_channel // 32

        self.cv1 = Conv(channel_in, hidden_channel, 1, 1)
        self.cv2 = Conv((1 + n) * hidden_channel, channel_out, 1)

        init_values = 0.01  # or smaller
        self.gamma = (
            nn.Parameter(init_values * torch.ones((channel_out)), requires_grad=True)
            if area_attention_2 and residual
            else None
        )

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(hidden_channel, num_heads, mlp_ratio, area) for _ in range(2)))
            if area_attention_2
            else C3k(hidden_channel, channel_out, 2, shortcut, groups)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


class Upsample(nn.Module):
    def __init__(self, scale_factor=2.0, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = f.upsample(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class DownsampleConv(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size, stride, padding)
        if channel_adjust:
            self.channel_adjust = Conv(in_channel, in_channel * 2, 1)
        else:
            self.channel_adjust = nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        x = self.channel_adjust(x)
        return x


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.Concat(x)
        return x


class C3AH(nn.Module):
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(
            embed_dim=c_, num_hyperedges=num_hyperedges, num_heads=num_heads, dropout=0.1, context=context
        )
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class AdaHyperedgeGen(nn.Module):
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'. " "Expected one of: 'mean', 'max', 'both'.")

        self.pre_head_proj = nn.Linear(node_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            avg_context = X.mean(dim=1)
            max_context, _ = X.max(dim=1)
            context_cat = torch.cat([avg_context, max_context], dim=-1)
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets

        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)

        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)

        logits = self.dropout(logits)

        return F.softmax(logits, dim=1)


class AdaHGConv(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, X):
        A = self.edge_generator(X)

        He = torch.bmm(A.transpose(1, 2), X)
        He = self.edge_proj(He)

        X_new = torch.bmm(A, He)
        X_new = self.node_proj(X_new)

        return X_new + X


class AdaHGComputation(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim, num_hyperedges=num_hyperedges, num_heads=num_heads, dropout=dropout, context=context
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens)
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out


class FuseModule(nn.Module):
    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out


class HyperACE(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1,
        context="both",
        channel_adjust=True,
    ):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class FullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out


class YoloV13(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 96, kernel=(3, 3), stride=(2, 2), padding=(1, 1)),  # 0
            Conv(96, 192, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=2, bias=False),  # 1
            Dsc3k2(
                in_channel=[192, 384],
                out_channel=[192, 384],
                kernel=[(1, 1), (1, 1)],
                stride=[(1, 1), (1, 1)],
                padding=[(1, 1), (1, 1)],
            ),  # 2
            Conv(384, 384, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False),  # 3
            Dsc3k2(
                in_channel=[384, 768],
                out_channel=[384, 768],
                kernel=[(1, 1), (1, 1)],
                stride=[(1, 1), (1, 1)],
                padding=[(1, 1), (1, 1)],
            ),  # 4
            DsConv(
                in_channel=[768, 768],
                out_channel=[768, 768],
                kernel=[(3, 3), (1, 1)],
                stride=[(2, 2), (1, 1)],
                padding=[(1, 1), (1, 1)],
                groups=[768, 1],
                bias=False,
                enable_act=True,
            ),  # 5
            A2C2f(),  # 6
            DsConv(
                in_channel=[768, 768],
                out_channel=[768, 768],
                kernel=[(3, 3), (1, 1)],
                stride=[(2, 2), (1, 1)],
                padding=[(1, 1), (1, 1)],
                groups=[768, 1],
                bias=False,
                enable_act=True,
            ),  # 7
            A2C2f(),  # 8
            HyperACE(),  # 9
            Upsample(scale_factor=2.0, mode="nearest"),  # 10
            DownsampleConv(),  # 11
            FullPAD_Tunnel(),  # 12
            FullPAD_Tunnel(),  # 13
            FullPAD_Tunnel(),  # 14
            Upsample(scale_factor=2.0, mode="nearest"),  # 15
            Concat(),  # 16
            DSC3k2(
                in_channel=[1536, 768],
                out_channel=[1536, 768],
                kernel=[(1, 1), (1, 1)],
                stride=[(1, 1), (1, 1)],
                padding=[(1, 1), (1, 1)],
            ),  # 17
            FullPAD_Tunnel(),  # 18
            Upsample(scale_factor=2.0, mode="nearest"),  # 19
            Concat(),  # 20
            DSC3k2(
                in_channel=[1536, 768],
                out_channel=[1536, 768],
                kernel=[(1, 1), (1, 1)],
                stride=[(1, 1), (1, 1)],
                padding=[(1, 1), (1, 1)],
            ),  # 21
            Conv(),  # 22
            FullPAD_Tunnel(),  # 23
            Conv(),  # 24
            Concat(),  # 25
            DSC3k2(
                in_channel=[1536, 768],
                out_channel=[1536, 768],
                kernel=[(1, 1), (1, 1)],
                stride=[(1, 1), (1, 1)],
                padding=[(1, 1), (1, 1)],
            ),  # 26
            FullPAD_Tunnel(),  # 27
            Conv(),  # 28
            Concat(),  # 29
            DSC3k2(
                in_channel=[1536, 768],
                out_channel=[1536, 768],
                kernel=[(1, 1), (1, 1)],
                stride=[(1, 1), (1, 1)],
                padding=[(1, 1), (1, 1)],
            ),  # 30
            FullPAD_Tunnel(),  # 31
            Detect(),  # 32
        )

    def forward(self, x):
        x = self.model[0](x)  # 0
        x = self.model[1](x)  # 1
        x = self.model[2](x)  # 2
        x = self.model[3](x)  # 3
        x = self.model[4](x)  # 4
        x4 = x
        x = self.model[5](x)  # 5
        x = self.model[6](x)  # 6
        x6 = x
        x = self.model[7](x)  # 7
        x = self.model[8](x)  # 8
        x = self.model[9](x)  # 9
        x = self.model[10](x)  # 10
        x10 = x
        x = f.upsample(x, scale_factor=2.0)  # 11
        x = torch.cat((x, x6), 1)  # 12
        x = self.model[13](x)  # 13
        x13 = x
        x = f.upsample(x, scale_factor=2.0)  # 14
        x = torch.cat((x, x4), 1)  # 15
        x = self.model[16](x)  # 16
        x16 = x
        x = self.model[17](x)  # 17
        x = torch.cat((x, x13), 1)  # 18
        x = self.model[19](x)  # 19
        x19 = x
        x = self.model[20](x)  # 20
        x = torch.cat((x, x10), 1)  # 21
        x = self.model[22](x)  # 22
        x22 = x
        x = self.model[23](x16, x19, x22)  # 23
        return x
