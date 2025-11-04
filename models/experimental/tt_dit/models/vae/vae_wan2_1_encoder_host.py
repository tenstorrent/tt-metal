# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "Wan2_1_VAE",
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = nn.Identity() if in_dim == out_dim else CausalConv3d(in_dim, out_dim, 1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        short_cut = x
        ## residual
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        ## shortcut
        if isinstance(self.shortcut, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = short_cut[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            short_cut = self.shortcut(short_cut, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            short_cut = self.shortcut(short_cut)
        return x + short_cut


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim, images=False)
        self.qkv_proj = CausalConv3d(dim, dim * 3, 1)
        self.out_proj = CausalConv3d(dim, dim, 1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()

        ## norm
        res = x
        x = self.norm(x)

        ## qkv_proj
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.qkv_proj(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.qkv_proj(x)

        ## attention
        q, k, v = rearrange(x, "b (m c) t h w -> m b c t (h w)", m=3).contiguous().chunk(3)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(attn, "b c t (h w) -> b c t h w", h=h)

        ## out_proj
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.out_proj(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.out_proj(x)

        return x + res


class Encoder3d(nn.Module):
    def __init__(self, dim, out_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]

        # in
        self.head = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, padding=1),
        )

        # downsamples
        self.downsamples = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = i == len(dims) - 2
            downsample_type = (
                "downsample3d" if (i < len(temperal_downsample) and temperal_downsample[i]) else "downsample2d"
            )

            self.downsamples.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                ResidualBlock(in_dim if j == 0 else out_dim, out_dim, dropout)
                                for j in range(num_res_blocks)
                            ]
                        ),
                        nn.ModuleList([AttentionBlock(out_dim) for _ in range(num_res_blocks)])
                        if 2**i in attn_scales
                        else nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)]),
                        Resample(out_dim, "none" if is_last else downsample_type),
                    ]
                )
            )

        # mid
        self.mid = nn.ModuleList(
            [
                ResidualBlock(dims[-1], dims[-1], dropout),
                AttentionBlock(dims[-1]),
                ResidualBlock(dims[-1], dims[-1], dropout),
            ]
        )

        # out
        self.tail = nn.Sequential(
            RMS_norm(dims[-1], images=False), nn.SiLU(), CausalConv3d(dims[-1], out_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()

        ## head
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.head(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        ## downsamples
        for res_blocks, attn_blocks, resample in self.downsamples:
            for res_block, attn_block in zip(res_blocks, attn_blocks):
                if feat_cache is not None:
                    x = res_block(x, feat_cache, feat_idx)
                    x = attn_block(x, feat_cache, feat_idx)
                else:
                    x = res_block(x)
                    x = attn_block(x)
            if feat_cache is not None:
                x = resample(x, feat_cache, feat_idx)
            else:
                x = resample(x)

        ## mid
        for layer in self.mid:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## tail
        for layer in self.tail:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_upsample, dropout):
        super().__init__()
        dims = [dim * u for u in dim_mult]

        # in
        self.head = nn.Sequential(
            CausalConv3d(z_dim, dims[-1], 3, padding=1),
            ResidualBlock(dims[-1], dims[-1], dropout),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1], dropout),
        )

        # upsamples
        self.upsamples = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(reversed(dims[:-1]), reversed(dims[1:]))):
            is_last = i == len(dims) - 2
            upsample_type = "upsample3d" if (i < len(temperal_upsample) and temperal_upsample[i]) else "upsample2d"

            self.upsamples.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                ResidualBlock(in_dim if j == 0 else out_dim, out_dim, dropout)
                                for j in range(num_res_blocks)
                            ]
                        ),
                        nn.ModuleList([AttentionBlock(out_dim) for _ in range(num_res_blocks)])
                        if 2 ** (len(dims) - 2 - i) in attn_scales
                        else nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)]),
                        Resample(out_dim, "none" if is_last else upsample_type),
                    ]
                )
            )

        # out
        self.tail = nn.Sequential(RMS_norm(dims[0], images=False), nn.SiLU(), CausalConv3d(dims[0], 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## head
        for layer in self.head:
            if isinstance(layer, (CausalConv3d, ResidualBlock, AttentionBlock)) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for res_blocks, attn_blocks, resample in self.upsamples:
            for res_block, attn_block in zip(res_blocks, attn_blocks):
                if feat_cache is not None:
                    x = res_block(x, feat_cache, feat_idx)
                    x = attn_block(x, feat_cache, feat_idx)
                else:
                    x = res_block(x)
                    x = attn_block(x)
            if feat_cache is not None:
                x = resample(x, feat_cache, feat_idx)
            else:
                x = resample(x)

        ## tail
        for layer in self.tail:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device="cpu", **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f"loading {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, map_location=device), assign=True)

    return model


class Wan2_1_VAE:
    def __init__(self, z_dim=16, vae_pth="cache/vae_step_411000.pth", dtype=torch.float, device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0) for u in videos]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0) for u in zs]
