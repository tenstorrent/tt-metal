# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.tt.xtts_conv import TtConv2d

from models.experimental.xtts.config import (  # noqa: F401 — re-exported for callers
    SPK_ASP_EPS as ASP_EPS,
    SPK_BN_EPS as BN_EPS,
    SPK_INSTANCENORM_EPS as INSTANCENORM_EPS,
    TILE,
)

BODY_DTYPE = ttnn.bfloat16
TAIL_DTYPE = ttnn.bfloat16
OUT_DTYPE = ttnn.float32
TAIL_WEIGHTS_L1 = True
BODY_FIDELITY = ttnn.MathFidelity.HiFi4
# Mel frames above which the body convs drop act/weights double buffering. Those CBs scale with the
# per-core height shard, i.e. linearly with the mel length on a fixed grid, so they eventually run
# into the L1 buffers: measured clash at 2400 frames for the encoder alone, and at 2000 when it is
# co-resident with the rest of the model. Single-buffered is numerically identical and reaches the
# full 30 s (3000 frames), but costs ~4% of the pass at 801 frames and ~7% at 1600, so it is only
# used above this length. The threshold sits above the demo's 8 s window (801 frames) and below the
# shortest length seen to clash in the full pipeline (1200).
SINGLE_BUFFER_FRAMES = 1024
_MEAN_BATCH_MIN_TILES = 64
_SE_BROADCAST_MAX_TILES = 192
RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
LOG = ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)
SQRT = ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT)
CLAMP_ASP_EPS = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, ASP_EPS)


def _to_tile(t: torch.Tensor, device, dtype=None, memory_config=None) -> ttnn.Tensor:
    """Upload a torch tensor as tiled device weights."""
    return ttnn.from_torch(
        t.float(),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype or TAIL_DTYPE,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_body(t: torch.Tensor, device) -> ttnn.Tensor:
    """Upload a tensor in body bfloat16 dtype."""
    return _to_tile(t, device, BODY_DTYPE)


def _bn_scale_shift(bn, eps=BN_EPS):
    """Fold batch-norm scale and shift from running stats."""
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + eps)
    shift = bn.bias.detach() - bn.running_mean.detach() * scale
    return scale, shift


def _body_compute_config(device):
    """Build HiFi4 compute kernel config for body ops."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=BODY_FIDELITY, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _max_subblock(per_core_M, per_core_N, max_dst=4):
    """Pick max DEST-fitting matmul out subblock sizes."""
    cands = [(1, w) for w in range(1, per_core_N + 1) if per_core_N % w == 0]
    cands += [(h, per_core_N) for h in range(2, per_core_M + 1) if per_core_M % h == 0]
    return max((c for c in cands if c[0] * c[1] <= max_dst), key=lambda c: c[0] * c[1], default=(1, 1))


def _largest_divisor(n, cap):
    """Return the largest divisor of n not exceeding cap."""
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


@lru_cache(maxsize=None)
def _matmul_1d(per_core_M, per_core_N, in0_block_w, grid_x, grid_y):
    """Build a cached 1D multicast matmul program config."""
    sub_h, sub_w = _max_subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        mcast_in0=False,
        gather_in0=False,
        fuse_batch=True,
        fused_activation=None,
    )


@lru_cache(maxsize=None)
def _matmul_2d(per_core_M, per_core_N, in0_block_w, grid_x, grid_y):
    """Build a cached 2D multicast matmul program config."""
    sub_h, sub_w = _max_subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _scale_channels(x, diag, compute_config, core_grid, bias=None):
    """Scale channels via diagonal matmul with optional bias."""
    mem = x.memory_config()
    program_config = None
    if mem.shard_spec is not None:
        channel_tiles = x.shape[-1] // TILE
        program_config = _matmul_1d(
            mem.shard_spec.shape[0] // TILE, channel_tiles, channel_tiles, core_grid.x, core_grid.y
        )
    return ttnn.linear(
        x,
        diag,
        bias=bias,
        compute_kernel_config=compute_config,
        memory_config=mem,
        core_grid=None if program_config is not None else core_grid,
        program_config=program_config,
    )


class _ChannelAffine(LightweightModule):
    def __init__(self, device, scale, shift):
        """Store diagonal scale and shift for channel affine."""
        super().__init__()
        self.diag = _to_body(torch.diag(scale), device)
        self.bias = _to_body(shift.reshape(1, -1), device)
        self.compute_config = _body_compute_config(device)
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

    def forward(self, x):
        """Apply channel-wise affine transform."""
        return _scale_channels(x, self.diag, self.compute_config, self.core_grid, bias=self.bias)


def _time_major(weight):
    """Transpose last two dims of a conv weight."""
    return weight.transpose(-1, -2).contiguous()


def _fold_bn(weight, bias, bn):
    """Fold batch-norm into conv weight and bias."""
    scale, shift = _bn_scale_shift(bn)
    folded_bias = shift if bias is None else bias * scale + shift
    return weight * scale.reshape(-1, 1, 1, 1), folded_bias


def _stage_memory_config(ncores, hw):
    """Choose L1 vs interleaved memory for a stage HW."""
    return None if hw >= ncores * (TILE // 2) else ttnn.L1_MEMORY_CONFIG


@lru_cache(maxsize=None)
def _mean_block(tiles, ncores):
    """Pick mean-reduction block size minimizing cost."""

    def cost(n):
        """Score a candidate mean-reduction block factor."""
        return -(-n // ncores) * (tiles // n) + -(-n // TILE)

    n = min((d for d in range(1, tiles + 1) if tiles % d == 0), key=lambda d: (cost(d), d))
    return (tiles // n) * TILE


def _global_mean(x, hw, ncores):
    # Staged single-axis means on purpose; do not revert to dim=[1, 2] (FW-column is misleading).
    """Compute global mean over spatial dims for SE."""
    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    tiles = -(-hw // TILE)
    if tiles < _MEAN_BATCH_MIN_TILES:
        return ttnn.mean(x, dim=2, keepdim=True)
    hw_pad = tiles * TILE
    if hw_pad != hw:
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, hw_pad - hw), (0, 0)], value=0.0)
    block = _mean_block(tiles, ncores)
    mean = ttnn.mean(ttnn.reshape(x, [1, hw_pad // block, block, x.shape[-1]]), dim=2, keepdim=True)
    return ttnn.mean(ttnn.transpose(mean, 1, -2), dim=2, keepdim=True, scalar=hw_pad / hw)


def _se_core_grid(out_channels, grid):
    """Pick SE linear core grid for out_channels."""
    tiles = max(1, out_channels // TILE)
    x = _largest_divisor(tiles, grid.x)
    return ttnn.CoreGrid(y=min(tiles // x, grid.y), x=x)


class TtSELayer(LightweightModule):
    def __init__(self, device, se):
        """Load squeeze-excitation FC weights and grids."""
        super().__init__()
        self.w1 = _to_body(se.fc[0].weight.t(), device)
        self.b1 = _to_body(se.fc[0].bias.reshape(1, -1), device)
        self.w2 = _to_body(se.fc[2].weight.t(), device)
        self.b2 = _to_body(se.fc[2].bias.reshape(1, -1), device)
        channels = se.fc[0].weight.shape[1]
        self.eye = _to_body(torch.eye(channels).reshape(1, 1, channels, channels), device)
        grid = device.compute_with_storage_grid_size()
        self.grid1 = _se_core_grid(se.fc[0].weight.shape[0], grid)
        self.grid2 = _se_core_grid(channels, grid)
        self.ncores = grid.x * grid.y
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.compute_config = _body_compute_config(device)

    def forward(self, x, hw):
        """Apply SE channel gating with global mean."""
        broadcast = hw <= _SE_BROADCAST_MAX_TILES * TILE
        y = _global_mean(x, hw, self.ncores)
        y = ttnn.linear(
            y,
            self.w1,
            bias=self.b1,
            activation="relu",
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.grid1,
            compute_kernel_config=self.compute_config,
        )
        y = ttnn.linear(
            y,
            self.w2,
            bias=self.b2,
            activation="sigmoid" if broadcast else None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.grid2,
            compute_kernel_config=self.compute_config,
        )
        if broadcast:
            return ttnn.mul(x, y, memory_config=x.memory_config())
        diag = ttnn.mul(self.eye, y, input_tensor_b_activations=[SIGMOID], memory_config=ttnn.L1_MEMORY_CONFIG)
        return _scale_channels(x, diag, self.compute_config, self.core_grid)


class TtSEBasicBlock(LightweightModule):
    def __init__(self, device, block, **conv_kwargs):
        """Load an SE-ResNet basic block with folded BN."""
        super().__init__()
        self.stride = stride = block.stride[0] if isinstance(block.stride, tuple) else block.stride
        self.conv1 = TtConv2d(
            device,
            _time_major(block.conv1.weight.detach()),
            None,
            stride=stride,
            padding=1,
            activation=RELU,
            **conv_kwargs,
        )
        self.bn1 = _ChannelAffine(device, *_bn_scale_shift(block.bn1))
        w2, b2 = _fold_bn(block.conv2.weight.detach(), None, block.bn2)
        self.conv2 = TtConv2d(device, _time_major(w2), b2, stride=1, padding=1, **conv_kwargs)
        self.se = TtSELayer(device, block.se)
        self.compute_config = _body_compute_config(device)
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.ncores = grid.x * grid.y
        self.downsample_conv = None
        if block.downsample is not None:
            wd, bd = _fold_bn(block.downsample[0].weight.detach(), None, block.downsample[1])
            self.downsample_conv = TtConv2d(device, _time_major(wd), bd, stride=stride, padding=0, **conv_kwargs)

    @property
    def convs(self):
        """List conv modules used in this block."""
        return [c for c in (self.conv1, self.conv2, self.downsample_conv) if c is not None]

    def forward(self, x, h, w):
        """Run SE basic block forward returning output and HW."""
        oh, ow = (h - 1) // self.stride + 1, (w - 1) // self.stride + 1
        mem = _stage_memory_config(self.ncores, oh * ow)
        out, oh, ow = self.conv1(x, h, w, mem)
        out = self.bn1(out)
        out, _, _ = self.conv2(out, oh, ow, mem)
        out = self.se(out, oh * ow)
        residual = x if self.downsample_conv is None else self.downsample_conv(x, h, w, mem)[0]
        return ttnn.add(out, residual, activations=[RELU]), oh, ow


class TtResNetSpeakerEncoder(LightweightModule):
    def __init__(self, device, ref):
        """Load ResNet speaker encoder body and ASP/FC tail."""
        super().__init__()
        self.device = device
        body = {"activations_dtype": BODY_DTYPE, "math_fidelity": BODY_FIDELITY}
        self.conv1 = TtConv2d(
            device,
            _time_major(ref.conv1.weight.detach()),
            ref.conv1.bias.detach(),
            stride=1,
            padding=1,
            activation=RELU,
            **body,
        )
        self.bn1 = _ChannelAffine(device, *_bn_scale_shift(ref.bn1))
        self.layers = [
            [TtSEBasicBlock(device, blk, **body) for blk in layer]
            for layer in (ref.layer1, ref.layer2, ref.layer3, ref.layer4)
        ]
        self.body_convs = [self.conv1] + [c for layer in self.layers for blk in layer for c in blk.convs]

        att = ref.attention
        asp_channels = ref.layer4[-1].conv2.weight.shape[0]
        asp_dim = att[0].weight.shape[1]
        asp_freq = asp_dim // asp_channels
        asp_perm = torch.tensor([(k % asp_channels) * asp_freq + (k // asp_channels) for k in range(asp_dim)])
        wmem = ttnn.L1_MEMORY_CONFIG if TAIL_WEIGHTS_L1 else None
        self.att_w1 = _to_tile(att[0].weight.detach().squeeze(-1)[:, asp_perm], device, memory_config=wmem)
        self.att_b1 = _to_tile(att[0].bias.detach().reshape(-1, 1), device)
        att_scale, att_shift = _bn_scale_shift(att[2])
        w2 = att[3].weight.detach().squeeze(-1)[asp_perm]
        self.att_w2 = _to_tile(w2 * att_scale.reshape(1, -1), device, memory_config=wmem)
        att_b2 = w2 @ att_shift + att[3].bias.detach()[asp_perm]
        self.att_b2 = _to_tile(att_b2.reshape(-1, 1), device)

        fc_perm = torch.cat([asp_perm, asp_perm + asp_dim])
        self.fc_w = _to_tile(ref.fc.weight.detach()[:, fc_perm], device, ttnn.bfloat16, wmem)
        self.fc_b = _to_tile(ref.fc.bias.detach().reshape(-1, 1), device, OUT_DTYPE)

        self.tail_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        grid = device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        fc_mt, fc_kt = ref.fc.weight.shape[0] // TILE, ref.fc.weight.shape[1] // TILE
        fc_gy = _largest_divisor(fc_mt, grid.y)
        self.fc_config = _matmul_2d(fc_mt // fc_gy, 1, _largest_divisor(fc_kt, 8), 1, fc_gy)

    def forward(self, mel):
        """Encode mel into an L2-normalized speaker embedding."""
        _, freq, time = mel.shape
        for conv in self.body_convs:
            conv.set_double_buffer(time < SINGLE_BUFFER_FRAMES)
        x = ttnn.layer_norm(ttnn.add(mel, 1e-6, activations=[LOG], dtype=TAIL_DTYPE), epsilon=INSTANCENORM_EPS)
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.reshape(x, [1, 1, time * freq, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

        x, h, w = self.conv1(x, time, freq)
        x = self.bn1(x)
        for layer in self.layers:
            for block in layer:
                x, h, w = block(x, h, w)

        c = x.shape[-1]
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, [1, 1, h, w * c])
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.reshape(x, [w * c, h])

        gx, gy = self.grid
        n_tiles = -(-h // TILE)
        w1_mt, w1_kt = self.att_w1.shape[0] // TILE, self.att_w1.shape[1] // TILE
        w1_gx = _largest_divisor(n_tiles, gx)
        w1_gy = _largest_divisor(w1_mt, gy)
        a = ttnn.matmul(
            self.att_w1,
            x,
            program_config=_matmul_2d(w1_mt // w1_gy, n_tiles // w1_gx, _largest_divisor(w1_kt, 8), w1_gx, w1_gy),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.tail_compute_config,
        )
        a = ttnn.add(a, self.att_b1, activations=[RELU])
        w2_mt, w2_kt = self.att_w2.shape[0] // TILE, self.att_w2.shape[1] // TILE
        w2_cores = _largest_divisor(w2_mt, gx * gy)
        w2_gx = _largest_divisor(w2_cores, gx)
        a = ttnn.matmul(
            self.att_w2,
            a,
            program_config=_matmul_1d(w2_mt // w2_cores, n_tiles, _largest_divisor(w2_kt, 8), w2_gx, w2_cores // w2_gx),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.tail_compute_config,
        )
        a = ttnn.add(a, self.att_b2)
        wgt = ttnn.softmax(a, dim=-1)

        xw = ttnn.mul(x, wgt)
        mu = ttnn.sum(xw, dim=-1, keepdim=True)
        e2 = ttnn.sum(ttnn.mul(x, xw), dim=-1, keepdim=True)
        sg = ttnn.sub(e2, ttnn.mul(mu, mu), activations=[CLAMP_ASP_EPS, SQRT])
        feat = ttnn.concat([mu, sg], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)

        g = ttnn.matmul(
            self.fc_w,
            feat,
            program_config=self.fc_config,
            dtype=OUT_DTYPE,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.tail_compute_config,
        )
        g = ttnn.add(g, self.fc_b)
        g = ttnn.reshape(g, [1, self.fc_w.shape[0]])
        norm = ttnn.sqrt(ttnn.sum(ttnn.mul(g, g), dim=-1, keepdim=True))
        return ttnn.div(g, norm)
