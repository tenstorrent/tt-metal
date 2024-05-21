import ttnn
import tt_lib
from tt_lib.fallback_ops import fallback_ops
import math
import torch
import torch.nn.functional as F
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from models.experimental.functional_swin_s.swin_helper_funcs import linear as TtLinear


class TtShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        shift_size,
        num_heads,
        qkv_bias,
        proj_bias,
        attention_dropout,
        dropout,
        device,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.device = device
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = TtLinear(dim, dim * 3, bias=qkv_bias)
        self.proj = TtLinear(dim, dim, bias=proj_bias)

        self.relative_position_bias_table = torch.zeros(
            (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads
        )
        # nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = fallback_ops.reshape(relative_position_bias, (N, N, -1))
        relative_position_bias = tt_lib.tensor.permute(relative_position_bias, (2, 0, 1))
        relative_position_bias = tt_to_torch_tensor(relative_position_bias)
        relative_position_bias = relative_position_bias.unsqueeze(0)
        relative_position_bias = torch_to_tt_tensor_rm(relative_position_bias, self.device, put_on_device=True)

        logit_scale = None

        B, H, W, C = x.get_legacy_shape()
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_values = (0, 0, 0, pad_r, 0, pad_b)
        x = fallback_ops.pad(x, pad_values)
        _, pad_H, pad_W, _ = x.get_legacy_shape()

        self.shift_size = tt_lib.tensor.copy(self.shift_size)
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        x = tt_to_torch_tensor(x)
        # cyclic shift
        if tt_lib.tensor.sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        x = torch_to_tt_tensor_rm(x, self.device)

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x = tt_to_torch_tensor(x)
        x = x.view(
            B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)
        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)

        # multi-head attention
        if logit_scale is not None and self.qkv.bias is not None:
            self.qkv.bias = tt_lib.tensor.clone(self.qkv.bias)
            shape = self.qkv.bias.get_legacy_shape()
            numel = shape[0] * shape[1] * shape[2] * shape[3]
            length = numel // 3
            self.qkv.bias[length : 2 * length] = 0

        # tt to torch and torch to ttnn of self.qkv.weight
        self.qkv.weight = tt_to_torch_tensor(self.qkv.weight)
        self.qkv.weight = ttnn.from_torch(
            self.qkv.weight, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        # tt to torch and torch to ttnn of self.qkv.bias
        self.qkv.bias = tt_to_torch_tensor(self.qkv.bias)
        self.qkv.bias = ttnn.from_torch(self.qkv.bias, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # tt to torch and torch to ttnn of x
        x = tt_to_torch_tensor(x)
        x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        qkv = ttnn.linear(x, self.qkv.weight, self.qkv.bias)
        # ttnn to torch and torch to tt of self.qkv.weight and x
        qkv = ttnn.to_torch(qkv)
        x = ttnn.to_torch(x)

        qkv = qkv.reshape(x.size(0), x.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if logit_scale is not None:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
        else:
            q = q * (C // self.num_heads) ** -0.5
            attn = q.matmul(k.transpose(-2, -1))
        # add relative position bias
        attn = torch_to_tt_tensor_rm(attn, self.device, put_on_device=True)
        attn = attn + relative_position_bias

        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)
        if tt_lib.tensor.sum(self.shift_size) > 0:
            # generate attention mask
            attn_mask = tt_lib.tensor.zeros([pad_H, pad_W], data_type=x.data_type, device=x.device)
            h_slices = (
                (0, -self.window_size[0]),
                (-self.window_size[0], -self.shift_size[0]),
                (-self.shift_size[0], None),
            )
            w_slices = (
                (0, -self.window_size[1]),
                (-self.window_size[1], -self.shift_size[1]),
                (-self.shift_size[1], None),
            )
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
