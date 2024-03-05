# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Tuple
import tt_lib
from tt_lib import fallback_ops
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, torch_to_tt_tensor
from models.experimental.mistral.mistral_helper_funcs import Linear as TtLinear, format_tensor, unpad_from_zero


class TtAttention(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        tt_cache_path=None,
        output_mem_config=None,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.base_address = base_address
        self.output_mem_config = output_mem_config
        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wq.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wq = TtLinear(
            args.dim,
            args.n_heads * args.head_dim,
            self.wq_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        self.wk_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wk.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wk = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wk_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        self.wv_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wv.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wv = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wv_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        self.wo_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wo.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wo = TtLinear(
            args.n_heads * args.head_dim,
            args.dim,
            self.wo_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        if self.args.FALLBACK_EMPTY:
            self.cache_k = torch.empty(
                args.max_batch_size,
                args.sliding_window,
                self.n_kv_heads,
                self.args.head_dim,
            )
            self.cache_v = torch.empty(args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim)
        else:
            cache_k = tt_lib.tensor.empty(
                [args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim],
                layout=tt_lib.tensor.Layout.ROW_MAJOR,
                device=self.device,
                output_mem_config=self.args.out_mem_config,
            )
            self.cache_k = tt_to_torch_tensor(cache_k).to(torch.float32)
            cache_v = tt_lib.tensor.empty(
                [args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim],
                layout=tt_lib.tensor.Layout.ROW_MAJOR,
                device=self.device,
                output_mem_config=self.args.out_mem_config,
            )
            self.cache_v = tt_to_torch_tensor(cache_v).to(torch.float32)

    def repeat_kv(self, keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tt_lib.tensor.Tensor:
        dim = 2
        keys = torch_to_tt_tensor_rm(keys, self.device)
        values = torch_to_tt_tensor_rm(values, self.device)
        keys = tt_lib.tensor.repeat_interleave(keys, repeats, dim, output_mem_config=self.args.out_mem_config)
        values = tt_lib.tensor.repeat_interleave(values, repeats, dim, output_mem_config=self.args.out_mem_config)
        return keys, values

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        bcast_freq_xq: tt_lib.tensor.complex_tensor,
        bcast_freq_xk: tt_lib.tensor.complex_tensor,
        positions: tt_lib.tensor.Tensor,
        mask: Optional[torch.Tensor],
        seqlen: int,
    ) -> tt_lib.tensor.Tensor:
        _, bsz, _, _ = x.get_legacy_shape()

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = tt_to_torch_tensor(xq).to(torch.float32)
        xk = tt_to_torch_tensor(xk).to(torch.float32)
        xv = tt_to_torch_tensor(xv).to(torch.float32)

        xq = xq[:, :, :seqlen, :]
        xk = xk[:, :, :seqlen, :]
        xv = xv[:, :, :seqlen, :]

        xq = torch_to_tt_tensor_rm(xq, self.device, put_on_device=True)
        xk = torch_to_tt_tensor_rm(xk, self.device, put_on_device=True)
        xv = torch_to_tt_tensor_rm(xv, self.device, put_on_device=True)

        xq = tt_lib.tensor.reshape(xq, bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = tt_lib.tensor.reshape(xk, bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = tt_lib.tensor.reshape(xv, bsz, seqlen, self.n_kv_heads, self.args.head_dim)

        xq = tt_to_torch_tensor(xq).to(torch.float32)
        xk = tt_to_torch_tensor(xk).to(torch.float32)
        xv = tt_to_torch_tensor(xv).to(torch.float32)

        xq, xk = apply_rotary_emb(
            xq, xk, bcast_freq_xq, bcast_freq_xk, device=self.device, mem_config=self.args.out_mem_config
        )

        # The cache is a rotating buffer
        positions = tt_to_torch_tensor(positions).squeeze(0).squeeze(0).squeeze(0)
        if self.args.FALLBACK_SCATTER:
            scatter_pos = (positions[-self.sliding_window :] % self.sliding_window)[None, :, None, None]
            scatter_pos = scatter_pos.to(torch.int64)
            scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
            self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window :])
            self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window :])
        else:
            self.cache_k = tt_to_torch_tensor(
                tt_lib.tensor.scatter(
                    torch_to_tt_tensor_rm(xk, self.device), torch_to_tt_tensor_rm(self.cache_k, self.device)
                )
            )
            self.cache_v = tt_to_torch_tensor(
                tt_lib.tensor.scatter(
                    torch_to_tt_tensor_rm(xv, self.device), torch_to_tt_tensor_rm(self.cache_v, self.device)
                )
            )

        if positions.shape[0] > 1:
            # prefill
            key, value = self.repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = int(positions[-1].item() + 1)
            key, value = self.repeat_kv(
                self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats
            )

        xq = torch_to_tt_tensor(xq, self.device)

        query = tt_lib.tensor.transpose(xq, 1, -2, output_mem_config=self.args.out_mem_config)
        desired_score_shape = [
            query.get_legacy_shape()[-1],
            query.get_legacy_shape()[-2],
            query.get_legacy_shape()[-3],
            query.get_legacy_shape()[-4],
        ]
        desired_score_shape[-1] = key.get_legacy_shape()[1]
        xq.deallocate()

        key = format_tensor(key, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        key = tt_lib.tensor.permute(key, [0, 2, 3, 1])
        key = format_tensor(key, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        value = format_tensor(value, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        value = tt_lib.tensor.transpose(value, 1, -2, output_mem_config=self.args.out_mem_config)
        value = format_tensor(value, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        query = format_tensor(query, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        scores = tt_lib.tensor.bmm(query, key, output_mem_config=self.args.out_mem_config)
        key.deallocate()
        scores = tt_lib.tensor.mul_unary(scores, self.scale, output_mem_config=self.args.out_mem_config)

        if mask is not None:
            mask = tt_lib.tensor.permute(mask, [2, 3, 0, 1])
            scores = tt_lib.tensor.permute(scores, [2, 3, 0, 1])

            scores = tt_lib.tensor.bcast(
                scores,
                mask,
                tt_lib.tensor.BcastOpMath.ADD,
                tt_lib.tensor.BcastOpDim.HW,
                output_mem_config=self.output_mem_config,
            )
            scores = tt_lib.tensor.permute(scores, [2, 3, 0, 1])
        desired_output_shape = [bsz, 32, seqlen, seqlen]
        desired_output_shape[-1] = value.get_legacy_shape()[-1]

        if self.args.FALLBACK_SOFTMAX:
            scores = fallback_ops.softmax(scores, dim=-1)
        else:
            scores = tt_lib.tensor.softmax(scores, output_mem_config=self.args.out_mem_config)
        output = tt_lib.tensor.bmm(
            scores, value, output_mem_config=self.args.out_mem_config
        )  # (bs, n_local_heads, slen, head_dim)

        value.deallocate()
        scores.deallocate()
        output = unpad_from_zero(output, desired_output_shape)
        output = torch_to_tt_tensor_rm(output, self.device, put_on_device=False)

        output = tt_lib.tensor.transpose(output, 1, -2, output_mem_config=self.args.out_mem_config)

        output = fallback_ops.reshape(output, 1, bsz, seqlen, -1)

        desired_output_shape = output.get_legacy_shape()
        output = format_tensor(output, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        output = self.wo(output)
        return output


def apply_rotary_emb(
    t_xq: torch.Tensor, t_xk: torch.Tensor, bcast_freq_xq, bcast_freq_xk, device, mem_config
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_real = torch_to_tt_tensor_rm(t_xq[..., :, :, ::2], device)
    xq_img = torch_to_tt_tensor_rm(t_xq[..., :, :, 1::2], device)

    xq = tt_lib.tensor.complex_tensor(xq_real, xq_img)

    xq_real.deallocate()
    xq_img.deallocate()

    xk_real = torch_to_tt_tensor_rm(t_xk[..., :, :, ::2], device)
    xk_img = torch_to_tt_tensor_rm(t_xk[..., :, :, 1::2], device)
    xk = tt_lib.tensor.complex_tensor(xk_real, xk_img)

    xk_real.deallocate()
    xk_img.deallocate()

    xq_out = tt_lib.tensor.complex_mul(xq, bcast_freq_xq, output_mem_config=mem_config)

    xk_out = tt_lib.tensor.complex_mul(xk, bcast_freq_xk, output_mem_config=mem_config)

    xq_out = tt_lib.tensor.concat([xq_out.real, xq_out.imag], -1, mem_config)
    xk_out = tt_lib.tensor.concat([xk_out.real, xk_out.imag], -1, mem_config)
    xq, xk = tt_to_torch_tensor(xq_out).to(torch.float32), tt_to_torch_tensor(xk_out).to(torch.float32)

    xq_out.deallocate()
    xk_out.deallocate()
    # FIXME: move this operation to on-device - should be easy.

    shapes = xq.shape
    dindex = shapes[3] // 2
    xq_out = torch.empty(xq.shape)
    # for col in range(dindex):
    #    xq_out[:,:,:,2*col] = xq[:,:,:,col]
    #    xq_out[:,:,:,2*col+1] = xq[:,:,:,col+dindex]
    xq_out[:, :, :, ::2] = xq[:, :, :, :dindex]
    xq_out[:, :, :, 1::2] = xq[:, :, :, dindex:]

    shapes = xk.shape
    dindex = shapes[3] // 2
    xk_out = torch.empty(xk.shape)
    xk_out[:, :, :, ::2] = xk[:, :, :, :dindex]
    xk_out[:, :, :, 1::2] = xk[:, :, :, dindex:]

    return xq_out, xk_out
