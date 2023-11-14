# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple
import tt_lib
from tt_lib import fallback_ops
from tt_lib.fused_ops.softmax import softmax as Ttsoftmax
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.mistral.mistral_helper_funcs import Linear as TtLinear


class TtAttention(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        state_dict=None,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.base_address = base_address
        self.state_dict = state_dict

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wq.weight"], self.device)
        self.wq = TtLinear(
            args.dim,
            args.n_heads * args.head_dim,
            self.wq_weights,
            device=self.device,
        )

        self.wk_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wk.weight"], self.device)
        self.wk = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wk_weights,
            device=self.device,
        )

        self.wv_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wv.weight"], self.device)
        self.wv = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wv_weights,
            device=self.device,
        )

        self.wo_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wo.weight"], self.device)
        self.wo = TtLinear(
            args.n_heads * args.head_dim,
            args.dim,
            self.wo_weights,
            device=self.device,
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
                output_mem_config=tt_lib.tensor.MemoryConfig(
                    tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
                ),
            )
            self.cache_k = tt_to_torch_tensor(cache_k).to(torch.float32)
            cache_v = tt_lib.tensor.empty(
                [args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim],
                layout=tt_lib.tensor.Layout.ROW_MAJOR,
                device=self.device,
                output_mem_config=tt_lib.tensor.MemoryConfig(
                    tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
                ),
            )
            self.cache_v = tt_to_torch_tensor(cache_v).to(torch.float32)

    def repeat_kv(self, keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tt_lib.tensor.Tensor:
        dim = 2
        keys = torch_to_tt_tensor_rm(keys, self.device)
        values = torch_to_tt_tensor_rm(values, self.device)
        keys = tt_lib.tensor.repeat_interleave(keys, repeats, dim)
        values = tt_lib.tensor.repeat_interleave(values, repeats, dim)
        return keys, values

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        freqs_cis: torch.Tensor,
        positions: tt_lib.tensor.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tt_lib.tensor.Tensor:
        _, bsz, seqlen, _ = x.shape()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = fallback_ops.reshape(xq, bsz, seqlen, self.n_heads, self.args.head_dim)

        xk = fallback_ops.reshape(xk, bsz, seqlen, self.n_kv_heads, self.args.head_dim)

        xv = fallback_ops.reshape(xv, bsz, seqlen, self.n_kv_heads, self.args.head_dim)

        xq = tt_to_torch_tensor(xq).to(torch.float32)
        xk = tt_to_torch_tensor(xk).to(torch.float32)
        xv = tt_to_torch_tensor(xv).to(torch.float32)

        if self.args.FALLBACK_ROTARY_EMBEDDING:
            xq, xk = fallback_apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        else:
            xq, xk = apply_rotary_emb_type2(xq, xk, freqs_cis=freqs_cis, device=self.device)

        # The cache is a rotating buffer
        positions = tt_to_torch_tensor(positions).squeeze(0).squeeze(0).squeeze(0)
        scatter_pos = (positions[-self.sliding_window :] % self.sliding_window)[None, :, None, None]
        scatter_pos = scatter_pos.to(torch.int64)
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window :])
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window :])

        if positions.shape[0] > 1:
            # prefill
            key, value = self.repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = int(positions[-1].item() + 1)
            key, value = self.repeat_kv(
                self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats
            )

        xq = torch_to_tt_tensor_rm(xq, self.device)
        query = tt_lib.tensor.transpose(xq, 1, -2)
        xq.deallocate()
        key = tt_lib.tensor.transpose(key, 1, -2)
        value = tt_lib.tensor.transpose(value, 1, -2)

        key = tt_lib.tensor.transpose(key, -2, -1)
        scores = tt_lib.tensor.bmm(query, key)
        key.deallocate()
        scores = tt_lib.tensor.mul_unary(scores, self.scale)

        scores = tt_to_torch_tensor(scores)
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze()
            scores += mask[None, None, ...]

        query = tt_to_torch_tensor(query)
        scores = torch_to_tt_tensor_rm(scores, self.device, put_on_device=False)

        if self.args.FALLBACK_SOFTMAX:
            scores = fallback_ops.softmax(scores, dim=-1)
        else:
            scores = tt_lib.tensor.softmax(scores)

        output = tt_lib.tensor.bmm(scores, value)  # (bs, n_local_heads, slen, head_dim)

        value.deallocate()
        scores.deallocate()
        output = tt_lib.tensor.transpose(output, 1, -2)

        output = fallback_ops.reshape(output, 1, bsz, seqlen, -1)
        return self.wo(output)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x_shape, x_ndim) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x_ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x_shape[1], x_shape[-1]), (
        freqs_cis.shape,
        (x_shape[1], x_shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x_shape)]
    return freqs_cis.view(*shape)


def fallback_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_.shape, xq_.ndim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_type1(
    t_xq: torch.Tensor, t_xk: torch.Tensor, freqs_cis: torch.Tensor, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_shape = list(copy.deepcopy(t_xq.shape))
    xq_shape[-1] = xq_shape[-1] // 2
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_shape, 4)

    freq_real = torch_to_tt_tensor_rm(freqs_cis.real, device)
    freq_img = torch_to_tt_tensor_rm(freqs_cis.imag, device)
    freqs_cis = tt_lib.tensor.concat([freq_real, freq_img], -1)

    xq_real = torch_to_tt_tensor_rm(t_xq[..., :, :, ::2], device)
    xq_img = torch_to_tt_tensor_rm(t_xq[..., :, :, 1::2], device)
    xq = tt_lib.tensor.concat([xq_real, xq_img], -1)

    xk_real = torch_to_tt_tensor_rm(t_xk[..., :, :, ::2], device)
    xk_img = torch_to_tt_tensor_rm(t_xk[..., :, :, 1::2], device)
    xk = tt_lib.tensor.concat([xk_real, xk_img], -1)

    BCH = tt_lib.tensor.BcastOpDim.H
    BCMUL = tt_lib.tensor.BcastOpMath.MUL

    t_one = tt_lib.tensor.ones_like(xq)
    bcast_freq = tt_lib.tensor.bcast(t_one, freqs_cis, BCMUL, BCH)
    xq_out = tt_lib.tensor.complex_mul(
        xq,
        bcast_freq,
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    )

    t_one = tt_lib.tensor.ones_like(xk)
    bcast_freq = tt_lib.tensor.bcast(t_one, freqs_cis, BCMUL, BCH)
    xk_out = tt_lib.tensor.complex_mul(
        xk,
        bcast_freq,
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    )

    xq, xk = tt_to_torch_tensor(xq_out).to(torch.float32), tt_to_torch_tensor(xk_out).to(torch.float32)

    shapes = xq.shape
    dindex = shapes[3] // 2
    xq_out = torch.empty(xq.shape)
    xq_out[:, :, :, ::2] = xq[:, :, :, :dindex]
    xq_out[:, :, :, 1::2] = xq[:, :, :, dindex:]

    shapes = xk.shape
    dindex = shapes[3] // 2
    xk_out = torch.empty(xk.shape)
    xk_out[:, :, :, ::2] = xk[:, :, :, :dindex]
    xk_out[:, :, :, 1::2] = xk[:, :, :, dindex:]
    return xq_out, xk_out


def apply_rotary_emb_type2(
    t_xq: torch.Tensor, t_xk: torch.Tensor, freqs_cis: torch.Tensor, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_shape = list(copy.deepcopy(t_xq.shape))
    xq_shape[-1] = xq_shape[-1] // 2
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_shape, 4)

    freq_real = torch_to_tt_tensor_rm(freqs_cis.real, device)
    freq_img = torch_to_tt_tensor_rm(freqs_cis.imag, device)
    freqs_cis = tt_lib.tensor.complex_tensor(freq_real, freq_img)

    freq_real.deallocate()
    freq_img.deallocate()

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

    BCH = tt_lib.tensor.BcastOpDim.HW
    BCMUL = tt_lib.tensor.BcastOpMath.MUL

    t_one_xq = tt_lib.tensor.ones_like(xq.real)
    t_one_xq = tt_lib.tensor.permute(t_one_xq, (3, 1, 2, 0))
    freqs_real = tt_lib.tensor.permute(freqs_cis.real, (3, 1, 2, 0))
    freqs_imag = tt_lib.tensor.permute(freqs_cis.imag, (3, 1, 2, 0))
    bcast_freq_re_xq = tt_lib.tensor.bcast(t_one_xq, freqs_real, BCMUL, BCH)
    bcast_freq_im_xq = tt_lib.tensor.bcast(t_one_xq, freqs_imag, BCMUL, BCH)
    bcast_freq_re_xq = tt_lib.tensor.permute(bcast_freq_re_xq, (3, 1, 2, 0))
    bcast_freq_im_xq = tt_lib.tensor.permute(bcast_freq_im_xq, (3, 1, 2, 0))
    t_one_xq.deallocate()
    bcast_freq_xq = tt_lib.tensor.complex_tensor(bcast_freq_re_xq, bcast_freq_im_xq)
    bcast_freq_re_xq.deallocate()
    bcast_freq_im_xq.deallocate()
    xq_out = tt_lib.tensor.complex_mul(
        xq,
        bcast_freq_xq,
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    )
    bcast_freq_xq.deallocate()

    t_one_xk = tt_lib.tensor.ones_like(xk.real)
    t_one_xk = tt_lib.tensor.permute(t_one_xk, (3, 1, 2, 0))
    bcast_freq_re_xk = tt_lib.tensor.bcast(t_one_xk, freqs_real, BCMUL, BCH)
    bcast_freq_im_xk = tt_lib.tensor.bcast(t_one_xk, freqs_imag, BCMUL, BCH)
    bcast_freq_re_xk = tt_lib.tensor.permute(bcast_freq_re_xk, (3, 1, 2, 0))
    bcast_freq_im_xk = tt_lib.tensor.permute(bcast_freq_im_xk, (3, 1, 2, 0))
    bcast_freq_xk = tt_lib.tensor.complex_tensor(bcast_freq_re_xk, bcast_freq_im_xk)
    t_one_xk.deallocate()
    bcast_freq_re_xk.deallocate()
    bcast_freq_im_xk.deallocate()
    freqs_cis.deallocate()
    freqs_real.deallocate()
    freqs_imag.deallocate()
    xk_out = tt_lib.tensor.complex_mul(
        xk,
        bcast_freq_xk,
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
    )

    bcast_freq_xk.deallocate()
    xq_out = tt_lib.tensor.concat([xq_out.real, xq_out.imag], -1)
    xk_out = tt_lib.tensor.concat([xk_out.real, xk_out.imag], -1)
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
