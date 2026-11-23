# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Device weights for TiRex-2, built from the host fp32 weights of ``reference/dataflow.py``.

Folding (exact up to rounding):
* RMSNorm gains go into the rows of the linears that read the norm (norms run gain-free on device); the
  final T5 norm's gain goes into the head. The mLSTM MultiHeadLayerNorm weight goes into the rows of
  ``out_proj``. The sLSTM group-norm weight has no linear after it and is applied as a multiply.
* The mLSTM ``1 / sqrt(64)`` query scale and the variate attention's ``1 / sqrt(128)`` go into W_q.
* The mLSTM gate soft cap ``15 tanh(x / 15)``: the ``1 / 15`` goes into the gate weights and bias.
* mLSTM q and k are padded to 128 per head (zero columns) so q, k and v share one fused head split.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tt.common import TILE, Precision, upload_vector, upload_weight

QK_PAD = 128  # q and k head dim padded from 64 so q | k | v split as three 128-dim sections


def _rows(gain: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``diag(gain) @ w`` for ``w [in, out]``."""
    return gain[:, None] * w


def swiglu_pairs(w_silu: torch.Tensor, w_mul: torch.Tensor) -> torch.Tensor:
    """Two ``[in, n]`` weights -> ``[in, 2n]`` whose column tiles alternate ``silu_t0, mul_t0, silu_t1, ...``."""
    k, n = w_silu.shape
    pairs = torch.stack([w_silu.reshape(k, n // TILE, TILE), w_mul.reshape(k, n // TILE, TILE)], dim=2)
    return pairs.reshape(k, 2 * n)


@dataclass
class TtResidualBlock:
    w_hidden: object
    b_hidden: object
    w_out: object
    w_res: object
    b_out: object  # b_out + b_res


@dataclass
class TtFFN:
    w_swiglu: object  # [512, 2 x 2048]: column tiles alternate silu(gate) | up (``minimal_matmul(fuse_swiglu)``)
    w_down: object


@dataclass
class TtMLSTM:
    w_qkv: object  # [512, 1536]: q*1/8 | k | v, each 4 heads x 128 (q, k zero-padded 64 -> 128)
    w_o: object  # [512, 512]
    w_gates: object  # [512, 128] fp32, 32 columns per head: col 0 = i_h / 15, col 1 = f_h / 15
    b_gates: object  # [1, 1, 1, 128] fp32
    w_out: object  # [512, 512], rows scaled by the multi-head norm weight


@dataclass
class TtSLSTM:
    w_in: object  # [512, 2048] block-diagonal, columns (gate i f z o, head, d)
    bias: object  # [1, 1, 1, 2048] fp32
    r: object  # [512, 2048] bf16 block-diagonal recurrent kernel
    w_gnorm: object  # [1, 1, 1, 512] fp32


@dataclass
class TtVariate:
    w_qkv: object  # [512, 1536] (q scaled by 1/sqrt(128), norm gain folded)
    w_o: object
    w_wi: object  # [512, 2048] (ffn norm gain folded)
    w_wo: object


@dataclass
class TtBlock:
    index: int
    ffn: TtFFN
    cell: TtMLSTM | TtSLSTM
    variate: TtVariate | None


@dataclass
class TtTiRex2Weights:
    embed: TtResidualBlock
    blocks: list[TtBlock]
    head: TtResidualBlock
    head_mean: object  # [512, 512] fp32 block-averaging matrix (per-head mean), for the sLSTM group norm
    precision: Precision


def _residual(device, w: df.ResidualBlockW, dtype, gain=None) -> TtResidualBlock:  # dtype: the role's weights
    import ttnn

    wh, wr = w.w_hidden, w.w_res
    if gain is not None:
        wh, wr = _rows(gain, wh), _rows(gain, wr)
    return TtResidualBlock(
        upload_weight(device, wh, dtype),
        upload_vector(device, w.b_hidden, ttnn.bfloat16),
        upload_weight(device, w.w_out, dtype),
        upload_weight(device, wr, dtype),
        upload_vector(device, w.b_out + w.b_res, ttnn.bfloat16),
    )


def _mlstm(device, b: df.BlockW, dtype) -> TtMLSTM:
    import ttnn

    c: df.MLSTMW = b.cell
    w = _rows(b.w_norm_lstm, c.w_qkvoif)  # [512, 1544]
    q, k, v, o, g = torch.split(w, [256, 256, 512, 512, 8], dim=1)
    q = q * (df.QK_DIM**-0.5)
    pad = lambda t: torch.nn.functional.pad(t.reshape(512, 4, 64), (0, QK_PAD - 64)).reshape(512, 4 * QK_PAD)
    w_qkv = torch.cat([pad(q), pad(k), v], dim=1)
    wg = torch.zeros(512, df.HEADS * 32)
    bg = torch.zeros(df.HEADS * 32)
    for h in range(df.HEADS):
        wg[:, 32 * h] = g[:, h] / df.GATE_CAP
        wg[:, 32 * h + 1] = g[:, df.HEADS + h] / df.GATE_CAP
        bg[32 * h] = c.b_if[h] / df.GATE_CAP
        bg[32 * h + 1] = c.b_if[df.HEADS + h] / df.GATE_CAP
    return TtMLSTM(
        upload_weight(device, w_qkv, dtype),
        upload_weight(device, o, dtype),
        upload_weight(device, wg, ttnn.float32),
        upload_vector(device, bg, ttnn.float32),
        upload_weight(device, _rows(c.w_mhnorm, c.w_out), dtype),
    )


def _slstm(device, b: df.BlockW, dtype) -> TtSLSTM:
    import ttnn

    c: df.SLSTMW = b.cell
    return TtSLSTM(
        upload_weight(device, _rows(b.w_norm_lstm, c.w_in), dtype),
        upload_vector(device, c.bias, ttnn.float32),
        upload_weight(device, c.r, ttnn.bfloat16),
        upload_vector(device, c.w_gnorm, ttnn.float32),
    )


def _variate(device, b: df.BlockW, dtype) -> TtVariate:
    v = b.variate
    g = v.w_norm_attn
    wq = _rows(g, v.w_q) * (df.D_MODEL // df.HEADS) ** -0.5
    w_qkv = torch.cat([wq, _rows(g, v.w_k), _rows(g, v.w_v)], dim=1)
    return TtVariate(
        upload_weight(device, w_qkv, dtype),
        upload_weight(device, v.w_o, dtype),
        upload_weight(device, _rows(v.w_norm_ffn, v.w_wi), dtype),
        upload_weight(device, v.w_wo, dtype),
    )


def build(device, host: df.TiRex2W, precision: Precision, variate: bool = True) -> TtTiRex2Weights:
    """Upload every weight. ``variate=False`` skips the 12 variate-attention blocks (univariate-only use)."""
    import ttnn

    dt = precision.weight_dtype
    blocks = []
    for b in host.blocks:
        f = b.ffn
        ffn = TtFFN(
            upload_weight(device, swiglu_pairs(_rows(b.w_norm_ffn, f.w_gate), _rows(b.w_norm_ffn, f.w_up)), dt("ffn")),
            upload_weight(device, f.w_down, dt("ffn")),
        )
        cell = _mlstm(device, b, dt("mlstm")) if df.is_mlstm(b.index) else _slstm(device, b, dt("w_in"))
        blocks.append(TtBlock(b.index, ffn, cell, _variate(device, b, dt("variate")) if variate else None))
    mean = torch.block_diag(*[torch.full((128, 128), 1 / 128)] * df.HEADS)
    return TtTiRex2Weights(
        _residual(device, host.embed, dt("embed")),
        blocks,
        _residual(device, host.head, dt("head"), gain=host.w_out_norm),
        upload_weight(device, mean, ttnn.float32),
        precision,
    )
