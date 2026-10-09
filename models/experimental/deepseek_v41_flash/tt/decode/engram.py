# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode Engram: the n-gram memory written into the residual streams before layers 1 and 14.

The table lookup depends only on token ids, so the host does it
(:class:`~models.experimental.deepseek_v41_flash.tt.engram_lookup.EngramHostLookup`) and streams each
step's rows to the layer over the layer's own H2D socket -- the two Engram layers may sit on different
devices. ``recv_async_h2d`` lands the rows in a persistent buffer as the first op of :meth:`forward`, and
the device does the rest of the checkpoint's ``Engram.forward``::

    kv    = rows @ wkv^T                                   -> key [B, 1, hc, D], value [B, 1, 1, D]
    rstd  = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)       per (user, hc copy), over D
    dot   = sum(h * q_weight * k_weight * key) * rstd / sqrt(D)
    gate  = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
    out   = h + gate * value

``kv`` is bf16 out of the matmul; everything after it is fp32.
"""

from typing import Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import _HIFI4, DeepSeekV4Module
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.system_config import active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache, _load_weight, _materialize

# Clear of the cores V4's pipeline sockets use: (0,0) / (0,1) hand-off, (0,2) packet, (0,3) output.
ROWS_SOCKET_CORE = (0, 4)
# |dot| floor before the signed square root (the checkpoint's ``Engram.clamp_value``).
_GATE_CLAMP = 1e-6


def _value(weight) -> torch.Tensor:
    return weight() if callable(weight) else weight


def _rsqrt_mean_square(x: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """``[..., D]`` fp32 -> ``rsqrt(mean(x^2) + eps)`` ``[..., 1]``."""
    mean_square = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True, compute_kernel_config=_HIFI4)
    return ttnn.rsqrt(ttnn.add(mean_square, eps))


class DeepSeekV41Engram(DeepSeekV4Module):
    """One Engram layer's device half, with the H2D socket its host rows arrive on."""

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        batch: int = 1,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        fifo_pages: int = 2,
        socket_core: tuple = ROWS_SOCKET_CORE,
    ):
        """``weights`` is :func:`~models.experimental.deepseek_v41_flash.tt.config.engram_weights` (tensors or
        thunks); ``device`` a 1x1 mesh. One socket page is one user's rows, and the socket's L1 FIFO on
        ``socket_core`` holds ``fifo_pages`` of them. The socket allocates L1, so build the layer before any
        trace capture on ``device``.
        """
        cache = _as_cache(cache)
        self.layer_idx = layer_idx
        self.device = device
        self.batch = batch
        self.hc, self.dim, self.eps = config.hc_mult, config.hidden_size, config.rms_norm_eps
        self.kin = (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim
        hc, d = self.hc, self.dim

        self.wkv = Linear(weights["wkv.weight"], device, cache.file("wkv"), dtype=weight_dtype)
        gate_file = cache.file("gate_weight")
        gate_weight = _materialize(
            lambda: (_value(weights["q_weight"]).float() * _value(weights["k_weight"]).float() * d**-0.5).reshape(
                1, 1, hc, d
            ),
            gate_file,
            ttnn.float32,
        )
        self.gate_weight = _load_weight(gate_weight, device, cache_file_name=gate_file, dtype=ttnn.float32)

        self.page_bytes = self.kin * 2  # bf16
        alignment = active_system_config().pipeline.pcie_alignment
        if self.page_bytes % alignment:
            raise ValueError(f"a {self.page_bytes} B row page is not a multiple of the {alignment} B PCIe alignment")
        # Row-major, so a tensor page is one user's row: recv_async_h2d needs it to equal the socket page.
        self.rows = ttnn.from_torch(
            torch.zeros(1, 1, batch, self.kin),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.socket = ttnn.H2DSocket(
            device,
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*socket_core)),
            ttnn.BufferType.L1,
            fifo_pages * self.page_bytes,
            ttnn.H2DMode.HOST_PUSH,
        )
        self.socket.set_page_size(self.page_bytes)

    def write_rows(self, rows: torch.Tensor) -> None:
        """Push one step's rows ``[B, kin]`` (or ``[B, 1, kin]``, bf16) into the socket.

        Each write is consumed by exactly one :meth:`forward` (eager call or trace replay). Dispatch that
        forward first: a write blocks while the FIFO is full, until the device drains it.
        """
        rows = rows.reshape(self.batch, self.kin).to(torch.bfloat16).contiguous()
        # The socket copies raw bytes; an integer view keeps bf16 out of the DLPack dtype check.
        self.socket.write_tensor(rows.view(torch.int16))

    def forward(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        """``streams`` ``[B, 1, hc, D]`` (TILE) -> the same shape and dtype, with this step's rows written in."""
        b, hc, d = self.batch, self.hc, self.dim
        assert tuple(streams.shape) == (b, 1, hc, d), tuple(streams.shape)
        ttnn.experimental.recv_async_h2d(self.rows, self.socket)
        kv = ttnn.to_layout(self.wkv(ttnn.to_layout(self.rows, ttnn.TILE_LAYOUT)), ttnn.ROW_MAJOR_LAYOUT)
        # Split in row-major, where the [1, 1, B, *] -> [B, 1, *, D] reshapes are views.
        key = ttnn.reshape(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, b, hc * d]), [b, 1, hc, d])
        value = ttnn.reshape(ttnn.slice(kv, [0, 0, 0, hc * d], [1, 1, b, (hc + 1) * d]), [b, 1, 1, d])
        key = ttnn.typecast(ttnn.to_layout(key, ttnn.TILE_LAYOUT), ttnn.float32)
        value = ttnn.repeat(
            ttnn.typecast(ttnn.to_layout(value, ttnn.TILE_LAYOUT), ttnn.float32), ttnn.Shape([1, 1, hc, 1])
        )

        x = ttnn.typecast(streams, ttnn.float32)
        rstd = ttnn.multiply(_rsqrt_mean_square(x, self.eps), _rsqrt_mean_square(key, self.eps))
        weighted = ttnn.multiply(ttnn.multiply(x, self.gate_weight), key)
        dot = ttnn.multiply(ttnn.sum(weighted, dim=-1, keepdim=True, compute_kernel_config=_HIFI4), rstd)
        gate = ttnn.sigmoid(ttnn.multiply(ttnn.sqrt(ttnn.clamp(ttnn.abs(dot), min=_GATE_CLAMP)), ttnn.sign(dot)))
        return ttnn.typecast(ttnn.add(x, ttnn.multiply(gate, value)), streams.dtype)
