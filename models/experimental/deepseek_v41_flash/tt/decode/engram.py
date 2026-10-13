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

``kv`` is bf16 out of the matmul; everything after it is one ``ttnn.experimental.deepseek.engram_gate``
call (fp32 inside), on ROW_MAJOR WIDTH_SHARDED streams / kv.

``wkv`` is a ``matmul_decode`` whose weight streams through its own 80-receiver ring (see
:mod:`.decode_prefetch`) and whose A is a ROW_MAJOR HEIGHT_SHARDED replica of the rows on every receiver,
so ``kv`` comes out ROW_MAJOR WIDTH_SHARDED. The layer needs the prefetcher: split over 80 cores the
weight is ~2 MB a core, more L1 than a per-call copy into the matmul's own buffer has.
"""

from typing import Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import DeepSeekV4Module, width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.attention import _decode_activation
from models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch import ring_prefetch
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES
from models.experimental.deepseek_v4_flash.tt.layers import LinearDecode
from models.experimental.deepseek_v4_flash.tt.system_config import active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache, _load_weight, _materialize

from .decode_prefetch import prefetch_rings

DECODE_LAYOUTS = {
    # K = (max_ngram - 1) * n_heads * head_dim, N = (hc + 1) * D: 80 B cores of [6144, 320], 10 per DRAM bank.
    "wkv": {"K": 6144, "N": 25600, "n_blocks": 80},
}

# Clear of the cores V4's pipeline sockets use: (0,0) / (0,1) hand-off, (0,2) packet, (0,3) output.
ROWS_SOCKET_CORE = (0, 4)
# engram_gate computes on the cores its gate weight is sharded on (at most 32), splitting D over
# them; each one's slice must sit inside one streams shard. The streams are sharded over the
# hyperconnection's 8 collapse cores, too few to fit the gate's per-core buffers beside the rest of L1.
GATE_CORES = 16


def _value(weight) -> torch.Tensor:
    return weight() if callable(weight) else weight


def rows_page_bytes(config) -> int:
    """One user's bf16 Engram rows ``[kin]``: the page of a rows socket."""
    return (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim * 2


def open_rows_sockets(device, page_bytes: int, fifo_pages: int = 2, socket_core: tuple = ROWS_SOCKET_CORE) -> list:
    """One H2D socket per device of the ``device`` mesh, each a ``fifo_pages``-page L1 FIFO on ``socket_core``.

    The FIFOs live as long as the layer. Open them before ``device``'s decode GCB rings are built: L1 is
    allocated top-down, so FIFOs opened after the rings sit below them and, once the rings are released for
    prefill, are left in the middle of L1, capping the circular buffers prefill can place.
    """
    sockets = [
        ttnn.H2DSocket(
            device,
            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(*socket_core)),
            ttnn.BufferType.L1,
            fifo_pages * page_bytes,
            ttnn.H2DMode.HOST_PUSH,
        )
        for coord in ttnn.MeshCoordinateRange(device.shape)
    ]
    for socket in sockets:
        socket.set_page_size(page_bytes)
    return sockets


class DeepSeekV41Engram(DeepSeekV4Module):
    """One Engram layer's device half, with the H2D socket its host rows arrive on."""

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        prefetch_buffers: dict,
        batch: int = 1,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        fifo_pages: int = 2,
        socket_core: tuple = ROWS_SOCKET_CORE,
        sockets: Optional[list] = None,
    ):
        """``weights`` is :func:`~models.experimental.deepseek_v41_flash.tt.config.engram_weights` (tensors or
        thunks). One socket page is one user's rows, and a socket's L1 FIFO on ``socket_core`` holds
        ``fifo_pages`` of them. A socket feeds a single device, so each device of the ``device`` mesh gets its
        own and the layer runs replicated (a TP stage's streams are). The sockets allocate L1, so build the
        layer before any trace capture on ``device``. ``sockets`` takes ones already opened by
        :func:`open_rows_sockets` (then ``fifo_pages`` / ``socket_core`` are unused) instead.

        ``wkv`` streams through the ``engram.wkv`` ring of ``prefetch_buffers``, the per-device mapping
        every other layer on ``device`` shares; call ``wkv.fetch_weights()`` ahead of each :meth:`forward`.
        """
        cache = _as_cache(cache)
        self.layer_idx = layer_idx
        self.device = device
        self.batch = batch
        self.hc, self.dim, self.eps = config.hc_mult, config.hidden_size, config.rms_norm_eps
        self.kin = (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim
        hc, d = self.hc, self.dim

        if prefetch_buffers is None:
            raise ValueError("Engram's wkv needs the prefetcher: pass the device's shared prefetch_buffers")
        layout = DECODE_LAYOUTS["wkv"]
        assert (layout["K"], layout["N"]) == (self.kin, (hc + 1) * d), (layout, self.kin, (hc + 1) * d)
        self.wkv = LinearDecode(
            weights["wkv.weight"],
            device,
            cache.file("wkv"),
            dtype=weight_dtype,
            **layout,
            **ring_prefetch(prefetch_rings(), "engram.wkv", prefetch_buffers, device, weight_dtype),
            use_rm_hs=True,
        )
        # engram_gate takes the gate weight as one fp32 tile row: row i is the weight of copy i % hc.
        # The cache file keeps the memory config it was written with, so the name records the placement.
        gate_file = cache.file(f"gate_weight_tile_l1_{GATE_CORES}c")
        gate_weight = _materialize(
            lambda: (_value(weights["q_weight"]).float() * _value(weights["k_weight"]).float() * d**-0.5)
            .repeat(ttnn.TILE_SIZE // hc, 1)
            .reshape(1, 1, ttnn.TILE_SIZE, d),
            gate_file,
            ttnn.float32,
        )
        self.gate_weight = _load_weight(
            gate_weight,
            device,
            cache_file_name=gate_file,
            dtype=ttnn.float32,
            memory_config=width_sharded_l1_config(ttnn.TILE_SIZE, d, device, num_cores=GATE_CORES),
        )
        # The ROW_MAJOR WIDTH_SHARDED layout forward takes streams in (and returns them in).
        self.streams_memory_config = width_sharded_l1_config(
            batch * hc, d, device, num_cores=_HC_COLLAPSE_CORES, tile_height=1
        )

        self.page_bytes = rows_page_bytes(config)
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
        if sockets is None:
            sockets = open_rows_sockets(device, self.page_bytes, fifo_pages, socket_core)
        if len(sockets) != device.get_num_devices():
            raise ValueError(f"got {len(sockets)} rows sockets for a {device.get_num_devices()}-device mesh")
        self.sockets = sockets

    def write_rows(self, rows: torch.Tensor) -> None:
        """Push one step's rows ``[B, kin]`` (or ``[B, 1, kin]``, bf16) into every device's socket.

        Each write is consumed by exactly one :meth:`forward` (eager call or trace replay). Write before
        dispatching the step that runs it: its ``recv_async_h2d`` stalls the device queue until the rows
        arrive, so a host sync anywhere in that step (a blocking write, a read, a CCL's first-call setup)
        would wait forever on rows sent after it. A write blocks while the FIFO is full, until the device
        drains a page an earlier, already dispatched step consumes.
        """
        rows = rows.reshape(self.batch, self.kin).to(torch.bfloat16).contiguous()
        for socket in self.sockets:
            # The socket copies raw bytes; an integer view keeps bf16 out of the DLPack dtype check.
            socket.write_tensor(rows.view(torch.int16))

    def forward(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        """``streams`` ``[B, 1, hc, D]`` bf16 ROW_MAJOR in :attr:`streams_memory_config` -> the same shape and
        layout, with this step's rows written in."""
        b, hc, d = self.batch, self.hc, self.dim
        assert tuple(streams.shape) == (b, 1, hc, d), tuple(streams.shape)
        assert streams.layout == ttnn.ROW_MAJOR_LAYOUT, streams.layout
        assert streams.memory_config() == self.streams_memory_config, streams.memory_config()
        for socket in self.sockets:
            ttnn.experimental.recv_async_h2d(self.rows, socket)
        # ROW_MAJOR A, so kv comes out ROW_MAJOR WIDTH_SHARDED over the ring's receivers.
        rows = _decode_activation(self.wkv, self.rows)
        kv = self.wkv(rows)
        ttnn.deallocate(rows)

        out = ttnn.experimental.deepseek.engram_gate(streams, kv, self.gate_weight, self.eps)
        ttnn.deallocate(kv)
        return out
