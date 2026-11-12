# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The smallest ttnn.dram_prefetcher over a worker-sender PrefetcherPipe: one sender, one bank, one row.

A bringup vehicle for architectures whose smallest device does not fit test_prefetcher_worker_pipes'
layouts. Those stack receivers, senders and a spare core in separate rows, so they need a grid at least
three rows tall; the Quasar emulator's smallest model exposes two worker cores in one row. Two cores is
the floor for the op itself, since a pipe's sender must not be one of its receivers.

What this covers: the reader's DRAM reads under transaction ids, the staging ring, the writer's strided
pipe writes and credit returns, the receiver's drain, and the reader/writer exit handshake. With two
receivers, which needs three cores in a row, it also covers pipe fan-out and the strided split of a
shard across receivers. What it does not: multiple banks and entry re-sizing between tensors.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tt_py_test_utils_common.utility_functions import is_slow_dispatch
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    bytes_per_tile,
    make_krow_major_weight,
    round_up,
)

# One weight: dtype and the receiver's output width in tiles.
_DTYPE = ttnn.bfloat16
_PER_CORE_N = 2
# One K-block per pipe entry, in0_block_w tiles deep.
_IN0_BLOCK_W = 2


def _one_core(core):
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


class _MinimalLayout:
    """One sender feeding `num_receivers` receivers from one DRAM bank.

    The receivers are the first cores of row 0, and with one receiver it doubles as the matmul's only
    worker. The sender is the next core along the row, or the core below when a single receiver sits in
    a single-column grid. The op makes num_blocks = readers x receivers_per_reader, so every receiver
    gets one K-block per receiver.
    """

    num_banks = 1
    in0_block_w = _IN0_BLOCK_W

    def __init__(self, device, num_receivers=1):
        grid = device.compute_with_storage_grid_size()
        if num_receivers == 1:
            assert grid.x * grid.y >= 2, f"need two worker cores for a sender and a receiver, grid is {grid}"
            self._sender = ttnn.CoreCoord(1, 0) if grid.x >= 2 else ttnn.CoreCoord(0, 1)
        else:
            if grid.x < num_receivers + 1:
                pytest.skip(f"{num_receivers} receivers plus a sender need {num_receivers + 1} cores in a row")
            self._sender = ttnn.CoreCoord(num_receivers, 0)
        self.num_receivers = num_receivers
        self.num_blocks = num_receivers
        self.receiver_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_receivers - 1, 0))]
        )
        self.sender_cores = _one_core(self._sender)
        self.worker_cores = self.receiver_cores
        self.matmul_grid = (1, 1)

    @property
    def k_tiles(self):
        return self.num_blocks * self.in0_block_w

    def sender_receiver_mapping(self):
        return [(self._sender, self.receiver_cores)]

    @property
    def entry_size(self):
        """Bytes the receiver is delivered per block: one in1 K-block of the matmul."""
        return self.in0_block_w * _PER_CORE_N * bytes_per_tile(_DTYPE)


def _make_pipe(device, layout):
    """The pipe space and its one pipe. The space must outlive the pipe, so both are returned."""
    # Two and a half entries, as in the full tests: the matmul needs two resident K-blocks, and the half
    # entry makes the ring a non-multiple of the entry size so each wrap skips a gap.
    ring_size = round_up(5 * layout.entry_size // 2, ttnn.get_l1_alignment())
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=layout.sender_cores,
        receiver_domain=layout.receiver_cores,
        ring_size=ring_size,
        max_receivers_per_pipe=layout.num_receivers,
    )
    return space, space.create_pipes(layout.sender_receiver_mapping())


def _make_weights(device, layout, num_layers):
    """One [K, N] weight per layer, width-sharded over the single DRAM bank, plus the address tensor."""
    torch.manual_seed(0)
    K = layout.k_tiles * ttnn.TILE_SIZE
    N = layout.num_blocks * _PER_CORE_N * ttnn.TILE_SIZE
    pt_weights, tt_weights = [], []
    for _ in range(num_layers):
        pt_weight = torch.randn(1, 1, K, N)
        pt_weights.append(pt_weight)
        tt_weights.append(make_krow_major_weight(device, pt_weight, layout.num_banks, _DTYPE))
    return pt_weights, tt_weights, _make_addrs(device, layout, [w.buffer_address() for w in tt_weights])


def _make_addrs(device, layout, addresses):
    """The address table on the sender core: one row holding each layer's weight address, in order."""
    addr_row = torch.tensor(addresses, dtype=torch.int64).reshape(1, -1)
    return ttnn.as_tensor(
        addr_row,
        device=device,
        dtype=ttnn.uint32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(layout.sender_cores, [1, len(addresses)], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


class _Isolation:
    """Keeps the prefetcher's persistent program from blocking the consumer that drains it.

    Fast dispatch puts the prefetcher on its own sub-device and stalls only on the receiver. Slow
    dispatch has no sub-devices and uses asynchronous slow dispatch instead: launches stop blocking,
    and one waits only when its cores intersect the previous program's. Sender and receiver are
    disjoint, so the consumer starts while the prefetcher is still pushing.
    """

    def __init__(self, device, layout):
        self.device = device
        self.layout = layout
        self.worker_sub_device_id = None
        self._manager = None

    def __enter__(self):
        if is_slow_dispatch():
            ttnn.enable_asynchronous_slow_dispatch(self.device)
            return self
        self._manager = self.device.create_sub_device_manager(
            [ttnn.SubDevice([self.layout.sender_cores]), ttnn.SubDevice([self.layout.worker_cores])], 0
        )
        self.device.load_sub_device_manager(self._manager)
        self.worker_sub_device_id = ttnn.SubDeviceId(1)
        return self

    def __exit__(self, *_exc):
        if self._manager is None:
            ttnn.disable_asynchronous_slow_dispatch(self.device)
            return
        self.device.reset_sub_device_stall_group()
        self.device.clear_loaded_sub_device_manager()
        self.device.remove_sub_device_manager(self._manager)

    def stall_on_workers(self):
        """Stall only on the consumer. Slow dispatch's disjoint-core check already does this."""
        if self.worker_sub_device_id is not None:
            self.device.set_sub_device_stall_group([self.worker_sub_device_id])

    def release_stall(self):
        if self.worker_sub_device_id is not None:
            self.device.reset_sub_device_stall_group()


@pytest.mark.parametrize("num_receivers", [1, 2], ids=["1recv", "2recv"])
@pytest.mark.parametrize("num_layers", [1, 2], ids=["1layer", "2layers"])
def test_minimal_pipe_completion(device, num_layers, num_receivers):
    """Every block reaches every receiver: a discard consumer drains exactly what the prefetcher pushes.

    Hangs rather than fails if a block is lost, which is the point: it is the cheapest signal that the
    whole sender-to-receiver path moves data on a new architecture.
    """
    layout = _MinimalLayout(device, num_receivers)
    _space, pipes = _make_pipe(device, layout)
    _pt, tt_weights, tt_addrs = _make_weights(device, layout, num_layers)

    with _Isolation(device, layout) as isolation:
        # The op takes one weight per tensor kind; the address table carries every layer's copy.
        ttnn.dram_prefetcher(tt_weights[:1] + [tt_addrs], num_layers, prefetcher_pipes=pipes)
        isolation.stall_on_workers()
        for _layer in range(num_layers):
            ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
                device, num_iters=layout.num_blocks, page_size_bytes=layout.entry_size, prefetcher_pipes=pipes
            )
        ttnn.synchronize_device(device)


@pytest.mark.parametrize("num_receivers", [1, 2], ids=["1recv", "2recv"])
@pytest.mark.parametrize("num_layers", [1, 3], ids=["1layer", "3layers"])
def test_minimal_pipe_bytes(device, num_layers, num_receivers):
    """Every block that lands is the weight itself.

    The discard consumer only counts entries. This receiver memcmps each one against the source tensor
    and hangs on the first word that differs. With two receivers each one checks its own strided slice of
    the bank's shard. Every layer reads the same weight, as the BH validator tests do, so one source
    tensor checks them all. Three layers wrap the two-and-a-half-entry pipe ring.
    """
    layout = _MinimalLayout(device, num_receivers)
    _space, pipes = _make_pipe(device, layout)
    _pt, tt_weights, _ = _make_weights(device, layout, num_layers=1)
    tt_addrs = _make_addrs(device, layout, [tt_weights[0].buffer_address()] * num_layers)

    with _Isolation(device, layout) as isolation:
        ttnn.dram_prefetcher(tt_weights + [tt_addrs], num_layers, prefetcher_pipes=pipes)
        isolation.stall_on_workers()
        ttnn.experimental.test_tensor_prefetcher_pipe_validator(
            device,
            tt_weights[0],
            num_layers=num_layers,
            print_stride=0,
            prefetcher_pipes=pipes,
            bank_to_receivers=[(0, layout.receiver_cores)],
        )


def test_minimal_pipe_matmul(device):
    """The delivered block is the matmul's K-block: a one-core mcast_in0 matmul checked against torch.

    Completion alone cannot catch a block that arrives corrupt, at the wrong offset, or in the wrong
    order, so this is what makes the delivery contract testable rather than just the handshake.
    """
    layout = _MinimalLayout(device)
    _space, pipes = _make_pipe(device, layout)
    pt_weights, tt_weights, tt_addrs = _make_weights(device, layout, num_layers=1)

    M = ttnn.TILE_SIZE
    K = layout.k_tiles * ttnn.TILE_SIZE
    pt_act = torch.randn(1, 1, M, K)
    tt_act = ttnn.from_torch(
        pt_act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=layout.matmul_grid,
        in0_block_w=layout.in0_block_w,
        out_subblock_h=1,
        out_subblock_w=_PER_CORE_N,
        out_block_h=1,
        out_block_w=_PER_CORE_N,
        per_core_M=1,
        per_core_N=_PER_CORE_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    with _Isolation(device, layout) as isolation:
        ttnn.dram_prefetcher(tt_weights + [tt_addrs], 1, prefetcher_pipes=pipes)
        isolation.stall_on_workers()
        tt_out = ttnn.matmul(
            tt_act,
            tt_weights[0],
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
            sub_device_id=isolation.worker_sub_device_id,
            prefetcher_pipes=pipes,
        )
        ttnn.synchronize_device(device)
        isolation.release_stall()

    expected = pt_act.float() @ pt_weights[0].float()
    passing, output_str = comp_pcc(expected, ttnn.to_torch(tt_out), 0.999)
    logger.info(f"[minimal worker pipe] {output_str}")
    assert passing, output_str
