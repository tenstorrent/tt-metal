# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.dram_prefetcher delivering into worker-sender PrefetcherPipes instead of a GlobalCircularBuffer.

Layout used throughout: reader (pipe sender) b reads DRAM bank b, and its R receivers are the row-major
workers b*R .. b*R + R - 1 of the matmul grid, the order an mcast_in0 matmul over the pipes expects its
K-blocks in. The matmul grid is as wide as fits a whole number of banks' receivers, so each bank's
receivers are one row segment; the senders follow row-major in the rows below it, bank b's being the b-th
(the (num_banks - 1 - b)-th with reversed senders).

See tt_metal/impl/buffers/prefetcher_matmul_design.md for the delivery contract.
"""

import contextlib

import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    bank_receivers_contiguous,
    bytes_per_tile,
    make_krow_major_weight,
    round_up,
)
from tests.ttnn.unit_tests.operations.prefetcher_support import require_tensor_prefetcher

# Two weights whose per-receiver blocks differ in size, so every tensor switch re-sizes the pipes:
# (dtype, per-receiver output width in tiles).
_TWO_TENSORS = [(ttnn.bfloat16, 2), (ttnn.bfloat8_b, 1)]


def _core_range(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


class _Layout:
    """Reader, receiver and matmul geometry for `recv_per_reader` receivers per DRAM bank."""

    def __init__(self, device, recv_per_reader, reverse_senders=False):
        grid = device.compute_with_storage_grid_size()
        self.num_banks = device.dram_grid_size().x
        self.recv_per_reader = recv_per_reader
        self.num_blocks = self.num_banks * recv_per_reader
        banks_per_row = max(
            k for k in range(1, self.num_banks + 1) if self.num_banks % k == 0 and k * recv_per_reader <= grid.x
        )
        self.matmul_grid = (banks_per_row * recv_per_reader, self.num_banks // banks_per_row)
        self.receiver_cores = _core_range(0, 0, self.matmul_grid[0] - 1, self.matmul_grid[1] - 1)
        # The senders, then one core that is neither a sender nor a receiver.
        cores = [ttnn.CoreCoord(i % grid.x, self.matmul_grid[1] + i // grid.x) for i in range(self.num_banks + 1)]
        assert cores[-1].y < grid.y, f"{self.num_banks} senders do not fit below the {self.matmul_grid} matmul grid"
        self.sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores[:-1]])
        self.spare_core = cores[-1]
        self._bank_senders = cores[-2::-1] if reverse_senders else cores[:-1]
        # One K-block per pipe entry, in0_block_w tiles deep.
        self.in0_block_w = 2
        self.k_tiles = self.num_blocks * self.in0_block_w

    def bank_receivers(self, b):
        """Bank b's receivers: the row-major matmul workers b*R .. b*R + R - 1, a row segment."""
        return bank_receivers_contiguous(b, self.recv_per_reader, ring_cols=self.matmul_grid[0])

    def sender_receiver_mapping(self):
        """Bank b's (sender, receivers), in bank order."""
        return [(self._bank_senders[b], self.bank_receivers(b)) for b in range(self.num_banks)]

    def entry_size(self, dtype, per_core_n):
        """Bytes each receiver is delivered per block: one in1 K-block of the matmul."""
        return self.in0_block_w * per_core_n * bytes_per_tile(dtype)


def _make_pipes(device, layout, ring_size, mapping=None, receiver_domain=None):
    """A worker PrefetcherPipeSpace over the layout's senders and receivers and the pipes carved from it.

    Returns (space, pipes); the space must outlive the pipes, so callers keep both.
    """
    mapping = layout.sender_receiver_mapping() if mapping is None else mapping
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=layout.sender_cores,
        receiver_domain=layout.receiver_cores if receiver_domain is None else receiver_domain,
        ring_size=ring_size,
        max_receivers_per_pipe=max(receivers.num_cores() for _, receivers in mapping),
    )
    return space, space.create_pipes(mapping)


def _ring_size(entry_sizes):
    """Two and a half of the largest entry: the matmul needs two resident K-blocks, and the half block
    makes the ring a non-multiple of every entry size, so each wrap skips a gap."""
    return round_up(5 * max(entry_sizes) // 2, ttnn.get_l1_alignment())


def _make_weights(device, layout, tensor_kinds, num_layers, seed=0):
    """Per layer and tensor, a [K, N] weight width-sharded over the DRAM banks, plus the address tensor.

    Returns (pt_weights, tt_weights, tt_addrs), the weight lists indexed [layer * num_tensors + t].
    """
    torch.manual_seed(seed)
    K = layout.k_tiles * ttnn.TILE_SIZE
    pt_weights, tt_weights = [], []
    for _ in range(num_layers):
        for dtype, per_core_n in tensor_kinds:
            N = layout.num_blocks * per_core_n * ttnn.TILE_SIZE
            pt_weight = torch.randn(1, 1, K, N)
            pt_weights.append(pt_weight)
            tt_weights.append(make_krow_major_weight(device, pt_weight, layout.num_banks, dtype))
    return pt_weights, tt_weights, _make_addrs(device, tt_weights, layout.sender_cores)


def _make_addrs(device, tt_weights, grid):
    """The address tensor: every weight's address, [layer][tensor], as one row in each of `grid`'s shards."""
    addr_row = torch.tensor([w.buffer_address() for w in tt_weights], dtype=torch.int64).reshape(1, -1)
    return ttnn.as_tensor(
        addr_row.repeat(grid.num_cores(), 1),
        device=device,
        dtype=ttnn.uint32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [1, len(tt_weights)], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@contextlib.contextmanager
def _prefetcher_sub_devices(device, layout):
    """The prefetcher on its own sub-device, so its persistent program does not block the consumers.

    Yields the worker sub-device id; stalls only on it while the body runs, as
    prefetcher_common.run_prefetcher_mm does.
    """
    sub_device_manager = device.create_sub_device_manager(
        [ttnn.SubDevice([layout.sender_cores]), ttnn.SubDevice([layout.receiver_cores])], 0
    )
    device.load_sub_device_manager(sub_device_manager)
    worker_sub_device_id = ttnn.SubDeviceId(1)
    try:
        yield worker_sub_device_id
    finally:
        device.reset_sub_device_stall_group()
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sub_device_manager)


@skip_for_slow_dispatch("sub-device managers need fast dispatch")
@pytest.mark.parametrize("num_layers", [1, 2], ids=["1layer", "2layers"])
@pytest.mark.parametrize("num_tensors", [1, 2], ids=["1tensor", "2tensors"])
def test_worker_pipes_completion(device, num_tensors, num_layers):
    """Every block reaches every receiver: a discard consumer drains exactly what the prefetcher pushes.

    With two tensors the entry size changes at every tensor switch, on both ends: the prefetcher re-sizes
    its pipes mid-program, and each consumer program attaches at its tensor's size.
    """
    layout = _Layout(device, recv_per_reader=2)
    tensor_kinds = _TWO_TENSORS[:num_tensors]
    entry_sizes = [layout.entry_size(dtype, per_core_n) for dtype, per_core_n in tensor_kinds]
    _space, pipes = _make_pipes(device, layout, _ring_size(entry_sizes))
    _pt, tt_weights, tt_addrs = _make_weights(device, layout, tensor_kinds, num_layers)

    with _prefetcher_sub_devices(device, layout) as worker_sub_device_id:
        ttnn.dram_prefetcher(tt_weights[:num_tensors] + [tt_addrs], num_layers, prefetcher_pipes=pipes)
        device.set_sub_device_stall_group([worker_sub_device_id])
        for _layer in range(num_layers):
            for entry_size in entry_sizes:
                ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
                    device, num_iters=layout.num_blocks, page_size_bytes=entry_size, prefetcher_pipes=pipes
                )
        ttnn.synchronize_device(device)


def _matmul_program_config(layout, per_core_n):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=layout.matmul_grid,
        in0_block_w=layout.in0_block_w,
        out_subblock_h=1,
        out_subblock_w=per_core_n,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
    )


@skip_for_slow_dispatch("sub-device managers need fast dispatch")
@pytest.mark.parametrize("enable_performance_mode", [False, True], ids=["counted", "perf"])
@pytest.mark.parametrize("num_layers", [1, 2], ids=["1layer", "2layers"])
@pytest.mark.parametrize("recv_per_reader", [1, 2], ids=["1recv", "2recv"])
def test_worker_pipes_matmul(device, recv_per_reader, num_layers, enable_performance_mode):
    """dram_prefetcher over pipes feeding mcast_in0 matmuls: each receiver's K-blocks are its output columns'.

    Performance mode skips the reader's NoC counter updates, so the reader's exit re-sync and the writer's
    release of it get exercised with counters that are actually stale.
    """
    _check_matmuls(device, _Layout(device, recv_per_reader), num_layers, enable_performance_mode)


@skip_for_slow_dispatch("sub-device managers need fast dispatch")
def test_worker_pipes_readers_follow_pipe_order(device):
    """Reader i drives the i-th pipe given, wherever its sender sits.

    The senders are reversed, so the pipes' order is the reverse of their senders' row-major order; only
    the pipes' order sends each bank to the receivers whose output columns it holds.
    """
    layout = _Layout(device, recv_per_reader=1, reverse_senders=True)
    _check_matmuls(device, layout, num_layers=1, enable_performance_mode=False)


def _check_matmuls(device, layout, num_layers, enable_performance_mode):
    """Runs dram_prefetcher over the layout's pipes into mcast_in0 matmuls and checks every output.

    Two weights of different block sizes per layer; run twice, and the second run must reuse every
    cached program. The second run's weights and address tensor are new allocations of the same specs,
    allocated while the first run's are still alive and holding different values, so a cache hit that kept
    the first run's address tensor bound would read the first run's weights and fail the check.
    """
    entry_sizes = [layout.entry_size(dtype, per_core_n) for dtype, per_core_n in _TWO_TENSORS]
    _space, pipes = _make_pipes(device, layout, _ring_size(entry_sizes))
    runs = [_make_weights(device, layout, _TWO_TENSORS, num_layers, seed=run) for run in range(2)]

    M = ttnn.TILE_SIZE
    K = layout.k_tiles * ttnn.TILE_SIZE
    pt_act = torch.randn(1, 1, M, K)
    tt_act = ttnn.from_torch(
        pt_act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    program_configs = [_matmul_program_config(layout, per_core_n) for _, per_core_n in _TWO_TENSORS]
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    num_tensors = len(_TWO_TENSORS)
    cache_entries = []
    with _prefetcher_sub_devices(device, layout) as worker_sub_device_id:
        for run, (pt_weights, tt_weights, tt_addrs) in enumerate(runs):
            ttnn.dram_prefetcher(
                tt_weights[:num_tensors] + [tt_addrs],
                num_layers,
                enable_performance_mode=enable_performance_mode,
                prefetcher_pipes=pipes,
            )
            device.set_sub_device_stall_group([worker_sub_device_id])
            outputs = []
            for layer in range(num_layers):
                for t in range(num_tensors):
                    outputs.append(
                        ttnn.matmul(
                            tt_act,
                            tt_weights[layer * num_tensors + t],
                            program_config=program_configs[t],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=compute_kernel_config,
                            sub_device_id=worker_sub_device_id,
                            prefetcher_pipes=pipes,
                        )
                    )
            ttnn.synchronize_device(device)
            device.reset_sub_device_stall_group()
            cache_entries.append(device.num_program_cache_entries())

            for idx, tt_out in enumerate(outputs):
                expected = pt_act.float() @ pt_weights[idx].float()
                passing, output_str = comp_pcc(expected, ttnn.to_torch(tt_out), 0.999)
                logger.info(
                    f"[worker_pipes recv={layout.recv_per_reader} run={run} layer={idx // num_tensors} t={idx % num_tensors}] {output_str}"
                )
                assert passing, f"run {run} layer {idx // num_tensors} tensor {idx % num_tensors}: {output_str}"

    assert cache_entries[0] > 0, "the program cache is disabled, so the cache-hit check below would prove nothing"
    assert cache_entries[1] == cache_entries[0], f"second run built new programs: cache entries {cache_entries}"


def _rejection_setup(device):
    """One tensor, one layer, one receiver per reader. Returns (layout, tensors, ring_size, entry_size)."""
    layout = _Layout(device, recv_per_reader=1)
    tensor_kinds = _TWO_TENSORS[:1]
    _pt, tt_weights, tt_addrs = _make_weights(device, layout, tensor_kinds, num_layers=1)
    entry_size = layout.entry_size(*tensor_kinds[0])
    return layout, tt_weights + [tt_addrs], 4 * entry_size, entry_size


def test_worker_pipes_rejects_both_targets(device, expect_error):
    layout, tensors, ring_size, _entry = _rejection_setup(device)
    _space, pipes = _make_pipes(device, layout, ring_size)
    gcb = ttnn.create_global_circular_buffer(device, layout.sender_receiver_mapping(), ring_size)
    with expect_error(RuntimeError, "needs exactly one delivery target"):
        ttnn.dram_prefetcher(tensors, 1, gcb, prefetcher_pipes=pipes)


def test_worker_pipes_rejects_no_target(device, expect_error):
    _layout, tensors, _ring, _entry = _rejection_setup(device)
    with expect_error(RuntimeError, "needs exactly one delivery target"):
        ttnn.dram_prefetcher(tensors, 1)


def test_worker_pipes_rejects_dram_sender_pipes(device, expect_error):
    require_tensor_prefetcher(device)
    layout, tensors, ring_size, _entry = _rejection_setup(device)
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=ttnn.CoreRangeSet(set()),
        receiver_domain=layout.receiver_cores,
        ring_size=ring_size,
        max_receivers_per_pipe=1,
        num_dram_senders=layout.num_banks,
    )
    bank_to_receivers = [(b, layout.bank_receivers(b)) for b in range(layout.num_banks)]
    pipes = ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher(space, bank_to_receivers)
    with expect_error(RuntimeError, "DRAM-sender pipes belong to the Tensor prefetcher"):
        ttnn.dram_prefetcher(tensors, 1, prefetcher_pipes=pipes)


def test_worker_pipes_rejects_differing_receiver_counts(device, expect_error):
    layout, tensors, ring_size, _entry = _rejection_setup(device)
    # Reader 0 also feeds a core outside the layout; every other reader has one receiver.
    spare = layout.spare_core
    extra_receiver = _core_range(spare.x, spare.y, spare.x, spare.y)
    mapping = layout.sender_receiver_mapping()
    mapping[0] = (mapping[0][0], mapping[0][1].merge(extra_receiver))
    _space, pipes = _make_pipes(
        device, layout, ring_size, mapping, receiver_domain=layout.receiver_cores.merge(extra_receiver)
    )
    with expect_error(RuntimeError, "same number of receivers per reader"):
        ttnn.dram_prefetcher(tensors, 1, prefetcher_pipes=pipes)


def test_worker_pipes_rejects_too_few_pipes(device, expect_error):
    layout, tensors, ring_size, _entry = _rejection_setup(device)
    _space, pipes = _make_pipes(device, layout, ring_size)
    with expect_error(RuntimeError, "needs one PrefetcherPipe per reader core"):
        ttnn.dram_prefetcher(tensors, 1, prefetcher_pipes=pipes[:-1])


def test_worker_pipes_rejects_block_larger_than_ring(device, expect_error):
    layout, tensors, _ring, entry_size = _rejection_setup(device)
    _space, pipes = _make_pipes(device, layout, entry_size - ttnn.get_l1_alignment())
    with expect_error(RuntimeError, "but the pipes' ring is only"):
        ttnn.dram_prefetcher(tensors, 1, prefetcher_pipes=pipes)


def test_worker_pipes_rejects_shared_sender(device, expect_error):
    layout, tensors, ring_size, _entry = _rejection_setup(device)
    _space, pipes = _make_pipes(device, layout, ring_size)
    # Readers 0 and 1 would both run on pipe 0's sender.
    with expect_error(RuntimeError, "shares sender"):
        ttnn.dram_prefetcher(tensors, 1, prefetcher_pipes=pipes[:1] + pipes[:-1])


def test_worker_pipes_rejects_addrs_off_the_readers(device, expect_error):
    layout, tensors, ring_size, _entry = _rejection_setup(device)
    _space, pipes = _make_pipes(device, layout, ring_size)
    tt_addrs = _make_addrs(device, tensors[:-1], layout.receiver_cores)
    with expect_error(RuntimeError, "tensor_addrs must be sharded over the reader cores"):
        ttnn.dram_prefetcher(tensors[:-1] + [tt_addrs], 1, prefetcher_pipes=pipes)


def test_worker_pipes_rejects_addrs_missing_a_layer(device, expect_error):
    layout, tensors, ring_size, _entry = _rejection_setup(device)
    _space, pipes = _make_pipes(device, layout, ring_size)
    # The address tensor holds one layer's addresses, but the prefetcher is asked for two layers.
    with expect_error(RuntimeError, "must hold the address of every tensor of every layer"):
        ttnn.dram_prefetcher(tensors, 2, prefetcher_pipes=pipes)


def test_worker_pipes_rejects_partial_tile_split(device, expect_error):
    layout = _Layout(device, recv_per_reader=2)
    K = layout.k_tiles * ttnn.TILE_SIZE
    # Three tiles per bank cannot be split evenly over two receivers.
    shard_width = 3 * ttnn.TILE_SIZE
    weight = make_krow_major_weight(
        device, torch.randn(1, 1, K, layout.num_banks * shard_width), layout.num_banks, ttnn.bfloat16
    )
    _space, pipes = _make_pipes(device, layout, _ring_size([layout.entry_size(ttnn.bfloat16, 2)]))
    with expect_error(RuntimeError, "does not split evenly over the 2 receiver cores per sender"):
        ttnn.dram_prefetcher([weight, _make_addrs(device, [weight], layout.sender_cores)], 1, prefetcher_pipes=pipes)
