# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MatmulUnifiedProgramConfig with B delivered by ttnn.dram_prefetcher over worker-sender PrefetcherPipes.

The prefetcher cuts each bank's shard into num_banks x receivers_per_bank K-blocks of in0_block_w tiles, so the
matmul's K chunk is in0_block_w and its C slice is the receiver's N width. The full layouts are
test_prefetcher_worker_pipes.py's (fast dispatch: the prefetcher runs on its own sub-device); the minimal one, one
sender feeding the cores of row 0 from one DRAM bank, runs under either dispatch mode and fits Quasar's grids.

    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_matmul_unified_worker_pipes.py
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.prefetcher_common import bytes_per_tile, make_krow_major_weight, round_up
from tests.ttnn.unit_tests.operations.transformers.test_prefetcher_worker_pipes import (
    _Layout,
    _make_pipes,
    _make_weights,
    _prefetcher_sub_devices,
    _ring_size,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_slow_dispatch, skip_for_slow_dispatch

qsr = ttnn._ttnn.operations.experimental.quasar

# (dtype, per-receiver N tiles): the two block sizes re-size the pipes at every tensor switch.
_TWO_TENSORS = [(ttnn.bfloat16, 2), (ttnn.bfloat8_b, 1)]


def _compute_kernel_config(device):
    arch = device.arch()
    return ttnn.init_device_compute_kernel_config(
        arch,
        # HiFi4 with fp32 accumulation miscomputes isolated elements on Wormhole (the op warns); HiFi3 there.
        math_fidelity=ttnn.MathFidelity.HiFi3 if arch == ttnn.device.Arch.WORMHOLE_B0 else ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        # A 32-bit DST over K chunks longer than 2 tiles hangs on Quasar (#62102), apart from the pipes.
        fp32_dest_acc_en=arch != ttnn.device.Arch.QUASAR,
        packer_l1_acc=True,
    )


def _quantized(t, dtype):
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def _check(out, golden, what, rtol=0.02, pcc=0.9999):
    out_f = ttnn.to_torch(out).float()
    atol = rtol * golden.abs().max().item()
    max_err = (out_f - golden).abs().max().item()
    logger.info(f"[{what}] max abs err {max_err:.4g} (atol {atol:.4g})")
    assert_with_pcc(golden, out_f, pcc)
    assert torch.allclose(out_f, golden, rtol=rtol, atol=atol), f"{what}: max abs err {max_err:.4g} vs atol {atol:.4g}"


@skip_for_slow_dispatch("sub-device managers need fast dispatch")
@pytest.mark.parametrize("with_cores", [True, False], ids=["cores", "placed"])
@pytest.mark.parametrize("enable_performance_mode", [False, True], ids=["counted", "perf"])
@pytest.mark.parametrize("num_layers", [1, 2], ids=["1layer", "2layers"])
@pytest.mark.parametrize("recv_per_reader", [1, 2], ids=["1recv", "2recv"])
@pytest.mark.parametrize("in0_block_w", [2, 4], ids=["K2", "K4"])
def test_unified_matmul_worker_pipes(
    device, in0_block_w, recv_per_reader, num_layers, enable_performance_mode, with_cores
):
    """dram_prefetcher over worker pipes feeding unified matmuls, two weights of different block sizes per layer, run
    twice: the second run's weights are new allocations, and every program must come from the cache. Without cores
    the op places the matmul on the pipes' receivers."""
    layout = _Layout(device, recv_per_reader)
    layout.in0_block_w = in0_block_w
    entry_sizes = [layout.entry_size(dtype, per_core_n) for dtype, per_core_n in _TWO_TENSORS]
    _space, pipes = _make_pipes(device, layout, _ring_size(entry_sizes))
    runs = [_make_weights(device, layout, _TWO_TENSORS, num_layers, seed=run) for run in range(2)]

    torch.manual_seed(1234)
    pt_act = torch.randn(1, 1, ttnn.TILE_SIZE, layout.k_tiles * ttnn.TILE_SIZE)
    tt_act = ttnn.from_torch(pt_act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    act_q = _quantized(pt_act, ttnn.bfloat16)
    program_config = qsr.MatmulUnifiedProgramConfig(
        cores=layout.receiver_cores if with_cores else ttnn.CoreRangeSet([]), K_chunk_tiles=in0_block_w
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
            # The matmul's program lands on the workers' sub-device by its cores; the op takes no sub_device_id.
            device.set_sub_device_stall_group([worker_sub_device_id])
            outputs = [
                ttnn.experimental.quasar.matmul(
                    tt_act,
                    tt_weights[layer * num_tensors + t],
                    program_config=program_config,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=_compute_kernel_config(device),
                    prefetcher_pipes=pipes,
                )
                for layer in range(num_layers)
                for t in range(num_tensors)
            ]
            ttnn.synchronize_device(device)
            device.reset_sub_device_stall_group()
            cache_entries.append(device.num_program_cache_entries())
            for idx, tt_out in enumerate(outputs):
                golden = act_q @ _quantized(pt_weights[idx], _TWO_TENSORS[idx % num_tensors][0])
                _check(
                    tt_out, golden, f"recv={recv_per_reader} run={run} layer={idx // num_tensors} t={idx % num_tensors}"
                )
    assert cache_entries[0] > 0
    assert cache_entries[1] == cache_entries[0], f"second run built new programs: cache entries {cache_entries}"


class _MinimalLayout:
    """One sender feeding `num_receivers` cores at the start of row 0 from one DRAM bank, one K-block each. The sender
    is the next core along the row, or the core below a single receiver in a single-column grid."""

    num_banks = 1
    per_core_N = 2
    dtype = ttnn.bfloat16

    def __init__(self, device, num_receivers, in0_block_w):
        grid = device.compute_with_storage_grid_size()
        if grid.x < num_receivers + 1 and not (num_receivers == 1 and grid.y >= 2):
            pytest.skip(f"{num_receivers} receivers and a sender do not fit the {grid} grid")
        sender = ttnn.CoreCoord(num_receivers, 0) if grid.x > num_receivers else ttnn.CoreCoord(0, 1)
        self.in0_block_w = in0_block_w
        self.num_blocks = num_receivers
        self.k_tiles = num_receivers * in0_block_w
        self.receiver_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_receivers - 1, 0))]
        )
        self.sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(sender, sender)])
        self.mapping = [(sender, self.receiver_cores)]
        self.entry_size = in0_block_w * self.per_core_N * bytes_per_tile(self.dtype)


class _MinimalIsolation:
    """Keeps the prefetcher's persistent program from blocking the matmuls that drain it: a sub-device of its own under
    fast dispatch, asynchronous slow dispatch (a launch waits only on programs whose cores it shares) otherwise."""

    def __init__(self, device, layout):
        self.device = device
        self.layout = layout
        self.manager = None

    def __enter__(self):
        if is_slow_dispatch():
            ttnn.enable_asynchronous_slow_dispatch(self.device)
            return self
        self.manager = self.device.create_sub_device_manager(
            [ttnn.SubDevice([self.layout.sender_cores]), ttnn.SubDevice([self.layout.receiver_cores])], 0
        )
        self.device.load_sub_device_manager(self.manager)
        self.device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
        return self

    def __exit__(self, *_exc):
        if self.manager is None:
            ttnn.disable_asynchronous_slow_dispatch(self.device)
            return
        self.device.reset_sub_device_stall_group()
        self.device.clear_loaded_sub_device_manager()
        self.device.remove_sub_device_manager(self.manager)


@pytest.mark.parametrize("in0_block_w", [2, 8], ids=["K2", "K8"])
@pytest.mark.parametrize("num_receivers", [1, 2], ids=["1recv", "2recv"])
@pytest.mark.parametrize("num_layers", [1, 3], ids=["1layer", "3layers"])
def test_unified_matmul_minimal_worker_pipe(device, num_layers, num_receivers, in0_block_w):
    """The minimal layout, one matmul per layer; three layers wrap the two-and-a-half-entry pipe ring."""
    if num_layers > 1 and is_slow_dispatch():
        pytest.skip(
            "asynchronous slow dispatch makes a launch that shares a core with any outstanding program wait for all of "
            "them, the persistent prefetcher too, so a second matmul never starts"
        )
    layout = _MinimalLayout(device, num_receivers, in0_block_w)
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=layout.sender_cores,
        receiver_domain=layout.receiver_cores,
        ring_size=round_up(5 * layout.entry_size // 2, ttnn.get_l1_alignment()),
        max_receivers_per_pipe=num_receivers,
    )
    pipes = space.create_pipes(layout.mapping)
    torch.manual_seed(0)
    K = layout.k_tiles * ttnn.TILE_SIZE
    N = num_receivers * layout.per_core_N * ttnn.TILE_SIZE
    pt_weights = [torch.randn(1, 1, K, N) for _ in range(num_layers)]
    tt_weights = [make_krow_major_weight(device, w, layout.num_banks, layout.dtype) for w in pt_weights]
    tt_addrs = ttnn.as_tensor(
        torch.tensor([w.buffer_address() for w in tt_weights], dtype=torch.int64).reshape(1, -1),
        device=device,
        dtype=ttnn.uint32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(layout.sender_cores, [1, num_layers], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    torch.manual_seed(1234)
    pt_act = torch.randn(1, 1, ttnn.TILE_SIZE, K)
    tt_act = ttnn.from_torch(pt_act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    act_q = _quantized(pt_act, ttnn.bfloat16)
    program_config = qsr.MatmulUnifiedProgramConfig(cores=layout.receiver_cores, K_chunk_tiles=in0_block_w)

    with _MinimalIsolation(device, layout):
        # The op takes one weight per tensor kind; the address table carries every layer's.
        ttnn.dram_prefetcher(tt_weights[:1] + [tt_addrs], num_layers, prefetcher_pipes=pipes)
        outputs = [
            ttnn.experimental.quasar.matmul(
                tt_act,
                tt_weights[layer],
                program_config=program_config,
                dtype=ttnn.bfloat16,
                compute_kernel_config=_compute_kernel_config(device),
                prefetcher_pipes=pipes,
            )
            for layer in range(num_layers)
        ]
        ttnn.synchronize_device(device)
    for layer, tt_out in enumerate(outputs):
        _check(
            tt_out, act_q @ _quantized(pt_weights[layer], layout.dtype), f"minimal recv={num_receivers} layer={layer}"
        )


@pytest.mark.parametrize("cores", [False, True], ids=["no_config", "config_without_K_chunk"])
def test_rejects_a_guessed_K_chunk_over_worker_pipes(device, expect_error, cores):
    """The op cannot see a worker sender's entry size, so it does not guess a K chunk. No prefetcher is launched."""
    layout = _MinimalLayout(device, 1, in0_block_w=2)
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=layout.sender_cores,
        receiver_domain=layout.receiver_cores,
        ring_size=round_up(5 * layout.entry_size // 2, ttnn.get_l1_alignment()),
        max_receivers_per_pipe=1,
    )
    pipes = space.create_pipes(layout.mapping)
    K = layout.k_tiles * ttnn.TILE_SIZE
    tt_weight = make_krow_major_weight(
        device, torch.randn(1, 1, K, layout.per_core_N * ttnn.TILE_SIZE), layout.num_banks, layout.dtype
    )
    tt_act = ttnn.from_torch(
        torch.randn(1, 1, ttnn.TILE_SIZE, K), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    program_config = qsr.MatmulUnifiedProgramConfig(cores=layout.receiver_cores) if cores else None
    with expect_error(RuntimeError, "need MatmulUnifiedProgramConfig::K_chunk_tiles set"):
        ttnn.experimental.quasar.matmul(tt_act, tt_weight, program_config=program_config, prefetcher_pipes=pipes)
