# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MatmulUnifiedProgramConfig with B delivered by the Tensor prefetcher over PrefetcherPipes (GH#64089, stage 1).

The pipes' receivers are the matmul's cores, laid out row-major in a rectangle of `banks` columns and `receivers_per_bank`
rows: the core at row-major position i computes all of M for N slice i, and the pipes deliver it that slice's K chunks in
K order, one pipe entry each. B_slice is a relay over the pipes, so B is never read or copied by the matmul; A is
multicast from the first core to all of them. The op queues the prefetch request itself.

Blackhole only: the Tensor prefetcher runs on the DRAM cores' RISC (DRISC). Shapes are Llama-3.1-8B decode (M = 32):
QKV, FF1 and O; N is split over as many receivers (at most 64) as the grid and DRAM banks lay out evenly.

    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_matmul_unified_prefetcher.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    bank_receivers_contiguous,
    bank_receivers_strided,
    bytes_per_tile,
    make_krow_major_weight,
    make_recv_contig_weight,
    tensor_prefetcher_session,
)
from tests.ttnn.unit_tests.operations.prefetcher_support import make_tensor_prefetcher_pipes, require_tensor_prefetcher
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole("The Tensor prefetcher needs Blackhole")

qsr = ttnn._ttnn.operations.experimental.quasar

TILE = 32
DECODE_M = 32
LLAMA_8B_DECODE = {"QKV": (4096, 12288), "FF1": (4096, 14336), "O": (4096, 4096)}  # (K, N)

# B's DRAM layout: a receiver-contiguous weight (one full-K shard per receiver) with either shard distribution, or the
# legacy K-row-major one (one shard per bank, which the prefetcher slices across the bank's receivers).
WEIGHT_LAYOUTS = {
    "recv_contig_contiguous": ttnn.ShardDistributionStrategy.CONTIGUOUS_1D,
    "recv_contig_strided": ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    "krow_major": None,
}


def _receiver_layout(device, N_tiles):
    """(banks, receivers_per_bank): the most receivers, at most 64, that split N evenly and lay out as `banks` columns
    of the grid by `receivers_per_bank` rows."""
    grid = device.compute_with_storage_grid_size()
    num_banks = device.dram_grid_size().x
    for receivers in range(min(64, num_banks * grid.y), 0, -1):
        if N_tiles % receivers != 0:
            continue
        for banks in range(min(num_banks, grid.x), 0, -1):
            if receivers % banks == 0 and receivers // banks <= grid.y:
                return banks, receivers // banks
    pytest.skip(f"no receiver layout splits {N_tiles} N tiles")


def _quantized(t, dtype):
    # Round-trips t through dtype so the golden sees the operand the device does.
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)) if dtype == ttnn.bfloat8_b else t


def _check(out, golden, rtol=0.02, pcc=0.9999):
    out_f = out.to(torch.float32)
    atol = rtol * golden.abs().max().item()
    assert_with_pcc(golden, out_f, pcc)
    assert torch.allclose(
        out_f, golden, rtol=rtol, atol=atol
    ), f"max abs err {(out_f - golden).abs().max().item():.4g} vs atol {atol:.4g}"


def _compute_kernel_config(device, fp32_dest_acc_en=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


class _Setup:
    """B in `weight_layout` on `banks` DRAM banks, the pipes delivering its N slices to the receivers (paired with the
    banks as `pairing_layout`'s distribution needs, by default B's), and A, M rows tall."""

    def __init__(
        self, device, K, N, weight_layout, b_dtype, pipe_entries, K_chunk_tiles, pairing_layout=None, M=DECODE_M, seed=0
    ):
        require_tensor_prefetcher(device)
        N_tiles = N // TILE
        self.banks, self.receivers_per_bank = _receiver_layout(device, N_tiles)
        self.num_receivers = self.banks * self.receivers_per_bank
        self.C_slice_N_tiles = N_tiles // self.num_receivers
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.banks - 1, self.receivers_per_bank - 1))]
        )
        torch.manual_seed(seed)
        self.M = M
        self.a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        self.b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        self.golden = _quantized(self.a, ttnn.bfloat16).float() @ _quantized(self.b, b_dtype).float()

        strategy = WEIGHT_LAYOUTS[weight_layout]
        if strategy is None:
            self.b_t = make_krow_major_weight(device, self.b, num_dram_banks=self.banks, dtype=b_dtype)
        else:
            self.b_t = make_recv_contig_weight(
                device,
                self.b,
                num_dram_banks=self.banks,
                ring_size=self.num_receivers,
                dtype=b_dtype,
                distribution_strategy=strategy,
            )
        # Shard i goes to the core at row-major position i of the rectangle.
        strided = WEIGHT_LAYOUTS[pairing_layout or weight_layout] == ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
        R = self.receivers_per_bank
        bank_to_receivers = [
            (
                b,
                bank_receivers_strided(b, R, self.banks, ring_cols=self.banks)
                if strided
                else bank_receivers_contiguous(b, R, ring_cols=self.banks),
            )
            for b in range(self.banks)
        ]
        entry_bytes = K_chunk_tiles * self.C_slice_N_tiles * bytes_per_tile(b_dtype)
        # Carved before any program runs on the receivers: a live program seals the L1 the pipes are carved from.
        _space, self.pipes = make_tensor_prefetcher_pipes(
            device, bank_to_receivers, int(pipe_entries * entry_bytes), single_sender=strategy is None
        )
        self.a_t = ttnn.from_torch(self.a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def matmul(self, device, program_config, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.experimental.quasar.matmul(
            self.a_t,
            self.b_t,
            program_config=program_config,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_compute_kernel_config(device),
            prefetcher_pipes=self.pipes,
        )

    def output_width_sharded(self):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, [self.M, self.C_slice_N_tiles * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )


def _run_twice(device, setup, program_config, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Two runs against the golden; the second must hit the program cache (the pipes are the same objects)."""
    with tensor_prefetcher_session(device):
        for run in range(2):
            out = setup.matmul(device, program_config, memory_config)
            _check(ttnn.to_torch(out), setup.golden)
            if run == 0:
                cache_entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == cache_entries, "the second run did not hit the program cache"


@pytest.mark.parametrize("shape", list(LLAMA_8B_DECODE))
@pytest.mark.parametrize("b_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bf8b", "bf16"])
def test_llama_decode(device, shape, b_dtype):
    """Llama-3.1-8B decode shapes over a receiver-contiguous weight, pipes of four K chunks."""
    K, N = LLAMA_8B_DECODE[shape]
    setup = _Setup(device, K, N, "recv_contig_contiguous", b_dtype, pipe_entries=4, K_chunk_tiles=4)
    _run_twice(device, setup, qsr.MatmulUnifiedProgramConfig(cores=setup.cores, K_chunk_tiles=4))


@pytest.mark.parametrize("weight_layout", list(WEIGHT_LAYOUTS))
@pytest.mark.parametrize("pipe_entries", [2, 3, 2.5], ids=["entries2", "entries3", "entries2p5"])
def test_weight_layouts_and_pipe_sizes(device, weight_layout, pipe_entries):
    """Each DRAM layout of B through pipes of two, three and two and a half K chunks (the pipe uses two and skips the
    half), 32 K chunks in all. A K-row-major B needs receivers on every DRAM bank: N is 8 receivers of 2 tiles per bank.
    """
    N = device.dram_grid_size().x * 8 * 2 * TILE
    setup = _Setup(device, 4096, N, weight_layout, ttnn.bfloat8_b, pipe_entries=pipe_entries, K_chunk_tiles=4)
    _run_twice(device, setup, qsr.MatmulUnifiedProgramConfig(cores=setup.cores, K_chunk_tiles=4))


@pytest.mark.parametrize("with_config", [False, True], ids=["no_config", "config_without_cores"])
def test_auto_placement_and_K_chunk(device, with_config):
    """Without a program config, or with one without cores, the op places the matmul on the pipes' receivers and
    picks a K chunk two of which fit in a pipe."""
    setup = _Setup(device, 4096, 14336, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=8)
    _run_twice(device, setup, program_config=qsr.MatmulUnifiedProgramConfig() if with_config else None)


def test_fused_bias_and_activation(device):
    """A one-row bias and a ReLU fused into the matmul (16-bit DST): the reader reads the bias of its C slice before
    publishing the K chunks."""
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=4)
    bias = torch.randn(1, 1, 1, 4096, dtype=torch.bfloat16)
    bias_t = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    golden = torch.relu(setup.golden + bias.float())
    program_config = qsr.MatmulUnifiedProgramConfig(
        cores=setup.cores, K_chunk_tiles=4, fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    )
    with tensor_prefetcher_session(device):
        out = ttnn.experimental.quasar.linear(
            setup.a_t,
            setup.b_t,
            bias=bias_t,
            program_config=program_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_compute_kernel_config(device, fp32_dest_acc_en=False),
            prefetcher_pipes=setup.pipes,
        )
        _check(ttnn.to_torch(out), golden)


def test_bias_added_after_the_op_with_fp32_accumulation(device):
    """With a 32-bit DST ttnn.linear adds the bias after the op, so the auto placement plans no fused bias."""
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=4)
    bias = torch.randn(1, 1, 1, 4096, dtype=torch.bfloat16)
    bias_t = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with tensor_prefetcher_session(device):
        out = ttnn.experimental.quasar.linear(
            setup.a_t,
            setup.b_t,
            bias=bias_t,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_compute_kernel_config(device, fp32_dest_acc_en=True),
            prefetcher_pipes=setup.pipes,
        )
        _check(ttnn.to_torch(out), setup.golden + bias.float())


def test_width_sharded_output_without_a_shard_spec(device):
    """A sharded C without a shard spec gets one from the C slices, which the op allocates before queuing the
    request."""
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=4)
    _run_twice(device, setup, program_config=None, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)


def test_width_sharded_output_on_the_receivers(device):
    """C width-sharded on the receivers, each core's shard its C slice."""
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=4)
    _run_twice(
        device,
        setup,
        qsr.MatmulUnifiedProgramConfig(cores=setup.cores, K_chunk_tiles=4),
        memory_config=setup.output_width_sharded(),
    )


def test_K_chunk_chosen_in_the_L1_C_leaves(device):
    """The auto K chunk is chosen once C is allocated. On p100a C's L1 shards, 22 tiles tall, take the room of the K
    chunk (8) that fits without them, and a shorter one (4) fits."""
    setup = _Setup(
        device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=32, M=22 * TILE
    )
    with tensor_prefetcher_session(device):
        out = setup.matmul(device, program_config=None, memory_config=setup.output_width_sharded())
        _check(ttnn.to_torch(out), setup.golden)


def test_rejects_a_pipe_of_one_K_chunk(device, expect_error):
    """The reader publishes one K chunk while compute multiplies the previous one, so a pipe holds two."""
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=1, K_chunk_tiles=4)
    with expect_error(RuntimeError, "at least 2 K chunks"):
        setup.matmul(device, qsr.MatmulUnifiedProgramConfig(cores=setup.cores, K_chunk_tiles=4))


def test_rejects_a_K_chunk_that_does_not_divide_K(device, expect_error):
    """Rejected before the request, which is cut into K / K_chunk_tiles blocks, is queued."""
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=3, K_chunk_tiles=4)
    with expect_error(RuntimeError, "must divide K_tiles"):
        setup.matmul(device, qsr.MatmulUnifiedProgramConfig(cores=setup.cores, K_chunk_tiles=3))


def test_rejects_cores_other_than_the_receivers(device, expect_error):
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=4)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, setup.receivers_per_bank - 1))])
    with expect_error(RuntimeError, "to be exactly its cores"):
        setup.matmul(device, qsr.MatmulUnifiedProgramConfig(cores=cores, K_chunk_tiles=4))


def test_rejects_other_program_configs(device, expect_error):
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=4)
    program_config = qsr.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(setup.banks, setup.receivers_per_bank),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=setup.C_slice_N_tiles,
    )
    with expect_error(RuntimeError, "supported only with MatmulUnifiedProgramConfig"):
        setup.matmul(device, program_config)


def test_rejects_bank_pairing_of_another_distribution(device, expect_error):
    """A ROUND_ROBIN_1D B with the banks paired as CONTIGUOUS_1D needs: the pipes would deliver core i another core's
    N slice."""
    setup = _Setup(
        device,
        4096,
        4096,
        "recv_contig_strided",
        ttnn.bfloat8_b,
        pipe_entries=2,
        K_chunk_tiles=4,
        pairing_layout="recv_contig_contiguous",
    )
    with expect_error(RuntimeError, "Pair each bank with the cores whose shards it holds"):
        setup.matmul(device, qsr.MatmulUnifiedProgramConfig(cores=setup.cores, K_chunk_tiles=4))


@pytest.mark.parametrize("override", ["subblock_N_tiles", "C_slice_N_tiles", "C_slice_M_tiles"])
def test_rejects_a_C_slice_other_than_the_pipes(device, expect_error, override):
    """Each core computes all of M by its own N slice, read at the pipe entry's width: a subblock that pads the C slice,
    or an explicit C slice of another size, is rejected."""
    setup = _Setup(device, 4096, 4096, "recv_contig_contiguous", ttnn.bfloat8_b, pipe_entries=2, K_chunk_tiles=4)
    fields = {
        "subblock_N_tiles": {
            "subblock_M_tiles": 1,
            "subblock_N_tiles": next(n for n in range(2, 5) if setup.C_slice_N_tiles % n != 0),
        },
        "C_slice_N_tiles": {"C_slice_N_tiles": 2 * setup.C_slice_N_tiles},
        "C_slice_M_tiles": {"C_slice_M_tiles": 2 * DECODE_M // TILE},
    }[override]
    with expect_error(RuntimeError, "in one batch, unpadded"):
        setup.matmul(device, qsr.MatmulUnifiedProgramConfig(cores=setup.cores, K_chunk_tiles=4, **fields))
