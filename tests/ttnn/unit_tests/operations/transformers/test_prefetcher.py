# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_wormhole_b0, is_blackhole, skip_for_blackhole, skip_for_slow_dispatch
from tests.ttnn.unit_tests.operations.prefetcher_common import run_prefetcher_mm
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@skip_for_slow_dispatch()
@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, dtypes, num_layers",
    [
        (2, 2, [(256, 512), (256, 512)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(1024, 256), (1024, 256)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(128, 128), (128, 128)], [ttnn.bfloat4_b] * 2, 2),
        (2, 2, [(256, 1024), (256, 2048)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 1),
        (2, 3, [(256, 1024), (256, 2048), (512, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        (2, 2, [(256, 1024), (128, 128)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 5),
        (2, 3, [(256, 1024), (128, 128), (1024, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        # Padding check
        (
            2,
            3,
            [(256 + 32, 512 + 224), (128, 128 + 64), (512 + 256, 224)],
            [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b],
            5,
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher_post_commit(
    device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    function_level_defaults,
):
    run_prefetcher_mm(
        device,
        num_tensors,
        input_shapes,
        num_layers,
        num_reader_cores,
        dtypes,
        is_functional_test=True,
    )


# Test DRAM Prefetcher x Matmul on T3K
@skip_for_slow_dispatch()
@pytest.mark.parametrize(
    "num_reader_cores, num_tensors, input_shapes, dtypes, num_layers",
    [
        (2, 2, [(256, 512), (256, 512)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(1024, 256), (1024, 256)], [ttnn.bfloat4_b] * 2, 5),
        (2, 2, [(128, 128), (128, 128)], [ttnn.bfloat4_b] * 2, 2),
        (2, 2, [(256, 1024), (256, 2048)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 1),
        (2, 3, [(256, 1024), (256, 2048), (512, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        (2, 2, [(256, 1024), (128, 128)], [ttnn.bfloat4_b, ttnn.bfloat8_b], 5),
        (2, 3, [(256, 1024), (128, 128), (1024, 256)], [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b], 5),
        # Padding check
        (
            2,
            3,
            [(256 + 32, 512 + 224), (128, 128 + 64), (512 + 256, 224)],
            [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b],
            5,
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
def test_run_prefetcher_post_commit_multi_device(
    mesh_device,
    num_tensors,
    input_shapes,
    num_layers,
    num_reader_cores,
    dtypes,
    function_level_defaults,
):
    if mesh_device.get_num_devices() <= 1:
        pytest.skip("This test requires multiple devices")

    run_prefetcher_mm(
        mesh_device,
        num_tensors,
        input_shapes,
        num_layers,
        num_reader_cores,
        dtypes,
        is_functional_test=True,
    )


# ttnn.dram_prefetcher with weights whose K is not a multiple of the prefetch ring.
#
# The prefetcher pads a DRAM weight shard of H tile rows to round_up(H, num_readers * receivers_per_reader) rows;
# the pad rows must reach the receivers as zeros and never be read from DRAM (the next allocation lies past H).
#
# Each tensor_addrs entry points at one DRAM tensor [W ; NaN rows] (the op takes only shapes from the K-row tensors
# it is passed) and an unpadded all-NaN tensor is prefetched first, so unread or unzeroed pad rows show up as NaN.
#
# Consume mode runs the matmul over the padded K, so compute multiplies the pad rows and must still give
# in0[:, :K] @ W; production mode (activation K == weight K, as the models run it) guards the production path.


class _PaddedKRing:
    """The prefetcher ring of the padded-K tests, the core layout of get_core_ranges(2, 2,
    is_functional_test=True) in models/demos/llama3_70b_galaxy/tt/model_config.py: reader r sits at
    (0, r), reads DRAM bank r and feeds receivers (1, r) and (2, r), which form the matmul ring.
    Banks 0 and 1 exist on every single card (WH 12 banks, BH P150 8, P100 7)."""

    num_readers = 2
    recv_per_reader = 2
    size = num_readers * recv_per_reader  # num_blocks: the K-split of every prefetched tensor

    def __init__(self):
        readers, recv = range(self.num_readers), self.recv_per_reader
        self.sender_cores = [ttnn.CoreCoord(0, r) for r in readers]
        self.receiver_sets = [
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, r), ttnn.CoreCoord(recv, r))]) for r in readers
        ]
        self.sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.sender_cores])
        ring_cores = [ttnn.CoreCoord(x, r) for r in readers for x in range(1, recv + 1)]
        self.ring_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in ring_cores])
        self.worker_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(recv, self.num_readers - 1))]
        )
        self.dram_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in readers])

    def padded_k(self, k):
        return -(-k // (self.size * ttnn.TILE_SIZE)) * self.size * ttnn.TILE_SIZE

    def block_tiles(self, k, n):
        """Tiles in one K-block of one bank's shard, as the reader stages it."""
        return self.padded_k(k) // self.size // ttnn.TILE_SIZE * (n // self.num_readers // ttnn.TILE_SIZE)

    def dram_width_sharded(self, rows, n):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(self.dram_crs, [rows, n // self.num_readers], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def l1_width_sharded(self, shard_width):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.ring_crs, [ttnn.TILE_SIZE, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
        )


def _run_padded_k_prefetch(device, weights, num_layers, consume_padded_k, enable_performance_mode):
    """Prefetches ``weights`` ((K, dtype) per weight of one layer, N = 768) for ``num_layers`` layers in
    one ttnn.dram_prefetcher call and checks each gather_in0 matmul output against in0[:, :K] @ W."""
    ring = _PaddedKRing()
    tile = ttnn.TILE_SIZE
    n = 768  # 12 tiles per bank, 6 per receiver
    tile_bytes = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
    nan = float("nan")

    # The NaN prefill: unpadded (one tile row per block), bf16, and wide enough that each of its blocks
    # is at least as large as any weight's block, so it covers every part of the triple-buffered
    # staging slots a later block uses.
    prefill_k = prefill_n = ring.size * tile
    max_weight_block_bytes = max(ring.block_tiles(k, n) * tile_bytes[dtype] for k, dtype in weights)
    while ring.block_tiles(prefill_k, prefill_n) * tile_bytes[ttnn.bfloat16] < max_weight_block_bytes:
        prefill_n += ring.size * tile
    tensors = [(prefill_k, prefill_n, ttnn.bfloat16)] + [(k, n, dtype) for k, dtype in weights]

    # Large enough for the reader's triple-buffered staging CB and one whole tensor per receiver.
    max_tile_bytes = max(tile_bytes[dtype] for _, _, dtype in tensors)
    max_block_tiles = max(ring.block_tiles(k, width) for k, width, _ in tensors)
    gcb_size = (max(3 * max_block_tiles, max_block_tiles // ring.recv_per_reader * ring.size) + 64) * max_tile_bytes
    gcb = ttnn.create_global_circular_buffer(device, list(zip(ring.sender_cores, ring.receiver_sets)), gcb_size)

    shape_carriers, data_tensors, torch_weights = [], [], []
    for layer in range(num_layers):
        for i, (k, width, dtype) in enumerate(tensors):
            torch_weight = torch.full((k, width), nan) if i == 0 else torch.randn(k, width)
            # Rows [0, K) are the weight; every row after them, up to ring.size tile rows past the padded K, is NaN.
            nan_rows = ring.padded_k(k) - k + ring.size * tile
            data_tensor = ttnn.from_torch(
                torch.cat([torch_weight, torch.full((nan_rows, width), nan)]),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ring.dram_width_sharded(k + nan_rows, width),
            )
            data_tensors.append(data_tensor)
            # The reference is what the device holds (bf16 / bfp8 / bfp4 rounding of the real rows).
            torch_weights.append(ttnn.to_torch(data_tensor).float().reshape(-1, width)[:k])
            if layer == 0:
                shape_carriers.append(
                    ttnn.from_torch(
                        torch.zeros(k, width),
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ring.dram_width_sharded(k, width),
                    )
                )

    # One address per (layer, tensor) on every reader core, in prefetch order.
    tensor_addrs = torch.tensor([[t.buffer_address() for t in data_tensors]] * ring.num_readers, dtype=torch.int64)
    tt_tensor_addrs = ttnn.from_torch(
        tensor_addrs,
        dtype=ttnn.uint32,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ring.sender_crs, [1, len(data_tensors)], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    # One matmul per prefetched tensor, in prefetch order. With a global_cb the gather_in0 matmul takes
    # in1 from the GlobalCircularBuffer, not from b: b only supplies K, N and the in1 dtype. In consume
    # mode b is a padded-K zeros tensor, so compute runs over the whole padded K, pad rows included.
    torch_in0s, matmul_args = [], []
    for i, (k, width, dtype) in enumerate(tensors):
        k_pad = ring.padded_k(k)
        k_mm = k_pad if consume_padded_k else k
        torch_in0 = torch.randn(1, 1, tile, k_mm)
        torch_in0s.append(torch_in0.bfloat16().float()[0, 0])
        tt_in0 = ttnn.from_torch(
            torch_in0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ring.l1_width_sharded(k_pad // ring.size),
        )
        tt_b = shape_carriers[i]
        if k_mm != k:
            tt_b = ttnn.from_torch(
                torch.zeros(k_mm, width),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ring.dram_width_sharded(k_mm, width),
            )
        # The gather_in0 factory derives in0_block_w from the in0 shard width (k_pad / ring size); the
        # config value only has to pass validation, which wants it to divide the activation's K in tiles.
        in0_block_w = k_pad // ring.size // tile
        while (k_mm // tile) % in0_block_w:
            in0_block_w -= 1
        per_core_n = width // ring.size // tile
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(ring.recv_per_reader, ring.num_readers),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=max(w for w in range(1, 5) if per_core_n % w == 0),
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            num_global_cb_receivers=ring.recv_per_reader,
            hop_cores=ttnn.CoreRangeSet(set()),
        )
        matmul_args.append((tt_in0, tt_b, program_config, ring.l1_width_sharded(width // ring.size)))
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    sub_device_manager = device.create_sub_device_manager(
        [ttnn.SubDevice([ring.sender_crs]), ttnn.SubDevice([ring.worker_crs])], 0
    )
    device.load_sub_device_manager(sub_device_manager)
    worker_sub_device_id = ttnn.SubDeviceId(1)
    try:
        ttnn.dram_prefetcher(
            [*shape_carriers, tt_tensor_addrs],
            num_layers,
            global_cb=gcb,
            enable_performance_mode=enable_performance_mode,
        )
        device.set_sub_device_stall_group([worker_sub_device_id])
        tt_outputs = []
        for _ in range(num_layers):
            for tt_in0, tt_b, program_config, output_mem_config in matmul_args:
                tt_out = ttnn.matmul(
                    tt_in0,
                    tt_b,
                    program_config=program_config,
                    memory_config=output_mem_config,
                    compute_kernel_config=compute_kernel_config,
                    global_cb=gcb,
                    sub_device_id=worker_sub_device_id,
                )
                tt_outputs.append(ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG))
        device.reset_sub_device_stall_group()
        outputs = [ttnn.to_torch(t).float().reshape(tile, -1) for t in tt_outputs]
        ttnn.synchronize_device(device)
    finally:
        device.reset_sub_device_stall_group()
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sub_device_manager)

    failures = []
    for idx, (output, torch_weight) in enumerate(zip(outputs, torch_weights)):
        layer, i = divmod(idx, len(tensors))
        num_bad = (~torch.isfinite(output)).sum().item()
        if i == 0:  # the NaN prefill's output is not checked
            logger.info(f"layer {layer}, NaN prefill: {num_bad}/{output.numel()} outputs are not finite")
            continue
        k, _, dtype = tensors[i]
        what = f"layer {layer}, weight {i - 1} (K {k}, padded K {ring.padded_k(k)}, {dtype})"
        if num_bad:
            failures.append(f"{what}: {num_bad}/{output.numel()} outputs are not finite")
            continue
        expected = torch_in0s[i][:, :k] @ torch_weight  # the pad rows must contribute nothing
        pcc_passed, pcc_message = check_with_pcc_without_tensor_printout(expected, output, 0.9999)
        max_err = (output - expected).abs().max().item()
        logger.info(f"{what}: PCC {pcc_message}, max |error| {max_err:.4f} (max |expected| {expected.abs().max():.2f})")
        # The operands are what the device holds and accumulation is fp32, so the error is the bf16
        # rounding of the output, <= 2**-7 relative even if the packer truncates.
        if not pcc_passed or not torch.allclose(output, expected, rtol=2**-6, atol=0.1):
            failures.append(f"{what}: PCC {pcc_message}, max |output - in0[:, :K] @ W| {max_err:.4f}")
    assert not failures, (
        f"activation K {'padded' if consume_padded_k else '== weight K'}, enable_performance_mode="
        f"{enable_performance_mode}: output != in0[:, :K] @ W. With a padded activation K this means the pad rows "
        "did not arrive as zeros: either the reader read past the end of the weight shard (the rows after it are "
        "NaN: non-finite outputs), or it did not zero-fill them and stale staging data went through (the NaN "
        "prefill: non-finite outputs; an earlier block's weights: a PCC miss).\n" + "\n".join(failures)
    )


@skip_for_slow_dispatch()
@pytest.mark.parametrize("enable_performance_mode", [False, True], ids=["perf_off", "perf_on"])
@pytest.mark.parametrize(
    "weights, num_layers",
    [
        # (K, dtype) per weight of one layer. Ring 4, so K is padded to a multiple of 128.
        ([(288, ttnn.bfloat16)], 1),  # 9 tile rows -> 12: blocks 0-2 real, block 3 all pad
        ([(320, ttnn.bfloat16)], 1),  # 10 -> 12: block 3 = 1 real + 2 pad rows
        ([(160, ttnn.bfloat8_b)], 1),  # 5 -> 8: block 2 = 1 real + 1 pad, block 3 all pad
        # bfp4 (5 -> 8, 576 B pad tiles), an unpadded weight, and 1 -> 4 (blocks 1-3 all pad); two layers
        ([(160, ttnn.bfloat4_b), (256, ttnn.bfloat16), (32, ttnn.bfloat8_b)], 2),
    ],
    ids=["k288-bf16", "k320-bf16", "k160-bfp8", "k160bfp4-k256bf16-k32bfp8-2layers"],
)
def test_dram_prefetcher_padded_k(device, weights, num_layers, enable_performance_mode, function_level_defaults):
    _run_padded_k_prefetch(
        device, weights, num_layers, consume_padded_k=True, enable_performance_mode=enable_performance_mode
    )


@skip_for_slow_dispatch()
def test_dram_prefetcher_padded_k_production(device, function_level_defaults):
    # Activation K == weight K, as the models run it: guards the production path of the padded-K reader.
    _run_padded_k_prefetch(
        device,
        [(320, ttnn.bfloat8_b), (160, ttnn.bfloat4_b)],
        2,
        consume_padded_k=False,
        enable_performance_mode=True,
    )
