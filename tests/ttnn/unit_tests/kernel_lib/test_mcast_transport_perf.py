# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware multicast vs chain unicast on the SAME rectangular receiver set.

One fixed sender at the row start pushes ROUNDS payloads to a 1xN row (sender included).
Each core records its loop time in wall-clock cycles; the sender's time is the end-to-end
figure per family. Chain mode on a rectangle needs McastConfig.transfer_mode_override.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import KERNEL_DIR, make_cb

ROUNDS = 256
PAGE = 2048
BYTES = [32, 2048, 16384]
SHAPES = [(2, 1), (4, 1), (11, 1), (11, 4), (11, 8)]
MODES = [ttnn.TransferMode.Multicast, ttnn.TransferMode.ChainUnicast]


def _run(device, shape, nbytes, mode, handshake=True):
    width, height = shape
    size = device.compute_with_storage_grid_size()
    if width > size.x or height > size.y:
        pytest.skip(f"needs {width}x{height}, grid has {size.x}x{size.y}")
    row = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))])
    cores = [(x, y) for y in range(height) for x in range(width)]

    config = ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=handshake, transfer_mode_override=mode)
    family = ttnn.McastFamily(device, config)
    family.add_group(row, [ttnn.CoreCoord(0, 0)])
    family.prepare_arguments()

    # generic_op requires at least one input; the kernel never reads it.
    dummy = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(cores), 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ct = [ROUNDS, nbytes] + list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [output.buffer_address(), i]
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_transport_perf.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=row,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor()
    family.attach(descriptor, "mcast", [kernel])
    ct_off = dict(kernel.named_compile_time_args)["mcast_ct_offset"]
    flags = kernel.compile_time_args[ct_off + 5]
    assert (flags >> 3) & 3 == int(mode == ttnn.TransferMode.ChainUnicast), "host did not honour override"
    payload_page = max(PAGE, nbytes)
    descriptor.cbs = [make_cb(0, row, pages=1, page_bytes=payload_page), make_cb(1, row, pages=1)]
    descriptor.kernels = [kernel]

    out = ttnn.generic_op([dummy, output], descriptor)
    ttnn.synchronize_device(device)
    words = ttnn.to_torch(out).view(torch.int32).reshape(len(cores), -1)
    cycles = (words[:, 0].to(torch.int64) & 0xFFFFFFFF) | (words[:, 1].to(torch.int64) << 32)
    roles = words[:, 2].tolist()
    assert roles[0] == 1 and all(r == 2 for r in roles[1:]), roles
    assert all(words[:, 3].tolist()) == (mode == ttnn.TransferMode.ChainUnicast)
    return cycles[0].item() / ROUNDS, cycles[1:].max().item() / ROUNDS


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("nbytes", BYTES)
def test_transport_compare(device, shape, nbytes):
    mcast_sender, mcast_recv = _run(device, shape, nbytes, ttnn.TransferMode.Multicast)
    chain_sender, chain_recv = _run(device, shape, nbytes, ttnn.TransferMode.ChainUnicast)
    logger.info(
        f"cores={shape[0] * shape[1]:3d} bytes={nbytes:5d} | cycles/round sender: mcast {mcast_sender:7.0f} chain {chain_sender:7.0f} "
        f"chain/mcast x{chain_sender / mcast_sender:.2f} | slowest receiver: mcast {mcast_recv:7.0f} chain {chain_recv:7.0f}"
    )


@pytest.mark.parametrize("shape", [(4, 1), (11, 8)])
@pytest.mark.parametrize("nbytes", [32, 16384])
def test_multicast_no_handshake_reference(device, shape, nbytes):
    # Fire-and-forget multicast is the floor: no consumer-ready round trip at all.
    sender, recv = _run(device, shape, nbytes, ttnn.TransferMode.Multicast, handshake=False)
    logger.info(f"cores={shape[0] * shape[1]:3d} bytes={nbytes:5d} | no-handshake mcast cycles/round sender {sender:7.0f} recv {recv:7.0f}")
