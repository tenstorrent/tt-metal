# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""H2D/D2H socket *driver* (host/connector side) for the abstract matmul op test.

This is the connector half of a two-process test that exercises the host <-> device
PCIe socket ops end to end. It mirrors ``socket_driver.py`` from the inworld_tts
demo: this process owns **no** ``MeshDevice``; it only attaches to sockets exported
by ``h2d_d2h_matmul_server.py`` and drives raw H2D/D2H traffic through them.

Flow:

    driver --(M, K input)--> server --(M, N = input @ weight)--> driver

Each iteration the driver pushes a fresh random ``(M, K)`` input (H2D), reads back
the ``(M, N)`` matmul result (D2H), and compares it against ``input @ weight`` (the
weight is loaded from the .pth the server saved) with PCC.

Start the server (on its own chip) first, then run this driver:

    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    python tests/ttnn/unit_tests/operations/ccl/h2d_d2h_matmul_driver.py

The H2D/D2H socket connector path is Blackhole-only.
"""

import argparse
import os
import sys
import time

import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Must match the server. Override on *both* processes if you change them.
DEFAULT_RUN_ID = "h2d_d2h_matmul"
DEFAULT_WEIGHT_PATH = "/dev/shm/tt_h2d_d2h_matmul_weight.pt"
DEFAULT_M, DEFAULT_K, DEFAULT_N = 32, 64, 64
DEFAULT_NUM_ITERATIONS = 8
DEFAULT_PCC = 0.99

_BYTES_PER_ELEMENT = 2  # bfloat16
_CONNECT_TIMEOUT_MS = 60_000
_READY_TIMEOUT_S = 600


def _socket_ids(run_id: str) -> tuple[str, str]:
    """Descriptor ids shared between the server and the driver."""
    return f"{run_id}_h2d", f"{run_id}_d2h"


def _wait_for_path(path: str, what: str, timeout_s: int = _READY_TIMEOUT_S) -> None:
    """Block until ``path`` exists (a descriptor or the weight file)."""
    deadline = time.time() + timeout_s
    while not os.path.exists(path):
        if time.time() > deadline:
            raise TimeoutError(f"{what} {path} not found; is the server running?")
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="H2D/D2H socket matmul driver (connector side)")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID, help="Shared descriptor id prefix")
    parser.add_argument("--weight-path", type=str, default=DEFAULT_WEIGHT_PATH, help="Path to the saved matmul weight")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="Input rows")
    parser.add_argument("--K", type=int, default=DEFAULT_K, help="Contraction dim")
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="Output cols")
    parser.add_argument("--num-iterations", type=int, default=DEFAULT_NUM_ITERATIONS, help="Matmul iterations to drive")
    parser.add_argument("--pcc", type=float, default=DEFAULT_PCC, help="Minimum acceptable PCC")
    parser.add_argument("--seed", type=int, default=0, help="Input RNG seed")
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    input_page_size_bytes = K * _BYTES_PER_ELEMENT
    output_page_size_bytes = N * _BYTES_PER_ELEMENT

    h2d_id, d2h_id = _socket_ids(args.run_id)

    logger.info("Waiting for server socket descriptors and weight file...")
    _wait_for_path(f"/dev/shm/tt_h2d_{h2d_id}.bin", "H2D descriptor")
    _wait_for_path(f"/dev/shm/tt_d2h_{d2h_id}.bin", "D2H descriptor")
    _wait_for_path(args.weight_path, "weight file")

    torch_weight = torch.load(args.weight_path).to(torch.float32)
    if tuple(torch_weight.shape) != (K, N):
        raise ValueError(
            f"weight shape {tuple(torch_weight.shape)} from {args.weight_path} does not match "
            f"--K={K} --N={N}; the driver and server args must agree."
        )

    h2d_socket = ttnn.H2DSocket.connect(h2d_id, timeout_ms=_CONNECT_TIMEOUT_MS)
    d2h_socket = ttnn.D2HSocket.connect(d2h_id, timeout_ms=_CONNECT_TIMEOUT_MS)
    h2d_socket.set_page_size(input_page_size_bytes)
    d2h_socket.set_page_size(output_page_size_bytes)

    # Drop any pages left over from a previous run and resync before driving.
    d2h_socket.discard_pending_pages()
    d2h_socket.barrier(1000)
    d2h_socket.discard_pending_pages()

    logger.info(f"Connected: H2D={h2d_id} D2H={d2h_id}, M={M} K={K} N={N}, pcc>={args.pcc}")
    generator = torch.Generator().manual_seed(args.seed)

    for iteration in range(args.num_iterations):
        torch_input = torch.randn(M, K, generator=generator, dtype=torch.float32)
        torch_input_bf16 = torch_input.to(torch.bfloat16).contiguous()

        h2d_socket.write_tensor(torch_input_bf16)

        result = torch.zeros(M, N, dtype=torch.bfloat16)
        d2h_socket.read_tensor(result)

        # bfloat16 matmul on device is not bit-exact vs fp32, so compare with PCC.
        torch_expected = torch_input @ torch_weight
        assert_with_pcc(torch_expected, result.to(torch.float32), pcc=args.pcc)
        logger.info(f"Iteration {iteration + 1}/{args.num_iterations} matched (pcc>={args.pcc})")

    logger.info("Driver done: all iterations matched the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
