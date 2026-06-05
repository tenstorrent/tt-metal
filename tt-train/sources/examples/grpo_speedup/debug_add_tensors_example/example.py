# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank "orchestrator + worker" send/compute/recv debug example.

Pattern
-------

Launched under tt-run with world_size == 2:

  * Rank 0 (ORCH)   builds tensors a and b on its half of the mesh,
                    sends them to rank 1, then receives a+b and prints.
  * Rank 1 (WORKER) receives a and b on its half of the mesh, computes
                    a+b on-device, and sends the result back.

There is no Python `subprocess` style spawn: tt-run (a wrapper around
mpirun) brings up both processes simultaneously, and they coordinate
through MPI rank ids. Each rank opens its own MeshDevice scoped by
TT_VISIBLE_DEVICES from configurations/local2/rank_bindings.yaml.

The transport between ranks is selected via SOCKET_TYPE below:
  * SocketType.MPI    -- host TCP/IP via MPI; works without fabric.
  * SocketType.FABRIC -- device-to-device over the Tenstorrent ethernet
                         fabric; faster, but requires the CLUSTER graph
                         declared in mgd.textproto.

Run with:

  ./tt-train/sources/examples/grpo_speedup/debug/runner.sh
"""

from __future__ import annotations

import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')

import numpy as np
import ttml
import ttnn
from ttml.common.config import load_config
from ttml.common.utils import initialize_device


SOCKET_TYPE = ttml.core.distributed.SocketType.FABRIC

ORCH = 0
WORKER = 1

DEVICE_CONFIG_REL = "configurations/local2/device.yaml"

TENSOR_SHAPE = (1, 1, 32, 32)


def _make_tensor(values: np.ndarray) -> "ttml.autograd.Tensor":
    return ttml.autograd.Tensor.from_numpy(
        values.astype(np.float32),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
    )


def _empty_like_tensor() -> "ttml.autograd.Tensor":
    """Allocate a receive-buffer tensor with the right shape/dtype/layout.

    On the receive side the tensor's value is overwritten by socket_manager.recv,
    so the contents here don't matter; only the shape/dtype/layout do.
    """
    template = _make_tensor(np.zeros(TENSOR_SHAPE, dtype=np.float32))
    return ttml.core.empty_like(template)


def main() -> None:
    autograd_ctx = ttml.autograd.AutoContext.get_instance()

    autograd_ctx.initialize_distributed_context(*sys.argv)
    distributed_ctx = autograd_ctx.get_distributed_context()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()
    assert world_size == 2, f"This example expects world_size == 2, got {world_size}"

    here = os.path.dirname(os.path.abspath(__file__))
    yaml_config = load_config(os.path.join(here, DEVICE_CONFIG_REL))
    initialize_device(yaml_config)

    autograd_ctx.initialize_socket_manager(SOCKET_TYPE)
    socket_manager = autograd_ctx.get_socket_manager()

    # Each rank's submesh is [1, 4] (4 chips) per device.yaml. Tensor.from_numpy
    # replicates the input across every chip, so the host-side tensor returned
    # by recv carries one buffer per device. Reading it back to numpy needs a
    # composer; concat along dim 0 gives shape (num_devices, *TENSOR_SHAPE),
    # and since the data is replicated, slot 0 is representative.
    device = autograd_ctx.get_device()
    tensor_composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    role = "orchestrator" if rank == ORCH else "worker"
    print(f"[rank {rank}/{world_size}] {role} ready", flush=True)

    try:
        if rank == ORCH:
            a_np = np.full(TENSOR_SHAPE, 1.5, dtype=np.float32)
            b_np = np.full(TENSOR_SHAPE, 2.25, dtype=np.float32)
            a = _make_tensor(a_np)
            b = _make_tensor(b_np)

            print(f"[rank {ORCH}] sending a (filled with {a_np.flat[0]}) to rank {WORKER}", flush=True)
            socket_manager.send(a, distributed_ctx, WORKER)
            print(f"[rank {ORCH}] sending b (filled with {b_np.flat[0]}) to rank {WORKER}", flush=True)
            socket_manager.send(b, distributed_ctx, WORKER)

            result = _empty_like_tensor()
            print(f"[rank {ORCH}] waiting for a+b from rank {WORKER}", flush=True)
            result = socket_manager.recv(result, distributed_ctx, WORKER)

            result_full = result.to_numpy(composer=tensor_composer)
            result_np = result_full[0].flatten()
            expected = a_np.flat[0] + b_np.flat[0]
            print(
                f"[rank {ORCH}] received a+b: first 4 elements = {result_np[:4].tolist()} " f"(expected ~{expected})",
                flush=True,
            )
            assert np.allclose(
                result_np, expected, atol=1e-2
            ), f"orchestrator: result {result_np[:8]} does not match expected {expected}"
            print(f"[rank {ORCH}] OK", flush=True)

        elif rank == WORKER:
            a = _empty_like_tensor()
            b = _empty_like_tensor()

            print(f"[rank {WORKER}] waiting for a from rank {ORCH}", flush=True)
            a = socket_manager.recv(a, distributed_ctx, ORCH)
            print(f"[rank {WORKER}] waiting for b from rank {ORCH}", flush=True)
            b = socket_manager.recv(b, distributed_ctx, ORCH)

            c = a + b

            print(f"[rank {WORKER}] sending a+b to rank {ORCH}", flush=True)
            socket_manager.send(c, distributed_ctx, ORCH)
            print(f"[rank {WORKER}] OK", flush=True)
    finally:
        # Always reach the barrier + close_device so a crash on one rank
        # doesn't leave the other rank wedged in MPI collective state.
        distributed_ctx.barrier()
        autograd_ctx.close_device()
        print(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()
