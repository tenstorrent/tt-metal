# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Numpy-bytes <-> DistributedContext.send/recv helpers.

The bindings on DistributedContext send and receive raw bytes. We wrap them
to ship numpy arrays (the shape that all training-side payloads end up as
when they reach the host MPI primitive — see MPISocket::send in
ttnn/core/distributed/mpi_socket.cpp:44-66 for the production analogue).
"""

from __future__ import annotations

import numpy as np


# Per-element value used as the rank tag. Receiver checks tensor == sender_rank_tag(sender)
def sender_rank_tag(sender_rank: int) -> float:
    return float(sender_rank + 1)


# Default payload: float32 1024 elements = 4 KB. Big enough to exercise a real
# byte transfer; small enough to be cheap.
DEFAULT_PAYLOAD_SHAPE = (1024,)
DEFAULT_PAYLOAD_DTYPE = np.float32


def make_tagged_payload(sender_rank: int, shape=DEFAULT_PAYLOAD_SHAPE, dtype=DEFAULT_PAYLOAD_DTYPE):
    """Numpy array filled with sender_rank_tag(sender_rank)."""
    return np.full(shape, sender_rank_tag(sender_rank), dtype=dtype)


def make_constant_payload(value: float, shape=DEFAULT_PAYLOAD_SHAPE, dtype=DEFAULT_PAYLOAD_DTYPE):
    return np.full(shape, value, dtype=dtype)


def send_array(distributed_ctx, array: np.ndarray, dest: int, tag: int = 0) -> None:
    """Serialize a numpy array and send via DistributedContext.send."""
    distributed_ctx.send(array.tobytes(), dest=dest, tag=tag)


def recv_array(distributed_ctx, source: int, shape, dtype, tag: int = 0) -> np.ndarray:
    """Receive a numpy array of the expected shape/dtype via DistributedContext.recv."""
    nbytes = int(np.dtype(dtype).itemsize) * int(np.prod(shape))
    raw = distributed_ctx.recv(nbytes=nbytes, source=source, tag=tag)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def assert_array_equals_value(array: np.ndarray, expected_value: float, context: str) -> None:
    if not np.all(array == expected_value):
        sample = array.flatten()[:8].tolist()
        raise AssertionError(
            f"{context}: expected all elements == {expected_value}, " f"got first-8 sample {sample} (size {array.size})"
        )
