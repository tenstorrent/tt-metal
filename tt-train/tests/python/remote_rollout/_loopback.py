# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


class LoopbackContext:
    """In-memory two-rank communicator for both ranks; a message to ``dest`` waits in ``queues[(dest, tag)]``."""

    def __init__(self, queues):
        self._queues = queues

    def send(self, data, dest, tag):
        self._queues[(dest, tag)].put(bytes(data))

    def recv(self, nbytes, source, tag):
        data = self._queues[(1 - source, tag)].get(timeout=30)
        assert len(data) == nbytes
        return data
