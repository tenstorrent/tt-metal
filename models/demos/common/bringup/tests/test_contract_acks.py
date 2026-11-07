# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The serving contract's ack-timing check (testing/contract.AckReads, ``acks_early``), host only (a fake table).

A history (KV) config whose acked block changes after its ack still counts as early, at any chunk. A Snapshot config
(tensor_type Snapshot: fixed-size state the model rewrites on every chunk, shipped by tt-d-gen with the request's last
chunk only) is read at the last chunk's acks only: rewritten before the last chunk it passes; changed after the last
chunk's ack it fails; never looked up past its declared max_sequence_length (tt-metal's table refuses that).
"""

from __future__ import annotations

import types

from models.demos.common.bringup.testing.contract import AckReads, is_snapshot_config

BLOCK = 32


class FakeTable:
    """configs: [(max_sequence_length, tensor_type name)]; memory: {(layer, position, config): bytes}."""

    def __init__(self, configs):
        self.cfgs = [
            types.SimpleNamespace(
                max_sequence_length=m, chunk_n_tokens=BLOCK, tensor_type=types.SimpleNamespace(name=t)
            )
            for m, t in configs
        ]
        self.mem = {}

    def num_configs(self):
        return len(self.cfgs)

    def config(self, cid=0):
        return self.cfgs[cid]

    def _check(self, position, cid):
        assert 0 <= position < self.cfgs[cid].max_sequence_length, (position, cid)  # (tt-metal TT_FATALs here)

    def lookup(self, layer, position, slot, cid):
        self._check(position, cid)
        return types.SimpleNamespace(size_bytes=BLOCK)

    def read_device_chunk(self, layer, position, slot, config_id):
        self._check(position, config_id)
        return self.mem.get((layer, position, config_id), b"\0")

    def write(self, layer, position, cid, data: bytes):
        self.mem[(layer, position, cid)] = data


SEQ, CHUNK, LAYERS = 256, 128, 2
HIST, SNAP = 0, 1


def run(mutate=None):
    """Two chunks of a 256-token request, two layers; config 0 history (every position), config 1 a 128-position
    Snapshot rewritten by every chunk before its ack. ``mutate(table, chunk, layer)`` runs after each ack."""
    t = FakeTable([(SEQ, "History"), (128, "Snapshot")])
    acks = AckReads(t, slot=0, last_chunk=SEQ // CHUNK - 1)
    for c in range(SEQ // CHUNK):
        a, b = c * CHUNK, (c + 1) * CHUNK
        for layer in range(LAYERS):
            for p in range(a, b, BLOCK):
                t.write(layer, p, HIST, bytes([layer, c, p % 251]))
            for p in range(0, 128, BLOCK):  # the snapshot: every chunk rewrites all of it
                t.write(layer, p, SNAP, bytes([layer, c, 7]))
            acks.on_ack(layer, c, a, b)
            if mutate is not None:
                mutate(t, c, layer)
    return acks


def test_snapshot_type():
    assert is_snapshot_config(types.SimpleNamespace(tensor_type=types.SimpleNamespace(name="Snapshot")))
    assert is_snapshot_config(types.SimpleNamespace(tensor_type="KvChunkTensorType.Snapshot"))
    assert not is_snapshot_config(types.SimpleNamespace(tensor_type=types.SimpleNamespace(name="History")))
    assert not is_snapshot_config(types.SimpleNamespace())


def test_snapshot_rewritten_before_last_chunk_passes():
    acks = run()
    assert acks.early() == 0
    snap_reads = [k for k in acks.reads if k[3] == SNAP]
    assert snap_reads and all(k[1] == SEQ // CHUNK - 1 for k in snap_reads), snap_reads  # (last chunk only)
    assert sorted({k[2] for k in snap_reads}) == [0, 32, 64, 96]  # (every record, none past 128)
    assert {k[1] for k in acks.reads if k[3] == HIST} == {0, 1}  # (history: every chunk, as before)


def test_history_changed_after_ack_fails():
    def late(t, c, layer):
        if c == 0 and layer == 0:  # chunk 0's first block of layer 0 rewritten after its ack
            t.write(0, 0, HIST, b"late")

    assert run(late).early() == 1


def test_snapshot_changed_after_last_ack_fails():
    def late(t, c, layer):
        if c == SEQ // CHUNK - 1 and layer == 0:
            t.write(0, 64, SNAP, b"late")

    assert run(late).early() == 1
