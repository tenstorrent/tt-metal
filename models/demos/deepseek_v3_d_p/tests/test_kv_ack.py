# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The KV pad-zero and migration ack, exercised on host with both transports faked.

`zero_pad_and_ack` was lifted verbatim out of `TtPrefillBlock.forward` so Kimi-K3 — whose blocks are
not that class, because only 24 of its 93 layers write KV — can hand off to migration through the
same code instead of a second copy. The move is mechanical, but what it moved is not: two ack
transports with different flush requirements, a layout guard that silently skips the zero for sparse
caches, and two argument forms for the zero op depending on whether the caller is traced.

None of that needs a device to pin. The op and the sync are the only two things it touches, so
faking them turns the whole decision tree into a host test — which is worth having, because the
failure mode of getting it wrong is a migration worker copying pre-zero data, and that surfaces as
a rare wrong answer downstream rather than an error here.
"""

from types import SimpleNamespace

import ttnn
from models.demos.deepseek_v3_d_p.tt.kv_ack import zero_pad_and_ack


class _Recorder:
    """Stands in for `zero_padded_kv_cache` and `synchronize_device`."""

    def __init__(self):
        self.zero_calls = []
        self.syncs = 0
        self.socket_syncs = []

    def install(self, monkeypatch):
        monkeypatch.setattr(
            ttnn.experimental.deepseek_prefill,
            "zero_padded_kv_cache",
            lambda *args: self.zero_calls.append(args),
        )
        monkeypatch.setattr(
            ttnn.experimental.deepseek_prefill,
            "outbound_socket_service_sync",
            lambda service, metadata=None: self.socket_syncs.append((service, metadata)),
        )
        monkeypatch.setattr(ttnn, "synchronize_device", lambda device: setattr(self, "syncs", self.syncs + 1))


def _cache(layout=ttnn.TILE_LAYOUT):
    return SimpleNamespace(storage=SimpleNamespace(layout=layout))


def _call(**overrides):
    args = dict(
        kvpe_cache=_cache(),
        mesh_device="MESH",
        cache_layer_idx=2,
        cache_user_id=1,
        layer_num=6,
        sp_factor=8,
        sp_axis=0,
        global_layer_idx=23,
        seq_len_local=640,
        actual_end=5000,
        metadata=None,
        d2h_service=None,
        record_dev=None,
        on_layer_complete=None,
        trace_controller=None,
    )
    args.update(overrides)
    return args


def test_no_transport_is_a_no_op(monkeypatch):
    """Safe to call unconditionally: a block with no migration engine wired does nothing."""
    rec = _Recorder()
    rec.install(monkeypatch)
    zero_pad_and_ack(**_call())
    assert rec.zero_calls == [] and rec.syncs == 0 and rec.socket_syncs == []


def test_host_callback_flushes_before_acking(monkeypatch):
    """The pipeline transport is a HOST callback, so the zero must be flushed first.

    The migration worker reads the cache over NoC out-of-band from the ttnn queue; without the sync
    it can copy pre-zero data. The ack carries the GLOBAL layer index, because the scheduler orders
    acks across pipeline ranks.
    """
    rec = _Recorder()
    rec.install(monkeypatch)
    acked = []
    zero_pad_and_ack(**_call(on_layer_complete=acked.append))

    assert len(rec.zero_calls) == 1
    assert rec.syncs == 1, "host-callback ack must be preceded by a device sync"
    assert acked == [23], "the ack carries the global layer index, not the cache slot"


def test_device_ack_needs_no_sync(monkeypatch):
    """The d2h transport enqueues its ack on the same queue, so zero-completion is implied."""
    rec = _Recorder()
    rec.install(monkeypatch)
    zero_pad_and_ack(**_call(d2h_service="SVC", record_dev="REC"))

    assert len(rec.zero_calls) == 1
    assert rec.socket_syncs == [("SVC", "REC")]
    assert rec.syncs == 0, "the device-op ack must NOT force a host sync"


def test_trace_controller_takes_precedence_only_with_an_ack(monkeypatch):
    """A controller without an ack callback is the test path and must not swallow the ack."""
    rec = _Recorder()
    rec.install(monkeypatch)

    acked, controller_acks = [], []
    live = SimpleNamespace(has_layer_ack=lambda: True, layer_ack=controller_acks.append)
    zero_pad_and_ack(**_call(on_layer_complete=acked.append, trace_controller=live))
    assert controller_acks == [23] and acked == [] and rec.syncs == 0

    inert = SimpleNamespace(has_layer_ack=lambda: False, layer_ack=controller_acks.append)
    zero_pad_and_ack(**_call(on_layer_complete=acked.append, trace_controller=inert))
    assert acked == [23] and rec.syncs == 1


def test_row_major_cache_skips_the_zero_but_still_acks(monkeypatch):
    """A DSA-sparse cache is ROW_MAJOR and the zero op asserts TILE, so the zero is skipped.

    The ack is not: migration still needs to know the layer is done.
    """
    rec = _Recorder()
    rec.install(monkeypatch)
    acked = []
    zero_pad_and_ack(**_call(kvpe_cache=_cache(ttnn.ROW_MAJOR_LAYOUT), on_layer_complete=acked.append))

    assert rec.zero_calls == [], "the zero op is TILE-only and must be skipped for a sparse cache"
    assert acked == [23]


def test_metadata_and_scalar_forms_pass_different_arguments(monkeypatch):
    """Traced callers pass on-device 1-element tensors; eager callers pass Python scalars."""
    rec = _Recorder()
    rec.install(monkeypatch)

    zero_pad_and_ack(**_call(on_layer_complete=lambda _idx: None))
    scalar_args = rec.zero_calls[-1]
    assert scalar_args == (rec.zero_calls[-1][0], 1, 2, 6, 5000, 640 * 8, 0)

    zero_pad_and_ack(**_call(metadata=("SLOT", "START", "END"), on_layer_complete=lambda _idx: None))
    traced_args = rec.zero_calls[-1]
    assert traced_args[1:] == ("SLOT", "END", 2, 6, 640 * 8, 0), "traced form reads slot and end on device"


def test_missing_end_and_metadata_is_rejected(expect_error):
    """Without either, the zero has no idea where the pad window starts."""
    with expect_error(AssertionError, "actual_end or metadata required"):
        zero_pad_and_ack(**_call(actual_end=None, on_layer_complete=lambda _idx: None))


def test_d2h_without_record_dev_is_rejected(expect_error):
    with expect_error(AssertionError, "record_dev required"):
        zero_pad_and_ack(**_call(d2h_service="SVC", record_dev=None))
