# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the producer's PREFILL_PRODUCER_DRAIN_ACKS contract (a perf-only leg's single producer). With it, the
producer drains the layer-completion channel and fails unless one ack per layer per chunk arrived, still sending the
SHUTDOWN sentinel; without it and without PCC it is a pure token feeder that never touches the channel."""

import types
from unittest import mock

import pytest

from models.demos.common.prefill.runners import prefill_producer as producer

CHUNKS = 4
LAYERS = 3
METADATA_BYTES = 16


@pytest.fixture
def run_main(monkeypatch):
    """Run producer.main() single-rank with every device and channel boundary stubbed; return
    (exit code, whether the completion channel was connected, the pushes sent)."""
    # main() reloads these module globals from the env; set them here instead, so teardown restores the module.
    for name, value in {
        "SP_FACTOR": 1,
        "CHUNK_SIZE": 1024,
        "MAX_SEQ_LEN": CHUNKS * 1024,
        "NUM_LAYERS": LAYERS,
        "NUM_ACK_LAYERS": LAYERS,
        "MTP_LEVELS": 0,
        "MTP_DECODE_LEVELS": 0,
        "FORWARD_CHUNKS": 1,
        "CHUNK_METADATA_SIZE_BYTES": METADATA_BYTES,
        "PASS_SCHEDULE": None,
        "ADAPTER": types.SimpleNamespace(verify_layer_completions=False),
    }.items():
        monkeypatch.setattr(producer, name, value, raising=False)
    monkeypatch.setattr(producer, "_load_env_config", lambda: None)
    monkeypatch.setattr(producer, "check_packed_forward_support", lambda *a, **k: None)
    monkeypatch.setattr(producer, "packed_forward_capabilities", lambda adapter: {})
    monkeypatch.setattr(producer, "install_crash_diagnostics", lambda: None)
    monkeypatch.setattr(producer, "_mr_config", lambda: (0, 1))
    monkeypatch.setattr("sys.argv", ["prefill_producer"])
    for key in ("PREFILL_PRODUCER_MANIFEST", "PREFILL_NUM_USERS", "PREFILL_PRODUCER_WARMUP_CHUNKS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PREFILL_PRODUCER_CHUNKS", str(CHUNKS))
    monkeypatch.setenv("PREFILL_SEND_SHUTDOWN", "1")

    def run(*, drain_acks, channel=True, received=None):
        monkeypatch.setenv("PREFILL_PRODUCER_CHECK_PCC", "0")
        monkeypatch.setenv("PREFILL_PRODUCER_DRAIN_ACKS", "1" if drain_acks else "0")
        pushes = []
        stats = types.SimpleNamespace(
            total_chunks=CHUNKS, total_pushes=CHUNKS, completed=1, wall_s=1.0, push_ms=[1.0], resident={}
        )
        with mock.patch.object(producer.ttnn, "H2DStreamService") as service, mock.patch.object(
            producer, "connect_layer_completion_channel", return_value=object() if channel else None
        ) as connect, mock.patch.object(
            producer,
            "start_layer_completion_drain",
            side_effect=lambda ch, **kw: None if ch is None else mock.MagicMock(),
        ), mock.patch.object(
            producer,
            "wait_for_layer_completions",
            side_effect=lambda drain, expected: 0 if drain is None else (expected if received is None else received),
        ), mock.patch.object(
            producer, "_resolve_slot_prompts", return_value=({0: "trace"}, None, {"trace": None})
        ), mock.patch.object(
            producer, "run_schedule", return_value=stats
        ), mock.patch.object(
            producer, "_push", side_effect=lambda *args: pushes.append(args)
        ), mock.patch.object(
            producer, "_shutdown_payload", return_value=(None, None, b"\0" * METADATA_BYTES)
        ):
            service.connect.return_value.payload_size_bytes.return_value = 1
            try:
                producer.main()
                rc = 0
            except SystemExit as exit_:
                rc = exit_.code
        return rc, connect.called, pushes

    return run


def test_drain_acks_passes_when_every_ack_arrives(run_main):
    rc, connected, pushes = run_main(drain_acks=True)
    assert (rc, connected) == (0, True)
    assert [p[-1] for p in pushes] == [b"\0" * METADATA_BYTES]  # the SHUTDOWN sentinel


def test_drain_acks_fails_on_a_missing_ack_and_still_shuts_the_runner_down(run_main):
    rc, connected, pushes = run_main(drain_acks=True, received=CHUNKS * LAYERS - 1)
    assert (rc, connected) == (1, True)
    assert [p[-1] for p in pushes] == [b"\0" * METADATA_BYTES]


def test_drain_acks_fails_without_a_completion_channel(run_main):
    rc, connected, pushes = run_main(drain_acks=True, channel=False)
    assert (rc, connected) == (1, True)
    assert pushes == []


def test_feeder_without_drain_acks_never_touches_the_channel(run_main):
    rc, connected, pushes = run_main(drain_acks=False, received=0)
    assert (rc, connected) == (0, False)
    assert len(pushes) == 1
