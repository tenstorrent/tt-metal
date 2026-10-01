# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttnn.InterProcessCounterChannel.

The channel is a POSIX-SHM-backed monotonic producer counter with two
sides: an owner (creates /dev/shm/<shm_name>, atomic-adds events via
`inject(n)`) and a connector (attaches via `connect(shm_name)`, drains
events via `try_consume_all()`).

Both roles are exercised here from the same Python process — the kernel
doesn't care that owner + connector share a PID, and same-process tests
are easier to reason about than fork()-based ones. Production has the
owner in tt-metal/ttnn (Python workload) and the connector in C++
(tt-llm-engine scheduler), but the wire contract is identical.
"""

import os

import pytest

import ttnn


def _unlink_if_exists(shm_name: str) -> None:
    """Defensive cleanup of /dev/shm/<shm_name>. The owner ctor uses
    shm_open(O_CREAT|O_EXCL) and will fail with `File exists` if a
    prior run crashed without unlinking. Missing-file errors here
    are the expected case on a clean machine."""
    path = "/dev/shm" + shm_name
    try:
        os.unlink(path)
    except FileNotFoundError:
        # Expected on clean machines: nothing to unlink.
        return


# ---------------------------------------------------------------------------
# Basic roundtrip
# ---------------------------------------------------------------------------


def test_owner_to_connector_roundtrip():
    """Create as owner → inject → connect → drain → verify counts."""
    shm_name = "/tt_counter_channel_smoke_roundtrip"
    _unlink_if_exists(shm_name)

    owner = ttnn.InterProcessCounterChannel(shm_name)
    assert owner.shm_name == shm_name

    # Publish 16 events across three injections.
    owner.inject(10)
    owner.inject(5)
    owner.inject(1)

    connector = ttnn.InterProcessCounterChannel.connect(shm_name, connect_timeout_ms=5_000)
    assert connector.shm_name == shm_name

    # First connector on a fresh segment — owner stamps clean=1 in its
    # ctor, so the first attach observes a clean prior.
    assert connector.had_clean_prior_shutdown() is True

    # All 16 visible; drain returns it; pending → 0.
    assert connector.pending() == 16
    assert connector.try_consume_all() == 16
    assert connector.pending() == 0

    # New events post-drain pick up on the next call.
    owner.inject(3)
    assert connector.try_consume_all() == 3

    # Clean teardown: connector first (stamps clean=1), owner second
    # (munmap + shm_unlink → removes the /dev/shm name).
    connector.shutdown()
    owner.shutdown()


# ---------------------------------------------------------------------------
# Role-restriction checks
# ---------------------------------------------------------------------------


def test_owner_cannot_consume():
    """inject() is owner-only; try_consume_all() / pending() are
    connector-only. Calling on the wrong role raises RuntimeError."""
    shm_name = "/tt_counter_channel_smoke_roles"
    _unlink_if_exists(shm_name)

    owner = ttnn.InterProcessCounterChannel(shm_name)
    connector = ttnn.InterProcessCounterChannel.connect(shm_name)

    with pytest.raises(RuntimeError):
        owner.try_consume_all()
    with pytest.raises(RuntimeError):
        owner.pending()
    with pytest.raises(RuntimeError):
        owner.had_clean_prior_shutdown()

    with pytest.raises(RuntimeError):
        connector.inject(1)

    connector.shutdown()
    owner.shutdown()


# ---------------------------------------------------------------------------
# Persistence across connector restarts
# ---------------------------------------------------------------------------


def test_consumer_cursor_persists_across_connector_restart():
    """The cursor lives in the SHM segment, so a fresh connector
    attaching to the same owner picks up where the previous one left
    off — no double-counting."""
    shm_name = "/tt_counter_channel_smoke_cursor_persist"
    _unlink_if_exists(shm_name)

    owner = ttnn.InterProcessCounterChannel(shm_name)
    owner.inject(7)

    c1 = ttnn.InterProcessCounterChannel.connect(shm_name)
    assert c1.try_consume_all() == 7
    c1.shutdown()

    owner.inject(4)

    c2 = ttnn.InterProcessCounterChannel.connect(shm_name)
    # c1's cursor advanced past the first 7 and that advance is durable.
    # c2 only sees the new 4.
    assert c2.try_consume_all() == 4
    c2.shutdown()

    owner.shutdown()


def test_prior_clean_shutdown_chain():
    """When the previous connector exited cleanly via shutdown(), the
    next connector observes had_clean_prior_shutdown() == True. The
    flag is single-step: it reflects the IMMEDIATELY-PREVIOUS exit,
    not the entire history."""
    shm_name = "/tt_counter_channel_smoke_clean_chain"
    _unlink_if_exists(shm_name)

    owner = ttnn.InterProcessCounterChannel(shm_name)

    # First attach — owner stamped clean=1 at ctor.
    c1 = ttnn.InterProcessCounterChannel.connect(shm_name)
    assert c1.had_clean_prior_shutdown() is True
    c1.shutdown()  # stamps clean=1 before munmap

    # Second attach — c1 exited cleanly.
    c2 = ttnn.InterProcessCounterChannel.connect(shm_name)
    assert c2.had_clean_prior_shutdown() is True
    c2.shutdown()

    owner.shutdown()


# ---------------------------------------------------------------------------
# Lifecycle / cleanup hygiene
# ---------------------------------------------------------------------------


def test_owner_shutdown_removes_segment():
    """After owner.shutdown(), the /dev/shm file is gone — a fresh
    owner ctor with the same name succeeds (no EEXIST)."""
    shm_name = "/tt_counter_channel_smoke_unlink"
    _unlink_if_exists(shm_name)

    owner1 = ttnn.InterProcessCounterChannel(shm_name)
    owner1.shutdown()
    # No EEXIST since owner1's shutdown ran shm_unlink.
    owner2 = ttnn.InterProcessCounterChannel(shm_name)
    owner2.shutdown()


def test_double_shutdown_is_noop():
    """shutdown() is idempotent — calling it twice on either role is
    a no-op via an internal exchange-on-call guard."""
    shm_name = "/tt_counter_channel_smoke_double_shutdown"
    _unlink_if_exists(shm_name)

    owner = ttnn.InterProcessCounterChannel(shm_name)
    owner.shutdown()
    owner.shutdown()  # no-op, must not raise

    # Same on the connector side. Recreate the segment first.
    owner2 = ttnn.InterProcessCounterChannel(shm_name)
    connector = ttnn.InterProcessCounterChannel.connect(shm_name)
    connector.shutdown()
    connector.shutdown()
    owner2.shutdown()


def test_owner_eexist_when_stale_segment_present():
    """If a segment with the same name already exists, the owner
    ctor refuses to recycle silently — caller is responsible for
    explicit cleanup of stale segments from crashed prior runs."""
    shm_name = "/tt_counter_channel_smoke_eexist"
    _unlink_if_exists(shm_name)

    owner_a = ttnn.InterProcessCounterChannel(shm_name)
    with pytest.raises(RuntimeError):
        ttnn.InterProcessCounterChannel(shm_name)
    owner_a.shutdown()


# ---------------------------------------------------------------------------
# Connect timeout behaviour
# ---------------------------------------------------------------------------


def test_connector_times_out_when_owner_absent():
    """connect() polls /dev/shm with a deadline; if no owner exports
    within connect_timeout_ms it raises RuntimeError."""
    shm_name = "/tt_counter_channel_smoke_timeout"
    _unlink_if_exists(shm_name)

    with pytest.raises(RuntimeError):
        ttnn.InterProcessCounterChannel.connect(shm_name, connect_timeout_ms=200)
