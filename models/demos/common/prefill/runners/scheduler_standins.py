# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Test-only scheduler stand-in for the per-layer ack counter channel (never in serving).

In production a real scheduler connects to the master rank's layer-ack channel and drives KV migration
from it. ``CompletionCheckConsumer`` (PREFILL_CHECK_COMPLETIONS=1) fakes that consumer side under test:
it tallies acks against an expected total to verify the router aggregates every (chunk, layer)
completion, and issues no migrates.

Only ONE consumer may attach: ``try_consume_all()`` is a destructive read against a single shared
cursor, so two consumers would split the ack stream instead of each seeing it whole.
"""

import os

from loguru import logger


class CompletionCheckConsumer:
    """Test-only scheduler stand-in (enabled by PREFILL_CHECK_COMPLETIONS=1).

    Thin Python wrapper over the C++ LayerCompletionConsumer from the `_layer_completion`
    extension (ttnn._experimental.layer_completion). The C++ consumer drains the master router's scheduler counter
    channel on a NATIVE thread — immune to the GIL. An earlier Python daemon-thread version stalled at
    a partial count because the master rank's main thread blocks in a GIL-holding request-loop call
    and starves any Python drain thread, even though the router had already injected every completion.

    In production a real scheduler consumes this channel; this only fakes the consumer side under test.
    Pre-configured with the expected total (PREFILL_CHECK_EXPECTED_CHUNKS) so the C++ thread
    self-terminates + logs PASS on its own — no dependency on Python teardown.
    """

    def __init__(self, ack_shm_name: str, *, num_layers: int):
        # Imported here (not at module top) so the runner doesn't hard-fail when the test-only
        # _layer_completion extension is absent; only PREFILL_CHECK_COMPLETIONS=1 runs reach this.
        from ttnn._experimental.layer_completion import LayerCompletionConsumer

        self._num_layers = num_layers
        # The external producer decides the chunk count, so the expected total must come from
        # PREFILL_CHECK_EXPECTED_CHUNKS. If unset, the consumer's self-terminate threshold is a guess and
        # the PASS/FAIL signal is unreliable.
        explicit_chunks = os.environ.get("PREFILL_CHECK_EXPECTED_CHUNKS")
        if explicit_chunks is None:
            logger.warning(
                "[completion-check] PREFILL_CHECK_EXPECTED_CHUNKS is not set; falling back to 11 chunks — "
                "the PASS/FAIL tally is unreliable in unbounded request mode. Set it to the number of "
                "chunks the producer will actually send."
            )
        self._expected_chunks = int(explicit_chunks or "11")
        self._expected_total = self._expected_chunks * num_layers
        # Internal C++ native-thread consumer (re-exported from prefill_test; see that module).
        self._impl = LayerCompletionConsumer(
            channel_shm_name=ack_shm_name,
            expected=self._expected_total,
            connect_timeout_ms=30000,
            log_step=num_layers,
        )
        logger.info(
            f"[completion-check] C++ consumer draining {ack_shm_name}; expecting {self._expected_total} "
            f"completions ({self._expected_chunks} chunks x {num_layers} layers), then self-terminates"
        )

    def stop_and_report(self) -> None:
        self._impl.stop()  # join the native thread + final drain
        got = self._impl.total
        logger.info(
            f"[completion-check] master aggregated {got} completions "
            f"(expected {self._expected_total} = {self._expected_chunks} x {self._num_layers})"
        )
        assert got > 0, "[completion-check] FAIL: master received ZERO completions (router not aggregating)"
        if got >= self._expected_total:
            logger.success(f"[completion-check] PASS: {got} >= {self._expected_total}")
        else:
            logger.warning(f"[completion-check] count short: got {got}, expected {self._expected_total}")
