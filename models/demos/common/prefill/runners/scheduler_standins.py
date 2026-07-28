# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Test-only scheduler stand-ins for the per-layer ack counter channel (never in serving).

In production a real scheduler connects to the master rank's layer-ack channel and drives KV migration
from it. These fake that consumer side under test:

  * ``CompletionCheckConsumer`` (PREFILL_CHECK_COMPLETIONS=1) — tally acks against an expected total to
    verify the router aggregates every (chunk, layer) completion. Issues no migrates.
  * ``InterleavedMigrationDriver`` (PREFILL_MIGRATION_SELFTEST=1 + PREFILL_MIGRATION_INTERLEAVED=1) —
    migrate each chunk as its layers ack, overlapping later chunks' prefill instead of one bulk migrate.

The runner builds at most ONE: ``try_consume_all()`` is a destructive read against a single shared
cursor, so two consumers would split the ack stream instead of each seeing it whole.
"""

import os
import time

from loguru import logger

import ttnn


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
        # This consumer only runs in (unbounded) request mode, where the external producer — NOT
        # PREFILL_STANDALONE_NCHUNKS — determines the chunk count. So the expected total must come from
        # PREFILL_CHECK_EXPECTED_CHUNKS; NCHUNKS is deliberately not consulted (it's commonly set via the
        # standalone global_env and would silently pick a wrong-but-confident count). If unset, the
        # consumer's self-terminate threshold is a guess and the PASS/FAIL signal is unreliable.
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


class InterleavedMigrationDriver:
    """Test-only scheduler stand-in (PREFILL_MIGRATION_INTERLEAVED=1): consume the per-layer ack
    counter channel and issue ONE KV migrate per fully-completed chunk, interleaved with ongoing
    prefill — no post-loop bulk migrate. Driven on the request-loop thread (no native thread, no GIL
    contention), so it is only safe in BOUNDED self-test mode where the loop yields between chunks.

    The counter channel is payload-free, so 'which chunk' is derived from the dense, in-order
    seq = request_id*num_layers + layer_idx: cursor // num_layers is the count of fully-completed
    chunks. request_id -> (slot, pos) correlation (the scheduler's InFlightChunkFIFO) is recorded as
    each chunk is dispatched. Single-rank: acks fire synchronously inside prefill(); pipeline: they
    arrive async via the router and drain() polls the tail.

    Single-rank requires PREFILL_ENABLE_LAYER_ACK=1 (or PREFILL_ENABLE_MIGRATION=1) so the runtime
    actually injects the channel; pipeline always injects via the master router."""

    POS_ALIGN = 32  # KV migration chunk granularity (blaze _align_up)

    def __init__(
        self,
        ack_shm_name,
        migration_endpoint,
        *,
        num_layers,
        src_slot,
        dst_slot,
        endpoint_id,
        wait_complete_ms,
        router=None,
        granularity: str = "layerwise",
    ):
        self._acks = ttnn.InterProcessCounterChannel.connect(ack_shm_name, 30000)
        self._mig = migration_endpoint
        self._num_layers = num_layers
        self._src, self._dst, self._ep = src_slot, dst_slot, endpoint_id
        self._wait_ms = wait_complete_ms
        self._inflight: dict = {}  # request_id -> (slot_id, actual_start, actual_end)
        self._cursor = 0  # completions consumed so far (== next expected seq)
        self._migrated_chunks = 0  # chunks already migrated
        self._migrated_layers = 0  # cumulative layers already migrated
        self._tokens: list = []  # outstanding migrate tokens (deferred wait_complete => overlap)
        # Diagnostics only. _router (master LayerCompletionRouter, may be None) exposes .processed = the
        # total acks the router has INJECTED into the channel; comparing it to our consumed _cursor tells
        # us whether completions are even reaching the channel during the loop. _migrated_in_loop counts
        # migrates issued WHILE prefill was running (the real interleave count) vs. at the tail drain.
        self._router = router
        self._migrated_in_loop = 0
        self._migration_granularity = granularity
        self._uuid_seq = 0  # monotonic, so every migrate() (incl. per-layer slices) gets a unique token

    def record_chunk(self, request_id, slot_id, actual_start, actual_end) -> None:
        self._inflight[request_id] = (slot_id, actual_start, actual_end)

    def _next_uuid(self) -> int:
        # uuid 0 is reserved (an all-zero migration-table entry means "empty"), so start at 1.
        self._uuid_seq += 1
        return self._uuid_seq

    def pump(self, current_prefill_chunk=None) -> None:
        """Non-blocking: consume injected acks, then migrate interleaved chunkwise or layerwise.
        ``current_prefill_chunk`` is the chunk the prefill loop is on RIGHT NOW (``None`` during the
        tail drain, i.e. after the loop ended) — used only to log migrate-vs-prefill overlap."""

        consumed = self._acks.try_consume_all()
        self._cursor += consumed
        chunks_complete = (
            self._cursor // self._num_layers
        )  # Once a chunk has gone through all layers, there will have been NUM_LAYERS Layer Acks
        layers_complete = self._cursor % self._num_layers  # layers completed for the current (partial) chunk

        if self._migration_granularity == "chunkwise":
            # Per-call diagnostic. The KEY question is whether `cursor`/`complete_chunks` ADVANCE during the
            # loop (current_prefill_chunk set) or only at the tail drain. router_injected is what the master
            # router has pushed into the channel so far: if injected climbs during the loop but consumed/
            # cursor don't, the driver isn't keeping up; if injected itself stays flat until the tail, the
            # completions aren't reaching the channel mid-loop (the chunk isn't "done" until the last stage).
            # Log every loop call; during drain only log when acks actually arrived (avoid 2ms-poll spam).
            if current_prefill_chunk is not None or consumed:
                injected = self._router.processed if self._router is not None else -1
                phase = f"prefill@chunk={current_prefill_chunk}" if current_prefill_chunk is not None else "tail-drain"
                logger.debug(
                    f"[interleave-diag] pump({phase}): {consumed=} cursor={self._cursor} "
                    f"{chunks_complete=} already_migrated={self._migrated_chunks} router_injected={injected} "
                    f"(num_layers={self._num_layers})"
                )
            while self._migrated_chunks < chunks_complete:
                self._migrate_chunk(self._migrated_chunks, 0, self._num_layers, current_prefill_chunk)
                self._inflight.pop(self._migrated_chunks, None)  # chunk fully migrated -> evict
                self._migrated_chunks += 1

        # migrate on a layerwise granularity
        elif self._migration_granularity == "layerwise":
            if current_prefill_chunk is not None or consumed:
                injected = self._router.processed if self._router is not None else -1
                phase = f"prefill@chunk={current_prefill_chunk}" if current_prefill_chunk is not None else "tail-drain"
                logger.debug(
                    f"[interleave-diag] pump({phase}): {consumed=} cursor={self._cursor} "
                    f"{layers_complete=} migrated_layers={self._migrated_layers} router_injected={injected} "
                    f"(num_layers={self._num_layers})"
                )
            # Each chunk's _inflight
            # entry is evicted only once its FINAL layer has been migrated (read, don't pop, so a
            # chunk can be migrated across many pump() calls).
            NL = self._num_layers
            while self._migrated_layers < self._cursor:
                chunk = self._migrated_layers // NL
                layer = self._migrated_layers % NL
                chunk_end = (chunk + 1) * NL
                batch_end = min(self._cursor, chunk_end)  # never cross a chunk boundary in one migrate
                self._migrate_chunk(chunk, layer, batch_end - chunk * NL, current_prefill_chunk)
                self._migrated_layers = batch_end
                if batch_end == chunk_end:
                    self._inflight.pop(chunk, None)  # chunk fully migrated -> evict
        else:
            raise ValueError('Migration granularity must be either "chunkwise" or "layerwise" ')

    def _migrate_chunk(self, chunk, l_start, l_end, current_prefill_chunk=None) -> None:
        # READ, don't pop: layerwise migrates one chunk across many calls (one per layer slice),
        # so the _inflight entry must survive until its final layer ships. The caller (pump) evicts
        # the chunk once batch_end reaches chunk_end.
        slot, a_start, a_end = self._inflight[chunk]
        if slot != self._src:
            return  # not the slot this loopback test migrates
        pos_start = (a_start // self.POS_ALIGN) * self.POS_ALIGN
        pos_end = ((a_end + self.POS_ALIGN - 1) // self.POS_ALIGN) * self.POS_ALIGN

        if pos_end <= pos_start:
            return  # all-pad chunk: nothing real to ship

        uuid = self._next_uuid()

        if self._migration_granularity == "chunkwise":
            tok = self._mig.migrate(uuid, self._ep, self._src, self._dst, l_start, l_end, pos_start, pos_end)
            self._tokens.append(tok)
            # Overlap evidence: the migrate is now running ASYNC on the worker (wait_complete is deferred to
            # drain()), so the prefill loop keeps going while this copy is in flight. Compare this line's
            # timestamp against the next "[interleave] prefilled chunk ..." line to see the overlap on the
            # wall clock; the "copy(ies) in flight" count below is how many copies are running concurrently.
            if current_prefill_chunk is None:
                overlap = "TAIL (prefill loop already finished)"
            else:
                self._migrated_in_loop += 1
                overlap = f"WHILE prefilling chunk {current_prefill_chunk} (prefill is {current_prefill_chunk - chunk} chunk(s) ahead)"
            logger.info(
                f"[interleave] MIGRATE issued uuid={uuid} chunk {chunk} slot{self._src}->slot{self._dst} "
                f"pos[{pos_start},{pos_end}) {overlap}; {len(self._tokens)} copy(ies) in flight, none waited yet"
            )
        elif self._migration_granularity == "layerwise":
            tok = self._mig.migrate(uuid, self._ep, self._src, self._dst, l_start, l_end, pos_start, pos_end)
            self._tokens.append(tok)  # track so drain() waits on every layer slice
            # Same overlap accounting as the chunkwise path: count slices issued while prefill is still
            # running so drain()'s overlapped-vs-tail summary is correct in layerwise mode too.
            if current_prefill_chunk is not None:
                self._migrated_in_loop += 1
                overlap = f"WHILE prefilling chunk {current_prefill_chunk}"
            else:
                overlap = "TAIL (prefill loop already finished)"
            logger.info(
                f"[interleave] MIGRATE issued uuid={uuid} chunk {chunk} layers[{l_start},{l_end}) "
                f"slot{self._src}->slot{self._dst} pos[{pos_start},{pos_end}) {overlap}; "
                f"{len(self._tokens)} copy(ies) in flight, none waited yet"
            )
        else:
            raise ValueError('Migration granularity must be either "chunkwise" or "layerwise" ')

    def drain(self, expected_chunks, poll_timeout_s=120.0) -> None:
        """Tail: pipeline acks may still be in flight. Poll until all completions are consumed
        (migrating as they land), then wait_complete every outstanding copy."""
        target = expected_chunks * self._num_layers
        deadline = time.perf_counter() + poll_timeout_s
        while self._cursor < target:
            self.pump(current_prefill_chunk=None)
            if self._cursor >= target or time.perf_counter() >= deadline:
                break
            time.sleep(0.002)
        self.pump(current_prefill_chunk=None)  # flush the final completed chunk
        if self._cursor < target:
            logger.warning(f"[interleave] drain timeout: {self._cursor}/{target} completions consumed")
        # This is the ONLY place we block on completion. The split below is the headline interleave
        # metric: migrated_in_loop is how many copies were issued WHILE prefill was still running (real
        # overlap); migrated_at_tail is how many only became ready after the loop. All-tail => no overlap.
        # If wait_complete returns near-instantly, the in-loop copies finished during prefill (good); if
        # it takes ~as long as a bulk migrate, nothing actually overlapped.
        # Granularity-agnostic totals: every issued migrate (chunkwise = 1/chunk, layerwise = 1/layer
        # slice) appends exactly one token, so len(_tokens) is the true issued count for both modes.
        total = len(self._tokens)
        tail = total - self._migrated_in_loop
        logger.info(
            f"[interleave] prefill loop finished; {total} migrate(s) total: "
            f"{self._migrated_in_loop} issued DURING prefill (overlapped), {tail} issued at the TAIL; "
            f"{total} copy(ies) still in flight — now wait_complete-ing all (the only blocking wait)"
        )
        t_wait = time.perf_counter()
        for tok in self._tokens:
            self._mig.wait_complete(tok, self._wait_ms)
        logger.success(
            f"[interleave] {total} migrate(s) complete ({self._migrated_in_loop} overlapped, "
            f"{tail} tail); tail wait_complete took {(time.perf_counter() - t_wait) * 1e3:.1f} ms"
        )
