#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Disaggregated prefill runner — one entry point, two run modes that share the same N-rank pipeline.

Model-agnostic: the model is selected by PREFILL_MODEL and driven through a PrefillModelAdapter
(see ../adapter.py and ADDING_A_PREFILL_MODEL.md). This driver wires rank topology, input,
transport, and the per-chunk schedule; the adapter supplies how to build the model, allocate the KV
cache, run a chunk, and validate/migrate it.

The model is split across N ranks under tt-run: each rank owns a contiguous layer slice and builds
the same TtPrefillRuntime (first_layer_idx / is_first_rank / is_last_rank). With >1 rank the cross-rank
hidden state moves device-to-device over fabric sockets (connected MGD + FABRIC_2D); N=1 is the
single-galaxy case (no transport). Ranks run decoupled (no per-chunk barrier; one warm-up barrier
after compile). The two modes run identical pipeline mechanics and differ only in the trigger:

  * Request mode (default): production serving. rank 0's tokens + per-iter PrefillMetadata arrive over
    the H2D socket from an external producer (prefill_producer.py / the scheduler); the loop is
    UNBOUNDED. KV-chunk-table migration + per-layer LayerAck are wired for the single-rank case only
    (disabled for the pipeline for now). Shutdown is graceful: the producer/scheduler closes the stream
    with an all -1 PrefillMetadata sentinel that each rank forwards downstream and then exits on; a rank
    blocked in the recv can only be released by a transfer (the recv device op has no timeout), so
    SIGTERM/SIGKILL remains the hard fallback if no sentinel arrives.

  * Standalone mode (PREFILL_STANDALONE=1): bring-up / benchmark. rank 0's input is the golden trace
    for a fixed PREFILL_STANDALONE_NCHUNKS chunks; the loop is BOUNDED and exits cleanly.
    PREFILL_STANDALONE_PCC=1 checks each rank's KV slice vs the golden.

The model class is the single source of truth — this driver wires rank topology, input, transport,
and the per-chunk schedule; it does not reimplement embed / layers / forward.
"""

import json
import os
import signal
import time

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import DEFAULT_MODEL, PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.migration import serialize_device_map
from models.demos.common.prefill.runners.runner_utils import (
    activation_global_spec,
    build_h2d_service,
    load_trace_token_ids,
    open_mesh_device,
    resolve_trace_dir,
)

# NOTE: the pipelined_prefill layer-completion classes (the standalone `_layer_completion` extension)
# are imported lazily at point-of-use — its .so is built only under WITH_PYTHON_BINDINGS and may be
# absent in a packaged/wheel build, so a top-level import would hard-fail the runner for everyone
# (including single-rank runs that never touch layer completion).


def _apply_manifest_env():
    """If PREFILL_MANIFEST is set, load the shared run.json and populate the env vars
    the runner (and migration/validation helpers) read. setdefault => an explicitly
    exported env var still wins over the manifest. Must be invoked before the
    module-level env reads below (e.g. PREFILL_MAX_SEQ_LEN) so the values take effect."""
    manifest_path = os.environ.get("PREFILL_MANIFEST")
    if not manifest_path:
        return

    with open(manifest_path) as mp:
        manifest = json.load(mp)

    def sd(key, val):
        if val is not None:
            os.environ.setdefault(key, str(val))

    # Generic model/run config: a flat PREFILL_* map applied verbatim (setdefault). This lets a
    # rank-binding stay topology-only (rank_bindings + mesh_graph_desc) and point at a per-model
    # manifest for all model config — PREFILL_MODEL, fabric mode, chunk count, etc.
    for key, val in manifest.get("env", {}).items():
        sd(key, val)

    # The migration/pairwise-validation runs additionally carry a users[] + migration{} block. A
    # plain model-config manifest omits it (env-only), so it's optional.
    users = manifest.get("users")
    if not users:
        return
    N = len(users)

    model = manifest.get("model", {})
    mig = manifest.get("migration", {})
    paths = manifest.get("paths", {})

    sd("PREFILL_MODEL", model.get("variant"))
    sd("DEEPSEEK_PREFILL_TRACE_DIR", paths.get("trace_dir"))
    sd("PREFILL_MIGRATION_CLIENT_DIR", paths.get("migration_client_dir"))
    sd("PREFILL_NUM_USERS", 2 * N)
    sd("PREFILL_MAX_SEQ_LEN", model.get("max_seq_len"))
    sd("PREFILL_STANDALONE_CHUNKED_NCHUNKS", sum(u["n_chunks"] for u in users))
    sd("PREFILL_MIGRATE_WAIT_S", mig.get("wait_s"))
    sd("PREFILL_MIGRATE_GOLDEN_PTS", ",".join(u.get("kv_cache", "") for u in users))

    # Mode: default to pairwise
    mode = mig.get("mode") or "pairwise"
    # Loud failure for incorrect mode
    if mode != "pairwise":
        raise ValueError(f"manifest migration.mode must be 'pairwise', got: {mode}")
    # Loud failure for empty users
    if N < 1:
        raise ValueError(f"manifest migration.mode 'pairwise' requires at least 1 user, got {N}")
    sd("PREFILL_MIGRATE", mode)

    # Each non-empty kv_cache must exist on disk.
    for i, u in enumerate(users):
        kv = u.get("kv_cache", "")
        if kv and not os.path.exists(kv):
            raise FileNotFoundError(f"PREFILL_MANIFEST user {i} kv_cache not found: {kv}")

    # PREFILL_NUM_USERS (derived or explicitly exported) must equal 2*N.
    num_users = int(os.environ["PREFILL_NUM_USERS"])
    if num_users != 2 * N:
        raise ValueError(
            f"PREFILL_NUM_USERS ({num_users}) inconsistent with manifest " f"({N} users => expected {2 * N})"
        )


# Populate env from the manifest BEFORE the module-level env reads below.
_apply_manifest_env()

# Both socket transports (H2D input on rank 0, D2D between ranks) share a 1x1 push/sync worker grid and
# the same 3-word PrefillMetadata (slot_id, actual_start, actual_end). The 1x1 grid is the cheapest
# footprint with no penalty: a grid sweep showed compute + handoff gap flat from 1x1 to 4x4 (the
# per-chunk overhead is the persistent service's fabric/NoC presence, not the push workers).
SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
METADATA_SIZE_BYTES = 12

# LayerAck D2H FIFO. Records are METADATA_SIZE_BYTES (12 B) each; 4 KB is a PCIe-aligned
# one-page buffer with generous headroom for in-flight records.
LAYER_ACK_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_LAYER_ACK_FIFO_BYTES", 4 * 1024))

# End-of-stream sentinel: the producer/scheduler closes the request stream with one final push whose
# PrefillMetadata words are all -1 (0xFFFFFFFF on the wire). -1 is out of range for slot_id and both KV
# positions, so it can't collide with a real chunk. On receipt a rank forwards it to the next rank
# (unblocking that rank's recv) and breaks its loop, so an N-rank pipeline drains and exits gracefully
# instead of every rank blocking in its recv until SIGKILL. Shared wire convention with the scheduler;
# see ADDING_A_PREFILL_MODEL.md.
SHUTDOWN_METADATA_WORD = -1

# H2D socket service (request mode, rank 0 input): one worker core copies each pushed chunk into a fresh
# tensor; the producer packs the PrefillMetadata alongside each push.
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

# D2D socket transport (>1 rank): one persistent sender/receiver pair per rank boundary carries the
# sharded hidden state over inter-galaxy fabric. The activation is sharded [seq across SP rows, emb
# across TP cols] — the same layout the embedding output uses — so the receiver backing feeds the
# downstream model with no reshard.
D2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 64 * 1024))

ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
MODEL_CFG = ADAPTER.model_config

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
# Chunks this run drives. The per-user KV cache is sized to exactly hold them
# (max_seq_len = chunk_size * num_chunks), so there is no separate cache-length knob to keep in sync.
# PREFILL_MAX_SEQ_LEN still overrides if a larger cache is wanted.
NUM_CHUNKS = int(os.environ.get("PREFILL_STANDALONE_NCHUNKS", 11))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * NUM_CHUNKS))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", ADAPTER.default_gate_mode)
# When on (default), the last transformer layer runs kv-only: it fills the KV cache for migration and
# skips its Q/SDPA/wo, FFN/MoE, final norm, and LM head. In a pipeline only the last rank applies it.
KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"
# Measurement-only: synchronize the device after each chunk's forward and log the isolated per-rank
# compute (CHUNK_COMPUTE). Off in production — the sync serializes dispatch and kills pipeline overlap.
SYNC_PER_CHUNK = os.environ.get("PREFILL_SYNC_PER_CHUNK", "0") == "1"
# Some models (e.g. Kimi: single expert group, device gate) route the MoE routing all-gather's global
# semaphores to L1_SMALL so they don't pin the main-L1 floor and clash with the next layer's MLA static
# CBs, which needs the mesh opened with an L1_SMALL region. The adapter owns both knobs.
_L1_SMALL_SIZE = ADAPTER.l1_small_size

os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


def _snap_counts_to_starts(counts, valid_starts, num_layers):
    """Nudge an even split's interior rank boundaries onto the nearest valid start (preserving
    sum == num_layers), for models that constrain where a rank may begin (layer_split_boundaries).
    Nearest by |distance| then lower index; each boundary is used at most once and stays increasing."""
    valid = sorted(valid_starts)
    boundaries, s = [], 0
    for c in counts[:-1]:
        s += c
        boundaries.append(s)
    snapped, prev = [], 0
    for b in boundaries:
        cand = min(
            (v for v in valid if prev < v < num_layers and v not in snapped),
            key=lambda v: (abs(v - b), v),
            default=None,
        )
        if cand is None:
            raise ValueError(f"cannot place {len(counts)} pipeline ranks on valid layer boundaries {valid}")
        snapped.append(cand)
        prev = cand
    out, prev = [], 0
    for b in [*snapped, num_layers]:
        out.append(b - prev)
        prev = b
    return out


def compute_layer_split(num_layers: int, num_ranks: int, valid_starts=None) -> list[tuple[int, int]]:
    """Contiguous (first_layer_idx, count) per rank. PREFILL_PP_LAYER_COUNTS, a
    comma-separated count list summing to num_layers, overrides the default even
    split (remainder handed to the earlier ranks).

    ``valid_starts`` (from the adapter's ``layer_split_boundaries``): layer indices at which a rank may
    begin. None => unconstrained. When set, the default even split is auto-snapped onto valid
    boundaries, and any split (explicit or snapped) whose rank starts fall off them is rejected early."""
    override = os.environ.get("PREFILL_PP_LAYER_COUNTS")
    if override:
        counts = [int(x) for x in override.split(",")]
        if len(counts) != num_ranks or sum(counts) != num_layers:
            raise ValueError(
                f"PREFILL_PP_LAYER_COUNTS={override!r} must list {num_ranks} counts summing to "
                f"{num_layers} (got {len(counts)} counts summing to {sum(counts)})"
            )
    else:
        base, rem = divmod(num_layers, num_ranks)
        counts = [base + (1 if r < rem else 0) for r in range(num_ranks)]
        if valid_starts is not None:
            counts = _snap_counts_to_starts(counts, valid_starts, num_layers)

    ranges = []
    start = 0
    for count in counts:
        ranges.append((start, count))
        start += count

    if valid_starts is not None:
        for first_idx, _ in ranges:
            if first_idx not in valid_starts:
                near = sorted(b for b in valid_starts if abs(b - first_idx) <= 4)
                raise ValueError(
                    f"pipeline rank starts at layer {first_idx}, not a valid boundary for this model "
                    f"(nearest valid: {near}). Set PREFILL_PP_LAYER_COUNTS so every cumulative boundary "
                    f"is a valid start."
                )
    return ranges


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _load_token_ids() -> list[int]:
    """Load this run's token IDs (same source as the single-rank standalone loop).
    All ranks load identically so they agree on the chunk schedule."""
    import json

    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default))
    input_override = os.environ.get("PREFILL_STANDALONE_INPUT")
    if input_override:
        with open(input_override) as f:
            token_ids = list(json.load(f)["token_ids"])
        logger.info(f"[pp] input override: {len(token_ids)} token_ids from {input_override}")
    else:
        logger.info(f"[pp] reading input token_ids from {trace_dir}/metadata.json")
        token_ids = load_trace_token_ids(trace_dir)
    return token_ids


# ---------------------------------------------------------------------------
# Layer-completion routing (pipeline / num_ranks > 1)
# ---------------------------------------------------------------------------

# When the completion ring is full, spin waiting for the router to drain rather than
# dropping/failing immediately. Bounded so a genuinely stalled router still surfaces.
LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S = float(os.environ.get("PREFILL_LAYER_COMPLETION_PUSH_TIMEOUT_S", 30.0))
LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001  # tiny yield so the spin doesn't peg a core


def build_layer_completion_sink(producer, *, source_rank, num_layers):
    """Build the per-layer completion sink the runtime fires once per layer.

    Computes a globally-dense ordering key and pushes a full completion
    into `producer` (a pipelined_prefill.LayerCompletionQueue). The
    master router re-emits completions strictly in ascending `seq`.

    seq = request_id * num_layers + layer_idx — dense across all (request,
    layer) pairs. For pipelined prefill each rank owns a disjoint set of
    global layer indices per request, so the union of every rank's seqs
    tiles [0, num_requests*num_layers) with no gaps or collisions.

    The runtime calls the returned sink as `sink(layer_idx, request_id)`,
    binding the current chunk's request_id per prefill() call — so this
    builder needs no request-id accessor and reads no shared mutable state.

    Args:
        producer: connected LayerCompletionQueue (the host-local ring).
        source_rank: this rank's world rank (diagnostic in the payload).
        num_layers: total GLOBAL layers (the seq stride per request), NOT this rank's slice.
    """

    def on_layer_complete(layer_idx: int, request_id: int) -> None:
        # Hot path: fired once per layer inside model.forward. Push directly (no per-call closure)
        # and return on the common success; only the rare full-ring case falls into the spin below.
        seq = request_id * num_layers + layer_idx
        if producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
            return

        # Ring is sized well above in-flight depth; a full ring means the router
        # thread is momentarily behind. Spin (don't drop) for up to PUSH_SPIN_TIMEOUT_S
        # waiting for it to drain; log on entry, every PUSH_SPIN_LOG_EVERY_S while waiting,
        # and on exit. Only surface an error if it never catches up.
        start = time.monotonic()
        next_log = start + LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
        logger.warning(
            f"[layer-completion] ring full (seq={seq}); spinning up to "
            f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s for router to drain"
        )
        while True:
            if producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
                logger.info(f"[layer-completion] ring drained after {time.monotonic() - start:.1f}s; pushed seq={seq}")
                return
            if _shutdown:
                # Operator asked to stop (SIGTERM/SIGINT). Abort the spin immediately instead of
                # ignoring the signal for up to the full timeout; teardown runs via run_request_loop's
                # finally. Raising (vs. silently dropping) keeps the failure visible.
                raise RuntimeError(f"layer-completion ring full (seq={seq}); shutdown requested while spinning")
            now = time.monotonic()
            if now - start >= LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:
                logger.error(f"[layer-completion] gave up after {now - start:.1f}s spinning on full ring (seq={seq})")
                raise RuntimeError(
                    f"layer-completion ring full (seq={seq}); router not draining after "
                    f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s"
                )
            if now >= next_log:
                logger.warning(f"[layer-completion] still spinning on full ring (seq={seq}) after {now - start:.0f}s")
                next_log += LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
            time.sleep(LAYER_COMPLETION_PUSH_SPIN_SLEEP_S)

    return on_layer_complete


class _CompletionCheckConsumer:
    """Test-only scheduler stand-in (enabled by PREFILL_CHECK_COMPLETIONS=1).

    Thin Python wrapper over the C++ LayerCompletionConsumer from the standalone `_layer_completion`
    extension (models/demos/common/prefill/runners/pipelined_prefill — NOT part of the ttnn
    module). The C++ consumer drains the master router's scheduler counter
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
        from models.demos.common.prefill.runners.pipelined_prefill import LayerCompletionConsumer

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


class _InterleavedMigrationDriver:
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


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _first_rank_chunk_tokens(runtime, token_ids: list[int], kv_actual: int) -> ttnn.Tensor:
    """Slice this chunk's tokens and build the SP-sharded input tensor. Delegates to the runtime's own
    builder so the input format has one source of truth."""
    cfg = runtime.config
    return runtime.make_chunk_input(token_ids[kv_actual : kv_actual + cfg.chunk_size])


def _is_shutdown_sentinel(meta: dict) -> bool:
    """True for the all -1 end-of-stream sentinel (see SHUTDOWN_METADATA_WORD); false for every real
    chunk, whose slot_id and KV positions are non-negative and in range."""
    return (
        meta["slot_id"] == SHUTDOWN_METADATA_WORD
        and meta["actual_start"] == SHUTDOWN_METADATA_WORD
        and meta["actual_end"] == SHUTDOWN_METADATA_WORD
    )


def _socket_next(h2d_service) -> tuple:
    """Block on the next producer push: returns (tt_tokens, {slot_id, actual_start, actual_end},
    tt_metadata). The device metadata tensor is returned (not discarded) so it can be propagated into
    the model's per-layer ack send. Used only by the unbounded request loop (rank 0 input)."""
    import torch

    tt_tokens, tt_metadata = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        h2d_service, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
    return tt_tokens, {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}, tt_metadata


def build_d2d_pipeline_endpoints(mesh_device, rank: int, num_ranks: int, chunk_size: int, hidden_size: int):
    """Stand up this rank's persistent D2D endpoints for the pipeline: an inbound receiver from rank-1
    (every rank but the first) and an outbound sender to rank+1 (every rank but the last). Returns
    (inbound_receiver_or_None, outbound_sender_or_None).

    Setup order is inbound-then-outbound on every rank. create_sender/create_receiver rendezvous
    point-to-point between the two boundary ranks (no world barrier), and each MeshSocket ctor blocks
    until its peer's matching ctor. Doing inbound first chains the bring-up: rank 0's sender unblocks
    rank 1's receiver, which frees rank 1 to build its sender for rank 2's receiver, and so on — no
    deadlock. Both sides pass the identical worker-core grid and global spec."""
    global_spec = activation_global_spec(chunk_size, hidden_size)

    def _common():
        # Fresh mapper per call: create_sender/create_receiver take the mapper by std::unique_ptr and
        # MOVE it, so a middle rank (builds BOTH a receiver and a sender) must not reuse one — the
        # second create would get a consumed/null mapper and fail overload resolution.
        return dict(
            global_spec=global_spec,
            mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=SYNC_WORKER_CORES,
            receiver_worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
            share_fabric_links=True,
            # The service asserts L1-only (d2d_stream_service.cpp:260).
            socket_buffer_type=ttnn.BufferType.L1,
        )

    inbound = None
    if rank > 0:
        logger.info(f"[pp rank {rank}] [d2d] creating inbound receiver from rank {rank - 1}")
        inbound = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=rank - 1, receiver_rank=rank, **_common()
        )
    outbound = None
    if rank < num_ranks - 1:
        logger.info(f"[pp rank {rank}] [d2d] creating outbound sender to rank {rank + 1}")
        outbound = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=rank, receiver_rank=rank + 1, **_common()
        )
    logger.info(
        f"[pp rank {rank}] [d2d] endpoints up (inbound={'yes' if inbound else 'no'} "
        f"outbound={'yes' if outbound else 'no'}, workers={SYNC_WORKER_CORES}, fifo={D2D_FIFO_SIZE_BYTES}B)"
    )
    return inbound, outbound


def _d2d_recv(inbound) -> tuple:
    """Drain the next chunk that landed in the inbound receiver backing into a fresh device tensor and
    decode the inline metadata. The returned tensor already has the embedding-output sharding, so it
    feeds runtime.prefill with no reshard. Pairs with the upstream rank's _d2d_send."""
    import torch

    t0 = time.perf_counter()
    act, metadata_device = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        inbound, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(metadata_device)[0]).view(torch.int32).flatten()
    meta = {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}
    logger.info(
        f"[pp] RECV-d2d [{meta['actual_start']},{meta['actual_end']}) slot={meta['slot_id']} "
        f"[xfer] sync={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )
    return act, meta, metadata_device


def _d2d_send(outbound, activation: ttnn.Tensor, rank: int, meta: dict) -> None:
    """Push this rank's output hidden state + metadata to the downstream rank's receiver, then free it.
    The model already emits the activation in the sender backing's spec, and outbound_socket_service_sync
    TT_FATALs on any spec mismatch, so no host-side relayout is needed."""
    t0 = time.perf_counter()
    backing = outbound.get_backing_tensor()
    import torch

    words = [meta["slot_id"], meta["actual_start"], meta["actual_end"]]
    # The outbound op ships metadata as a replicated device tensor (3 uint32 words), not a Python list.
    md_tensor = ttnn.from_torch(
        torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=backing.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            backing.device(),
            ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]),
        ),
    )
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(outbound, activation, metadata=md_tensor)
    ttnn.deallocate(activation)
    logger.info(
        f"[pp rank {rank}] SEND-d2d [{meta['actual_start']},{meta['actual_end']}) "
        f"[xfer] push={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )


def _forward_shutdown(d2d_out, rank: int, hidden_size: int) -> None:
    """Forward the shutdown sentinel to the downstream rank so it unblocks in its own recv, then release
    the outbound link so the transfer ships (mirroring _compute_and_send's tail). The activation content
    is irrelevant — the downstream discards it once it sees the sentinel — but outbound_socket_service_sync
    requires the input's per-shard spec to equal the sender backing's, so build the dummy exactly like a
    real activation: the [1, 1, CHUNK_SIZE, hidden_size] bf16 TILE spec sharded by D2D_MAPPER_CONFIG."""
    import torch

    dev = d2d_out.get_backing_tensor().device()
    dummy = ttnn.from_torch(
        torch.zeros(1, 1, CHUNK_SIZE, hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(dev, D2D_MAPPER_CONFIG),
    )
    sentinel = {
        "slot_id": SHUTDOWN_METADATA_WORD,
        "actual_start": SHUTDOWN_METADATA_WORD,
        "actual_end": SHUTDOWN_METADATA_WORD,
    }
    _d2d_send(d2d_out, dummy, rank, sentinel)  # ships + frees the dummy
    d2d_out.release_fabric_links()
    logger.info(f"[pp rank {rank}] forwarded SHUTDOWN sentinel to rank {rank + 1}")


def _lease_reclaim(d2d_in, d2d_out) -> None:
    """Before a chunk: reclaim this rank's fabric links (the previous-iter D2D transfer has drained),
    then grant the inbound receiver so this chunk's activation drains into its backing. No-op without
    D2D (single rank). The outbound grant happens AFTER the push, in _compute_and_send."""
    if d2d_in is not None:
        d2d_in.wait_for_fabric_links()
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    if d2d_in is not None:
        d2d_in.release_fabric_links()


def _compute_and_send(
    runtime, kv_caches, rank: int, c: int, inp, meta: dict, d2d_out, d2h_service=None, record_dev=None
) -> float:
    """Run one chunk: prefill into the engine-owned kv_caches, forward the output downstream (non-last
    rank) and grant the outbound sender so it ships over fabric. Returns the compute-start epoch
    (NTP-comparable). CHUNK_START is logged BEFORE the forward, with this chunk's metadata, so the
    slot/KV-range is visible per rank even if prefill_chunk hangs. The trailing metadata is kept after
    compute_start so the c=/compute_start= fields stay parseable (plot_pipeline_trace.py)."""
    t_start = time.time()
    logger.info(
        f"[pp rank {rank}] CHUNK_START c={c} compute_start={t_start:.6f} "
        f"slot={meta['slot_id']} [{meta['actual_start']},{meta['actual_end']})"
    )
    out = runtime.prefill_chunk(
        inp,
        kv_caches,
        slot_id=meta["slot_id"],
        actual_start=meta["actual_start"],
        actual_end=meta["actual_end"],
        request_id=c,
        d2h_service=d2h_service,
        record_dev=record_dev,
    )
    if SYNC_PER_CHUNK:
        # Block on device completion so the delta is this rank's forward alone, not the downstream-start
        # proxy. Serializes dispatch (no overlap) — measurement runs only.
        ttnn.synchronize_device(runtime.mesh_device)
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={(time.time() - t_start) * 1000.0:.3f}")
    if not runtime.config.is_last_rank:
        _d2d_send(d2d_out, out, rank, meta)  # push + free; the grant below forwards it over fabric
    if d2d_out is not None:
        d2d_out.release_fabric_links()
    return t_start


def _drain_and_log_e2e(runtime, rank: int, d2d_out, first_compute_start, n_done: int, t0: float) -> None:
    """Per-rank teardown: drain the last outbound D2D forward, one synchronize so the e2e clock reflects
    device completion, then log E2E_CLOCK (first prefill start + last compute end, NTP-comparable epochs)
    and the chunk count. No teardown barrier across ranks."""
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    ttnn.synchronize_device(runtime.mesh_device)
    logger.info(
        f"[pp rank {rank}] E2E_CLOCK first_compute_start={first_compute_start:.6f} last_compute_end={time.time():.6f}"
    )
    logger.info(f"[pp rank {rank}] processed {n_done} chunks in {(time.perf_counter() - t0) * 1000.0:.2f} ms")


def run_request_loop(
    runtime,
    kv_caches,
    rank: int,
    num_ranks: int,
    *,
    hidden_size: int,
    h2d_service=None,
    d2d_in=None,
    d2d_out=None,
    migration_driver=None,
    d2h_service=None,
) -> dict:
    """Production serving loop — UNBOUNDED. rank 0 reads each chunk from the H2D socket (the external
    producer decides the count); downstream ranks read from D2D. Runs until the producer/scheduler
    closes the stream with the all -1 shutdown sentinel (each rank forwards it and exits gracefully) or,
    as a hard fallback, until SIGTERM/SIGKILL. No fixed NUM_CHUNKS bound, no trace input, no PCC — see
    run_standalone_loop for those."""
    cfg = runtime.config
    if cfg.is_first_rank and h2d_service is None:
        raise ValueError("request mode requires the H2D service on the first rank for input")
    logger.info(
        f"[pp rank {rank}/{num_ranks}] request (unbounded) loop start "
        f"(is_first={cfg.is_first_rank} is_last={cfg.is_last_rank} input={'h2d' if cfg.is_first_rank else 'd2d'})"
    )
    # Self-test bound: PREFILL_MIGRATION_SELFTEST=1 makes the loop run exactly NUM_CHUNKS chunks then
    # exit CLEANLY so the post-loop migrate + verify can run — without it the unbounded loop blocks in
    # recv and only SIGKILL exits, which kills before the verify. NUM_CHUNKS is the run's single chunk
    # count (the per-user KV cache is sized to exactly hold it, max_seq_len = chunk_size * NUM_CHUNKS),
    # and the producer pushes the same count, so they match by construction. 0 == unbounded serving.
    n_selftest = NUM_CHUNKS if os.environ.get("PREFILL_MIGRATION_SELFTEST", "0") == "1" else 0
    t0 = time.perf_counter()
    c = 0
    first = None
    real_end_per_slot: dict = {}
    while not _shutdown:
        if n_selftest and c >= n_selftest:
            break
        _lease_reclaim(d2d_in, d2d_out)
        if cfg.is_first_rank:
            inp, meta, metadata_device = _socket_next(h2d_service)  # slot/start/end from the producer
        else:
            inp, meta, metadata_device = _d2d_recv(d2d_in)
        if _is_shutdown_sentinel(meta):
            # End of stream: drop the throwaway payload + its metadata tensor, hand the sentinel to the
            # next rank so it too unblocks and exits, then fall through to the graceful drain below.
            logger.info(f"[pp rank {rank}] SHUTDOWN sentinel received after {c} chunks; exiting request loop")
            ttnn.deallocate(inp)
            ttnn.deallocate(metadata_device)
            if d2d_out is not None:
                _forward_shutdown(d2d_out, rank, hidden_size)
            break
        t = _compute_and_send(
            runtime, kv_caches, rank, c, inp, meta, d2d_out, d2h_service=d2h_service, record_dev=metadata_device
        )
        # Interleaved migration: register this chunk's correlation, then migrate any chunk whose layers
        # have all acked (single-rank: chunk c's acks are visible the moment _compute_and_send returns).
        if migration_driver is not None:
            migration_driver.record_chunk(c, meta["slot_id"], meta["actual_start"], meta["actual_end"])
            logger.info(
                f"[interleave] prefilled chunk {c} (slot{meta['slot_id']} "
                f"pos[{meta['actual_start']},{meta['actual_end']})); pumping migration driver"
            )
            migration_driver.pump(current_prefill_chunk=c)
        # Track the real (non-pad) end position per slot: the producer clamps actual_end to the real
        # ISL, so the max over a slot's chunks is that slot's prompt length (== blaze's S).
        s = meta["slot_id"]
        real_end_per_slot[s] = max(real_end_per_slot.get(s, 0), meta["actual_end"])
        if first is None:
            first = t
        c += 1
    # Bounded self-test: every rank must finish receiving + forwarding the final chunk before any
    # rank reclaims its outbound fabric link in the drain (mirrors run_standalone_loop's tail barrier).
    if num_ranks > 1 and n_selftest:
        ttnn.distributed_context_barrier()
    _drain_and_log_e2e(runtime, rank, d2d_out, first, c, t0)
    return real_end_per_slot


def run_standalone_loop(runtime, kv_caches, rank: int, num_ranks: int, *, d2d_in=None, d2d_out=None) -> None:
    """Bring-up / benchmark loop — BOUNDED, golden-trace input. rank 0 drives NUM_CHUNKS chunks from the
    trace; downstream ranks receive the same count over D2D. Every rank knows NUM_CHUNKS (propagated via
    global_env), so each loops a fixed range independently — no end-of-stream marker needed. With
    PREFILL_STANDALONE_PCC=1 each rank checks the KV slice it populated vs the golden trace."""
    cfg = runtime.config
    slot_id = 0  # first rank fills slot 0; downstream ranks adopt the slot from the received metadata
    n_chunks = NUM_CHUNKS
    token_ids = None
    if cfg.is_first_rank:
        token_ids = _load_token_ids()
        token_ids = (token_ids + [1] * (n_chunks * cfg.chunk_size))[: n_chunks * cfg.chunk_size]
        if n_chunks * cfg.chunk_size > cfg.max_seq_len:
            raise ValueError(
                f"{n_chunks} chunks x {cfg.chunk_size} exceeds per-user cache max_seq_len={cfg.max_seq_len}; "
                f"raise PREFILL_MAX_SEQ_LEN."
            )
    # Every rank loops a fixed range(n_chunks) independently — there is no end-of-stream marker, so all
    # ranks MUST resolve the same PREFILL_STANDALONE_NCHUNKS (set in the binding's global_env, not a
    # per-rank override). A mismatch strands the pipeline: a low downstream count exits early and leaves
    # rank 0's next send unconsumed. Log each rank's count so a mismatch is visible across the tag logs.
    logger.info(
        f"[pp rank {rank}/{num_ranks}] standalone (bounded) loop start "
        f"(is_first={cfg.is_first_rank} is_last={cfg.is_last_rank} input=trace chunks={n_chunks})"
    )
    t0 = time.perf_counter()
    first = None
    for c in range(n_chunks):
        _lease_reclaim(d2d_in, d2d_out)
        if cfg.is_first_rank:
            kv_actual = c * cfg.chunk_size
            inp = _first_rank_chunk_tokens(runtime, token_ids, kv_actual)
            meta = {"slot_id": slot_id, "actual_start": kv_actual, "actual_end": kv_actual + cfg.chunk_size}
            metadata_device = None
        else:
            inp, meta, metadata_device = _d2d_recv(d2d_in)
            slot_id = meta["slot_id"]
        t = _compute_and_send(runtime, kv_caches, rank, c, inp, meta, d2d_out, record_dev=metadata_device)
        if first is None:
            first = t
    # Every rank must finish receiving + forwarding the final chunk before any rank reclaims its
    # outbound fabric link in the drain. Without this, the producer reclaims the shared link
    # (share_fabric_links) right after its last send and strands the downstream's final recv —
    # the pipeline tail deadlocks (ranks 2/3 hang on the last chunk).
    if num_ranks > 1:
        ttnn.distributed_context_barrier()
    _drain_and_log_e2e(runtime, rank, d2d_out, first, n_chunks, t0)

    if os.environ.get("PREFILL_STANDALONE_PCC", "0") == "1":
        # Each rank PCC-checks the KV slice it populated against the golden trace (offset by
        # first_layer_idx); all ranks passing == the rank-sliced model reproduces single-rank KV.
        # kv_cache_pcc_check is an OPTIONAL runtime hook (golden-trace bring-up only — never used in
        # production serving), so a model whose runtime doesn't implement it can't be checked this way.
        pcc_check = getattr(runtime, "kv_cache_pcc_check", None)
        if pcc_check is None:
            raise RuntimeError(
                f"PREFILL_STANDALONE_PCC=1 but {type(runtime).__name__} implements no kv_cache_pcc_check "
                "(optional bring-up hook; see ADDING_A_PREFILL_MODEL.md §2)."
            )
        # Pass the raw trace path; the validation helper resolves it (descends the vllm hash subdir).
        pcc_check(
            kv_caches,
            slot_id=slot_id,
            n_chunks=n_chunks,
            trace_dir=os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default),
            first_layer_idx=cfg.first_layer_idx,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_config() -> None:
    """Print every env var the runner (and its downstream model/runner_utils) reads at startup so each
    rank's config is visible in logs. Values shown are the resolved effective values, not just what was
    set in the environment."""
    rows = [
        ("PREFILL_MODEL", ADAPTER.name),
        ("PREFILL_HF_MODEL", os.environ.get("PREFILL_HF_MODEL", ADAPTER.hf_model_default)),
        ("PREFILL_TTNN_CACHE", os.environ.get("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)),
        ("resolved weight_cache_path", str(ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE))),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_PP_LAYER_COUNTS", os.environ.get("PREFILL_PP_LAYER_COUNTS", "<even split>")),
        ("PREFILL_KV_ONLY_LAST_LAYER", str(KV_ONLY_LAST_LAYER)),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_STANDALONE_NCHUNKS", str(NUM_CHUNKS)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_FABRIC_MODE", os.environ.get("PREFILL_FABRIC_MODE", "<auto: 1d if sp<=8 else 2d>")),
        ("PREFILL_STANDALONE (pipeline/bring-up mode)", os.environ.get("PREFILL_STANDALONE", "0")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<trace default>")),
        ("PREFILL_STANDALONE_PCC", os.environ.get("PREFILL_STANDALONE_PCC", "0")),
        ("PREFILL_STANDALONE_CHUNKED_PCC", os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88")),
        (
            "PREFILL_STANDALONE_CHUNKED_RECORD_ONLY",
            os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0"),
        ),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        ("PREFILL_MOCK_MIGRATION", os.environ.get("PREFILL_MOCK_MIGRATION", "0")),
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
        ),
        ("PREFILL_MIGRATION_WAIT_READY_MS", os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000")),
        ("MIGRATION_DONE_FILE", os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")),
    ]
    sep = "=" * 70
    lines = [sep, "prefill_runner configuration", sep]
    lines += [f"  {label:<35} = {val}" for label, val in rows]
    lines.append(sep)
    logger.info("\n" + "\n".join(lines))


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _print_config()

    # tt-run launches the MPI ranks but does not stand up the distributed context;
    # do it here before reading rank/size (idempotent across re-entry).
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())

    layer_split = compute_layer_split(NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS))
    first_layer_idx, num_my_layers = layer_split[rank]
    is_first_rank = rank == 0
    is_last_rank = rank == num_ranks - 1
    logger.info(
        f"[pp rank {rank}/{num_ranks}] mesh={GLOBAL_MESH_SHAPE} layers=[{first_layer_idx}, "
        f"{first_layer_idx + num_my_layers}) is_first={is_first_rank} is_last={is_last_rank} "
        f"chunk_size={CHUNK_SIZE} max_seq_len={MAX_SEQ_LEN} num_users={NUM_USERS}"
    )

    mesh_device = open_mesh_device(GLOBAL_MESH_SHAPE, MODEL_CFG, l1_small_size=_L1_SMALL_SIZE)

    hf_config = ADAPTER.load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN

    params = PrefillRunParams(
        mesh_shape=GLOBAL_MESH_SHAPE,
        num_layers=num_my_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
        capacity_factor=CAPACITY_FACTOR,
        num_links=2 if is_blackhole() else 1,  # Blackhole trains 2 fabric routing planes, others 1
        gate_mode_name=_gate_mode_name,
        # Chunked prefill never samples (the populated KV cache is the output), so the final stage is
        # headless: its last layer runs KV-only and no norm/LM-head is built. Only the last rank does
        # this (single-rank inherits it); PREFILL_KV_ONLY_LAST_LAYER can force it off.
        kv_only_last_layer=is_last_rank and KV_ONLY_LAST_LAYER,
        weight_cache_path=ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE),
    )

    runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    # The engine owns the KV cache(s): allocate them once (the adapter defines the layout) as an opaque
    # KvCaches, hand that container to every runtime call, and let it free with the mesh at shutdown. The
    # runner stays model-agnostic — it never unpacks the container; the (model-specific) runtime pulls out
    # the primary cache and any secondary cache (e.g. a sparse/DSA model's index cache) it needs, and folds
    # both into the merged migration table (see build_kv_chunk_table).
    kv_caches = ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(kv_caches)

    if os.environ.get("PREFILL_STANDALONE", "0") == "1":
        _serve_standalone(runtime, kv_caches, mesh_device, hf_config, rank, num_ranks, is_first_rank)
    else:
        _serve_request(runtime, kv_caches, mesh_device, hf_config, rank, num_ranks, is_first_rank)

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


def _serve_standalone(
    runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool
) -> None:
    """Bring-up / benchmark path: golden-trace input on rank 0, D2D-socket transport between ranks,
    per-rank KV PCC. Self-contained (no external producer); covers num_ranks 1..N."""
    # Warm-up sync — the ONLY barrier. Every rank finishes compile before any chunk enters the
    # pipeline, so a downstream rank isn't still warming up while an upstream one races ahead. The
    # per-chunk loop takes no barrier. Trade-off: a rank that dies during compile hangs the others here.
    ttnn.distributed_context_barrier()

    # D2D transport: with >1 rank, every rank stands up its pipeline endpoints (revert the custom
    # sub-device as above). The post-compile barrier guarantees all ranks reach the chained create
    # rendezvous. A single rank owns the whole model — no transport.
    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)
        # The chained D2D socket rendezvous finishes at staggered times per rank. Without this barrier
        # rank 0 enters its produce loop first, fills the socket, and stalls ~6s waiting for the
        # downstream ranks to enter their consume loops — moving that skew out of the timed chunk loop.
        ttnn.distributed_context_barrier()

    logger.info(f"[pp rank {rank}] setup complete, entering standalone loop")
    run_standalone_loop(runtime, kv_caches, rank, num_ranks, d2d_in=d2d_in, d2d_out=d2d_out)

    if d2d_in is not None or d2d_out is not None:
        # Free the services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        d2d_in = d2d_out = None
        gc.collect()


def _serve_request(runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Production serving: token chunks + PrefillMetadata arrive over the H2D socket from an external
    producer (prefill_producer.py / the scheduler); unbounded (runs to SIGTERM). Same pipeline
    mechanics as standalone (num_ranks 1..N over D2D); the only difference is the trigger (H2D input)
    and that it runs forever.

    Migration (KV-chunk-table publish) + per-layer LayerAck are wired for the single-rank case only;
    they are disabled for num_ranks>1 (pipelined migration is future work). Shutdown for num_ranks>1 is
    rough: downstream ranks block in D2D recv when rank 0 stops, so they exit on teardown / SIGKILL."""
    single_rank = num_ranks == 1

    ttnn.distributed_context_barrier()  # warm-up: all ranks finish compile before chunks flow

    # H2D input service lives on the first rank only (downstream ranks read from D2D). compile() leaves
    # a custom sub-device manager loaded; the service's init program validates its cores against the
    # default whole-chip sub-device, so revert first.
    h2d_service = None
    if is_first_rank:
        mesh_device.clear_loaded_sub_device_manager()
        h2d_service = build_h2d_service(
            mesh_device,
            mesh_shape=GLOBAL_MESH_SHAPE,
            chunk_size=CHUNK_SIZE,
            mapper_config=H2D_MAPPER_CONFIG,
            worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
        )
        service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        descriptor_path = h2d_service.export_descriptor(service_id)
        logger.info(
            f"[pp rank {rank}] [h2d] descriptor service_id={service_id!r} -> {descriptor_path}; "
            f"drive it with prefill_producer.py / the scheduler."
        )

    # D2D pipeline transport for num_ranks>1 (same as standalone).
    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)

    # Per-layer LayerAck -> scheduler-driven migration. Two wirings by topology:
    #   * single-rank: the runtime owns the scheduler's counter channel and inject()s it
    #     directly (the original path).
    #   * pipeline (num_ranks > 1): each rank owns only a layer slice, so it cannot inject
    #     the scheduler channel directly. Every rank pushes full {seq, source_rank, layer_idx,
    #     request_id} completions into a host-local LayerCompletionQueue; a per-host
    #     LayerCompletionRouter forwards them to the master rank, which re-emits them in
    #     global seq order into the SAME counter channel the scheduler connects to. See
    #     build_layer_completion_sink() and the pipelined_prefill package.
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
    master_rank = int(os.environ.get("PREFILL_MASTER_RANK", "0"))
    ack_channel = None
    router = None
    producer = None
    # Completion checking (master rank only, test-only): a consumer that stands in for the scheduler
    # on the master's counter channel, to verify aggregated per-(chunk, layer) completions. See
    # _CompletionCheckConsumer. Gated by PREFILL_CHECK_COMPLETIONS=1 so it never competes with a real
    # scheduler consuming the same channel in production.
    completion_check = None
    check_completions = os.environ.get("PREFILL_CHECK_COMPLETIONS", "0") == "1"
    # The single-rank LayerAck channel is the scheduler's per-layer signal (it drives migration).
    # Opt-in: creating it unconditionally makes two concurrent single-rank runs sharing a service_id
    # collide on the same /dev/shm segment. Defaults on when migration is enabled (its only consumer);
    # set PREFILL_ENABLE_LAYER_ACK=1 to force it on without full migration.
    enable_layer_ack = (
        os.environ.get("PREFILL_ENABLE_LAYER_ACK", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")) == "1"
    )
    # D2H layer-ack backend (opt-in via PREFILL_LAYER_ACK_D2H=1): a metadata-only D2HStreamService whose
    # LayerAckService reader thread injects one ack per device record, in place of the runtime's direct
    # set_layer_ack_channel() callback. Mutually exclusive with the callback path (see the single-rank
    # block below); kept as None when unused so teardown is a no-op.
    d2h_service = None
    layer_ack_service = None

    def _unlink_stale_shm(name: str) -> None:
        # A prior run that didn't tear down cleanly leaves the segment behind (shm_open O_EXCL fails).
        path = f"/dev/shm/{name.lstrip('/')}"
        if os.path.exists(path):
            logger.warning(f"[migration] removing stale shm {path} from a prior run")
            os.remove(path)

    # Migration KV-chunk-table publish: runs for ANY rank count. The runner owns the control flow;
    # every rank joins the cross-host all-gather (barrier) that merges the table, then ONLY the first
    # rank asks its model runtime to build the merged table and sends it to the worker (mirroring
    # tt-blaze where all ranks all-gather but only mesh 0 builds + sends). Previously single-rank only.
    migration_endpoint = None
    # Single opt-in: PREFILL_MIGRATION_SELFTEST=1 runs the migrate + slot==slot verify AND implies the
    # table publish it depends on, so you don't also have to set PREFILL_ENABLE_MIGRATION. The latter
    # still works on its own for production publish-without-selftest.
    _selftest = os.environ.get("PREFILL_MIGRATION_SELFTEST", "0") == "1"
    _migration_enabled = os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1" or _selftest
    if _migration_enabled:
        if is_first_rank:
            # Clear a stale DONE sentinel from a prior run so the validator can't read its pairs.
            # First rank only -- it owns the publish + validation handshake.
            _done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
            if os.path.exists(_done_file):
                logger.warning(f"[migration] removing stale DONE sentinel {_done_file} from a prior run")
                os.remove(_done_file)

        # Migration bring-up, split by ownership before the request loop opens (the worker gates on
        # SetTable + AssignDevMap, so this must finish first):
        #   * ALL RANKS deliver their local device map + join the all-gather barrier (COLLECTIVE —
        #     every rank must call it or the communicator deadlocks).
        #   * The model RUNTIME builds + serializes the model-specific KV chunk table and returns its
        #     path (runtime.build_kv_chunk_table — the model owns the cache layout / address math).
        #   * RANK 0 ONLY publishes that serialized table to the worker and blocks on WORKER_READY.
        from models.demos.common.prefill.runners.migration import (
            deliver_device_map_and_gather_stage_layout,
            publish_serialized_table_and_wait_ready,
        )

        # This rank's pipeline stage owns layers [first_layer_idx, first_layer_idx + num_my_layers).
        # The layer-aware merge gathers each rank's range so the table spans all stages; pass this
        # rank's range (same split the runtime/cache was built with).
        first_layer_idx, num_my_layers = compute_layer_split(NUM_LAYERS, num_ranks)[rank]
        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))

        # ALL RANKS: deliver local device map + contribute this stage to the merged table (barrier).
        stage_layout = deliver_device_map_and_gather_stage_layout(
            mesh_device, kv_caches, GLOBAL_MESH_SHAPE, first_layer_idx, num_my_layers
        )

        if is_first_rank:
            # RANK 0: model runtime builds + serializes the merged table (spanning all gathered
            # stages), then publish the serialized path + block on WORKER_READY.
            table_path = runtime.build_kv_chunk_table(
                kv_caches,
                table_path,
                first_layer_idx=first_layer_idx,
                num_my_layers=num_my_layers,
                stage_layout=stage_layout,
            )
            migration_endpoint = publish_serialized_table_and_wait_ready(
                table_path=table_path,
                wait_ready_timeout_ms=wait_ready_ms,
            )
        elif os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1":
            # Mock integration (prefill_producer.py): serialize the KV chunk table so an external
            # producer can read it back via ttnn.experimental.disaggregation.import_from_protobuf_file
            # and locate each chunk — WITHOUT the migration_endpoint worker (no MigrationLayerClient,
            # no WORKER_READY). One galaxy => one complete table spanning all NUM_LAYERS / NUM_USERS
            # (both caches, merged, for a sparse model).
            table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
            runtime.build_kv_chunk_table(kv_caches, path=table_path)
            # Also publish the fabric_node -> ASIC unique_id device map so the producer can resolve chips
            # for its device-less UMD read (read_dram_umd) without touching the ControlPlane.
            device_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
            serialize_device_map(mesh_device, device_map_path)
            logger.info(
                f"[mock-migration] KV chunk table -> {table_path}, device map -> {device_map_path} "
                f"(no migration worker); prefill_producer can import them"
            )
        else:
            logger.info(
                f"[migration] rank {rank}: delivered local device map + contributed stage "
                f"(first_layer={first_layer_idx}, count={num_my_layers}); rank 0 sends the merged table."
            )

    # D2H layer-ack backend: the device sends one per-layer ack record over a metadata-only D2H socket
    # (outbound_socket_service_sync in each block) and a LayerAckService reader thread derives a
    # globally-dense seq per record and pushes it into the router-owned ring. This feeds the SAME
    # LayerCompletionRouter the multi-rank host-callback path uses, so it is multi-host compatible: it
    # works on any rank count (single-rank => world_size=1 router, local ring + counter channel, no MPI).
    use_d2h = os.environ.get("PREFILL_LAYER_ACK_D2H", "0") == "1"
    # The router path covers: (a) any multi-rank run (each rank owns only a layer slice, so it can't
    # inject the scheduler channel directly and must route to the master), and (b) the D2H backend on
    # any rank count (its LayerAckService is a pure producer into the ring). The single-rank non-D2H
    # path keeps the original direct-inject wiring.
    use_router = (not single_rank) or (use_d2h and enable_layer_ack)

    if use_router:
        # Imported here (not at module top) so single-rank / no-extension builds never need the
        # standalone _layer_completion .so (built only with WITH_PYTHON_BINDINGS).
        from models.demos.common.prefill.runners.pipelined_prefill import LayerCompletionQueue, LayerCompletionRouter

        # Each rank's router OWNS its own ring, so the name must be per-rank — append _{rank} even to
        # the env override (a single literal would make colocated ranks unlink each other's live ring
        # and collide on O_EXCL create).
        ring_base = os.environ.get("PREFILL_LAYER_COMPLETION_RING", "/tt_prefill_layer_completion_ring")
        ring_shm_name = f"{ring_base}_{rank}"
        _unlink_stale_shm(ring_shm_name)
        if rank == master_rank:
            _unlink_stale_shm(ack_shm_name)
        # The router owns the host-local ring (and, on the master, the scheduler counter channel,
        # which it inject()s in order). Subordinate ranks MPI-forward completions to the master.
        # Constructed BEFORE the ring source below: the router creates the ring, the source connects.
        router = LayerCompletionRouter(
            rank=rank,
            world_size=num_ranks,
            master_rank=master_rank,
            ring_shm_name=ring_shm_name,
            scheduler_channel_shm_name=ack_shm_name if rank == master_rank else "",
        )
        if use_d2h:
            # Device-record source. The reader thread reconstructs (chunk, global-layer) from a per-rank
            # record counter — valid because each rank emits exactly its layer-slice count (num_my_layers)
            # of records per chunk, in layer order. d2h_service is threaded into the request loop so
            # prefill_chunk drives the device-side send on every rank.
            first_layer_idx, num_my_layers = compute_layer_split(NUM_LAYERS, num_ranks)[rank]
            d2h_service = ttnn.D2HStreamService(
                mesh_device,
                global_spec=None,
                fifo_size_bytes=LAYER_ACK_FIFO_SIZE_BYTES,
                worker_cores=SYNC_WORKER_CORES,
                metadata_size_bytes=METADATA_SIZE_BYTES,
            )
            layer_ack_service = ttnn.LayerAckService(
                d2h_service,
                ring_shm_name,
                source_rank=rank,
                num_layers=NUM_LAYERS,
                first_layer_idx=first_layer_idx,
                local_layers=num_my_layers,
            )
            layer_ack_service.start()  # connects to the router-owned ring (created above)
            source_desc = "D2H device records"
        else:
            # Host-callback source: the runtime fires on_layer_complete(layer_idx, request_id) per layer
            # and the sink pushes into the ring (no device D2H). seq stride is the GLOBAL layer total
            # (NUM_LAYERS), NOT this rank's slice; layer_idx arriving at the sink is already global; the
            # chunk index is bound per prefill() call as request_id (passed by _compute_and_send), so the
            # sink reads no shared mutable state.
            producer = LayerCompletionQueue.connect(ring_shm_name, connect_timeout_ms=30000)
            runtime.set_layer_completion_sink(
                build_layer_completion_sink(
                    producer,
                    source_rank=rank,
                    num_layers=NUM_LAYERS,
                )
            )
            source_desc = "host on_layer_complete callback"
        logger.info(
            f"[migration] layer-completion routing up: rank={rank}/{num_ranks} master={master_rank} "
            f"ring={ring_shm_name} source={source_desc} "
            + (f"(owns scheduler channel {ack_shm_name})" if rank == master_rank else "(subordinate -> master)")
        )

        if rank == master_rank and check_completions:
            completion_check = _CompletionCheckConsumer(ack_shm_name, num_layers=NUM_LAYERS)
    elif single_rank and enable_layer_ack:
        # Single-rank non-D2H direct path: the runtime owns + inject()s the scheduler counter channel
        # directly (on_layer_complete fires per layer inside the model).
        _unlink_stale_shm(ack_shm_name)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        runtime.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")
    elif single_rank:
        logger.info("[migration] LayerAck channel disabled (set PREFILL_ENABLE_LAYER_ACK=1 to enable)")

    # Interleaved migration self-test: rank 0 stands in for the scheduler — consume the per-layer ack
    # channel and migrate each chunk as its layers complete, overlapping later chunks' prefill (replaces
    # the post-loop bulk migrate). Rank 0 only (it holds the migration client); other ranks just verify.
    mig_driver = None
    if _selftest and is_first_rank and os.environ.get("PREFILL_MIGRATION_INTERLEAVED", "0") == "1":
        assert migration_endpoint is not None, "rank 0 must hold the migration client for interleaved migrate"
        # Granularity flag: "layerwise" (default) migrates each chunk's layers as they ack — finer
        # overlap; "chunkwise" waits for a chunk's full 61-layer ack then migrates it in one shot.
        granularity = os.environ.get("PREFILL_MIGRATION_GRANULARITY", "layerwise").strip().lower()
        if granularity not in ("layerwise", "chunkwise"):
            raise ValueError(f"PREFILL_MIGRATION_GRANULARITY must be 'layerwise' or 'chunkwise', got {granularity!r}")
        mig_driver = _InterleavedMigrationDriver(
            ack_shm_name,
            migration_endpoint,
            num_layers=NUM_LAYERS,
            src_slot=int(os.environ.get("PREFILL_MIGRATE_SRC_SLOT", "0")),
            dst_slot=int(os.environ.get("PREFILL_MIGRATE_DST_SLOT", "1")),
            endpoint_id=int(os.environ.get("PREFILL_MIGRATION_ENDPOINT_ID", "1")),
            wait_complete_ms=int(os.environ.get("PREFILL_MIGRATE_WAIT_COMPLETE_MS", "120000")),
            # Diagnostics: the master router (pipeline only; None in single-rank) exposes .processed so the
            # driver can log injected-vs-consumed acks. router is None here in the single-rank path.
            router=router,
            granularity=granularity,
        )
        logger.info(
            f"[interleave] migration mode = INTERLEAVED, granularity={granularity} "
            f"(PREFILL_MIGRATION_INTERLEAVED=1, PREFILL_MIGRATION_GRANULARITY={granularity}): rank 0 migrates "
            "as layers ack, overlapping later chunks' prefill; one blocking wait at drain"
        )
    elif _selftest and is_first_rank:
        logger.info(
            "[interleave] migration mode = BULK (single post-loop migrate); set PREFILL_MIGRATION_INTERLEAVED=1 "
            "to interleave migrates with prefill"
        )
    logger.info(f"[pp rank {rank}] setup complete, entering request loop")

    try:
        # Prefill into the src slot (slot 0). Returns {slot_id -> real (non-pad) end position}.
        # In-loop migration self-test: rank 0 loopback-migrates src->dst, then EVERY rank asserts its
        # local dst KV slice equals its src slice (validate_migrations_pairwise). Env-gated so ALL ranks
        # take the same branch (the barrier below requires it); production serving is unaffected.
        real_end_per_slot = run_request_loop(
            runtime,
            kv_caches,
            rank,
            num_ranks,
            hidden_size=hf_config.hidden_size,
            h2d_service=h2d_service,
            d2d_in=d2d_in,
            d2d_out=d2d_out,
            migration_driver=mig_driver,
            d2h_service=d2h_service,
        )

        if _selftest:
            src_slot = int(os.environ.get("PREFILL_MIGRATE_SRC_SLOT", "0"))
            dst_slot = int(os.environ.get("PREFILL_MIGRATE_DST_SLOT", "1"))

            if mig_driver is None:
                # ensure KV cache is written if not interleaving migration
                ttnn.synchronize_device(runtime.mesh_device)
                if num_ranks > 1:
                    ttnn.distributed_context_barrier()

            # RANK 0 ONLY issues the migrate (it holds the MigrationLayerClient).
            if is_first_rank and mig_driver is not None:
                # Interleaved: per-chunk migrates were already issued during the loop; drain the tail
                # (consume any remaining acks + wait_complete the deferred copies).
                mig_driver.drain(expected_chunks=NUM_CHUNKS)
            elif is_first_rank:
                assert migration_endpoint is not None, "rank 0 must hold the migration client for the self-test"
                # Loopback target is THIS endpoint's own id (A->B loopback; no peer, no connect_to).
                self_ep = int(os.environ.get("PREFILL_MIGRATION_ENDPOINT_ID", "1"))
                # Position range = the src slot's real prefilled length, aligned UP to the 32-token KV
                # migration chunk (blaze's _align_up(S)). Migrate the FULL global layer range [0, NUM_LAYERS)
                # the merged table was built for — the worker routes each layer to its owning stage.
                POS_CHUNK = 32
                real_end = real_end_per_slot.get(src_slot, 0)
                pos_end = ((real_end + POS_CHUNK - 1) // POS_CHUNK) * POS_CHUNK
                logger.info(
                    f"[migration-selftest] loopback migrate slot{src_slot}->slot{dst_slot} "
                    f"layers[0,{NUM_LAYERS}) pos[0,{pos_end}) (real_end={real_end}, self_ep={self_ep})"
                )
                # wait_complete's C++ default is only 30s; a full-prefill loopback copy (here ~2 GB:
                # 56320 pos x 61 layers) can exceed that, so make it configurable.
                wait_complete_ms = int(os.environ.get("PREFILL_MIGRATE_WAIT_COMPLETE_MS", "120000"))
                tok = migration_endpoint.migrate(1, self_ep, src_slot, dst_slot, 0, NUM_LAYERS, 0, pos_end)
                migration_endpoint.wait_complete(tok, wait_complete_ms)
                logger.success(f"[migration-selftest] migrate slot{src_slot}->slot{dst_slot} complete")

            # Barrier: every rank must wait for rank 0's migrate to finish before reading its local
            # dst slot (the migrate covers all stages; each rank then verifies its own layers).
            if num_ranks > 1:
                ttnn.distributed_context_barrier()
            ttnn.synchronize_device(runtime.mesh_device)

            from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import validate_migrations_pairwise

            validate_migrations_pairwise(runtime, kv_caches, [(src_slot, dst_slot)])
    finally:
        # Always tear down — the request loop can raise (e.g. the layer-completion sink's ring-full
        # spin timing out on a stalled router); without this, producer/router/ack segments + the
        # router listener thread leak, and a downstream peer blocked in D2D recv deadlocks the pipeline.
        # Release services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).

        # Release services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        # Stop the D2H LayerAckService reader thread first (D2H backend only): it reads the D2H
        # service's sockets and pushes into the router-owned ring, so it must be joined before the
        # D2H service is dropped AND before router.stop() drains the ring (so the last records land).
        # No-op under the host-callback / direct-inject backends (layer_ack_service stays None).
        if layer_ack_service is not None:
            layer_ack_service.stop()
            layer_ack_service = None
        h2d_service = d2d_in = d2d_out = d2h_service = None
        gc.collect()
        # NOTE: for num_ranks>1 the request loop only returns on a clean _shutdown; the known
        # rough-shutdown path (downstream ranks block in D2D recv, exit on SIGKILL) can bypass this.
        # Clean cross-rank router teardown (barrier + end-of-request sentinel) is future work.
        if producer is not None:
            producer.shutdown()
        if router is not None:
            router.stop()  # joins the listener; the master's final ring-drain + inject happens HERE
        if completion_check is not None:
            # Tally AFTER router.stop(): the master injects its own trailing completions during the
            # listener's final drain (inside stop()). The consumer's mapping survives the owner's
            # shm_unlink (POSIX), so it still reads those — tallying earlier would miss them and
            # falsely report "count short". router.stop() unlinks the channel on the master.
            completion_check.stop_and_report()
        if ack_channel is not None:
            ack_channel.shutdown()  # munmap + shm_unlink
            ack_channel = None


if __name__ == "__main__":
    main()
