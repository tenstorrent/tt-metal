# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Producer-side KV migration driver: issues real slot->slot migrations after prefill.

The producer owns pushing chunks and draining LayerAcks; this module
owns everything migration: env/manifest/CLI config, the MigrationLayerClient attach, the optional
cross-endpoint pairing, resolving the src->dst mapping, issuing the migrate() calls, and writing the
two sidecar files consumers wait on. The producer's only coupling is two calls behind one flag::

    driver = migration_driver.create_driver(...)   # None unless migration is enabled
    if driver: driver.attach()                     # before prefill (fail fast + pair early)
    ...prefill + ack drain...
    if driver: driver.run(stats, ...)              # after the KV is fully resident

Contrast with ``runners.migration``, which is the RUNNER's passive setup half (publish the KV chunk
table + device map to the migration worker and block on WORKER_READY). This module reuses that module's
queue-name / client-import resolution but sends no SET_TABLE: by the time the producer runs, the runner
has already driven the endpoint to WORKER_READY and released its setup client, leaving the cmd queue
free — exactly the handoff the C++ PrefillScheduler relies on.

Two topologies, selected by whether the destination endpoint id differs from our own:
  * LOOPBACK (dest == src, the default): src and dst slots live in ONE table, routed to the endpoint's
    internal B worker. The prefill RUNNER validates the migrated KV on-device (PREFILL_VALIDATE_MIGRATION=1
    + validate_after_prefill) once it sees the DONE sentinel — the producer never PCCs migrated KV itself.
  * CROSS-ENDPOINT P->D (dest != src): dst lives in the remote DECODE galaxy's table, an independent
    address space. Requires the pairing/connect handshake below, and the decode side has no way to know
    each slot's prompt length / last token, so we write a JSON handoff sidecar for it.

Enabling it — ``--migrate`` on the producer CLI, ``migration.issue: true`` in the manifest, or
``PREFILL_PRODUCER_ISSUE_MIGRATION=1``. Nothing here runs otherwise.

Running it — migration is separate CODE but the SAME process as the producer (it needs the live H2D
run's resident-slot state, and must migrate while the runner still holds the KV in device DRAM). So the
three-terminal flow is unchanged; only terminal C's manifest selects whether migration happens::

    export ENGINE=/path/to/tt-llm-engine TT_METAL_HOME=/path/to/tt-metal HOST=<this-host>
    source $TT_METAL_HOME/python_env/bin/activate

    # A) migration endpoint (leave running)
    cd $ENGINE/disaggregation/migration
    ./launch_migration_endpoints.sh --name_server_host $HOST \
        --prefill_hosts $HOST --prefill_endpoint_id 1

    # B) prefill runner (wait for WORKER_READY)
    cd $TT_METAL_HOME
    ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
      models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_1rank.yaml \
      $HOST:1

    # C) producer; migration runs iff the manifest sets migration.issue (or you pass --migrate)
    python3 -m models.demos.common.prefill.runners.prefill_producer \
      --manifest models/demos/common/prefill/runners/producer_manifests/<MANIFEST>.yaml

Env (all also settable from the producer manifest's typed ``migration:`` block — see
``apply_manifest_env``; an explicitly exported env var always wins):
  PREFILL_PRODUCER_ISSUE_MIGRATION  "1" to attach a MigrationLayerClient and migrate slot KV after
                                    prefill (default 0 = no migration). Manifest: ``issue``.
  PREFILL_MIGRATION_DEST_ENDPOINT_ID  destination endpoint id for migrate() (default 1). Equal to our
                                    OWN id => loopback; a different id => cross-endpoint P->D.
  PREFILL_MIGRATION_SRC_ENDPOINT_ID  our OWN endpoint id, i.e. the prefill side (default 1). Only used
                                    to decide loopback vs cross-endpoint and to pick the pairing role.
  PREFILL_MIGRATION_PAIRS           arbitrary any-src -> any-dst mapping as "src:dst,src:dst,..." (e.g.
                                    "0:5,1:2,3:7"). When set, migrates exactly these pairs (each src must
                                    be a resident/prefilled slot; dst must fit the KV table; duplicate src
                                    fans out). Also settable via ``--migrations`` (CLI wins) or the
                                    manifest ``migration.pairs`` list. Overrides DST_SLOT_OFFSET.
  PREFILL_MIGRATION_DST_SLOT_OFFSET  offset fallback used ONLY when PREFILL_MIGRATION_PAIRS is unset:
                                    dst_slot = src_slot + offset (default = PREFILL_NUM_USERS, i.e. src
                                    slots [0,N) migrate to dst slots [N,2N); the KV table needs >= 2N slots).
  PREFILL_MIGRATION_LAYERS          "0,3" => migrate ONLY those layer rows, one migrate() per layer;
                                    unset (default) => the whole [0, num_layers) range in one shot.
  PREFILL_MIGRATION_HANDOFF_PATH    path of the JSON handoff for a cross-endpoint decode consumer;
                                    unset (default) => not written at all.
  PREFILL_MIGRATION_TIMEOUT_MS      per-migration wait_complete timeout (default 3600000).
  MIGRATION_DONE_FILE               path of the DONE sentinel the runner polls (default
                                    /tmp/migration_done.sentinel).
  Queues + client come from the runner's migration env, resolved via ``runners.migration``:
  PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE and PREFILL_MIGRATION_CLIENT_DIR.
"""

import json
import os

from loguru import logger


def apply_manifest_env(manifest: dict) -> None:
    """Map a producer manifest's typed ``migration:`` block onto the PREFILL_MIGRATION_* env vars this
    module reads. Uses setdefault, so an explicitly exported env var (and the manifest's own verbatim
    ``env:`` passthrough, which the producer applies first) still wins.

    Called from the producer's ``_apply_manifest_env`` at import time, before anything reads the env.
    """
    migration = manifest.get("migration") or {}

    def sd(key, val):  # setdefault, stringified; skips None so an absent field leaves the default
        if val is not None:
            os.environ.setdefault(key, str(val))

    def sd_bool(key, val):  # YAML true/false -> the "1"/"0" the env parsing expects
        if val is not None:
            os.environ.setdefault(key, "1" if val else "0")

    sd_bool("PREFILL_PRODUCER_ISSUE_MIGRATION", migration.get("issue"))
    sd("PREFILL_MIGRATION_DEST_ENDPOINT_ID", migration.get("dest_endpoint_id"))
    sd("PREFILL_MIGRATION_SRC_ENDPOINT_ID", migration.get("src_endpoint_id"))
    sd("PREFILL_MIGRATION_DST_SLOT_OFFSET", migration.get("dst_slot_offset"))
    # Arbitrary src->dst mapping. Accept a "src:dst,src:dst" string, or a list of {src, dst} dicts /
    # [src, dst] pairs / "src:dst" strings; all normalize to the PREFILL_MIGRATION_PAIRS env string.
    pairs = migration.get("pairs")
    if pairs is not None:
        if isinstance(pairs, str):
            sd("PREFILL_MIGRATION_PAIRS", pairs)
        else:
            parts = []
            for p in pairs:
                if isinstance(p, dict):
                    parts.append(f"{p['src']}:{p['dst']}")
                elif isinstance(p, (list, tuple)):
                    parts.append(f"{p[0]}:{p[1]}")
                else:
                    parts.append(str(p))  # already a "src:dst" string
            sd("PREFILL_MIGRATION_PAIRS", ",".join(parts))
    # Layer subset: accept "0,3" or [0, 3].
    layers = migration.get("layers")
    if layers is not None:
        sd("PREFILL_MIGRATION_LAYERS", layers if isinstance(layers, str) else ",".join(str(x) for x in layers))
    sd("PREFILL_MIGRATION_HANDOFF_PATH", migration.get("handoff_path"))
    sd("PREFILL_MIGRATION_TIMEOUT_MS", migration.get("timeout_ms"))
    sd("MIGRATION_DONE_FILE", migration.get("done_file"))
    sd("PREFILL_MIGRATION_CLIENT_DIR", migration.get("client_dir"))
    sd("PREFILL_MIGRATION_CMD_QUEUE", migration.get("cmd_queue"))
    sd("PREFILL_MIGRATION_TABLE_QUEUE", migration.get("table_queue"))
    sd("PREFILL_MIGRATION_RESP_QUEUE", migration.get("resp_queue"))
    sd("PREFILL_MIGRATION_TABLE_PATH", migration.get("table_path"))
    sd("PREFILL_MIGRATION_DEVICE_MAP_PATH", migration.get("device_map_path"))


def add_cli_arguments(parser) -> None:
    """Register the producer's migration CLI flags on its ArgumentParser."""
    parser.add_argument(
        "--migrate",
        action="store_true",
        default=False,
        help="Enable producer-issued KV migration after prefill (same as PREFILL_PRODUCER_ISSUE_MIGRATION=1 "
        "or the manifest's migration.issue). Off by default.",
    )
    parser.add_argument(
        "--migrations",
        default=None,
        help="Arbitrary src->dst migration mapping as 'src:dst,src:dst,...' (e.g. '0:5,1:2,3:7'). Overrides "
        "PREFILL_MIGRATION_PAIRS / the manifest and the uniform PREFILL_MIGRATION_DST_SLOT_OFFSET fallback.",
    )


def apply_cli_args(args) -> None:
    """Fold parsed migration CLI flags into the env (CLI wins over the manifest, which used setdefault).
    Must run before ``create_driver``. ``--migrate`` only ever turns migration ON, so omitting it leaves an
    env/manifest opt-in intact."""
    if getattr(args, "migrate", False):
        os.environ["PREFILL_PRODUCER_ISSUE_MIGRATION"] = "1"
    if getattr(args, "migrations", None) is not None:
        os.environ["PREFILL_MIGRATION_PAIRS"] = args.migrations


def create_driver(*, chunk_size: int, num_layers: int, default_dst_slot_offset: int):
    """Build a ``MigrationDriver`` from the env, or return ``None`` when migration is not enabled.

    ``None`` is the whole opt-out: ``start``/``finish`` both accept it and do nothing, so with
    PREFILL_PRODUCER_ISSUE_MIGRATION unset the producer runs its push/PCC path and never attaches a
    client, opens a queue, or writes a sidecar.

    ``chunk_size`` / ``num_layers`` are passed in rather than re-read from PREFILL_CHUNK_SIZE /
    PREFILL_NUM_LAYERS so the driver can never disagree with the transport the producer actually used.
    ``default_dst_slot_offset`` (the producer's num_users) is the fallback when neither
    PREFILL_MIGRATION_DST_SLOT_OFFSET nor an explicit pair list is given.
    """
    if os.environ.get("PREFILL_PRODUCER_ISSUE_MIGRATION", "0") != "1":
        return None
    return MigrationDriver(
        chunk_size=chunk_size,
        num_layers=num_layers,
        default_dst_slot_offset=default_dst_slot_offset,
    )


# ---------------------------------------------------------------------------------------------------
# Producer seam: `start` + `finish` are the ONLY two calls the producer makes at runtime. They exist so
# the host stays a pure H2D push engine — no `if migrating:` guards, no migration locals, no knowledge
# of attach-vs-run ordering. Both tolerate the disabled case (start returns None, finish no-ops on
# None), so the producer never branches on migration at all.
# ---------------------------------------------------------------------------------------------------


def start(args=None, *, chunk_size: int, num_layers: int, default_dst_slot_offset: int):
    """Fold in CLI overrides, build the driver, and attach it. Returns the driver, or ``None`` when
    migration is not enabled (the caller passes that straight to ``finish``).

    Call BEFORE prefill: attaching early makes a missing migration endpoint fail fast rather than after
    a multi-minute prefill, and lets a cross-endpoint pairing rendezvous while the decode side's
    blocking connect_to is still waiting. ``args`` is the producer's parsed argparse namespace (may be
    omitted when there is no CLI)."""
    if args is not None:
        apply_cli_args(args)
    driver = create_driver(
        chunk_size=chunk_size, num_layers=num_layers, default_dst_slot_offset=default_dst_slot_offset
    )
    if driver is not None:
        driver.attach()
    return driver


def finish(driver, stats, *, num_slots: int = None, slot_traces: dict = None, pools_by_trace: dict = None) -> list:
    """Migrate + publish the sidecars, or do nothing when ``driver`` is None. Returns the migrated
    triples (empty when disabled).

    Call AFTER the ack drain, while the runner is still alive: the endpoint reads source KV from device
    DRAM, so a SHUTDOWN sentinel must not have been sent yet."""
    if driver is None:
        return []
    return driver.run(stats, num_slots=num_slots, slot_traces=slot_traces, pools_by_trace=pools_by_trace)


class MigrationDriver:
    """Issues producer-side KV migrations. Construct via ``create_driver`` (which applies the enable flag),
    then ``attach()`` before prefill and ``run()`` after the ack drain."""

    def __init__(self, *, chunk_size: int, num_layers: int, default_dst_slot_offset: int):
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.dest_endpoint_id = int(os.environ.get("PREFILL_MIGRATION_DEST_ENDPOINT_ID", "1"))
        # Our OWN endpoint id (prefill side). dest == src => loopback (src/dst share one table);
        # dest != src => cross-endpoint P->D (dst lives in the remote decode table). Drives which mapping
        # invariants apply in _resolve_pairs and whether we do the pairing handshake.
        self.src_endpoint_id = int(os.environ.get("PREFILL_MIGRATION_SRC_ENDPOINT_ID", "1"))
        self.timeout_ms = int(os.environ.get("PREFILL_MIGRATION_TIMEOUT_MS", "3600000"))
        self.dst_slot_offset = int(os.environ.get("PREFILL_MIGRATION_DST_SLOT_OFFSET", str(default_dst_slot_offset)))
        # PREFILL_MIGRATION_LAYERS="0,3" => extract only those layers (one migrate per layer) into a
        # layer-id-indexed decode table; unset => migrate the whole [0, num_layers) range in one shot.
        spec = os.environ.get("PREFILL_MIGRATION_LAYERS", "").strip()
        self.layers = [int(x) for x in spec.split(",") if x.strip()] if spec else None
        self.done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
        self.handoff_path = os.environ.get("PREFILL_MIGRATION_HANDOFF_PATH", "")
        self.client = None

    @property
    def cross_endpoint(self) -> bool:
        """True when the destination table lives in a DIFFERENT endpoint (P->D) rather than our own."""
        return self.dest_endpoint_id != self.src_endpoint_id

    # ---------------------------------------------------------------- attach

    def attach(self) -> None:
        """Attach the MigrationLayerClient and, when cross-endpoint, complete the pairing handshake.

        Called BEFORE prefill so a missing endpoint fails fast (rather than after a long prefill), and so
        the decode side's blocking connect_to rendezvous's promptly instead of risking a connect timeout
        while we push chunks.
        """
        self.client = self._attach_client()
        if self.cross_endpoint:
            self._pair_cross_endpoint()

    def _attach_client(self):
        """Attach a MigrationLayerClient to the migration endpoint's queues.

        Reuses ``runners.migration``'s queue-name / client-import resolution (PREFILL_MIGRATION_{CMD,TABLE,
        RESP}_QUEUE + PREFILL_MIGRATION_CLIENT_DIR). Does NOT send SET_TABLE / device map / wait_ready: the
        runner already drove the endpoint to WORKER_READY (migration.publish_table_and_wait_ready) and
        released its setup client, leaving the cmd queue free for us. Returns the client (do NOT call
        shutdown() on it: the endpoint's lifetime is owned by the launcher, not the producer)."""
        from models.demos.common.prefill.runners.migration import _import_migration_client, _resolve_queue_names

        cmd_q, table_q, resp_q = _resolve_queue_names()
        client = _import_migration_client().MigrationLayerClient(cmd_q, table_q, resp_q)
        logger.info(f"[migration_driver] client attached: cmd={cmd_q} table={table_q} resp={resp_q}")
        return client

    def _pair_cross_endpoint(self) -> None:
        """Pair this prefill endpoint with the decode endpoint (P->D only).

        Without this, both endpoints self-loopback and migrate() aborts "No remote table found for
        destination". Convention matches the decode side + smoke test: lower id = PUBLISHER (accepts),
        higher = CONNECTOR (initiates); both sides derive ONE service_name from the id pair."""
        publisher = min(self.src_endpoint_id, self.dest_endpoint_id)
        connector = max(self.src_endpoint_id, self.dest_endpoint_id)
        service_name = f"pd-migration-ep{publisher}-ep{connector}"
        role = "PUBLISHER" if self.src_endpoint_id == publisher else "CONNECTOR"
        logger.info(
            f"[migration_driver] cross-endpoint pairing: connect_to(remote_ep={self.dest_endpoint_id}, "
            f"role={role}, service={service_name}) own_ep={self.src_endpoint_id}"
        )
        self.client.connect_to(remote_endpoint_id=self.dest_endpoint_id, role=role, service_name=service_name)
        logger.success(f"[migration_driver] cross-endpoint pairing established with remote_ep={self.dest_endpoint_id}")

    # ------------------------------------------------------------------- run

    def run(self, stats, *, num_slots: int = None, slot_traces: dict = None, pools_by_trace: dict = None) -> list:
        """Resolve the mapping, migrate every pair, then publish the sidecars. Returns the resolved
        ``(src_slot, dst_slot, real_len)`` triples.

        Must run while the runner is alive (before any SHUTDOWN sentinel): the endpoint reads source KV
        from device DRAM. ``stats`` is the producer's RunStats — only ``stats.resident`` (slot_id ->
        (chunks_pushed, actual_isl)) is read. ``num_slots`` is the KV table's slot count when known, used
        only to bounds-check loopback destinations. ``slot_traces`` / ``pools_by_trace`` are the producer's
        per-slot prompt maps, needed only to write the cross-endpoint handoff.

        The producer does NOT validate migrated KV — the RUNNER does, on-device, once it sees the DONE
        sentinel (PREFILL_VALIDATE_MIGRATION=1).
        """
        if self.client is None:
            raise RuntimeError("[migration_driver] run() called before attach()")
        triples = self._resolve_pairs(stats, num_slots=num_slots)
        self._issue(triples)
        # Cross-endpoint P->D: write the decode-side JSON handoff BEFORE the DONE sentinel, so a consumer
        # that wakes on DONE always finds a complete handoff.
        self._write_handoff(triples, slot_traces, pools_by_trace)
        self._write_done_sentinel(triples)
        return triples

    def _resolve_pairs(self, stats, *, num_slots: int = None) -> list:
        """Resolve the concrete ``(src_slot, dst_slot, real_len)`` migrations to perform, ONE list shared by
        both the migrate step and the DONE sentinel so the two can never drift apart.

        Two ways to describe the mapping:
          * Explicit — PREFILL_MIGRATION_PAIRS="src:dst,src:dst,..." (a manifest ``migration.pairs`` list or
            the ``--migrations`` CLI flag both feed this env var). Drives ARBITRARY any-src -> any-dst
            migrations; each src must be a resident slot (it has KV to migrate) and dst must fit the table.
            Duplicate src is allowed (fan-out: migrate one slot to several dsts).
          * Offset (fallback, no explicit pairs) — every resident src slot -> src + dst_slot_offset.

        ``real_len`` is the SRC slot's resident non-pad token count (min(chunks_pushed*chunk_size,
        actual_isl)), matching the KV the runner wrote; slots with no data are skipped. If ``num_slots`` is
        known (from the KV table), dst is bounds-checked so a too-large dst fails here with a clear message
        instead of a cryptic device-side error at migrate time."""

        def real_len_of(src: int) -> int:
            chunks_pushed, actual_isl = stats.resident[src]
            return min(chunks_pushed * self.chunk_size, actual_isl)

        def check_dst(src: int, dst: int) -> None:
            if dst < 0:
                raise ValueError(f"migration dst slot {dst} (src {src}) is negative")
            # The bound below is the PREFILL table's slot count -- only meaningful for loopback (dst lives in
            # this same table). Cross-endpoint dst lives in the DECODE table, whose size the producer doesn't
            # know, so skip it there (a too-large dst still fails clearly device-side at migrate time).
            if not self.cross_endpoint and num_slots is not None and dst >= num_slots:
                raise ValueError(
                    f"migration dst slot {dst} (src {src}) is out of range: the KV table has {num_slots} "
                    f"slot(s) [0,{num_slots}). Grow PREFILL_NUM_USERS or pick a smaller dst."
                )

        spec = os.environ.get("PREFILL_MIGRATION_PAIRS", "").strip()
        triples = []
        if spec:  # explicit arbitrary mapping
            for tok in spec.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if ":" not in tok:
                    raise ValueError(f"PREFILL_MIGRATION_PAIRS entry {tok!r} must be 'src:dst' (got no ':')")
                src_s, dst_s = tok.split(":", 1)
                src, dst = int(src_s), int(dst_s)
                if src not in stats.resident:
                    raise ValueError(
                        f"migration src slot {src} is not resident (resident slots: {sorted(stats.resident)}); "
                        f"only slots the producer prefilled hold KV to migrate."
                    )
                real_len = real_len_of(src)
                if real_len <= 0:
                    logger.warning(f"[migration_driver] src slot {src} has no resident data; skipping {src}:{dst}")
                    continue
                check_dst(src, dst)
                triples.append((src, dst, real_len))
        else:  # no explicit mapping: uniform dst = src + offset over every resident slot
            for src in sorted(stats.resident):
                real_len = real_len_of(src)
                if real_len <= 0:
                    continue
                dst = src + self.dst_slot_offset
                check_dst(src, dst)
                triples.append((src, dst, real_len))

        # Reject mappings that sequential single-shot migration cannot execute correctly. Each migrate() reads
        # its SRC slot from device DRAM at migrate time and there is no staging buffer, so:
        #   * a dst that is ALSO a src (overlap) -> an earlier migration overwrites a slot a later pair still
        #     reads (swaps 0:1,1:0, chains 0:1,1:2), migrating wrong KV;
        #   * a duplicate dst -> only the last write survives, yet the DONE sentinel asks the runner to validate
        #     EVERY pair.
        # Disjoint src/dst sets — the intended case, e.g. 0:3,1:2 or src [0,N) -> dst [N,2N) — are unaffected.
        srcs = [s for (s, _, _) in triples]
        dsts = [d for (_, d, _) in triples]
        dup_dsts = sorted({d for d in dsts if dsts.count(d) > 1})
        if dup_dsts:
            raise ValueError(
                f"migration has duplicate dst slot(s) {dup_dsts}: multiple pairs target the same slot, so only "
                f"the last survives while every pair would be validated. Give each migration a distinct dst."
            )
        # src/dst overlap corrupts KV ONLY in loopback, where src and dst share one table so an earlier
        # migration can overwrite a slot a later pair still reads. Cross-endpoint src (this prefill table) and
        # dst (the decode table) are independent address spaces, so src N -> dst N is the normal case there.
        overlap = sorted(set(srcs) & set(dsts))
        if overlap and not self.cross_endpoint:
            raise ValueError(
                f"migration src/dst overlap on slot(s) {overlap}: a slot is both a source and a destination, so "
                f"sequential migration would overwrite a slot a later pair still reads (e.g. swap 0:1,1:0 or "
                f"chain 0:1,1:2 corrupt KV). Use disjoint src/dst slots (e.g. src [0,N) -> dst [N,2N)); a mapping "
                f"that needs staging through a free slot is not supported by this single-shot driver."
            )
        return triples

    def _issue(self, triples: list) -> int:
        """Migrate each resolved ``(src_slot, dst_slot, real_len)`` triple's KV, blocking on completion.

        ``self.layers`` selects which layer rows move:
          * None (default): one single-shot migrate() over the whole ``[0, num_layers)`` layer range per
            pair -- correct when the dest table is contiguous 0..N (loopback, or a full decode).
          * A list (PREFILL_MIGRATION_LAYERS, e.g. [0, 3]): one migrate() PER listed layer, range
            ``[L, L+1)``. Because migrate()'s layer range is symmetric (src row == dst row), this EXTRACTS
            specific layers from the full contiguous SOURCE table (row i = layer i) into the SAME rows of a
            layer-id-indexed dest -- the cross-endpoint P->D case where a reduced decode holds only {0,3}.
            The list MUST match the decode side's gathered layer ids.

        (The C++ PrefillScheduler streams per-layer migrations into a burst as each layer-ack lands,
        overlapping prefill; the Python client binds no burst API, so this runs after the ack drain
        instead.) Returns the number of pairs migrated."""
        layer_ranges = [(int(l), int(l) + 1) for l in self.layers] if self.layers else [(0, self.num_layers)]
        if self.layers:
            logger.info(f"[migration_driver] migrating layer subset {self.layers} (one migrate per layer)")
        migrated = 0
        next_uuid = 1
        for src_slot, dst_slot, real_len in triples:
            for layer_start, layer_end in layer_ranges:
                logger.info(
                    f"[migration_driver] MIGRATE slot {src_slot} -> {dst_slot} ep={self.dest_endpoint_id} "
                    f"layers=[{layer_start},{layer_end}) pos=[0,{real_len})"
                )
                uuid = next_uuid
                next_uuid += 1
                token = self.client.migrate(
                    uuid=uuid,
                    remote_endpoint_id=self.dest_endpoint_id,
                    src_slot=src_slot,
                    dst_slot=dst_slot,
                    layer_start=layer_start,
                    layer_end_exclusive=layer_end,
                    pos_start=0,
                    pos_end_exclusive=real_len,
                )
                self.client.wait_complete(token, self.timeout_ms)  # self-polls when no poll thread is running
            logger.success(
                f"[migration_driver] MIGRATE slot {src_slot} -> {dst_slot} complete "
                f"({len(layer_ranges)} layer range(s))"
            )
            migrated += 1
        logger.info(f"[migration_driver] migrations complete: {migrated} pair(s)")
        return migrated

    def _write_handoff(self, triples: list, slot_traces: dict, pools_by_trace: dict) -> None:
        """Write the JSON handoff the DECODE-side consumer reads (blaze run_decode_from_migrated): one entry
        per migrated pair as ``{"slots": [{dst_slot, prompt_len, last_prompt_token}, ...]}``. ``first_token``
        is intentionally omitted -- the decode side derives it from the migrated KV.

        This exists for CROSS-ENDPOINT P->D: unlike loopback (where the prefill-side runner validates the KV
        on-device and needs no prompt metadata), the destination galaxy has no way to know each slot's prompt
        length / last token, so it cannot pick the decode start position without this sidecar. ``real_len``
        (element 2 of each triple) is exactly the resident prompt length that was migrated, and the src slot's
        last prompt token is ``pool[real_len - 1]`` of the trace the producer already loaded.

        Gated on ``PREFILL_MIGRATION_HANDOFF_PATH``: unset => write nothing, which is what a loopback run
        wants since the runner validates on-device and needs no sidecar. Written atomically (tmp +
        os.replace) so the decode side never reads a partial file."""
        if not self.handoff_path:
            return
        if slot_traces is None or pools_by_trace is None:
            raise ValueError(
                "PREFILL_MIGRATION_HANDOFF_PATH is set but the producer passed no slot_traces/pools_by_trace; "
                "the handoff needs each src slot's prompt to record its last token."
            )
        slots = []
        for src, dst, real_len in triples:
            pool = pools_by_trace[slot_traces[src]]
            last_tok = int(pool[real_len - 1]) if 1 <= real_len <= len(pool) else int(pool[-1])
            slots.append({"dst_slot": int(dst), "prompt_len": int(real_len), "last_prompt_token": last_tok})
        tmp = self.handoff_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"slots": slots}, f)
        os.replace(tmp, self.handoff_path)  # atomic: the decode side never reads a half-written handoff
        logger.success(f"[migration_driver] wrote handoff {self.handoff_path} ({len(slots)} slot(s)): {slots}")

    def _write_done_sentinel(self, triples: list) -> list:
        """Write the migration DONE sentinel — one ``src dst`` line per migrated pair — that the runner's
        validate_after_prefill (PREFILL_VALIDATE_MIGRATION=1) polls for. This is the SAME handshake the
        llm-engine scheduler/driver used (prefill_scheduler_driver wrote this file after migrating). Once it
        appears, the runner PCC-validates each pair ON-DEVICE: src vs golden + dst vs golden (burst), and/or
        dst==src (PREFILL_MIGRATE_PAIRWISE=1). Takes the SAME triples list ``_issue`` consumed, so the
        sentinel matches exactly what was migrated. Returns the (src, dst) pairs written."""
        if not triples:
            raise ValueError(
                "migration is enabled but the mapping resolved to zero pairs, so there is no DONE sentinel "
                "to publish (an empty one would make the runner validate nothing and report success). "
                "Prefill itself succeeded — this is a CONFIG problem: either no slot ended up resident, or "
                "every PREFILL_MIGRATION_PAIRS src was skipped for holding no data. Check that the producer "
                "actually prefilled the src slots you asked to migrate, or turn migration off "
                "(PREFILL_PRODUCER_ISSUE_MIGRATION=0 / migration.issue: false)."
            )
        pairs = [(src, dst) for (src, dst, _) in triples]
        with open(self.done_file, "w") as f:
            for s, d in pairs:
                f.write(f"{s} {d}\n")
        logger.success(f"[migration_driver] wrote DONE sentinel {self.done_file} ({len(pairs)} pair(s)): {pairs}")
        return pairs
