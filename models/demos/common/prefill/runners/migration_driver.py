# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""KV-migration test driver: prefills slots over H2D, then migrates their KV. Run this as a script.

This is the ENTRY POINT for a migration run — the whole terminal-C side in one process. It owns
everything migration: env/manifest/CLI config, the MigrationLayerClient attach, the optional
cross-endpoint pairing, resolving the src->dst mapping, issuing the migrate() calls, and writing the two
sidecar files consumers wait on.

The H2D half is driven with ``prefill_producer``'s helpers — manifest, push schedule, ack drain, golden
PCC — so the push path is defined in exactly one place. The dependency runs ONE way: ``prefill_producer``
is the plain runner test and imports nothing from here, so a runner-only run can never pull migration in.
Everything migration-specific lives in this file, including the optional src-KV dump for a decode-side
consumer (``--dump-src-kv``).

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

Running it — migration must share the producer's process: it needs that run's resident-slot state, and it
must migrate while the runner still holds the KV in device DRAM. Hence one entry point covering both, and
the same three-terminal flow as before — terminal C just runs THIS module instead of prefill_producer::

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

    # C) prefill + migrate (this module). For a runner-only run, use prefill_producer instead.
    python3 -m models.demos.common.prefill.runners.migration_driver \
      --manifest models/demos/common/prefill/runners/producer_manifests/<MANIFEST>.yaml

Env (all also settable from the producer manifest's typed ``migration:`` block — see
``apply_manifest_env``; an explicitly exported env var always wins). Invoking this module IS the opt-in,
so ``migration.issue`` is redundant here and a ``false`` is warned about rather than honoured:
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
import struct
import sys
import time

from loguru import logger

import ttnn


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


def _parse_layers(spec: str):
    """PREFILL_MIGRATION_LAYERS / manifest ``layers`` -> a list of layer ids, or None for "all"."""
    spec = (spec or "").strip()
    return [int(x) for x in spec.split(",") if x.strip()] if spec else None


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
        self.layers = _parse_layers(os.environ.get("PREFILL_MIGRATION_LAYERS", ""))
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
        # Safelist the configured directory and confirm both joined paths stay inside it before opening.
        base_dir = os.path.abspath(os.path.dirname(self.handoff_path) or ".")
        name = os.path.basename(self.handoff_path)
        handoff_path = os.path.abspath(os.path.join(base_dir, name))
        tmp = os.path.abspath(os.path.join(base_dir, name + ".tmp"))
        if not (handoff_path.startswith(base_dir + os.sep) and tmp.startswith(base_dir + os.sep)):
            raise ValueError(
                f"PREFILL_MIGRATION_HANDOFF_PATH={self.handoff_path!r} escapes its own directory "
                f"{base_dir!r} (resolved to {handoff_path!r}); give a path whose basename stays inside it."
            )
        if not os.path.isdir(base_dir):
            raise ValueError(
                f"PREFILL_MIGRATION_HANDOFF_PATH={self.handoff_path!r} needs directory {base_dir!r}, which "
                f"does not exist. Create it before the run — for P->D it must also be visible to the DECODE "
                f"side, so a missing directory here usually means the shared mount is absent."
            )
        with open(tmp, "w") as f:
            json.dump({"slots": slots}, f)
        os.replace(tmp, handoff_path)  # atomic: the decode side never reads a half-written handoff
        logger.success(f"[migration_driver] wrote handoff {handoff_path} ({len(slots)} slot(s)): {slots}")

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
        # Safelist the configured directory and confirm the joined path stays inside it before opening.
        base_dir = os.path.abspath(os.path.dirname(self.done_file) or ".")
        done_path = os.path.abspath(os.path.join(base_dir, os.path.basename(self.done_file)))
        if not done_path.startswith(base_dir + os.sep):
            raise ValueError(
                f"MIGRATION_DONE_FILE={self.done_file!r} escapes its own directory {base_dir!r} "
                f"(resolved to {done_path!r}); give a path whose basename stays inside it."
            )
        if not os.path.isdir(base_dir):
            raise ValueError(
                f"MIGRATION_DONE_FILE={self.done_file!r} needs directory {base_dir!r}, which does not exist. "
                f"Create it before the run."
            )
        with open(done_path, "w") as f:
            for s, d in pairs:
                f.write(f"{s} {d}\n")
        logger.success(f"[migration_driver] wrote DONE sentinel {done_path} ({len(pairs)} pair(s)): {pairs}")
        return pairs


def _dump_src_kv(dump_dir: str, table, stats, slot_traces: dict, layers) -> None:
    """Save each source slot's KV to ``<dump_dir>/src_slot<N>.pt`` as ``{"ref_kvpe_list": [...]}`` indexed
    BY LAYER, for a decode-side consumer to PCC its received copy against (blaze's
    ``--migration-validate-src-kv-pt``). That check tests the TRANSFER rather than the model: comparing
    decode's destination to the exact bytes prefill held beats comparing it to a golden trace, which would
    also fold in any model error.

    Read device-lessly over UMD via the runner's published table -- the same path the producer's PCC uses,
    reusing its table lookup and cache decode. Rows outside ``layers`` stay None: they are never read, and
    a full 78-layer slot would be ~10 GB. Values are stored in the DEVICE rope frame exactly as read, with
    no re-interleave, because that is the frame decode compares in.

    MLA only -- the M3 triple cache has no single kvpe tensor to write.
    """
    import torch

    from models.demos.common.prefill.runners import prefill_producer as producer
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

    if producer.ADAPTER.name == "minimax_m3":
        logger.warning(
            "[migration_driver] src-KV dump is not supported for minimax_m3 (its multi-config cache has no "
            "single kvpe tensor); skipping the dump."
        )
        return

    device_map = producer._read_device_map(int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60")))
    if not device_map:
        logger.error("[migration_driver] no device map available; skipping the src-KV dump.")
        return

    head_dim = producer.ADAPTER.model_config.KV_LORA_RANK + producer.ADAPTER.model_config.QK_ROPE_HEAD_DIM
    tokens_per_block = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    wanted = set(layers) if layers else set(range(producer.NUM_LAYERS))
    base_dir = os.path.abspath(os.path.expanduser(dump_dir))
    os.makedirs(base_dir, exist_ok=True)

    for slot_id, (chunks_pushed, actual_isl) in sorted(stats.resident.items()):
        real_len = min(chunks_pushed * producer.CHUNK_SIZE, actual_isl)
        if real_len <= 0:
            continue
        read_len = ((real_len + tokens_per_block - 1) // tokens_per_block) * tokens_per_block  # round to a block
        ref_kvpe_list = [None] * producer.NUM_LAYERS
        for layer in sorted(wanted):
            decoded_rows = []
            for pos in range(0, read_len, tokens_per_block):
                loc = table.lookup(layer, pos, slot_id)
                unique_id = producer._resolve_unique_id(
                    table.get_device_group(loc.device_group_index).fabric_node_ids, device_map
                )
                raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
                decoded_rows.append(producer._decode_kv_chunk(raw, head_dim))
            device_kv = torch.cat(decoded_rows, dim=0)[:real_len]  # natural order (the table un-rotates)
            ref_kvpe_list[layer] = device_kv.unsqueeze(0).unsqueeze(0)
        # dump_dir is the safelisted base; only the derived basename varies. Confirm the join stays inside
        # it before writing.
        out = os.path.abspath(os.path.join(base_dir, f"src_slot{int(slot_id)}.pt"))
        if not out.startswith(base_dir + os.sep):
            raise ValueError(f"src-KV dump path {out!r} escapes its base directory {base_dir!r}")
        torch.save({"ref_kvpe_list": ref_kvpe_list}, out)
        logger.success(
            f"[migration_driver] slot {slot_id} src KV dumped -> {out} "
            f"(layers {sorted(wanted)}, positions [0,{real_len}))"
        )


def main() -> None:
    """Prefill a set of slots over H2D, then migrate their KV — the whole terminal-C side of a migration
    run in one process.

    The H2D half is driven with ``prefill_producer``'s helpers (manifest, schedule, ack drain, golden PCC);
    the migration half is this module. The dependency runs THIS way on purpose: prefill_producer is the
    plain runner test and knows nothing about migration, so a runner-only run can never drag migration in.
    """
    import argparse

    from models.demos.common.prefill.runners import prefill_producer as producer

    parser = argparse.ArgumentParser(
        prog="migration_driver",
        description="Prefill over H2D and migrate the resulting KV. Config comes from the same producer "
        "YAML manifest as prefill_producer (--manifest / PREFILL_PRODUCER_MANIFEST); this entry point "
        "additionally applies the manifest's `migration:` block. Needs a live migration_endpoint that the "
        "prefill runner has already driven to WORKER_READY.",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        default=os.environ.get("PREFILL_PRODUCER_MANIFEST"),
        help="Path to the producer YAML manifest (applied at startup; exported env vars override it).",
    )
    parser.add_argument(
        "--migrations",
        default=None,
        help="Arbitrary src->dst mapping as 'src:dst,src:dst,...' (e.g. '0:5,1:2,3:7'). Overrides the "
        "manifest's migration.pairs and the uniform migration.dst_slot_offset fallback.",
    )
    parser.add_argument(
        "--dump-src-kv",
        default=os.environ.get("PREFILL_MIGRATION_DUMP_SRC_KV"),
        help="Directory to save each source slot's KV into as src_slot<N>.pt, for a decode-side consumer "
        "to PCC its received copy against. Honours the migrated layer subset. Off when unset.",
    )
    args = parser.parse_args()

    # Manifest order matters: the producer applies `env:` first (so a raw PREFILL_* key wins) plus its own
    # typed blocks, and hands back the parsed document; we then apply `migration:` from it. setdefault
    # throughout, so an exported env var still beats both.
    manifest = producer._apply_manifest_env(args.manifest) if args.manifest else {}
    apply_manifest_env(manifest)
    if args.migrations is not None:
        os.environ["PREFILL_MIGRATION_PAIRS"] = args.migrations  # CLI beats manifest + env
    producer._load_env_config()

    cfg = producer._config_from_env()
    # Invoking THIS module is the opt-in, so migration.issue is redundant here. Warn rather than silently
    # honour a `false` that would turn the whole invocation into a no-op.
    if os.environ.get("PREFILL_PRODUCER_ISSUE_MIGRATION", "1") == "0":
        logger.warning(
            "[migration_driver] the manifest sets migration.issue: false, which is ignored when this module "
            "is the entry point — invoking it IS the opt-in. Run prefill_producer for a no-migration run."
        )
    driver = MigrationDriver(
        chunk_size=producer.CHUNK_SIZE,  # module attrs, read AFTER _load_env_config() rebinds them
        num_layers=producer.NUM_LAYERS,
        default_dst_slot_offset=cfg.num_users,
    )

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    logger.info(
        f"[migration_driver] service_id={service_id!r} users={cfg.num_users} "
        f"chunks=[{cfg.chunks_min},{cfg.chunks_max}] max_requests={cfg.max_requests} verify={cfg.verify}"
    )
    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    payload_bytes = service.payload_size_bytes()
    logger.info(f"[migration_driver] attached; payload={payload_bytes}B")

    kv_table = producer._read_kv_chunk_table(timeout_s)
    ack_channel = producer._connect_layer_ack_channel(timeout_s)

    # Attach + pair BEFORE pushing: a missing endpoint fails in seconds instead of after a multi-minute
    # prefill, and a cross-endpoint pairing rendezvous's while the decode side is still blocked on it.
    driver.attach()

    slot_traces, slot_lengths, pools_by_trace = producer._resolve_slot_prompts(cfg)
    cfg.slot_lengths = slot_lengths

    def push_chunk(slot_id: int, chunk_idx: int, actual_start: int, actual_end: int) -> float:
        pool = pools_by_trace[slot_traces[slot_id]]
        chunk_bytes = producer._chunk_to_host_array(pool[actual_start : actual_start + producer.CHUNK_SIZE])
        assert (
            chunk_bytes.nbytes == payload_bytes
        ), f"payload {chunk_bytes.nbytes}B != service-expected {payload_bytes}B"
        logger.info(f"[migration_driver] push slot={slot_id} cidx={chunk_idx} start={actual_start} end={actual_end}")
        push_start = time.perf_counter()
        service.forward_to_tensor_bytes(
            chunk_bytes, metadata=producer._pack_metadata(slot_id, actual_start, actual_end)
        )
        return (time.perf_counter() - push_start) * 1000.0

    stats = producer.run_schedule(cfg, push_fn=push_chunk)
    service.barrier()
    logger.info(
        f"[migration_driver] prefill done wall={stats.wall_s:.1f}s pushes={stats.total_pushes} "
        f"requests={stats.completed}"
    )
    producer._drain_layer_acks(ack_channel, producer.NUM_LAYERS * stats.total_pushes)

    # Golden PCC + the src-KV dump run BEFORE migrating, so both land before the DONE sentinel a consumer
    # may be waiting on. Migration only READS the source slots, so the order is equivalent.
    verify_ok = True
    if cfg.verify and kv_table is not None:
        try:
            verify_ok = producer._verify_resident_slots(kv_table, stats, cfg.pcc_threshold, slot_traces)
        except Exception as e:
            logger.error(f"[migration_driver] KV read/PCC failed: {type(e).__name__}: {e}")
            verify_ok = False
    elif cfg.verify:
        logger.error("[migration_driver] check_pcc requested but no KV chunk table available; skipping PCC.")
        verify_ok = False

    if args.dump_src_kv:
        if kv_table is None:
            logger.error("[migration_driver] --dump-src-kv needs the KV chunk table, which never appeared.")
        else:
            _dump_src_kv(args.dump_src_kv, kv_table, stats, slot_traces, driver.layers)

    driver.run(
        stats,
        num_slots=kv_table.config().num_slots if kv_table is not None else None,
        slot_traces=slot_traces,
        pools_by_trace=pools_by_trace,
    )

    # Optional graceful shutdown: sent LAST, because the UMD read-backs above need the mesh/DRAM alive.
    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        sentinel = struct.pack("<iii", -1, -1, -1)
        payload = producer._chunk_to_host_array([1] * producer.CHUNK_SIZE)
        logger.info("[migration_driver] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        service.forward_to_tensor_bytes(payload, metadata=sentinel)
        service.barrier()
    else:
        logger.info("[migration_driver] exiting (the runner keeps its sync-op loop running).")

    if cfg.verify and not verify_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
