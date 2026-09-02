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
    internal B worker. Because both slots are in our own table, THIS module verifies the migrated KV
    after the copy — see ``--verify-migration`` (dst == src byte compare, and/or dst vs the src's golden)
    over the same device-less UMD path ``check_pcc`` uses for sources. The runner holds no validation of
    its own: it publishes the table and the device map, and every read-back happens out here.
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

Multi-host (validating EVERY rank's layers, not just rank 0's). Both read-backs here — the source golden
PCC and the destination check — go over UMD, which reaches only the chips physically attached to the host
the process runs on. On a pipeline runner spanning N hosts, ONE driver process therefore verifies just its
own host's layer slice and SKIPS the rest (it says so, but a PASS is then a fraction of the model). Covering
the model needs one process per host — which this module does NOT arrange for itself. Launching is the
shell's job, exactly as it is for the runner (``run_pipeline_prefill.sh``); use its sibling::

    # C) multi-host: host order == rank order, rank 0 first and it must be THIS host. The third argument
    #    is the NIC and must match the runner's.
    ./models/demos/common/prefill/runners/run_migration_driver.sh \
      models/demos/common/prefill/runners/producer_manifests/<MANIFEST>.yaml \
      <H0>:1,<H1>:1 [tcp_iface]

Run standalone (no launcher) it is simply rank 0 of 1: one process, one host's coverage, no MPI, and the
command at the top of this docstring is exactly right. Under a launcher (``OMPI_COMM_WORLD_SIZE`` > 1) the
same entry point splits by rank, like ``prefill_producer``:
  * rank 0 (the launch host): the full run — H2D feed, ack drain, MigrationLayerClient, migrate(), both
    sidecars — plus its own host's read-backs.
  * every other rank: a device-less VALIDATOR. No H2D connect, no migration client, no migrate; it only
    reads its OWN host's KV back (source PCC and/or destination check) and votes.
Coordination is three collectives over the distributed context (host-side MPI, no mesh device), so all
ranks must see the same env — the launcher script forwards every exported PREFILL_*/MIGRATION_* var and
each rank applies the same manifest itself. GO#1 = rank 0 broadcasts the resident-slot map once every
LayerAck has landed (releases the source PCC); GO#2 = rank 0 broadcasts the resolved (src, dst, real_len)
triples once wait_complete has returned (releases the destination check); DONE = an allgather of each
rank's verdict, which both waits for every validator's reads to finish and folds the pass/fail. Any rank
failing fails the run.
Storage, and this is the usual way a multi-host run goes wrong: PREFILL_MIGRATION_TABLE_PATH must be on
SHARED storage (every validator reads rank 0's table), while PREFILL_MIGRATION_DEVICE_MAP_PATH must stay
HOST-LOCAL (each rank reads its OWN host's chips).

Env. Everything below except the two PREFILL_VERIFY_MIGRATION* vars also has a typed field in the
producer manifest's ``migration:`` block — see ``apply_manifest_env``; an explicitly exported env var
always wins. Invoking this module IS the opt-in, so ``migration.issue`` is redundant here and a ``false``
is warned about rather than honoured:
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
  PREFILL_VERIFY_MIGRATION          destination read-back mode, == --verify-migration
                                    (off|dst-bytes|dst-golden|both; default dst-bytes, loopback only).
                                    No typed manifest field: set it via the CLI flag, an exported env var,
                                    or the manifest's raw ``env:`` passthrough.
  PREFILL_VERIFY_MIGRATION_LAYERS   comma layer list to spot-check instead of the full depth. Same: CLI
                                    flag / env / raw ``env:`` block, no typed field. Honoured by the
                                    dst-bytes half only; the golden half always reads the full depth, and
                                    is refused outright when PREFILL_MIGRATION_LAYERS migrated a subset
                                    (its reader would PCC dst rows the migrate never wrote).
  MIGRATION_DONE_FILE               path of the DONE sentinel published for an external consumer to poll
                                    (default /tmp/migration_done.sentinel).
  The host list is NOT env or manifest config: it is an argument to run_migration_driver.sh, which is what
  places one process per host (see the multi-host note above).
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
        ``_SlotFill``) is read. ``num_slots`` is the KV table's slot count when known, used
        only to bounds-check loopback destinations. ``slot_traces`` / ``pools_by_trace`` are the producer's
        per-slot prompt maps, needed only to write the cross-endpoint handoff.

        Validating the copy is the caller's job, not this method's: ``main()`` feeds the returned triples
        to ``_verify_migrated_slots`` once wait_complete has landed.
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

        ``real_len`` is the SRC slot's resident non-pad token count as recorded at push time (the last
        ``actual_end`` the producer sent), matching the KV the runner wrote; slots with no data are
        skipped. It is read rather than re-derived from chunk counts because a multi-turn slot resumed at
        a non-zero prefix, so ``chunks_pushed * chunk_size`` describes only its latest turn. If
        ``num_slots`` is known (from the KV table), dst is bounds-checked so a too-large dst fails here
        with a clear message instead of a cryptic device-side error at migrate time."""

        def real_len_of(src: int) -> int:
            return stats.resident[src].real_len

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

        This exists for CROSS-ENDPOINT P->D: unlike loopback (where this module verifies the destination
        itself and needs no prompt metadata), the destination galaxy has no way to know each slot's prompt
        length / last token, so it cannot pick the decode start position without this sidecar. ``real_len``
        (element 2 of each triple) is exactly the resident prompt length that was migrated, and the src slot's
        last prompt token is ``pool[real_len - 1]`` of the trace the producer already loaded.

        Gated on ``PREFILL_MIGRATION_HANDOFF_PATH``: unset => write nothing, which is what a loopback run
        wants since its destination is verified here rather than consumed by a decode side. Written
        atomically (tmp + os.replace) so the decode side never reads a partial file."""
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
        """Write the migration DONE sentinel — one ``src dst`` line per migrated pair — for an external
        consumer to poll. This is the SAME handshake the llm-engine scheduler/driver used
        (prefill_scheduler_driver wrote this file after migrating). It reports which pairs were COPIED, not
        that they were verified: the destination read-back happens back in ``main()``, after this. Takes the
        SAME triples list ``_issue`` consumed, so the sentinel matches exactly what was migrated. Returns
        the (src, dst) pairs written."""
        if not triples:
            raise ValueError(
                "migration is enabled but the mapping resolved to zero pairs, so there is no DONE sentinel "
                "to publish (an empty one would report success while nothing moved). "
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

    for slot_id, res in sorted(stats.resident.items()):
        real_len = res.real_len
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


def _verify_dst_vs_src_bytes(table, device_map: dict, triples: list, layers, *, max_report: int = 10) -> bool:
    """Golden-free DESTINATION check: assert every migrated dst slot holds byte-identical KV to its src
    slot, over every config / layer / position the migrate covered. Returns True on full agreement.

    This asks the dst==src question the runner's retired ``validate_migrations_pairwise`` used to ask, but
    over the device-less UMD path instead of the runner's in-memory cache — so it runs in the driver
    process, needs no per-model adapter hook, and leaves the runner a pure serving loop.

    Byte equality rather than PCC: migration is a byte copy, so the correct dst is bit-identical. That
    removes the whole reason the old runner-side version needed a 0.99 threshold and an all-zero
    short-circuit (a decoded all-zero pad tail or a dense layer's index_k has undefined correlation), and
    it catches corruption in regions where correlation is undefined. Nothing is decoded here, so the check
    is also model-agnostic — no per-layout branch, unlike ``prefill_producer._read_slot_kv_and_check_pcc``.

    CROSS-TALK between concurrent pairs (pair A's copy landing in pair B's destination) is detectable here
    ONLY when the source slots hold DIFFERENT data. With the default single-prompt schedule every slot
    pushes the same trace, so all sources are byte-identical and a mis-routed copy is indistinguishable
    from a correct one. Set PREFILL_PRODUCER_SLOT_TRACES="dirA,dirB,..." to give each slot its own prompt
    if that is a property you want this gate to cover. (The same limitation applies to the golden mode —
    it is a property of identical sources, not of the comparison.)

    Scope limits, all logged rather than silently absorbed:
      * ``layers`` (the migrated subset, PREFILL_MIGRATION_LAYERS) restricts which layer rows are read;
        with a subset only config 0 is checked, because a sparse model's index config can be COMPACTED
        (its layer axis is the full-indexer rank, not the global layer id) and a global id would index
        the wrong row.
      * Only whole chunks inside [0, real_len) are compared. A real_len that is not a multiple of the
        config's chunk_n_tokens leaves a trailing partial chunk unchecked, since whether the engine
        copies it whole is its business, not this gate's.
      * A layer whose chips are not in this host's device map is SKIPPED, not failed (multi-host: each
        driver process only sees its co-located galaxy). The skip count is reported so a "PASSED" line
        cannot be mistaken for whole-model coverage.
    """
    from models.demos.common.prefill.runners import prefill_producer as producer

    n_configs = table.num_configs()
    if layers and n_configs > 1:
        logger.warning(
            f"[migration_driver] verify: layer subset {sorted(set(layers))} given, so only config 0 is "
            f"checked ({n_configs} configs in the table). A compacted index config indexes rows by "
            "full-indexer rank, not global layer id, so a global id would read the wrong row."
        )
        n_configs = 1

    failures, checked, skipped, tail_tokens = [], 0, 0, 0
    for src, dst, real_len in triples:
        for cfg_id in range(n_configs):
            tcfg = table.config() if cfg_id == 0 else table.config(cfg_id)
            stride, cfg_layers = int(tcfg.chunk_n_tokens), int(tcfg.num_layers)
            n_full = (real_len // stride) * stride  # whole chunks only; see the docstring
            tail_tokens += real_len - n_full
            wanted = [l for l in sorted(set(layers)) if l < cfg_layers] if layers else list(range(cfg_layers))
            logger.info(
                f"[migration_driver] verify bytes: slot {src} -> {dst} config {cfg_id}: "
                f"{len(wanted)} layer(s) x {n_full // stride} chunk(s) of {stride} token(s) "
                f"= {2 * len(wanted) * (n_full // stride)} UMD read(s)"
            )
            for layer in wanted:
                mismatches_in_layer = 0
                for pos in range(0, n_full, stride):
                    src_loc = table.lookup(layer, pos, src, cfg_id)
                    dst_loc = table.lookup(layer, pos, dst, cfg_id)
                    try:
                        src_uid = producer._resolve_unique_id(
                            table.get_device_group(src_loc.device_group_index).fabric_node_ids, device_map
                        )
                        dst_uid = producer._resolve_unique_id(
                            table.get_device_group(dst_loc.device_group_index).fabric_node_ids, device_map
                        )
                    except KeyError:
                        # Not this host's chips — the same skip prefill_producer's own read-back makes.
                        skipped += 1
                        continue
                    read_umd = ttnn.experimental.disaggregation.read_dram_umd
                    src_raw = read_umd(src_uid, src_loc.noc_addr, src_loc.size_bytes)
                    dst_raw = read_umd(dst_uid, dst_loc.noc_addr, dst_loc.size_bytes)
                    checked += 1
                    if bytes(src_raw) != bytes(dst_raw):
                        mismatches_in_layer += 1
                        if len(failures) < max_report:
                            failures.append((src, dst, cfg_id, layer, pos))
                        elif len(failures) == max_report:
                            failures.append(None)  # sentinel: "and more"
                if mismatches_in_layer:
                    logger.error(
                        f"[migration_driver] verify bytes: slot {src} -> {dst} config {cfg_id} layer {layer}: "
                        f"{mismatches_in_layer} chunk(s) differ"
                    )

    if tail_tokens:
        logger.warning(
            f"[migration_driver] verify bytes: {tail_tokens} trailing token(s) across all pairs fell in a "
            "partial chunk and were NOT compared (real_len is not chunk-aligned)."
        )
    if skipped:
        logger.warning(
            f"[migration_driver] verify bytes: {skipped} chunk(s) skipped — their chips are not in this "
            "host's device map. This run verified only the layers resident on THIS host."
        )
    if failures:
        shown = [f for f in failures if f is not None]
        more = " (and more)" if any(f is None for f in failures) else ""
        detail = "; ".join(f"slot{s}->slot{d} cfg{c} layer{l} pos{p}" for s, d, c, l, p in shown)
        logger.error(f"[migration_driver] verify bytes FAILED: dst != src at {detail}{more}")
        return False
    if not checked:
        logger.error(
            "[migration_driver] verify bytes: nothing was compared (no chunk resolved to a local chip). "
            "Treating as a FAILURE — a check that read nothing must not report success."
        )
        return False
    logger.success(
        f"[migration_driver] verify bytes PASSED: {len(triples)} pair(s), {checked} chunk(s) byte-identical "
        "dst == src"
    )
    return True


def _verify_dst_vs_golden(table, device_map: dict, triples: list, slot_traces: dict, threshold: float) -> bool:
    """Golden-anchored DESTINATION check: PCC each migrated dst slot against the SRC slot's golden trace.
    Returns True when every pair meets ``threshold``.

    This is the device-less counterpart of the "AFTER" half of the runner's retired
    ``validate_migration_kv``. It reuses ``prefill_producer._read_slot_kv_and_check_pcc``
    unchanged — that reader is already parameterised by
    slot id, so passing ``dst`` reads the destination, and passing ``slot_traces[src]`` compares it to
    what the source was supposed to contain. A correct migration makes dst PCC to golden exactly as src
    does, which is the pair of numbers the old ``[kv-migrate-validate] BEFORE/AFTER`` lines reported
    (``_verify_resident_slots`` in main() is still the BEFORE half).

    Stronger than the byte compare in one way — it proves the copy carries MODEL-CORRECT data rather than
    merely the same bytes the source held, so it also re-confirms prefill itself at the destination — and
    weaker in others: it decodes through the per-model layout branch (so a new cache layout needs code),
    it needs the golden trace on disk, and its PCC is undefined over the all-zero pad tail. Run ``both``
    when you want the transport and the model correctness reported separately.

    FULL DEPTH, ALWAYS. ``_read_slot_kv_and_check_pcc`` takes a slot and a trace, not a layer list: it
    walks every layer of the model and reports each of its caches' min. There is no layer subset to pass, which is why
    the caller (``_verify_migrated_slots``) decides whether this check may run at all rather than passing
    one down — see the gate there. Unlike ``_verify_dst_vs_src_bytes``, this cannot honour
    PREFILL_VERIFY_MIGRATION_LAYERS, and it must not run against a partially migrated destination.
    """
    from models.demos.common.prefill.runners import prefill_producer as producer

    min_pcc, failures, checked = 1.0, [], 0
    for src, dst, real_len in triples:
        trace_dir = slot_traces.get(src)
        if trace_dir is None:
            logger.error(f"[migration_driver] verify golden: src slot {src} has no trace; skipping {src}->{dst}")
            continue
        logger.info(f"[migration_driver] verify golden: dst slot {dst} (migrated from {src}) over [0,{real_len})")
        try:
            slot_mins = producer._read_slot_kv_and_check_pcc(table, device_map, dst, real_len, trace_dir)
        except Exception as e:
            logger.error(f"[migration_driver] verify golden: dst slot {dst} read/PCC raised {type(e).__name__}: {e}")
            failures.append((src, dst, float("nan")))
            continue
        checked += 1
        # The reader reports one min per MODEL cache (a sparse model migrates its index cache too); gate on
        # the weakest and print the breakdown so a regression names the cache that moved wrong.
        pcc = min(slot_mins.values())
        min_pcc = min(min_pcc, pcc)
        per_cache = "".join(f" {cache}_pcc={value:.6f}" for cache, value in slot_mins.items())
        print(f"[migration_driver] AFTER dst_slot={dst} (src={src}) min_pcc={pcc:.6f}{per_cache}")
        if pcc < threshold:
            failures.append((src, dst, pcc))

    if failures:
        detail = "; ".join(f"slot{s}->slot{d} pcc={p:.6f}" for s, d, p in failures)
        logger.error(f"[migration_driver] verify golden FAILED (threshold {threshold}): {detail}")
        return False
    if not checked:
        logger.error("[migration_driver] verify golden: no pair was checked; treating as a FAILURE.")
        return False
    logger.success(
        f"[migration_driver] verify golden PASSED: {checked} migrated dst slot(s) >= {threshold} "
        f"(min {min_pcc:.6f})"
    )
    return True


def _verify_migrated_slots(
    mode: str, *, table, triples, slot_traces, layers, migrated_layers, threshold, cross_endpoint
) -> bool:
    """Run the requested destination check(s) after the migrate. Returns True when everything requested
    passed (and True for ``off``, which asserts nothing).

    Skips entirely when cross-endpoint: the dst slot lives in the DECODE galaxy's table, an independent
    address space this driver's table cannot address. Looking up ``dst`` in OUR table would silently read
    our own slot ``dst`` — an unwritten slot, or worse, another pair's source — and report a confident
    wrong answer. Loopback is the only topology where the destination is locally readable.

    TWO layer lists arrive here, and conflating them is a bug:
      * ``migrated_layers`` (PREFILL_MIGRATION_LAYERS) — which rows the migrate actually COPIED. A subset
        means every other row of each dst slot was never written, so it constrains what is even
        MEANINGFUL to read at the destination.
      * ``layers`` (``--verify-migration-layers``, defaulting to ``migrated_layers``) — which rows this
        run should bother reading. A pure cost knob: a subset makes a PASS a sample.
    ``dst-bytes`` takes ``layers`` and honours both. ``dst-golden`` can honour neither — its reader walks
    the full depth — so the gate below runs it only when the whole model was migrated, and says so
    otherwise instead of PCCing rows that hold nothing.
    """
    if mode == "off":
        logger.info("[migration_driver] destination verification is OFF; this run proves TRANSPORT only.")
        return True
    if cross_endpoint:
        logger.warning(
            f"[migration_driver] --verify-migration={mode} ignored: this is a CROSS-ENDPOINT migration, so "
            "the destination lives in the remote decode table and cannot be read from here. Verify it on "
            "the decode side (e.g. against a --dump-src-kv reference)."
        )
        return True
    if table is None:
        logger.error(f"[migration_driver] --verify-migration={mode} needs the KV chunk table, which never appeared.")
        return False
    if not triples:
        logger.error(f"[migration_driver] --verify-migration={mode} but zero pairs were migrated.")
        return False

    from models.demos.common.prefill.runners import prefill_producer as producer

    device_map = producer._read_device_map(int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60")))
    if not device_map:
        logger.error(f"[migration_driver] --verify-migration={mode} needs the device map, which is unavailable.")
        return False

    ok = True
    if mode in ("dst-bytes", "both"):
        ok = _verify_dst_vs_src_bytes(table, device_map, triples, layers) and ok
    if mode in ("dst-golden", "both"):
        if migrated_layers:
            # PARTIAL MIGRATION + golden check: refuse, don't guess. The golden reader walks every layer
            # of the model, but only `migrated_layers` were copied, so it would PCC the dst slot's
            # UNWRITTEN rows against golden and fail a perfectly correct migration. Skipping quietly would
            # be worse than failing: `both` would then report a PASS that never looked at the destination
            # the way the caller asked. dst-bytes is the mode that does honour a subset.
            logger.error(
                f"[migration_driver] --verify-migration={mode} cannot run its GOLDEN half: "
                f"PREFILL_MIGRATION_LAYERS migrated only layer(s) {sorted(set(migrated_layers))}, so every "
                "other row of each dst slot was never written — and the golden reader "
                "(prefill_producer._read_slot_kv_and_check_pcc) has no layer-subset parameter, so it would "
                "PCC that unwritten memory and fail a correct migration. Use --verify-migration=dst-bytes, "
                "which does honour the subset, or migrate the full depth (unset PREFILL_MIGRATION_LAYERS) "
                "to get the golden check."
            )
            ok = False
        else:
            if layers:
                # Everything was migrated, so the full-depth golden read is CORRECT here — just not the
                # cheap sample that was asked for. Say so rather than silently spending the reads.
                logger.warning(
                    f"[migration_driver] --verify-migration-layers {sorted(set(layers))} applies to the "
                    "dst-bytes half only; the golden half reads the FULL depth (its reader takes no layer "
                    "subset). Expect it to cost a full check_pcc pass per pair."
                )
            ok = _verify_dst_vs_golden(table, device_map, triples, slot_traces, threshold) and ok
    return ok


# ---------------------------------------------------------------------------
# Multi-rank coordination (device-less; GO/DONE over MPI collectives, not files)
#
# One driver process per pipeline host under an MPI launcher. Rank 0 is the master (feeds H2D, owns the
# LayerAck channel and the MigrationLayerClient, issues every migrate() and writes both sidecars); every
# other rank is a device-less validator that reads only its OWN host's chips. The reason the split exists
# at all is that read_dram_umd is host-local: rank 0 physically cannot read another galaxy's DRAM, so a
# single-process driver can only ever verify its own layer slice.
#
# The barriers reuse prefill_producer's collectives verbatim (_mr_config / _mr_bcast_resident /
# _mr_allgather_verdict) and add ONE of their own, _mr_bcast_triples, because the destination check needs
# something the producer never had: the mapping rank 0 actually migrated. Ordering is fixed and identical
# on every rank — resident bcast (GO#1), triples bcast (GO#2), verdict allgather (DONE) — since these are
# built out of allgather_int, and a rank issuing a different NUMBER of allgathers desynchronizes the run.
# ---------------------------------------------------------------------------


def _mr_bcast_triples(rank: int, triples: list) -> list:
    """Broadcast rank 0's resolved ``(src_slot, dst_slot, real_len)`` triples to every rank, built from
    allgather_int the same way ``prefill_producer._mr_bcast_resident`` is (element [0] of each allgather is
    rank 0's contribution; ttnn exposes no native broadcast). Non-master ranks pass ``[]`` and receive the
    list.

    Doubles as GO#2: a validator blocks in the first allgather until the master arrives, which happens only
    after every migrate()'s wait_complete has returned — i.e. exactly when the destination slots hold data.
    Broadcasting the triples rather than re-resolving them per rank means the validators check what was
    MIGRATED, not what a second evaluation of the env/manifest thinks should have been."""
    items = list(triples) if rank == 0 else []
    n = ttnn.distributed_context_allgather_int(len(items) if rank == 0 else 0)[0]
    out = []
    for k in range(n):
        src, dst, real_len = items[k] if rank == 0 else (0, 0, 0)
        src = ttnn.distributed_context_allgather_int(int(src))[0]
        dst = ttnn.distributed_context_allgather_int(int(dst))[0]
        real_len = ttnn.distributed_context_allgather_int(int(real_len))[0]
        out.append((src, dst, real_len))
    return out


def _run_validator(rank: int, world_size: int, args) -> None:
    """Non-master path: no H2D feed, no migration client, no migrate() — read-back only.

    Waits for the master's GO#1 (the resident-slot broadcast, which the master only reaches after draining
    every LayerAck, so all layers are written), PCCs this host's local layers of every source slot, then
    waits for GO#2 (the migrated triples) and runs the same destination check the master runs, again over
    this host's layers only. Joins the verdict allgather either way — including on failure — so the master
    can never hang waiting for a rank that gave up early. Exits non-zero when this rank's checks failed.

    Both read-backs filter by the local device map: a layer whose chips are not this host's resolves to no
    unique_id and is skipped. With one process per host the union across ranks covers the whole model,
    which is the entire point of running multi-rank."""
    from models.demos.common.prefill.runners import prefill_producer as producer

    cfg = producer._config_from_env()
    producer._require_shared_table_path(world_size)
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    # Constructed for its env-derived fields only (layers / cross_endpoint) — a validator never attaches a
    # client and never migrates. The same env on every rank makes these identical to the master's.
    driver = MigrationDriver(
        chunk_size=producer.CHUNK_SIZE,
        num_layers=producer.NUM_LAYERS,
        default_dst_slot_offset=cfg.num_users,
    )
    verify_layers = _parse_layers(args.verify_migration_layers) or driver.layers
    logger.info(
        f"[migration_driver] validator rank={rank}/{world_size}: read-back only (no H2D feed, no migrate); "
        f"src_pcc={cfg.verify} dst_verify={args.verify_migration}"
    )

    # GO#1 before the table read: the master broadcasts only after draining every LayerAck, which also
    # means rank 0 finished publishing this run's table, so a read after GO cannot observe a stale one.
    resident = producer._mr_bcast_resident(rank, {})
    logger.info(f"[migration_driver] validator rank={rank}: GO received, {len(resident)} resident slot(s)")

    try:
        kv_table = producer._read_kv_chunk_table(timeout_s)
    except Exception as e:
        logger.error(f"[migration_driver] validator rank={rank}: KV table read raised {type(e).__name__}: {e}")
        kv_table = None

    stats = producer.RunStats(resident=resident, total_pushes=0, push_ms=[], completed=0, wall_s=0.0)
    # Env-derived, so this is the same slot -> golden mapping the master pushed against. Resolved only for
    # the checks that actually need a golden: a dst-bytes-only validator compares device bytes to device
    # bytes, and requiring the trace to be mounted on every host would fail the rank for nothing.
    needs_traces = cfg.verify or args.verify_migration in ("dst-golden", "both")
    slot_traces = producer._resolve_slot_prompts(cfg)[0] if needs_traces else {}

    # Source PCC (was prefill correct on THIS host's layers?), the same gate the master runs on its own.
    verify_ok = True
    if cfg.verify:
        if kv_table is None:
            logger.error(f"[migration_driver] validator rank={rank}: no KV chunk table available; cannot PCC.")
            verify_ok = False
        else:
            try:
                verify_ok = producer._verify_resident_slots(kv_table, stats, cfg.pcc_threshold, slot_traces)
            except Exception as e:
                logger.error(f"[migration_driver] validator rank={rank}: KV read/PCC failed: {type(e).__name__}: {e}")
                verify_ok = False

    # GO#2: the migrated mapping. Blocks until the master's last wait_complete has returned.
    triples = _mr_bcast_triples(rank, [])
    logger.info(f"[migration_driver] validator rank={rank}: {len(triples)} migrated pair(s) received")

    try:
        migration_ok = _verify_migrated_slots(
            args.verify_migration,
            table=kv_table,
            triples=triples,
            slot_traces=slot_traces,
            layers=verify_layers,
            migrated_layers=driver.layers,
            threshold=cfg.pcc_threshold,
            cross_endpoint=driver.cross_endpoint,
        )
    except Exception as e:
        logger.error(f"[migration_driver] validator rank={rank}: destination verify raised {type(e).__name__}: {e}")
        migration_ok = False

    ok = verify_ok and migration_ok
    producer._mr_allgather_verdict(ok)  # DONE barrier + verdict fold (the master reads the result)
    logger.info(f"[migration_driver] validator rank={rank}: DONE src_pcc_ok={verify_ok} dst_verify_ok={migration_ok}")
    if not ok:
        sys.exit(1)


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
    parser.add_argument(
        "--verify-migration",
        choices=("off", "dst-bytes", "dst-golden", "both"),
        default=os.environ.get("PREFILL_VERIFY_MIGRATION", "dst-bytes"),
        help="Read the MIGRATED DESTINATION slots back after the migrate (loopback only). 'dst-bytes' "
        "(default) asserts dst is byte-identical to src — golden-free, model-agnostic, and valid in the "
        "pad/all-zero regions where PCC is undefined. 'dst-golden' PCCs each dst against the src's golden "
        "trace, proving the copy carries model-correct data. 'both' runs each in turn. 'off' reports "
        "transport only. Cost: dst-bytes reads BOTH slots, so it roughly doubles the UMD reads of a "
        "check_pcc pass (a full-depth Kimi pair is ~215k reads); use --verify-migration-layers to "
        "spot-check. Neither mode sees cross-talk unless PREFILL_PRODUCER_SLOT_TRACES gives the source "
        "slots different prompts. Coverage is HOST-LOCAL either way (UMD reaches only this host's chips): "
        "single-process it checks rank 0's layer slice alone, while under an MPI launcher each rank checks "
        "its own host's slice and the verdicts are folded — see the module docstring.",
    )
    parser.add_argument(
        "--verify-migration-layers",
        default=os.environ.get("PREFILL_VERIFY_MIGRATION_LAYERS"),
        help="Comma list of layer ids to verify (e.g. '0,30,60'), for a fast spot-check instead of the "
        "full depth. Defaults to every layer the migrate covered. A subset makes a PASS a sample, not a "
        "proof — the summary says so. Applies to the DST-BYTES half only: the golden half's reader takes "
        "no layer subset and always walks the full depth. For the same reason 'dst-golden'/'both' are "
        "REFUSED (not silently skipped) when PREFILL_MIGRATION_LAYERS migrated only part of the model — "
        "the unmigrated dst rows hold nothing, so PCCing them would fail a correct migration. Use "
        "'dst-bytes' for a partial-depth migration.",
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

    # Rank split. Under an MPI launcher every non-zero rank is a read-back-only validator for its OWN
    # host's layers; standalone this is (0, 1) and nothing below changes. Done AFTER the manifest lands in
    # the env so every rank derives the same config from the same source.
    #
    # The log line matters: _mr_config() calls MPI_Init, which BLOCKS until every rank has joined, and a
    # network misconfiguration (ranks placed on different NICs -- see run_migration_driver.sh's tcp_iface
    # argument) makes that block forever. Announcing it turns an otherwise silent hang into a visible
    # "waiting for N ranks".
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        logger.info(
            f"[migration_driver] joining the distributed context "
            f"({os.environ['OMPI_COMM_WORLD_SIZE']} rank(s) expected); this blocks until all of them arrive"
        )
    mr_rank, world_size = producer._mr_config()
    if mr_rank != 0:
        _run_validator(mr_rank, world_size, args)
        return

    cfg = producer._config_from_env()
    # Validators read the table the runner published under this path from their own hosts, so on
    # multi-rank it has to be shared storage. Checked on every rank (same env => same verdict), so a
    # rejection exits all of them symmetrically instead of half-opening a barrier.
    producer._require_shared_table_path(world_size)
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

    # Unconditional, unlike prefill_producer.main() which skips the table read when check_pcc is off: here
    # the table has a SECOND consumer, --verify-migration's destination read-back, which is on by default
    # and independent of check_pcc. Gating this on cfg.verify would make the default verify silently
    # unavailable ("needs the KV chunk table, which never appeared") on any check_pcc: false manifest.
    kv_table = producer._read_kv_chunk_table(timeout_s)
    ack_channel = producer._connect_layer_ack_channel(timeout_s)

    # Attach + pair BEFORE pushing: a missing endpoint fails in seconds instead of after a multi-minute
    # prefill, and a cross-endpoint pairing rendezvous's while the decode side is still blocked on it.
    driver.attach()

    slot_traces, slot_lengths, pools_by_trace = producer._resolve_slot_prompts(cfg)
    cfg.slot_lengths = slot_lengths

    def push_chunk(slot_id: int, chunk_idx: int, actual_start: int, actual_end: int, is_last: bool) -> float:
        pool = pools_by_trace[slot_traces[slot_id]]
        chunk_bytes = producer._chunk_to_host_array(pool[actual_start : actual_start + producer.CHUNK_SIZE])
        assert (
            chunk_bytes.nbytes == payload_bytes
        ), f"payload {chunk_bytes.nbytes}B != service-expected {payload_bytes}B"
        logger.info(f"[migration_driver] push slot={slot_id} cidx={chunk_idx} start={actual_start} end={actual_end}")
        push_start = time.perf_counter()
        service.forward_to_tensor_bytes(
            chunk_bytes, metadata=producer._pack_metadata(slot_id, actual_start, actual_end, is_last)
        )
        return (time.perf_counter() - push_start) * 1000.0

    stats = producer.run_schedule(cfg, push_fn=push_chunk)
    service.barrier()
    logger.info(
        f"[migration_driver] prefill done wall={stats.wall_s:.1f}s pushes={stats.total_pushes} "
        f"requests={stats.completed}"
    )
    producer._drain_layer_acks(ack_channel, producer.NUM_LAYERS * stats.total_pushes)

    # Multi-rank GO#1: every layer of every chunk is now written across every stage's DRAM, so release the
    # validators to PCC their own hosts' layers. Broadcast BEFORE this rank's own read-back so the reads
    # overlap across hosts. The drain above is unconditional here (unlike prefill_producer, which gates it
    # on check_pcc), so this guarantee holds even with check_pcc off.
    if world_size > 1:
        producer._mr_bcast_resident(mr_rank, stats.resident)

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
            # Rank 0 only: every rank would write the SAME src_slot<N>.pt with a different layer subset and
            # clobber the others. The dump therefore carries this host's layers alone on a multi-rank run.
            if world_size > 1:
                logger.warning(
                    f"[migration_driver] --dump-src-kv runs on rank 0 only, so {args.dump_src_kv} will hold "
                    f"THIS host's layers and None for the rest (world_size={world_size}). Dump from a "
                    "single-rank runner if the decode side needs every layer."
                )
            _dump_src_kv(args.dump_src_kv, kv_table, stats, slot_traces, driver.layers)

    # Migrate. Wrapped so a failure here still reaches GO#2 below with an empty mapping: a validator parked
    # in that broadcast would otherwise hang forever on a master that died. Every rank then verifies zero
    # pairs, votes False, and the run exits non-zero together.
    triples = []
    migrate_ok = True
    try:
        triples = driver.run(
            stats,
            num_slots=kv_table.config().num_slots if kv_table is not None else None,
            slot_traces=slot_traces,
            pools_by_trace=pools_by_trace,
        )
    except Exception as e:
        logger.exception(f"[migration_driver] migration failed: {type(e).__name__}: {e}")
        migrate_ok = False

    # Multi-rank GO#2: wait_complete has returned for every pair, so the destination slots hold data.
    # Release the validators with the mapping that was actually migrated.
    if world_size > 1:
        _mr_bcast_triples(mr_rank, triples)

    # Destination read-back. Runs AFTER driver.run() because the dst slot only holds anything once the
    # migrate's wait_complete has returned, and BEFORE the shutdown sentinel below because the UMD reads
    # need the mesh alive. Note this lands after run() published the DONE sentinel: harmless for loopback
    # (nothing polls it) but it does mean the sentinel means "copied", not "verified" — a cross-endpoint
    # consumer waking on it gets no verdict from here, which is why the cross-endpoint case is skipped.
    verify_layers = _parse_layers(args.verify_migration_layers) or driver.layers
    if verify_layers and args.verify_migration != "off":
        logger.warning(
            f"[migration_driver] verifying layer subset {sorted(set(verify_layers))} only — a PASS is a "
            "SAMPLE of the migration, not a proof that every layer copied correctly."
        )
    try:
        migration_ok = _verify_migrated_slots(
            args.verify_migration,
            table=kv_table,
            triples=triples,
            slot_traces=slot_traces,
            layers=verify_layers,
            migrated_layers=driver.layers,
            threshold=cfg.pcc_threshold,
            cross_endpoint=driver.cross_endpoint,
        )
    except Exception as e:
        # Same reason as the migrate above: reach the verdict allgather rather than stranding a validator.
        logger.exception(f"[migration_driver] destination verify raised {type(e).__name__}: {e}")
        migration_ok = False
    migration_ok = migration_ok and migrate_ok

    # Multi-rank DONE: the verdict allgather is also the barrier that holds this rank until every validator
    # has finished reading, so the shutdown sentinel below cannot tear the mesh/DRAM down under one. Fold
    # every rank's verdict — this rank's own is contributed as element [0] — so a failure anywhere fails
    # the run, and each rank covers a different slice of the model.
    local_ok = migration_ok and not (cfg.verify and not verify_ok)
    all_ranks_ok = local_ok
    if world_size > 1:
        verdicts = producer._mr_allgather_verdict(local_ok)
        for r, v in enumerate(verdicts):
            logger.info(f"[migration_driver] rank={r}: ok={v}")
        all_ranks_ok = all(verdicts)

    # Optional graceful shutdown: sent LAST, because the UMD read-backs above need the mesh/DRAM alive.
    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        sentinel = struct.pack("<iii", -1, -1, -1)
        payload = producer._chunk_to_host_array([1] * producer.CHUNK_SIZE)
        logger.info("[migration_driver] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        service.forward_to_tensor_bytes(payload, metadata=sentinel)
        service.barrier()
    else:
        logger.info("[migration_driver] exiting (the runner keeps its sync-op loop running).")

    # Both gates feed the exit code: the source PCC (was prefill correct?) and the destination read-back
    # (did the copy land correctly?). A silent success on either would make the run green while unverified.
    # Multi-rank, they are folded across ranks first — each rank only ever saw its own host's layers, so
    # only the fold covers the model.
    if cfg.verify and not verify_ok:
        logger.error("[migration_driver] FAILED: source KV PCC did not pass on rank 0.")
    if not migrate_ok:
        logger.error("[migration_driver] FAILED: the migrate step itself did not complete (see above).")
    elif not migration_ok:
        logger.error("[migration_driver] FAILED: migrated destination verification did not pass on rank 0.")
    if local_ok and not all_ranks_ok:
        logger.error("[migration_driver] FAILED: rank 0 passed but another rank's read-back did not.")
    if not all_ranks_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
