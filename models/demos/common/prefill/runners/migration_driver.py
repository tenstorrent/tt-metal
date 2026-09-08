# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


import json
import os
import struct
import sys
import time

from loguru import logger

import ttnn


def apply_manifest_env(manifest: dict) -> None:
    migration = manifest.get("migration") or {}

    def sd(key, val):
        if val is not None:
            os.environ.setdefault(key, str(val))

    def sd_bool(key, val):
        if val is not None:
            os.environ.setdefault(key, "1" if val else "0")

    sd_bool("PREFILL_PRODUCER_ISSUE_MIGRATION", migration.get("issue"))
    sd("PREFILL_MIGRATION_DEST_ENDPOINT_ID", migration.get("dest_endpoint_id"))
    sd("PREFILL_MIGRATION_SRC_ENDPOINT_ID", migration.get("src_endpoint_id"))
    sd("PREFILL_MIGRATION_DST_SLOT_OFFSET", migration.get("dst_slot_offset"))
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
                    parts.append(str(p))
            sd("PREFILL_MIGRATION_PAIRS", ",".join(parts))
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
    spec = (spec or "").strip()
    return [int(x) for x in spec.split(",") if x.strip()] if spec else None


class MigrationDriver:
    def __init__(self, *, chunk_size: int, num_layers: int, default_dst_slot_offset: int):
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.dest_endpoint_id = int(os.environ.get("PREFILL_MIGRATION_DEST_ENDPOINT_ID", "1"))
        self.src_endpoint_id = int(os.environ.get("PREFILL_MIGRATION_SRC_ENDPOINT_ID", "1"))
        self.timeout_ms = int(os.environ.get("PREFILL_MIGRATION_TIMEOUT_MS", "3600000"))
        self.dst_slot_offset = int(os.environ.get("PREFILL_MIGRATION_DST_SLOT_OFFSET", str(default_dst_slot_offset)))
        self.layers = _parse_layers(os.environ.get("PREFILL_MIGRATION_LAYERS", ""))
        self.done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
        self.handoff_path = os.environ.get("PREFILL_MIGRATION_HANDOFF_PATH", "")
        self.client = None

    @property
    def cross_endpoint(self) -> bool:
        return self.dest_endpoint_id != self.src_endpoint_id

    def attach(self) -> None:
        self.client = self._attach_client()
        if self.cross_endpoint:
            self._pair_cross_endpoint()

    def _attach_client(self):
        from models.demos.common.prefill.runners.migration import _import_migration_client, _resolve_queue_names

        cmd_q, table_q, resp_q = _resolve_queue_names()
        client = _import_migration_client().MigrationLayerClient(cmd_q, table_q, resp_q)
        logger.info(f"[migration_driver] client attached: cmd={cmd_q} table={table_q} resp={resp_q}")
        return client

    def _pair_cross_endpoint(self) -> None:
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

    def run(self, stats, *, num_slots: int = None, slot_traces: dict = None, pools_by_trace: dict = None) -> list:
        if self.client is None:
            raise RuntimeError("[migration_driver] run() called before attach()")
        triples = self._resolve_pairs(stats, num_slots=num_slots)
        self._issue(triples)
        self._write_handoff(triples, slot_traces, pools_by_trace)
        self._write_done_sentinel(triples)
        return triples

    def _resolve_pairs(self, stats, *, num_slots: int = None) -> list:
        def real_len_of(src: int) -> int:
            return stats.resident[src].real_len

        def check_dst(src: int, dst: int) -> None:
            if dst < 0:
                raise ValueError(f"migration dst slot {dst} (src {src}) is negative")
            if not self.cross_endpoint and num_slots is not None and dst >= num_slots:
                raise ValueError(
                    f"migration dst slot {dst} (src {src}) is out of range: the KV table has {num_slots} "
                    f"slot(s) [0,{num_slots}). Grow PREFILL_NUM_USERS or pick a smaller dst."
                )

        spec = os.environ.get("PREFILL_MIGRATION_PAIRS", "").strip()
        triples = []
        if spec:
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
        else:
            for src in sorted(stats.resident):
                real_len = real_len_of(src)
                if real_len <= 0:
                    continue
                dst = src + self.dst_slot_offset
                check_dst(src, dst)
                triples.append((src, dst, real_len))

        srcs = [s for (s, _, _) in triples]
        dsts = [d for (_, d, _) in triples]
        dup_dsts = sorted({d for d in dsts if dsts.count(d) > 1})
        if dup_dsts:
            raise ValueError(
                f"migration has duplicate dst slot(s) {dup_dsts}: multiple pairs target the same slot, so only "
                f"the last survives while every pair would be validated. Give each migration a distinct dst."
            )
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
                self.client.wait_complete(token, self.timeout_ms)
            logger.success(
                f"[migration_driver] MIGRATE slot {src_slot} -> {dst_slot} complete "
                f"({len(layer_ranges)} layer range(s))"
            )
            migrated += 1
        logger.info(f"[migration_driver] migrations complete: {migrated} pair(s)")
        return migrated

    def _write_handoff(self, triples: list, slot_traces: dict, pools_by_trace: dict) -> None:
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
        os.replace(tmp, handoff_path)
        logger.success(f"[migration_driver] wrote handoff {handoff_path} ({len(slots)} slot(s)): {slots}")

    def _write_done_sentinel(self, triples: list) -> list:
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


def _cache_plan(table, migrated_layers) -> list:
    from models.demos.common.prefill.runners import prefill_producer as producer

    num_layers = int(producer.NUM_LAYERS)
    n_model_configs = producer._num_model_configs(table)
    adapter = producer.ADAPTER
    rows_hook = getattr(adapter, "cache_layer_rows", None)
    head_dim_hook = getattr(adapter, "cache_head_dim", None)
    mc = adapter.model_config

    full_layers = None
    if table.num_configs() > 1:
        try:
            full_layers = producer._full_indexer_layer_indices(num_layers)
        except Exception as e:
            logger.debug(f"[migration_driver] no full-indexer layer map available ({e}); DSA convention off")

    plan = []
    for cfg_id in range(table.num_configs()):
        cfg = table.config() if cfg_id == 0 else table.config(cfg_id)
        is_index = cfg_id == 1 and n_model_configs > 1
        n_rows = int(cfg.num_layers)
        rows, why = None, ""
        if rows_hook is not None:
            mapped = rows_hook(cfg_id, num_layers)
            if mapped:
                rows, why = {int(l): int(r) for l, r in dict(mapped).items()}, "adapter cache_layer_rows()"
        if rows is None and n_rows >= num_layers:
            rows, why = {l: l for l in range(num_layers)}, "all-layers cache (row == global layer)"
        if rows is None and full_layers is not None and len(full_layers) == n_rows:
            rows, why = {lid: r for r, lid in enumerate(full_layers)}, "DSA index cache (row == full-indexer rank)"
        if rows is None:
            why = (
                f"UNKNOWN axis -- {n_rows} row(s) for {num_layers} layers, no cache_layer_rows() hook, and "
                f"a prefix cache looks like a compacted one from here"
            )

        head_dim = head_dim_hook(cfg_id) if head_dim_hook is not None else None
        if head_dim is None and cfg_id == 0 and hasattr(mc, "KV_LORA_RANK") and hasattr(mc, "QK_ROPE_HEAD_DIM"):
            head_dim = mc.KV_LORA_RANK + mc.QK_ROPE_HEAD_DIM
        if head_dim is None and is_index:
            head_dim = getattr(mc, "INDEX_HEAD_DIM", None)

        unaddressed = []
        if rows is not None and migrated_layers:
            unaddressed = sorted(l for l, r in rows.items() if l != r and l in set(migrated_layers))
            rows = {l: r for l, r in rows.items() if l == r} or None
            if unaddressed:
                why = f"{why}; a subset migration by global layer id cannot address layer(s) {unaddressed}"
        plan.append(
            {
                "config_id": cfg_id,
                "rows": rows,
                "head_dim": None if head_dim is None else int(head_dim),
                "kind": "index" if is_index else ("kvpe" if cfg_id == 0 else "other"),
                "unaddressed": unaddressed,
                "why": why,
            }
        )
    return plan


def _log_cache_plan(plan, tag: str) -> None:
    for entry in plan:
        cfg_id, rows = entry["config_id"], entry["rows"]
        if rows is None:
            logger.warning(f"[migration_driver] {tag}: cache config {cfg_id} SKIPPED -- {entry['why']}")
        else:
            log = logger.warning if entry["unaddressed"] else logger.info
            log(
                f"[migration_driver] {tag}: cache config {cfg_id} carries {len(rows)} layer(s) "
                f"({entry['why']}; head_dim={entry['head_dim']})"
            )


def _dump_src_kv(dump_dir: str, table, stats, slot_traces: dict, layers) -> None:
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

    tokens_per_block = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    wanted = set(layers) if layers else set(range(producer.NUM_LAYERS))
    base_dir = os.path.abspath(os.path.expanduser(dump_dir))
    os.makedirs(base_dir, exist_ok=True)

    plan = _cache_plan(table, layers)
    _log_cache_plan(plan, "src-KV dump")
    dumpable = []
    for entry in plan:
        cfg_id, rows = entry["config_id"], entry["rows"]
        if rows is None:
            continue
        if entry["head_dim"] is None:
            logger.warning(
                f"[migration_driver] src-KV dump: cache config {cfg_id} not dumped -- axis known, but no "
                f"decode width (no cache_head_dim() hook)."
            )
            continue
        selected = {l: r for l, r in rows.items() if l in wanted}
        if not selected:
            logger.warning(
                f"[migration_driver] src-KV dump: cache config {cfg_id} carries none of layer(s) "
                f"{sorted(wanted)}; not dumped."
            )
            continue
        dumpable.append((cfg_id, selected, entry["head_dim"]))
    if not any(cfg_id == 0 for cfg_id, _, _ in dumpable):
        logger.error(
            "[migration_driver] src-KV dump: cache config 0 is not readable (see the plan above), so there "
            "is no reference to write; skipping the dump."
        )
        return

    for slot_id, res in sorted(stats.resident.items()):
        real_len = res.real_len
        if real_len <= 0:
            continue
        read_len = ((real_len + tokens_per_block - 1) // tokens_per_block) * tokens_per_block

        refs = {}
        for cfg_id, selected, head_dim in dumpable:
            per_layer = [None] * producer.NUM_LAYERS
            for layer, row in sorted(selected.items()):
                if table.lookup(row, 0, slot_id, cfg_id).size_bytes == 0:
                    continue
                decoded_rows = []
                for pos in range(0, read_len, tokens_per_block):
                    loc = table.lookup(row, pos, slot_id, cfg_id)
                    unique_id = producer._resolve_unique_id(
                        table.get_device_group(loc.device_group_index).fabric_node_ids, device_map
                    )
                    raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
                    decoded_rows.append(producer._decode_kv_chunk(raw, head_dim))
                device_kv = torch.cat(decoded_rows, dim=0)[:real_len]
                per_layer[layer] = device_kv.unsqueeze(0).unsqueeze(0)
            refs[cfg_id] = per_layer

        out = os.path.abspath(os.path.join(base_dir, f"src_slot{int(slot_id)}.pt"))
        if not out.startswith(base_dir + os.sep):
            raise ValueError(f"src-KV dump path {out!r} escapes its base directory {base_dir!r}")
        blob = {"ref_cache_lists": refs, "ref_kvpe_list": refs[0]}
        if 1 in refs:
            blob["ref_index_k_list"] = refs[1]
        torch.save(blob, out)
        counts = ", ".join(f"cfg{c}={sum(t is not None for t in lst)}" for c, lst in sorted(refs.items()))
        logger.success(
            f"[migration_driver] slot {slot_id} src KV dumped -> {out} "
            f"(layers {sorted(wanted)}, positions [0,{real_len}), layer(s) per cache: {counts})"
        )


def _verify_dst_vs_src_bytes(
    table, device_map: dict, triples: list, layers, *, migrated_layers=None, max_report: int = 10
) -> bool:
    from models.demos.common.prefill.runners import prefill_producer as producer

    plan = _cache_plan(table, migrated_layers)
    _log_cache_plan(plan, "verify bytes")
    checkable = []
    for entry in plan:
        rows = entry["rows"]
        if rows is None:
            continue
        picked = sorted((l, r) for l, r in rows.items() if not layers or l in set(layers))
        if picked:
            checkable.append((entry["config_id"], picked))
    if not checkable:
        logger.error(
            "[migration_driver] verify bytes: no cache has an addressable axis (see the plan above), so "
            "nothing would be compared. Treating as a FAILURE."
        )
        return False

    failures, checked, skipped, tail_tokens = [], 0, 0, 0
    for src, dst, real_len in triples:
        for cfg_id, picked in checkable:
            tcfg = table.config() if cfg_id == 0 else table.config(cfg_id)
            stride = int(tcfg.chunk_n_tokens)
            n_full = (real_len // stride) * stride
            tail_tokens += real_len - n_full
            logger.info(
                f"[migration_driver] verify bytes: slot {src} -> {dst} config {cfg_id}: "
                f"{len(picked)} layer(s) x {n_full // stride} chunk(s) of {stride} token(s) "
                f"= {2 * len(picked) * (n_full // stride)} UMD read(s)"
            )
            for layer, row in picked:
                mismatches_in_layer = 0
                for pos in range(0, n_full, stride):
                    src_loc = table.lookup(row, pos, src, cfg_id)
                    dst_loc = table.lookup(row, pos, dst, cfg_id)
                    try:
                        src_uid = producer._resolve_unique_id(
                            table.get_device_group(src_loc.device_group_index).fabric_node_ids, device_map
                        )
                        dst_uid = producer._resolve_unique_id(
                            table.get_device_group(dst_loc.device_group_index).fabric_node_ids, device_map
                        )
                    except KeyError:
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
                            failures.append(None)
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
        ok = _verify_dst_vs_src_bytes(table, device_map, triples, layers, migrated_layers=migrated_layers) and ok
    if mode in ("dst-golden", "both"):
        if migrated_layers:
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
                logger.warning(
                    f"[migration_driver] --verify-migration-layers {sorted(set(layers))} applies to the "
                    "dst-bytes half only; the golden half reads the FULL depth (its reader takes no layer "
                    "subset). Expect it to cost a full check_pcc pass per pair."
                )
            ok = _verify_dst_vs_golden(table, device_map, triples, slot_traces, threshold) and ok
    return ok


def _mr_bcast_triples(rank: int, triples: list) -> list:
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
    from models.demos.common.prefill.runners import prefill_producer as producer

    cfg = producer._config_from_env()
    producer._require_shared_table_path(world_size)
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
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

    resident = producer._mr_bcast_resident(rank, {})
    logger.info(f"[migration_driver] validator rank={rank}: GO received, {len(resident)} resident slot(s)")

    try:
        kv_table = producer._read_kv_chunk_table(timeout_s)
    except Exception as e:
        logger.error(f"[migration_driver] validator rank={rank}: KV table read raised {type(e).__name__}: {e}")
        kv_table = None

    stats = producer.RunStats(resident=resident, total_pushes=0, push_ms=[], completed=0, wall_s=0.0)
    needs_traces = cfg.verify or args.verify_migration in ("dst-golden", "both")
    slot_traces = producer._resolve_slot_prompts(cfg)[0] if needs_traces else {}

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
    producer._mr_allgather_verdict(ok)
    logger.info(f"[migration_driver] validator rank={rank}: DONE src_pcc_ok={verify_ok} dst_verify_ok={migration_ok}")
    if not ok:
        sys.exit(1)


def main() -> None:
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

    manifest = producer._apply_manifest_env(args.manifest) if args.manifest else {}
    apply_manifest_env(manifest)
    if args.migrations is not None:
        os.environ["PREFILL_MIGRATION_PAIRS"] = args.migrations
    producer._load_env_config()

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
    producer._require_shared_table_path(world_size)
    if os.environ.get("PREFILL_PRODUCER_ISSUE_MIGRATION", "1") == "0":
        logger.warning(
            "[migration_driver] the manifest sets migration.issue: false, which is ignored when this module "
            "is the entry point — invoking it IS the opt-in. Run prefill_producer for a no-migration run."
        )
    driver = MigrationDriver(
        chunk_size=producer.CHUNK_SIZE,
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

    if world_size > 1:
        producer._mr_bcast_resident(mr_rank, stats.resident)

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
            if world_size > 1:
                logger.warning(
                    f"[migration_driver] --dump-src-kv runs on rank 0 only, so {args.dump_src_kv} will hold "
                    f"THIS host's layers and None for the rest (world_size={world_size}). Dump from a "
                    "single-rank runner if the decode side needs every layer."
                )
            _dump_src_kv(args.dump_src_kv, kv_table, stats, slot_traces, driver.layers)

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

    if world_size > 1:
        _mr_bcast_triples(mr_rank, triples)

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
        logger.exception(f"[migration_driver] destination verify raised {type(e).__name__}: {e}")
        migration_ok = False
    migration_ok = migration_ok and migrate_ok

    local_ok = migration_ok and not (cfg.verify and not verify_ok)
    all_ranks_ok = local_ok
    if world_size > 1:
        verdicts = producer._mr_allgather_verdict(local_ok)
        for r, v in enumerate(verdicts):
            logger.info(f"[migration_driver] rank={r}: ok={v}")
        all_ranks_ok = all(verdicts)

    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        sentinel = struct.pack("<iii", -1, -1, -1)
        payload = producer._chunk_to_host_array([1] * producer.CHUNK_SIZE)
        logger.info("[migration_driver] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        service.forward_to_tensor_bytes(payload, metadata=sentinel)
        service.barrier()
    else:
        logger.info("[migration_driver] exiting (the runner keeps its sync-op loop running).")

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
