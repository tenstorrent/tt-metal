# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""on_kv_cache_ready hook: build the KV chunk address table from the live KV
cache, export it to a protobuf file, and hand it to an already-running
migration endpoint (tt-llm-engine stack) over its shmem control queues.

Runs inside the model process. Only the KV-owning rank (the single
DenseDecoderStage in the decoder-only single-pod config) does any work; all
other ranks return immediately.

Conventions matched to the migration worker (tt-llm-engine
disaggregation/migration — see its main.cpp / control_thread.cpp):
  * noc_addr encoding: (dram_bank_id << 32) | per_bank_offset — identical to
    the worker's addr_channel/addr_local decode (noc_addr.hpp).
  * FabricNodeIds registered as (mesh 0, chip 0..N-1) — the worker
    self-registers its chips as FabricNodeId(MeshId{0}, rank*D + d).
  * Hostnames registered as "host-0" — the worker's SubordinateInfo hostname
    for rank 0 (control_thread.cpp build_local_grouping).
  * The chip order sidecar lists this mesh's PHYSICAL device ids in fabric
    chip-id order: the launcher must pass exactly this as TT_VISIBLE_DEVICES
    to the migration endpoint so worker chip d == table chip d.

The MigrationLayerClient python module comes from the tt-llm-engine build —
set TT_MIGRATION_PYTHON_DIR to
<tt-llm-engine>/disaggregation/migration/build_RelWithDebInfo/python.
"""

from __future__ import annotations

import os
import sys
import time

from loguru import logger

import ttnn


def _pack_bank_offset(bank_id: int, bank_offset: int) -> int:
    """(bank_id << 32) | offset — must match the worker's noc_addr.hpp."""
    return ((bank_id & 0xFFFFFFFF) << 32) | (bank_offset & 0xFFFFFFFF)


def _get_kv_chunk_metadata(kv_cache, position_id: int, slot_id: int, base_addr: int):
    """Chunk address math for one (position, slot) — mirrors the FlashMLA
    block layout the attention op reads (lifted from tt-blaze
    runners/kv_cache_table_helpers.get_kv_cache_metadata, single-mesh form).

    Returns (dram_bank_id, per_bank_offset, chunk_size_bytes, sp_idx).
    """
    from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode

    kv_cache_shape = kv_cache.shape
    per_device_seq_len = kv_cache_shape[2]
    kv_cache_dim = kv_cache_shape[3]

    tokens_per_kv_tile = kv_cache.get_tile().tile_shape[0]
    k_tile_size = kv_cache.get_tile().get_tile_size(kv_cache.dtype)
    kv_transfer_chunk_size = (kv_cache_dim // tokens_per_kv_tile) * k_tile_size

    pc = FlashMLADecode.ProgramConfig(k_chunk_size=128)
    grid = pc.grid

    # Slot stride WITHIN ONE DRAM BANK. A slot's chunks interleave across all
    # banks (OPTIMAL_DRAM_BANK_ORDER), so per-bank stride is total/NUM_BLOCKS —
    # using the whole-slot size here (the reference helper's bug) lands every
    # slot>0 access NUM_BLOCKS strides away.
    num_banks = FlashMLADecode.ProgramConfig(k_chunk_size=128).grid.NUM_BLOCKS
    slot_size_in_bytes = per_device_seq_len // tokens_per_kv_tile * kv_transfer_chunk_size // num_banks

    block_id = position_id // pc.k_chunk_size
    block_in_device = block_id // (pc.sp_dim * grid.NUM_BLOCKS)
    block_size_in_bytes = pc.k_chunk_size * kv_transfer_chunk_size // tokens_per_kv_tile
    offset_in_block = (position_id % pc.k_chunk_size) // tokens_per_kv_tile

    offset = (
        base_addr
        + block_in_device * block_size_in_bytes
        + offset_in_block * kv_transfer_chunk_size
        + slot_size_in_bytes * slot_id
    )

    dram_bank_id = grid.OPTIMAL_DRAM_BANK_ORDER[block_id % grid.NUM_BLOCKS]
    sp_idx = (block_id // grid.NUM_BLOCKS) % pc.sp_dim
    return dram_bank_id, offset, kv_transfer_chunk_size, sp_idx


def build_table_from_kv_cache(
    mesh_device, kv_cache, *, layer_id: int, num_layers: int, max_seq_len: int, num_slots: int
):
    """Build a KvChunkAddressTable covering (num_layers × max_seq_len × num_slots)
    from this rank's live KV tensor. In the decoder-only config num_layers=1 and
    layer_id is this stage's layer (0).

    Device groups use REMAPPED fabric ids: (mesh 0, fabric_chip_id) with
    hostname "host-0" — see module docstring.
    """
    d = ttnn.experimental.disaggregation
    FabricNodeId = ttnn._ttnn.fabric.FabricNodeId
    MeshId = ttnn._ttnn.fabric.MeshId

    base_addr = int(kv_cache.buffer_address())
    tokens_per_tile = kv_cache.get_tile().tile_shape[0]
    num_chunks = max_seq_len // tokens_per_tile
    _, _, chunk_size_bytes, _ = _get_kv_chunk_metadata(kv_cache, 0, 0, base_addr)

    cfg = d.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_slots
    cfg.chunk_n_tokens = tokens_per_tile
    cfg.chunk_size_bytes = chunk_size_bytes
    table = d.KvChunkAddressTable(cfg)

    # Device groups are keyed by the chip index the MIGRATION WORKER will use.
    # UMD parses TT_VISIBLE_DEVICES into an unordered_set (order discarded) and
    # renumbers visible devices in ascending-physical order — which is exactly
    # what mesh_device.get_device_id(coord) returns inside this rank (its
    # LOGICAL id). So register each sp row's TP pair under the logical ids of
    # its mesh coords, NOT the fabric chip id (fabric != logical in general).
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    sp_row_groups = {}
    for r in range(rows):
        fnids = [
            FabricNodeId(MeshId(0), int(mesh_device.get_device_id(ttnn.MeshCoordinate(r, c)))) for c in range(cols)
        ]
        dg = table.add_device_group(fnids)
        for f in fnids:
            table.set_fabric_node_host(f, "host-0")
        sp_row_groups[r] = dg

    for slot_id in range(num_slots):
        for chunk_idx in range(num_chunks):
            pos = chunk_idx * tokens_per_tile
            bank, offset, size, sp_idx = _get_kv_chunk_metadata(kv_cache, pos, slot_id, base_addr)
            loc = d.KvCacheLocation()
            loc.noc_addr = _pack_bank_offset(bank, offset)
            loc.size_bytes = size
            loc.device_group_index = sp_row_groups[sp_idx]
            table.set(layer_id, pos, slot_id, loc)

    logger.info(
        "[migration-hook] table built: layers={} max_seq={} slots={} chunk_n_tokens={} "
        "chunk_size_bytes={} num_chunks={} base_addr=0x{:x}",
        num_layers,
        max_seq_len,
        num_slots,
        tokens_per_tile,
        chunk_size_bytes,
        num_chunks,
        base_addr,
    )
    return table


def _write_chip_sidecar(mesh_device, path: str) -> None:
    """PHYSICAL device ids (ascending) → the TT_VISIBLE_DEVICES for the
    migration endpoint.

    UMD parses TT_VISIBLE_DEVICES into an unordered_set — list order is
    discarded and logical renumbering follows ascending-physical enumeration.
    So the sidecar is simply this rank's visible physical ids, sorted. The
    table's chip indices (logical ids) then line up with the worker's chip
    numbering by construction: worker chip d == d-th smallest physical id ==
    this rank's logical id d.
    """
    visible_env = os.environ.get("TT_VISIBLE_DEVICES")
    if visible_env:
        physical = sorted(int(x) for x in visible_env.split(",") if x != "")
    else:
        # No remap in effect — logical ids ARE physical ids.
        rows, cols = mesh_device.shape[0], mesh_device.shape[1]
        physical = sorted(
            int(mesh_device.get_device_id(ttnn.MeshCoordinate(r, c))) for r in range(rows) for c in range(cols)
        )
    with open(path, "w") as f:
        f.write(",".join(str(p) for p in physical) + "\n")
    logger.info(
        "[migration-hook] chip sidecar written: {} -> {} (rank TT_VISIBLE_DEVICES={})",
        path,
        ",".join(str(p) for p in physical),
        visible_env,
    )


def _attach_migration_client(cmd_q: str, table_q: str, resp_q: str, timeout_s: float = 120.0):
    """Import MigrationLayerClient from the tt-llm-engine build and attach to
    the endpoint's shmem queues, retrying until they exist (the endpoint and
    the model race at startup)."""
    mig_dir = os.environ.get("TT_MIGRATION_PYTHON_DIR")
    if mig_dir and os.path.isdir(mig_dir) and mig_dir not in sys.path:
        sys.path.insert(0, mig_dir)
    from _migration_client import MigrationLayerClient  # noqa: E402

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return MigrationLayerClient(cmd_q, table_q, resp_q)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"[migration-hook] endpoint queues never appeared ({cmd_q}) — " "is migration_endpoint running?"
                )
            time.sleep(0.25)


def validate_kv_migration(
    pipeline,
    mesh_device,
    *,
    done_file: str,
    pairs: list[tuple[int, int]] | None = None,
    positions: int = 32,
    result_file: str | None = None,
) -> bool | None:
    """Pull the KV cache to host and byte-compare src_slot vs dst_slot over
    positions [0, positions). Returns True on MATCH, False on MISMATCH, None
    on non-KV ranks.

    MUST be called AFTER pipeline teardown (persistent slow-dispatch kernels
    stopped) — get_kv_cache_host needs a fast-dispatch context which is only
    safe once nothing is running on the mesh.
    """
    if pipeline.kv_cache() is None:
        return None
    if pairs is None:
        pairs = [(0, 1), (2, 3)]
    out_path = result_file or (done_file + ".result")
    logger.info("[migration-validate] pulling KV cache to host (slow dispatch, pipeline alive)")
    kv = pipeline.get_kv_cache_host()
    if kv is None:
        logger.error("[migration-validate] get_kv_cache_host returned None")
        return False
    import torch

    # Per-slot stats over the compared window.
    for s in range(kv.shape[0]):
        win = kv[s, 0, :positions, :]
        logger.info(
            "[migration-validate] slot{} stats: nonzeros={}/{} min={:.4f} max={:.4f} mean={:.6f}",
            s,
            int((win != 0).sum()),
            win.numel(),
            float(win.min()),
            float(win.max()),
            float(win.mean()),
        )

    # Dump compared windows (capped at 64 positions per slot — debugging aid,
    # the full compare below runs over the whole window).
    dump_positions = min(positions, 64)
    dump = {f"slot{s}": kv[s, 0, :dump_positions, :].clone() for s in range(kv.shape[0])}
    dump["meta"] = {"pairs": pairs, "positions": positions, "kv_shape": tuple(kv.shape)}
    dump_path = done_file + ".kv_dump.pt"
    torch.save(dump, dump_path)
    logger.info("[migration-validate] KV dump written: {} ({} positions/slot)", dump_path, dump_positions)

    all_ok = True
    lines = []
    for src_slot, dst_slot in pairs:
        if dst_slot >= kv.shape[0]:
            continue
        src = kv[src_slot, 0, :positions, :]
        dst = kv[dst_slot, 0, :positions, :]
        mismatched = int((src != dst).sum().item())
        ok = mismatched == 0
        all_ok = all_ok and ok
        msg = (
            f"slot{src_slot} vs slot{dst_slot} positions[0,{positions}): "
            f"{'MATCH' if ok else 'MISMATCH'} ({mismatched}/{src.numel()} elems differ)"
        )
        logger.info("[migration-validate] {}", msg)
        lines.append(msg)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return all_ok


def make_on_kv_cache_ready(
    *,
    cmd_queue: str,
    table_queue: str,
    resp_queue: str,
    table_path: str,
    num_layers: int,
    max_seq_len: int,
    num_slots: int,
):
    """Build the on_kv_cache_ready callback for run_demo.

    Signature matches ModelPipeline's invocation: (mesh_device, kv_cache, layer_id).
    Non-KV ranks (kv_cache=None) return immediately.
    """

    def on_kv_cache_ready(mesh_device, kv_cache, layer_id):
        if kv_cache is None:
            return
        logger.info("[migration-hook] kv_cache ready (layer_id={}) — building table", layer_id)

        # Derive global max_seq_len when not given: the sequence is striped
        # across sp rows, so global = per-device seq len × sp_dim.
        effective_max_seq = max_seq_len
        if effective_max_seq is None:
            from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode

            sp_dim = FlashMLADecode.ProgramConfig(k_chunk_size=128).sp_dim
            effective_max_seq = int(kv_cache.shape[2]) * sp_dim
            logger.info(
                "[migration-hook] derived max_seq_len={} (per_device={} × sp_dim={})",
                effective_max_seq,
                int(kv_cache.shape[2]),
                sp_dim,
            )

        table = build_table_from_kv_cache(
            mesh_device,
            kv_cache,
            layer_id=int(layer_id) if layer_id is not None else 0,
            num_layers=num_layers,
            max_seq_len=effective_max_seq,
            num_slots=num_slots,
        )
        ttnn.experimental.disaggregation.export_kv_chunk_table_to_protobuf_file(table, table_path)
        logger.info("[migration-hook] table exported to {}", table_path)
        _write_chip_sidecar(mesh_device, table_path + ".chips")

        # Bootstrap mode: the migration endpoint needs TT_VISIBLE_DEVICES (from
        # the .chips sidecar) at ITS launch, but the sidecar is produced here —
        # an export-only first run breaks the cycle.
        if os.environ.get("TT_MIGRATION_EXPORT_ONLY") == "1":
            logger.info("[migration-hook] TT_MIGRATION_EXPORT_ONLY=1 — skipping endpoint delivery")
            return

        client = _attach_migration_client(cmd_queue, table_queue, resp_queue)
        client.send_kv_chunk_table(table_path)
        client.wait_ready()
        logger.info("[migration-hook] migration endpoint WORKER_READY — table delivered")
        # Client detaches at scope exit; shmem queues persist for the scheduler.

    return on_kv_cache_ready
