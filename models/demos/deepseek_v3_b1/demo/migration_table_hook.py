# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""on_kv_cache_ready hook: build the KV chunk address table from the live KV
cache, export it to a protobuf file, and hand it (plus the device map) to an
already-running migration endpoint (tt-llm-engine stack) over its shmem
control queues.

Runs inside every model process — the hook is an MPI-collective rendezvous:
each rank contributes its own (devices, layer, kv address, KV geometry) via
allgathers, then RANK 0 (the head node / first pipeline stage) materializes the
full table for all decoder stages and is the SOLE publisher to the migration
endpoint. Rank 0 owns no KV cache (it is the embedding stage), so it authors
the table from the geometry gathered from the first KV-owning rank — letting
the endpoint live on the head node, co-located with stage 0's H2D/D2H sockets
and the scheduler. Mirrors tt-blaze runners/decode_runner.py::_setup_decode_migration.

Conventions matched to the migration worker (tt-llm-engine
disaggregation/migration — see its main.cpp / control_thread.cpp):
  * noc_addr encoding: (dram_bank_id << 32) | per_bank_offset — identical to
    the worker's addr_channel/addr_local decode (noc_addr.hpp).
  * Device-map backend: the worker opens UMD chips from the
    send_device_map([(fabric_mesh_id, fabric_chip_id, umd_chip_id), ...])
    entries and registers each DeviceDram under its REAL FabricNodeId. The
    table's device groups therefore carry real fabric node ids — no remap.
  * Fabric-node host tags: each fnid is tagged "host-<crc32(gethostname)hex>",
    the join key the worker's build_local_grouping (control_thread.cpp) uses to
    partition the table per rank. Multi-host: the worker does NO master->
    subordinate device-map forwarding, so each host's representative model rank
    delivers its host's device map directly to that host's internal A/B worker
    queues (/ep_<pid>_{a,b}_*); the head publishes the single global table.

The MigrationLayerClient python module comes from the tt-llm-engine build —
set TT_MIGRATION_PYTHON_DIR to
<tt-llm-engine>/disaggregation/migration/build_RelWithDebInfo/python.
"""

from __future__ import annotations

import os
import sys
import time
from collections import namedtuple

from loguru import logger

import ttnn

# Upper bound for the fixed-size device-list allgather (8 chips/mesh today,
# 2x headroom). ttnn only exposes single-int allgather, so variable-length
# lists are padded to this bound and gathered field by field.
_MAX_DEVICES_PER_MESH = 16

# layer allgather sentinel for ranks that own no KV (embedding, passthrough).
_SENTINEL_LAYER = -1


# KV-cache geometry needed to author the chunk table. Identical across every
# decoder layer (same FlashMLA layout), so it is gathered ONCE from the first
# KV-owning rank and used by the publisher — which may itself own no KV (e.g.
# stage 0 / embedding on the head node). All five fields are small positive
# ints (well under 2^31) so they ride the signed int allgather without masking.
KvGeometry = namedtuple(
    "KvGeometry",
    ["per_device_seq_len", "kv_cache_dim", "tokens_per_kv_tile", "k_tile_size", "mesh_cols"],
)

# Sentinel geometry contributed by a rank that owns no KV cache.
_GEOM_NONE = KvGeometry(0, 0, 0, 0, 0)


def _kv_geometry_from_cache(kv_cache, mesh_device) -> KvGeometry:
    """Read the table geometry from a live KV cache (KV-owning ranks only)."""
    return KvGeometry(
        per_device_seq_len=int(kv_cache.shape[2]),
        kv_cache_dim=int(kv_cache.shape[3]),
        tokens_per_kv_tile=int(kv_cache.get_tile().tile_shape[0]),
        k_tile_size=int(kv_cache.get_tile().get_tile_size(kv_cache.dtype)),
        mesh_cols=int(mesh_device.shape[1]),
    )


def _pack_bank_offset(bank_id: int, bank_offset: int) -> int:
    """(bank_id << 32) | offset — must match the worker's noc_addr.hpp."""
    return ((bank_id & 0xFFFFFFFF) << 32) | (bank_offset & 0xFFFFFFFF)


def _get_kv_chunk_metadata(geom: KvGeometry, position_id: int, slot_id: int, base_addr: int):
    """Chunk address math for one (position, slot) — mirrors the FlashMLA
    block layout the attention op reads (lifted from tt-blaze
    runners/kv_cache_table_helpers.get_kv_cache_metadata, single-mesh form).

    `geom` carries the KV-cache geometry (gathered from the first KV rank), so
    this is computable on a rank that owns no KV cache. The arithmetic is
    byte-for-byte the same as the original kv_cache-reading version.

    Returns (dram_bank_id, per_bank_offset, chunk_size_bytes, sp_idx).
    """
    from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode

    per_device_seq_len = geom.per_device_seq_len
    kv_cache_dim = geom.kv_cache_dim

    tokens_per_kv_tile = geom.tokens_per_kv_tile
    k_tile_size = geom.k_tile_size
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


def _enumerate_devices(mesh_device) -> list[tuple[int, int, int]]:
    """Row-major (umd_physical_chip_id, fabric_mesh_id, fabric_chip_id) for
    this rank's mesh.

    UMD chip ids must be PHYSICAL: mesh_device.get_device_id returns the
    LOGICAL id, which equals the physical id only when no TT_VISIBLE_DEVICES
    remap is in effect. Under a remap, logical d is the d-th smallest visible
    physical id (UMD parses the env into an unordered_set and renumbers in
    ascending-physical order).
    """
    visible_env = os.environ.get("TT_VISIBLE_DEVICES")
    phys_by_logical = None
    if visible_env:
        phys_by_logical = sorted(int(x) for x in visible_env.split(",") if x != "")

    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    out = []
    for r in range(rows):
        for c in range(cols):
            coord = ttnn.MeshCoordinate(r, c)
            logical = int(mesh_device.get_device_id(coord))
            phys = phys_by_logical[logical] if phys_by_logical is not None else logical
            fnid = mesh_device.get_fabric_node_id(coord)
            out.append((phys, int(fnid.mesh_id), int(fnid.chip_id)))
    return out


def _gather_kv_topology(devices: list[tuple[int, int, int]], layer_id, kv_addr: int, geom_local: KvGeometry):
    """MPI-collective: every rank (KV-owning or not) MUST call this.

    Returns (all_rank_devices, all_layers, all_addrs, all_host_tags, all_geom)
    where index = world rank within the current distributed context:
      all_rank_devices[r] — rank r's row-major (umd, fabric_mesh, fabric_chip)
      all_layers[r]       — rank r's layer id, or _SENTINEL_LAYER (no KV)
      all_addrs[r]        — rank r's kv_cache.buffer_address(), or 0 (no KV)
      all_host_tags[r]    — crc32 of rank r's hostname (matches the worker's
                            crc32(MPI_Get_processor_name) slicing key)
      all_geom[r]         — rank r's KvGeometry, or _GEOM_NONE (no KV). Lets a
                            non-KV publisher (e.g. stage 0) author the table
                            from the first KV rank's geometry.

    Every rank must pass through the SAME allgather calls in the same order —
    do not make any of these conditional on rank.
    """
    import socket
    import zlib

    ag = ttnn.distributed_context_allgather_int
    size = int(ttnn.distributed_context_get_size())

    if len(devices) > _MAX_DEVICES_PER_MESH:
        raise RuntimeError(
            f"rank owns {len(devices)} devices, exceeds _MAX_DEVICES_PER_MESH="
            f"{_MAX_DEVICES_PER_MESH}; bump the constant."
        )
    padded = list(devices) + [(0, 0, 0)] * (_MAX_DEVICES_PER_MESH - len(devices))

    counts = list(ag(len(devices)))
    slot_umd = [list(ag(int(padded[s][0]))) for s in range(_MAX_DEVICES_PER_MESH)]
    slot_mesh = [list(ag(int(padded[s][1]))) for s in range(_MAX_DEVICES_PER_MESH)]
    slot_fchip = [list(ag(int(padded[s][2]))) for s in range(_MAX_DEVICES_PER_MESH)]
    all_rank_devices = [
        [(slot_umd[s][r], slot_mesh[s][r], slot_fchip[s][r]) for s in range(counts[r])] for r in range(size)
    ]

    all_layers = list(ag(int(layer_id) if layer_id is not None else _SENTINEL_LAYER))
    addr_lo = list(ag(kv_addr & 0xFFFFFFFF))
    addr_hi = list(ag((kv_addr >> 32) & 0xFFFFFFFF))
    all_addrs = [((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF) for lo, hi in zip(addr_lo, addr_hi)]

    # 31-bit mask: allgather_int is signed; the worker masks identically.
    host_tag = zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF
    all_host_tags = [t & 0x7FFFFFFF for t in ag(host_tag)]

    # KV geometry — one allgather per field, same fixed order on every rank.
    # Values are small positive ints; no masking needed.
    geom_seq = list(ag(int(geom_local.per_device_seq_len)))
    geom_dim = list(ag(int(geom_local.kv_cache_dim)))
    geom_tpt = list(ag(int(geom_local.tokens_per_kv_tile)))
    geom_tile = list(ag(int(geom_local.k_tile_size)))
    geom_cols = list(ag(int(geom_local.mesh_cols)))
    all_geom = [KvGeometry(geom_seq[r], geom_dim[r], geom_tpt[r], geom_tile[r], geom_cols[r]) for r in range(size)]

    return all_rank_devices, all_layers, all_addrs, all_host_tags, all_geom


def _build_table_gathered(
    geom: KvGeometry,
    *,
    layer_to_rank: dict[int, int],
    all_rank_devices: list[list[tuple[int, int, int]]],
    all_addrs: list[int],
    all_host_tags: list[int],
    max_seq_len: int,
    num_slots: int,
):
    """Publisher-only: build the KvChunkAddressTable covering every gathered
    decoder layer.

    The KV layout (bank striping, chunk/slot strides) is identical on every
    rank, so the chunk math runs against the gathered geometry of the first KV
    rank (`geom`); the two things that differ per layer are patched in from the
    gather — the owning rank's buffer address (independent MeshAllocators) and
    its mesh's real fabric node ids (device groups, one per sp row). `mesh_cols`
    also comes from `geom` (the KV mesh shape), NOT the publisher's own mesh,
    which may be the embedding mesh with a different shape.
    """
    d = ttnn.experimental.disaggregation
    FabricNodeId = ttnn._ttnn.fabric.FabricNodeId
    MeshId = ttnn._ttnn.fabric.MeshId

    mesh_cols = geom.mesh_cols
    tokens_per_tile = geom.tokens_per_kv_tile
    num_chunks = max_seq_len // tokens_per_tile
    _, _, chunk_size_bytes, _ = _get_kv_chunk_metadata(geom, 0, 0, 0)

    num_layers_cfg = max(layer_to_rank) + 1
    cfg = d.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers_cfg
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_slots
    cfg.chunk_n_tokens = tokens_per_tile
    cfg.chunk_size_bytes = chunk_size_bytes
    table = d.KvChunkAddressTable(cfg)

    # One device group per (owning rank, sp row): the gathered device lists are
    # row-major, so row r of a rank's mesh is the slice [r*cols, (r+1)*cols).
    # All meshes are assumed to share the gathered KV geometry's (rows, cols)
    # shape — asserted via the gathered device count.
    rank_row_groups: dict[int, dict[int, int]] = {}

    def _row_groups_for(rank: int) -> dict[int, int]:
        if rank in rank_row_groups:
            return rank_row_groups[rank]
        devs = all_rank_devices[rank]
        if len(devs) % mesh_cols != 0:
            raise RuntimeError(f"rank {rank} gathered {len(devs)} devices, not divisible by mesh_cols={mesh_cols}")
        # Host tag must match the worker grouping's "host-<crc32 hex>" — the
        # join key for hostname->rank chunk routing.
        host = f"host-{all_host_tags[rank]:08x}"
        groups = {}
        for row in range(len(devs) // mesh_cols):
            fnids = [
                FabricNodeId(MeshId(mesh), fchip) for (_, mesh, fchip) in devs[row * mesh_cols : (row + 1) * mesh_cols]
            ]
            dg = table.add_device_group(fnids)
            for f in fnids:
                table.set_fabric_node_host(f, host)
            groups[row] = dg
        rank_row_groups[rank] = groups
        return groups

    for layer_id, owner_rank in sorted(layer_to_rank.items()):
        base_addr = all_addrs[owner_rank]
        row_groups = _row_groups_for(owner_rank)
        for slot_id in range(num_slots):
            for chunk_idx in range(num_chunks):
                pos = chunk_idx * tokens_per_tile
                bank, offset, size, sp_idx = _get_kv_chunk_metadata(geom, pos, slot_id, base_addr)
                loc = d.KvCacheLocation()
                loc.noc_addr = _pack_bank_offset(bank, offset)
                loc.size_bytes = size
                loc.device_group_index = row_groups[sp_idx]
                table.set(layer_id, pos, slot_id, loc)
                # Sample log: first + last chunk of each (layer, slot) — decoded
                # (bank, offset) cross-checks against the worker's [dram-io] lines.
                if chunk_idx in (0, num_chunks - 1):
                    logger.info(
                        "[migration-hook] table[{},{},{}]: bank={} off=0x{:x} size={} sp_row={} (owner r{} base 0x{:x})",
                        layer_id,
                        pos,
                        slot_id,
                        bank,
                        offset,
                        size,
                        sp_idx,
                        owner_rank,
                        base_addr,
                    )

    logger.info(
        "[migration-hook] table built: layers={} ({}) max_seq={} slots={} chunk_n_tokens={} "
        "chunk_size_bytes={} num_chunks={}",
        num_layers_cfg,
        sorted(layer_to_rank.items()),
        max_seq_len,
        num_slots,
        tokens_per_tile,
        chunk_size_bytes,
        num_chunks,
    )
    return table


# Default location of the _migration_client python module on the shared (NFS)
# build tree. Used when TT_MIGRATION_PYTHON_DIR isn't forwarded to this rank —
# which happens on multi-host runs where the migration master (first dense
# decoder) lands on a host other than the tt-run launch host, so the launch
# shell's env doesn't propagate. Override with TT_MIGRATION_PYTHON_DIR.
_DEFAULT_MIGRATION_PYTHON_DIRS = [
    "/data/asaigal/tt-llm-engine/disaggregation/migration/build_RelWithDebInfo/python",
]


def _attach_migration_client(cmd_q: str, table_q: str, resp_q: str, timeout_s: float = 120.0):
    """Import MigrationLayerClient from the tt-llm-engine build and attach to
    the endpoint's shmem queues, retrying until they exist (the endpoint and
    the model race at startup)."""
    candidates = []
    env_dir = os.environ.get("TT_MIGRATION_PYTHON_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.extend(_DEFAULT_MIGRATION_PYTHON_DIRS)
    for d in candidates:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from _migration_client import MigrationLayerClient  # noqa: E402
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "[migration-hook] cannot import _migration_client. Tried sys.path entries "
            f"{candidates}. Set TT_MIGRATION_PYTHON_DIR (and ensure tt-run forwards it to "
            'every rank via --mpi-args "-x TT_MIGRATION_PYTHON_DIR"), or add this rank\'s '
            "build path to _DEFAULT_MIGRATION_PYTHON_DIRS."
        ) from e

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


def _deliver_device_map_to_local_workers(entries, attach_timeout: float = 60.0):
    """Deliver THIS host's device map to the local endpoint's per-rank internal A+B worker
    queues (/ep_<pid>_{a,b}_*) — the same queues run_migration_verify.sh's deliver_maps targets.

    A single multi-rank endpoint has one worker rank per host (an A + B loopback pair); the model
    publishes the table once on the master but EACH rank's device map must reach that rank's
    workers directly (the worker does no master->subordinate devmap forwarding — main delivers
    per-rank). We discover this host's worker queues by globbing /dev/shm, so we need no
    orchestrator-PID / rank bookkeeping. `entries` is a list of (mesh, fnid_chip, umd_chip)
    3-tuples (no chip-offset: the model's real fnids match the worker grouping's host partition)."""
    import glob

    deadline = time.monotonic() + attach_timeout
    while True:
        found = {}
        for w in ("a", "b"):
            cmds = sorted(
                (f for f in glob.glob(f"/dev/shm/ep_*_{w}_cmd*") if not f.endswith(".lock")),
                key=os.path.getmtime,
            )
            if cmds:
                found[w] = "/" + os.path.basename(cmds[-1])  # /ep_<pid>_<w>_cmd<sfx>
        if len(found) == 2:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"[migration-hook] local endpoint internal worker queues (/ep_*_{{a,b}}_*) not found "
                f"in /dev/shm (found {found}) — is the multi-rank migration endpoint up on this host?"
            )
        time.sleep(0.25)
    for w, cmd in found.items():
        tbl = cmd.replace(f"_{w}_cmd", f"_{w}_table", 1)
        resp = cmd.replace(f"_{w}_cmd", f"_{w}_resp", 1)
        client = _attach_migration_client(cmd, tbl, resp)
        client.send_device_map(entries)
        logger.info("[migration-hook] delivered {} device-map entries to {}", len(entries), cmd)


def validate_kv_migration(
    pipeline,
    mesh_device,
    *,
    done_file: str,
    pairs: list[tuple[int, int]] | None = None,
    positions: int = 32,
    result_file: str | None = None,
    golden_pt: str | None = None,
    golden_pcc: float = 0.88,
) -> bool | None:
    """Pull the KV cache to host and validate it. Returns True on PASS, False on
    FAIL, None on non-KV ranks.

    Two modes:
      * default (golden_pt None) — byte-compare src_slot vs dst_slot over
        positions [0, positions) for each (src, dst) in `pairs` (loopback test).
      * golden_pt set — a comma-separated list of .pt files, ONE per user/slot in
        slot order. Each KV-owning rank PCC-compares slot s's KV (this rank's layer)
        against golden[s]'s `ref_kvpe_list[layer]`, nope/pe separately (layer score =
        min), and writes per-(layer,slot) lines to <done_file>.golden_layer<L>.result.
        get_kv_cache_host() returns NATURAL order + the migration is position-aligned,
        so no de-permute is applied here (see memory project_kv_golden_validation).

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

    # ---- Golden-PCC mode: this rank's migrated KV vs the on-disk golden -----------
    if golden_pt is not None:
        from models.common.utility_functions import comp_pcc

        layer = pipeline.layer_idx()
        if layer is None:
            # owns KV but reports no decoder layer index — can't map to a golden layer.
            logger.warning("[migration-validate-golden] KV rank has no layer_idx; skipping golden compare")
            return None
        golden_paths = [p for p in golden_pt.split(",") if p]
        kvpe = int(kv.shape[3])
        kv_lora = kvpe - 64  # qk_rope_head_dim = 64; nope = kv_lora_rank (512 for DeepSeek)
        all_ok = True
        result_lines = []
        for s, gp in enumerate(golden_paths):
            try:
                ref_list = torch.load(gp, map_location="cpu", weights_only=True, mmap=True)["ref_kvpe_list"]
            except Exception as e:  # noqa: BLE001 — surface any load/format error as a FAIL
                logger.error("[migration-validate-golden] slot{} failed to load golden {}: {}", s, gp, e)
                all_ok = False
                continue
            if layer >= len(ref_list):
                logger.error("[migration-validate-golden] layer {} >= golden layers {} ({})", layer, len(ref_list), gp)
                all_ok = False
                continue
            ref_layer = ref_list[layer]
            n = min(int(positions), int(kv.shape[2]), int(ref_layer.shape[2]))
            # NATURAL order both sides (get_kv_cache_host de-permutes; the kvpe channel dim is
            # untouched). pe (RoPE, [kv_lora:]) compared DIRECTLY — no HF->Meta re-interleave: the
            # .pt is already in the device rotary basis and the migration byte-copies it (see
            # memory project_kv_golden_validation). bf8_b on device -> PCC; nope/pe min = score.
            dev = kv[s, 0, :n, :].to(torch.float32)
            ref = ref_layer[0, 0, :n, :].to(torch.float32)
            _, pcc_nope = comp_pcc(ref[:, :kv_lora], dev[:, :kv_lora], pcc=golden_pcc)
            _, pcc_pe = comp_pcc(ref[:, kv_lora:], dev[:, kv_lora:], pcc=golden_pcc)
            pcc_nope, pcc_pe = float(pcc_nope), float(pcc_pe)
            layer_pcc = min(pcc_nope, pcc_pe)
            ok = layer_pcc >= golden_pcc
            all_ok = all_ok and ok
            msg = (
                f"layer{layer} slot{s} positions[0,{n}): {'PASS' if ok else 'FAIL'} "
                f"pcc_nope={pcc_nope:.6f} pcc_pe={pcc_pe:.6f} min={layer_pcc:.6f} threshold={golden_pcc}"
            )
            logger.info("[migration-validate-golden] {}", msg)
            result_lines.append(msg)
        golden_out = result_file or f"{done_file}.golden_layer{layer}.result"
        with open(golden_out, "w") as f:
            f.write("\n".join(result_lines) + "\n")
        return all_ok

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
    max_seq_len: int,
    num_slots: int,
):
    """Build the on_kv_cache_ready callback for run_demo.

    Signature matches ModelPipeline's invocation: (mesh_device, kv_cache,
    layer_id), called on EVERY rank. The callback is MPI-collective — all
    ranks enter the allgathers; non-publisher ranks return after the
    rendezvous. RANK 0 (head node / first stage) is the sole publisher of
    SET_TABLE + the device map; it authors the table from the geometry
    gathered from the first KV-owning rank, so it needs no local KV cache.
    The migration endpoint MUST run on rank 0's host (its shmem queues are
    host-local).
    """

    def on_kv_cache_ready(mesh_device, kv_cache, layer_id):
        my_rank = int(ttnn.distributed_context_get_rank())

        if kv_cache is not None:
            devices = _enumerate_devices(mesh_device)
            kv_addr = int(kv_cache.buffer_address())
            geom_local = _kv_geometry_from_cache(kv_cache, mesh_device)
        else:
            devices = []
            kv_addr = 0
            geom_local = _GEOM_NONE
        logger.info(
            "[migration-hook] rank {} local: layer={} kv_addr=0x{:x} geom={} devices(umd,mesh,fchip)={}",
            my_rank,
            layer_id,
            kv_addr,
            tuple(geom_local),
            devices,
        )
        all_rank_devices, all_layers, all_addrs, all_host_tags, all_geom = _gather_kv_topology(
            devices, layer_id, kv_addr, geom_local
        )
        logger.info(
            "[migration-hook] rank {} gathered: host_tags={} layers={}",
            my_rank,
            [f"{t:08x}" for t in all_host_tags],
            all_layers,
        )

        kv_ranks = [r for r, devs in enumerate(all_rank_devices) if devs]
        if not kv_ranks:
            raise RuntimeError("[migration-hook] no rank owns a KV cache — nothing to migrate")

        # layer → owning rank, with duplicate-claim detection (a duplicate
        # means dense_layer_id_override collapsed several stages onto one
        # layer id — the table key must be unique per stage).
        layer_to_rank: dict[int, int] = {}
        for r in kv_ranks:
            lid = all_layers[r]
            if lid == _SENTINEL_LAYER:
                raise RuntimeError(f"[migration-hook] rank {r} owns KV but reported no layer id")
            if lid in layer_to_rank:
                raise RuntimeError(
                    f"[migration-hook] ranks {layer_to_rank[lid]} and {r} both claim layer {lid} "
                    "— run decoder stages with distinct layer ids (drop dense_layer_id_override)"
                )
            layer_to_rank[lid] = r

        # SINGLE multi-rank endpoint across all hosts (master rank 0 on the head + one worker rank
        # per other host). Two distinct deliveries, mirroring run_migration_verify.sh:
        #   (a) DEVICE MAP — per rank. Each host's representative model rank delivers ITS host's
        #       real device map straight to that host's internal A+B worker queues. The worker does
        #       no master->subordinate devmap forwarding (main delivers per-rank), so this is how a
        #       subordinate rank learns its fnid->chip mapping and can resolve its table split.
        #   (b) TABLE — once. The head publishes the global table on the master's outward queue; the
        #       master splits it by host (build_local_grouping's host_key_for partition) and forwards
        #       each rank its slice over MPI. The outward device map is EMPTY (real maps already in
        #       place from (a)).
        master_rank = 0
        ag = ttnn.distributed_context_allgather_int
        world = len(all_host_tags)

        # (a) Per-host device-map delivery — done by the lowest-ranked model rank on each host.
        my_tag = all_host_tags[my_rank]
        same_host_ranks = [r for r in range(world) if all_host_tags[r] == my_tag]
        if my_rank == min(same_host_ranks):
            seen = set()
            host_device_map = []
            for r in kv_ranks:
                if all_host_tags[r] != my_tag:
                    continue
                for umd, mesh, fchip in all_rank_devices[r]:
                    key = (mesh, fchip, umd)
                    if key not in seen:
                        seen.add(key)
                        host_device_map.append(key)
            logger.info(
                "[migration-hook] rank {} (host {:08x}) delivering {} device-map entries to local workers: {}",
                my_rank,
                my_tag,
                len(host_device_map),
                host_device_map,
            )
            _deliver_device_map_to_local_workers(host_device_map)

        # Barrier: every host's device map must be in place before the master sets the table —
        # WORKER_READY gates on every rank's dev_map_applied_.
        list(ag(1))

        # (b) Master (head) publishes the global table once, then an EMPTY outward device map.
        if my_rank == master_rank:
            # Author from the first KV rank's geometry (identical across layers).
            geom = all_geom[kv_ranks[0]]
            if geom.tokens_per_kv_tile == 0 or geom.mesh_cols == 0 or geom.per_device_seq_len == 0:
                raise RuntimeError(
                    f"[migration-hook] publisher rank {my_rank} gathered invalid geometry from kv_rank "
                    f"{kv_ranks[0]}: {tuple(geom)} — gather/order bug"
                )
            logger.info(
                "[migration-hook] publisher rank {} building table from kv_rank {} geom={}: layers={} addrs={}",
                my_rank,
                kv_ranks[0],
                tuple(geom),
                sorted(layer_to_rank.items()),
                {r: hex(all_addrs[r]) for r in kv_ranks},
            )
            # Derive global max_seq_len when not given: the sequence is striped across sp rows,
            # so global = per-device seq len × sp_dim.
            effective_max_seq = max_seq_len
            if effective_max_seq is None:
                from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode

                sp_dim = FlashMLADecode.ProgramConfig(k_chunk_size=128).sp_dim
                effective_max_seq = geom.per_device_seq_len * sp_dim
                logger.info(
                    "[migration-hook] derived max_seq_len={} (per_device={} × sp_dim={})",
                    effective_max_seq,
                    geom.per_device_seq_len,
                    sp_dim,
                )
            table = _build_table_gathered(
                geom,
                layer_to_rank=layer_to_rank,
                all_rank_devices=all_rank_devices,
                all_addrs=all_addrs,
                all_host_tags=all_host_tags,
                max_seq_len=effective_max_seq,
                num_slots=num_slots,
            )
            ttnn.experimental.disaggregation.export_kv_chunk_table_to_protobuf_file(table, table_path)
            logger.info("[migration-hook] table exported to {}", table_path)

            client = _attach_migration_client(cmd_queue, table_queue, resp_queue)
            client.send_kv_chunk_table(table_path)
            client.send_device_map([])  # real per-rank maps already delivered in (a)
            client.wait_ready(120000)
            logger.info(
                "[migration-hook] migration endpoint WORKER_READY — global table + per-rank device maps in place"
            )

        # Barrier: keep all model ranks in lockstep until the endpoint is fully ready.
        list(ag(2))
        # Client detaches at scope exit; shmem queues persist for the scheduler.

    return on_kv_cache_ready
