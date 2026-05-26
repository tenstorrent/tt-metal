# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Migration endpoint construction for the prefill runner.

Builds a connected MigrationLayerEndpoint for the prefill side of a
disaggregated prefill/decode setup.

Conventions match tt-blaze tests/blaze/run_prefill_decode_integration.sh:

    DECODE_SUBCTX_ID  = 0   ↔  DECODE_EP_ID  = 0
    PREFILL_SUBCTX_ID = 1   ↔  PREFILL_EP_ID = 1

Lifecycle (under tt-run with --rank-bindings-mapping):

    1. tt-run sets up sub-contexts via the rank-bindings-mapping YAML.
    2. ttnn.init_distributed_context() — lazy-init by ttnn.
    3. ttnn.open_mesh_device(...)                        ← runner does this
    4. allocate kvpe_cache                               ← pipeline does this
    5. setup_prefill_migration(mesh, kvpe_cache, ...)    ← AFTER pipeline.compile()
    6. pipeline.setup_migration(endpoint, DECODE_EP_ID)

The pipeline's per-layer migration callback then fires automatically during
pipeline.prefill(): every layer triggers migrate_layer (non-blocking); the
last layer also waits on its uuid (in-order transport ⇒ all migrations
acked by the time forward() returns).
"""

import sys

import ttnn

# Sub-context + endpoint IDs — matches run_prefill_decode_integration.sh convention.
DECODE_SUBCTX_ID = 0
PREFILL_SUBCTX_ID = 1
DECODE_EP_ID = 0
PREFILL_EP_ID = 1

# Migration master must own a kv_cache to build the chunk address table from.
# In the pod pipeline (num_procs ∈ {4, 16, 64}), local rank 0 is always
# SpecLMHead+Embedding (no decoder KV) and local rank 1 is the first
# DenseDecoderStage (has KV). Pick rank 1 unconditionally — every rank can
# infer this without communication since the pipeline config is identical
# across ranks. Both prefill (when building decode_grouping) and decode
# (when constructing its own endpoint) must agree on this value.
DECODE_MIGRATION_MASTER_LOCAL_RANK = 1

# Sentinel dst_slot value sent by the C++ server when a request must NOT trigger
# KV migration (e.g. warmup probes, scheduler-skipped requests).
INVALID_SLOT_ID = 0xFFFFFFFF

# DRAM banks per Blackhole device. Used when packing NOC addresses for the
# prefill KvChunkAddressTable.
BH_NUM_DRAM_BANKS = 8


def _import_migration():
    """Lazy import so the migration extension is only required when migration
    is actually enabled (PREFILL_ENABLE_MIGRATION=1).

    KvCacheLocation, KvChunkAddressTable, KvChunkAddressTableConfig are also
    registered with nanobind by ttnn (ttnn/cpp/ttnn-nanobind/disaggregation.cpp).
    nanobind's global type registry rejects duplicate registrations: whichever
    .so loads first wins, and the other module's symbols stay unbound. ttnn is
    always loaded before _migration here, so we pull those three types from
    ttnn.experimental.disaggregation instead. The migration-only symbols
    (MigrationLayerEndpoint, SubordinateInfo, EndpointGrouping,
    make_mpi_endpoint_device) come from _migration as before.
    """
    try:
        from _migration import MigrationLayerEndpoint  # noqa: F401  (re-exposed via factory)
        from _migration import EndpointGrouping, SubordinateInfo, make_mpi_endpoint_device
    except ImportError as exc:
        raise ImportError(
            "Cannot import _migration. Build tt-blaze migration extension "
            "(./build_blaze.sh --with-metal) and add "
            "<tt-blaze>/build_Release/python to PYTHONPATH."
        ) from exc
    KvCacheLocation = ttnn.experimental.disaggregation.KvCacheLocation
    KvChunkAddressTable = ttnn.experimental.disaggregation.KvChunkAddressTable
    KvChunkAddressTableConfig = ttnn.experimental.disaggregation.KvChunkAddressTableConfig
    return dict(
        EndpointGrouping=EndpointGrouping,
        KvCacheLocation=KvCacheLocation,
        KvChunkAddressTable=KvChunkAddressTable,
        KvChunkAddressTableConfig=KvChunkAddressTableConfig,
        SubordinateInfo=SubordinateInfo,
        make_mpi_endpoint_device=make_mpi_endpoint_device,
    )


def ensure_distributed_context():
    """Lazy-init the ttnn distributed context (idempotent).

    Under tt-run with a rank-bindings-mapping YAML, the C++ side already
    knows the sub-context layout — this just creates the Python handle.
    Safe to call multiple times; falls back to a single-rank world when
    not under tt-run/mpirun.
    """
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()


def get_distributed_info():
    """Return (subctx_id, local_rank, local_size, world_rank, world_size).

    Local = within this rank's sub-context. World = across all sub-contexts.
    """
    ensure_distributed_context()
    return (
        ttnn.distributed_context_subcontext_id(),
        int(ttnn.distributed_context_get_rank()),
        int(ttnn.distributed_context_get_size()),
        int(ttnn.distributed_context_world_rank()),
        int(ttnn.distributed_context_world_size()),
    )


def _build_prefill_table(mesh, tt_kvpe_cache, seq_len, num_layers, mesh_shape, m):
    """KvChunkAddressTable for a kvpe_cache laid out by init_kvpe_cache().

    NOC address encoding: (bank_id << 32) | (base_addr + bank_offset),
    where bank_offset accumulates by chunk_size_bytes each time the bank
    counter wraps.
    """
    KvCacheLocation = m["KvCacheLocation"]
    KvChunkAddressTable = m["KvChunkAddressTable"]
    KvChunkAddressTableConfig = m["KvChunkAddressTableConfig"]

    # Match kvpe geometry — keep in sync with kv_cache_utils.NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    # and the bfp8 [1, 1, 32, 576] chunk size used by DeepSeek V3.
    chunk_n_tokens = 32
    chunk_size_bytes = 19584

    cfg = KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = seq_len
    cfg.num_slots = 1
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = chunk_size_bytes
    table = KvChunkAddressTable(cfg)

    base_addr = tt_kvpe_cache.buffer_address()
    print(f"[migration][prefill] base_addr = {base_addr}")

    rows, cols = mesh_shape
    dg_per_row = []
    for row in range(rows):
        devices = []
        for col in range(cols):
            fnid = mesh.get_fabric_node_id(ttnn.MeshCoordinate(row, col))
            devices.append((int(fnid.mesh_id), int(fnid.chip_id)))
        dg_idx = table.add_device_group(devices)
        for mid, cid in devices:
            table.set_fabric_node_host(mid, cid, f"mesh-{mid}")
        dg_per_row.append(dg_idx)

    # Two-strip layout: each row owns a low strip (front) and a high strip (back).
    num_tokens_in_strip = seq_len // (rows * 2)
    num_chunks_in_strip = num_tokens_in_strip // chunk_n_tokens
    chunks_per_dg = num_chunks_in_strip * 2

    low_start = 0
    high_end = seq_len - 1
    low_strips, high_strips = [], []
    for _ in range(rows):
        low_end = low_start + num_tokens_in_strip - 1
        high_start = high_end - num_tokens_in_strip + 1
        low_strips.append((low_start, low_end))
        high_strips.append((high_start, high_end))
        low_start = low_end + 1
        high_end = high_start - 1

    for row in range(rows):
        curr_bank_id = 0
        curr_bank_offset = 0
        for layer in range(num_layers):
            pos = low_strips[row][0]
            for c in range(chunks_per_dg):
                noc_addr = (curr_bank_id << 32) | (base_addr + curr_bank_offset)
                print(
                    f"[migration][prefill][table][init] l={layer}, pos={pos}, bank_id={curr_bank_id}, bank_offset={curr_bank_offset}, noc_addr={noc_addr}"
                )
                table.set(layer, pos, 0, KvCacheLocation(noc_addr, chunk_size_bytes, dg_per_row[row]))
                curr_bank_id = (curr_bank_id + 1) % BH_NUM_DRAM_BANKS
                if curr_bank_id == 0:
                    curr_bank_offset += chunk_size_bytes
                pos += chunk_n_tokens
                if c == num_chunks_in_strip - 1:
                    pos = high_strips[row][0]
    return table


def _build_subordinate_infos(local_size, device_list, m):
    """One SubordinateInfo per local-subcontext rank.

    Per the integration test convention: master (i=0) declares its real
    devices; non-masters declare empty local_devices.
    """
    SubordinateInfo = m["SubordinateInfo"]
    sub_infos = []
    for i in range(local_size):
        si = SubordinateInfo()
        si.id = i
        si.hostname = f"prefill-local-{i}"
        si.local_devices = [(mid, cid) for (_, mid, cid) in device_list] if i == 0 else []
        sub_infos.append(si)
    return sub_infos


def setup_prefill_migration(*, mesh_device, kvpe_cache, seq_len: int, num_layers: int, mesh_shape):
    """Build a connected prefill MigrationLayerEndpoint.

    Reads rank/topology from ttnn's distributed context — assumes tt-run
    has already set up sub-contexts via --rank-bindings-mapping.

    Args:
        mesh_device: open ttnn.MeshDevice (already opened by the runner).
        kvpe_cache: ttnn.Tensor handle returned by init_kvpe_cache().
        seq_len: max_seq_len the cache is sized for.
        num_layers: transformer layer count.
        mesh_shape: (rows, cols) tuple for the mesh.

    Returns:
        MigrationLayerEndpoint, ready to be passed to
        TtDeepSeekPrefillPipeline.setup_migration(endpoint, DECODE_EP_ID).
    """
    m = _import_migration()
    EndpointGrouping = m["EndpointGrouping"]
    make_mpi_endpoint_device = m["make_mpi_endpoint_device"]

    subctx_id, local_rank, local_size, world_rank, world_size = get_distributed_info()
    assert subctx_id == PREFILL_SUBCTX_ID, (
        f"prefill_runner must launch in PREFILL_SUBCTX_ID={PREFILL_SUBCTX_ID}, "
        f"got subctx_id={subctx_id}. Check rank-bindings-mapping YAML."
    )

    # Group's world ranks: every rank in the prefill sub-context.
    prefill_group_world_ranks = [
        int(ttnn.distributed_context_local_to_world_rank(PREFILL_SUBCTX_ID, r)) for r in range(local_size)
    ]
    prefill_master_world_rank = prefill_group_world_ranks[0]

    # Build local device list (chip_id, mesh_id, fabric_chip_id triples).
    rows, cols = mesh_shape
    device_list = []
    for r in range(rows):
        for c in range(cols):
            coord = ttnn.MeshCoordinate(r, c)
            chip_id = mesh_device.get_device_id(coord)
            fnid = mesh_device.get_fabric_node_id(coord)
            device_list.append((chip_id, int(fnid.mesh_id), int(fnid.chip_id)))

    sub_infos = _build_subordinate_infos(local_size, device_list, m)

    print(
        f"[migration] creating prefill endpoint world_rank={world_rank} "
        f"local_rank={local_rank}/{local_size} ep_id={PREFILL_EP_ID} "
        f"master_world_rank={prefill_master_world_rank}",
        file=sys.stderr,
    )
    endpoint = make_mpi_endpoint_device(
        rank=world_rank,
        endpoint_id=PREFILL_EP_ID,
        group_ranks=prefill_group_world_ranks,
        master_rank=prefill_master_world_rank,
        my_devices=device_list,
        subordinate_infos=sub_infos,
    )

    table = _build_prefill_table(mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, m)
    endpoint.initialize_my_kv_chunk_mapping_table(table)

    # Build remote (decode) grouping by translating decode-local ranks to world ranks.
    _, _, _, _, _ws = get_distributed_info()
    decode_local_size = int(ttnn.distributed_context_subcontext_sizes()[DECODE_SUBCTX_ID])
    decode_group_world_ranks = [
        int(ttnn.distributed_context_local_to_world_rank(DECODE_SUBCTX_ID, r)) for r in range(decode_local_size)
    ]

    decode_grouping = EndpointGrouping()
    decode_grouping.endpoint_id = DECODE_EP_ID
    decode_grouping.master = decode_group_world_ranks[DECODE_MIGRATION_MASTER_LOCAL_RANK]
    decode_grouping.ranks = decode_group_world_ranks
    decode_grouping.subordinate_infos = []

    print(
        f"[migration] connecting to decode endpoint ep_id={DECODE_EP_ID} "
        f"ranks={decode_group_world_ranks} master={decode_grouping.master}",
        file=sys.stderr,
    )
    endpoint.connect_to_remote_endpoint(decode_grouping)
    print(
        "[migration] connected; per-layer migrations will fire from pipeline.prefill()",
        file=sys.stderr,
    )
    return endpoint
