# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop handoff (G1 -> G2) over a D2D socket between the two 4x4 halves of an 8x4 Galaxy.

G1 (rows 0-3) writes 28 loop layers of scaled-FP8 KVPE per chunk with the production TP-deduped write
(``update_padded_kv_cache(cluster_axis=0, tp_axis=1)``) and holds 7 top-k sets; the handoff collects the rows each
chip stored and the sets, and sends them as extra segments of the activation's one D2D service. G2 (rows 4-7) writes
the rows into its copy of the cache. A multi-chunk schedule covers chunk-aligned starts, a 32-aligned start that is
not chunk-aligned (rotated per-chip offsets) and partial ends. Asserted bit for bit: every received top-k set and
KVPE segment against what G1 sent, and G2's whole cache against G1's. ``traced`` replays a captured G1 collect+send
and a captured G2 drain+write for every chunk after the first.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.loop_handoff import LoopHandoff, LoopHandoffSender, write_kvpe_copy
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import CapturedCall

CHUNK = 5120
KV_LAYERS = 28  # loop layers 27-54
INDEX_SETS = 7  # full indexer layers among them (reuse groups of 4)
TOP_K = 2048
HIDDEN = 6144
NUM_USERS = 2
MAX_SEQ = 8 * CHUNK  # a multiple of the chunk and of 8192 (KV slots start on DRAM bank 0)
SENTINEL = 0xFFFFFFFF
METADATA_WORDS = 4  # slot, start, end, chunk number
# (slot, start, end): slot 1 resumes at a 32-aligned, chunk-unaligned start, so each chunk's per-chip offsets rotate.
SCHEDULE = [(0, 0, 5120), (1, 0, 96), (0, 5120, 10240), (1, 96, 5216), (0, 10240, 12000), (1, 5216, 9000)]


def _scalar(mesh, value):
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _fill(dst, values):
    host = ttnn.from_torch(
        torch.tensor(values, dtype=torch.int64).reshape(tuple(dst.shape)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dst.device()),
    )
    ttnn.copy_host_to_device_tensor(host, dst)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Bit pattern of a host tensor, so NaN payloads compare too."""
    if t.dtype in (torch.float32, torch.int32, torch.uint32):
        return t.contiguous().view(torch.int32)
    if t.dtype in (torch.bfloat16, torch.float16, torch.int16):
        return t.contiguous().view(torch.int16)
    return t.contiguous().view(torch.uint8)


def _shards(t) -> list:
    """Each chip's host copy, in mesh order. Read through a mesh composer: single-device to_torch rejects FP8."""
    mesh = t.device()
    whole = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return list(torch.chunk(whole, mesh.get_num_devices(), dim=0))


def _assert_shards_equal(a, b, what):
    sa, sb = _shards(a), _shards(b)
    assert len(sa) == len(sb), f"{what}: {len(sa)} != {len(sb)} shards"
    for k, (x, y) in enumerate(zip(sa, sb)):
        assert x.shape == y.shape, f"{what}: shard {k} shape {tuple(y.shape)} != {tuple(x.shape)}"
        assert torch.equal(_bits(x), _bits(y)), f"{what}: shard {k} differs"


def _kv_cache(mesh):
    return init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        mesh_device=mesh,
        seq_len=MAX_SEQ,
        mesh_shape=list(mesh.shape),
        sp_axis=0,
        num_kvpe_cache_layers=KV_LAYERS,
        num_users=NUM_USERS,
        tp_axis=1,
        bank_aligned=True,
    ).storage


def _g1_write_layers(mesh, cache, meta, generator):
    """Stand-in for G1's loop layers: each writes a TP-replicated chunk of distinct FP8 rows, as MLA does."""
    sp_rows_mapper = ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()])
    )
    width = int(cache.shape[-1])
    for layer in range(KV_LAYERS):
        rows = torch.randn(1, 1, CHUNK, width, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        bf16 = ttnn.from_torch(
            rows,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=sp_rows_mapper,
        )
        fp8 = ttnn.typecast(bf16, ttnn.fp8_e4m3)
        ttnn.deallocate(bf16)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            fp8,
            meta[0],
            meta[1],
            layer_idx=layer,
            num_layers=KV_LAYERS,
            cluster_axis=0,
            valid_global=meta[2],
            tp_axis=1,
        )
        ttnn.deallocate(fp8)


def _host_index_sets(handoff, end, generator):
    """Per set, a global [1, 1, sp*rows, tp*k] block of top-k rows: valid positions < end, then a sentinel tail."""
    sp, tp = handoff.mesh_shape
    _, _, rows, k = handoff.index_shape
    sets = []
    for _ in range(INDEX_SETS):
        idx = torch.randint(0, end, (1, 1, sp * rows, tp * k), generator=generator, dtype=torch.int64)
        valid = torch.randint(1, tp * k, (sp * rows, 1), generator=generator)
        tail = torch.arange(tp * k).unsqueeze(0) >= valid
        idx[0, 0][tail] = SENTINEL
        sets.append(idx)
    return sets


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_read_padded_kv_cache_round_trip(mesh_device):
    """The read op at the production 8x4 geometry (SP 8 x TP 4, 160 rows per chip per chunk), without D2D: read a
    layer range back after each chunk's TP-deduped write and write it into a second cache in complete-mesh mode;
    the two caches must match bit for bit over those layers, and the unread layer must stay untouched. Alternate
    chunks read into a fresh output, and half way the metadata tensors are replaced, so program-cache hits must
    patch both addresses."""
    if not is_blackhole():
        pytest.skip("FP8_E4M3 needs Blackhole")
    mesh_device.enable_program_cache()
    num_layers, first, count = 3, 1, 2
    caches = [
        init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.SCALED_FP8,
            hf_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
            mesh_device=mesh_device,
            seq_len=MAX_SEQ,
            mesh_shape=list(mesh_device.shape),
            sp_axis=0,
            num_kvpe_cache_layers=num_layers,
            num_users=NUM_USERS,
            tp_axis=1,
            bank_aligned=True,
        ).storage
        for _ in range(2)
    ]
    source, copy = caches
    sp, tp = mesh_device.shape
    handoff = LoopHandoff(
        mesh_shape=(sp, tp),
        chunk_size_global=CHUNK,
        kv_layers=count,
        kv_width=int(source.shape[-1]),
        kv_dtype=source.dtype,
        index_shape=(1, 1, CHUNK // (sp * tp), TOP_K),
        num_index_sets=0,
    )
    assert handoff.chunk_rows == 160
    sender = LoopHandoffSender(mesh_device, handoff, index_sets=[], first_kv_layer=first, num_kv_layers=num_layers)
    meta = [_scalar(mesh_device, 0) for _ in range(3)]
    generator = torch.Generator().manual_seed(27)
    rows_mapper = ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()])
    )
    for n, (slot, start, end) in enumerate(SCHEDULE):
        if n == len(SCHEDULE) // 2:
            # Fresh metadata tensors on a program-cache hit: the read must patch their new addresses.
            meta = [_scalar(mesh_device, 0) for _ in range(3)]
        for dst, value in zip(meta, (slot, start, end)):
            _fill(dst, [value])
        for layer in range(num_layers):
            rows = torch.randn(1, 1, CHUNK, int(source.shape[-1]), generator=generator).to(torch.bfloat16)
            bf16 = ttnn.from_torch(
                rows, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rows_mapper
            )
            fp8 = ttnn.typecast(bf16, ttnn.fp8_e4m3)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                source,
                fp8,
                meta[0],
                meta[1],
                layer_idx=layer,
                num_layers=num_layers,
                cluster_axis=0,
                valid_global=meta[2],
                tp_axis=1,
            )
            ttnn.deallocate(bf16)
            ttnn.deallocate(fp8)
        if n % 2:
            # A freshly allocated output (output=None) on a cache hit: its address must be patched too.
            rows = ttnn.experimental.deepseek_prefill.read_padded_kv_cache(
                source,
                meta[0],
                meta[1],
                first_layer=first,
                num_read_layers=count,
                num_layers=num_layers,
                chunk_size_global=CHUNK,
                cluster_axis=0,
                tp_axis=1,
            )
        else:
            sender.collect(source, meta)
            rows = sender.kvpe
        write_kvpe_copy(copy, rows, meta, handoff, first_kv_layer=first, num_kv_layers=num_layers)
        if rows is not sender.kvpe:
            ttnn.deallocate(rows)
    ttnn.synchronize_device(mesh_device)
    entries = NUM_USERS * num_layers
    for k, (a, b) in enumerate(zip(_shards(source), _shards(copy))):
        ta, tb = _bits(a), _bits(b)
        for entry in range(entries):
            layer = entry % num_layers
            if first <= layer < first + count:
                assert torch.equal(ta[entry], tb[entry]), f"chip {k} entry {entry}: copy differs from source"
                assert torch.count_nonzero(ta[entry]) > 0, f"chip {k} entry {entry}: nothing written"
            else:
                assert torch.count_nonzero(tb[entry]) == 0, f"chip {k} entry {entry}: unread layer was written"


@pytest.mark.parametrize("traced", [False, True], ids=["eager", "traced"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            fabric2d_device_params(trace_region_size=4 * 1024 * 1024),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(1800)
def test_glm54_loop_handoff(mesh_device, traced):
    if not is_blackhole():
        pytest.skip("D2DStreamService and FP8_E4M3 need Blackhole")
    half = mesh_device.shape[0] // 2
    cols = mesh_device.shape[1]
    g1 = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(0, 0))
    g2 = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(half, 0))
    g1.enable_program_cache()
    g2.enable_program_cache()
    mesh_shape = (half, cols)

    g1_cache, g2_cache = _kv_cache(g1), _kv_cache(g2)
    handoff = LoopHandoff(
        mesh_shape=mesh_shape,
        chunk_size_global=CHUNK,
        kv_layers=KV_LAYERS,
        kv_width=int(g1_cache.shape[-1]),
        kv_dtype=g1_cache.dtype,
        index_shape=(1, 1, CHUNK // (half * cols), TOP_K),
        num_index_sets=INDEX_SETS,
    )
    # G1's model sets (TtPrefillTransformer.loop_index_sets), which its pass 1 fills and the sender sends.
    model_sets = [
        ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(
                shape=ttnn.Shape(list(handoff.index_shape)),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            ),
            g1,
        )
        for _ in range(INDEX_SETS)
    ]
    sender_handoff = LoopHandoffSender(g1, handoff, index_sets=model_sets, first_kv_layer=0, num_kv_layers=KV_LAYERS)

    act_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, CHUNK, HIDDEN]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    act_cfg = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
    grid = g1.compute_with_storage_grid_size()
    workers = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(8, grid.x) - 1, min(8, grid.y) - 1))
    metadata_bytes = 4 * METADATA_WORDS
    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=g1,
        receiver_mesh=g2,
        global_spec=act_spec,
        mapper=ttnn.create_mesh_mapper(g1, act_cfg),
        fifo_size_bytes=16384,
        sender_worker_cores=workers,
        receiver_worker_cores=workers,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=metadata_bytes,
        share_fabric_links=True,
        extra_tensors=[(spec, ttnn.create_mesh_mapper(g1, cfg)) for spec, cfg in handoff.segments()],
    )
    # One service carries the activation and the whole handoff.
    assert len(sender.get_backing_tensors()) == 2 + INDEX_SETS

    # Persistent per-chunk state: what a traced runtime keeps at fixed addresses.
    g1_meta = [_scalar(g1, 0) for _ in range(3)]
    g2_meta = [_scalar(g2, 0) for _ in range(3)]
    record = ttnn.from_torch(
        torch.zeros(1, 1, 1, METADATA_WORDS, dtype=torch.int64),
        device=g1,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(g1),
    )
    activation = ttnn.zeros_like(sender.get_backing_tensors()[0])
    index_src = [ttnn.zeros_like(t) for t in sender_handoff.index_sets]
    recv_act = ttnn.zeros_like(receiver.get_backing_tensors()[0])
    recv_extras = [ttnn.zeros_like(t) for t in receiver.get_backing_tensors()[1:]]
    recv_md = ttnn.from_torch(
        torch.zeros(1, 1, 1, METADATA_WORDS, dtype=torch.int64),
        device=g2,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(g2),
    )

    def g1_send():
        for set_idx, indices in enumerate(index_src):
            ttnn.copy(indices, model_sets[set_idx])  # as pass 1 does when the full layer makes its top-k
        sender_handoff.collect(g1_cache, g1_meta)
        ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(
            sender, [activation, *sender_handoff.segments()], metadata=record
        )

    def g2_receive():
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            receiver,
            metadata_size_bytes=metadata_bytes,
            tokens_out=recv_act,
            extras_out=recv_extras,
            metadata_out=recv_md,
        )
        write_kvpe_copy(g2_cache, recv_extras[0], g2_meta, handoff, first_kv_layer=0, num_kv_layers=KV_LAYERS)

    send_trace = recv_trace = None
    generator = torch.Generator().manual_seed(54)
    index_mapper = ttnn.create_mesh_mapper(g1, handoff.index_mapper_config())
    try:
        for n, (slot, start, end) in enumerate(SCHEDULE):
            for metas in (g1_meta, g2_meta):
                for dst, value in zip(metas, (slot, start, end)):
                    _fill(dst, [value])
            _fill(record, [slot, start, end, n])
            _g1_write_layers(g1, g1_cache, g1_meta, generator)
            host_sets = _host_index_sets(handoff, end, generator)
            for host, dst in zip(host_sets, index_src):
                src = ttnn.from_torch(
                    host, device=g1, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=index_mapper
                )
                ttnn.copy(src, dst)
                ttnn.deallocate(src)
            act = ttnn.from_torch(
                torch.randn(1, 1, CHUNK, HIDDEN, generator=generator).to(torch.bfloat16),
                device=g1,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.create_mesh_mapper(g1, act_cfg),
            )
            ttnn.copy(act, activation)
            ttnn.deallocate(act)

            if traced and n == 1:
                # Chunk 0 compiled every program eagerly; capture the same calls once and replay them from here on.
                ttnn.synchronize_device(g1)
                ttnn.synchronize_device(g2)
                send_trace, recv_trace = CapturedCall(g1), CapturedCall(g2)
                send_trace.capture(g1_send)
                recv_trace.capture(g2_receive)
            if send_trace is not None:
                send_trace.replay()
            else:
                g1_send()
            sender.release_fabric_links()
            receiver.release_fabric_links()
            if recv_trace is not None:
                recv_trace.replay()
            else:
                g2_receive()
            sender.wait_for_fabric_links()
            receiver.wait_for_fabric_links()
            ttnn.synchronize_device(g1)
            ttnn.synchronize_device(g2)

            _assert_shards_equal(activation, recv_act, f"chunk {n} activation")
            _assert_shards_equal(sender_handoff.kvpe, recv_extras[0], f"chunk {n} KVPE rows")
            for i, (sent, got) in enumerate(zip(sender_handoff.index_sets, recv_extras[1:])):
                _assert_shards_equal(sent, got, f"chunk {n} top-k set {i}")
                composer = ttnn.ConcatMesh2dToTensor(g2, dims=(2, 3), mesh_shape=g2.shape)
                received = ttnn.to_torch(got, mesh_composer=composer).to(torch.int64) & 0xFFFFFFFF
                assert torch.equal(received, host_sets[i]), f"chunk {n} top-k set {i} differs from G1's indices"
            md = ttnn.to_torch(ttnn.get_device_tensors(recv_md)[0]).flatten().to(torch.int64).tolist()
            assert md == [slot, start, end, n], f"chunk {n} metadata {md}"

        _assert_shards_equal(g1_cache, g2_cache, "G2 KVPE copy vs G1 cache")
        # The copy is not vacuous: every slot's written rows hold data.
        for shard in _shards(g2_cache):
            assert torch.count_nonzero(_bits(shard)) > 0
    finally:
        for trace in (send_trace, recv_trace):
            if trace is not None:
                trace.release()
        del receiver
        del sender
