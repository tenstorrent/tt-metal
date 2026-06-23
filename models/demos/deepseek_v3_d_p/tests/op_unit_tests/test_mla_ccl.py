# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the all-gather and reduce-scatter CCLs in the chunked-prefill MLA forward
(models/demos/deepseek_v3_d_p/tt/mla/mla.py).

The MLA forward runs four CCLs, all along the TP axis (tp_factor=4):

    | id          | op             | dim | per-device in -> out          | mla.py |
    |-------------|----------------|-----|-------------------------------|--------|
    | q_a_proj_rs | reduce_scatter |  3  | [1,1,S,1536] -> [1,1,S,384]   | :718   |
    | q_ag        | all_gather     |  3  | [1,1,S,384]  -> [1,1,S,1536]  | :729   |
    | kv_ag       | all_gather     |  1  | [1,1,S,576]  -> [1,4,S,576]   | :797   |
    | o_proj_rs   | reduce_scatter |  3  | [1,1,S,7168] -> [1,1,S,1792]  | :912   |

S is the PER-DEVICE sequence length (seq_len_local). On the 8x4 Galaxy with a 5120-token
chunk (sp=8), each chip sees chunk_size_global / sp = 5120 / 8 = 640 tokens. These tests fix
S=640 so that running on a 2x4 mesh reproduces exactly the per-device CCL shapes seen on 8x4
(the CCLs are on the TP axis, which is 4 on both meshes; only SP differs, and SP just adds
independent replicas of the same per-device op).

The op call signatures mirror mla.py exactly: DRAM interleaved, the model's semaphore layout
(all_gather -> [2 sems] + barrier; reduce_scatter -> [3 sems] + barrier, persistent_output_buffers
=None), and ccl_num_links = 2 on Blackhole else 1.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config as Cfg
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# MLA feature dims (kimi_k2_6 == deepseek_v3 for all of these; only num_heads differs and it
# does not enter the CCL widths).
Q_LORA_RANK = Cfg.Q_LORA_RANK  # 1536
KVPE_DIM = Cfg.KV_LORA_RANK + Cfg.QK_ROPE_HEAD_DIM  # 512 + 64 = 576
HIDDEN_SIZE = Cfg.EMB_SIZE  # 7168
TP_FACTOR = 4  # production TP; tp_axis size on both 8x4 and 2x4
SEQ_LOCAL = 640  # per-device seq: chunk_size_global(5120) / sp(8) on the 8x4 Galaxy

# ring_mla feature dims. After the wkv_b1 absorption + rope concat, the query lives in the latent
# KV space, so Q and K share the kvpe width (d_q == d_k == 576) and V is its first kv_lora_rank
# columns (d_v = 512). The softmax scale, however, is on the *original* qk_head_dim (192), not the
# latent width -- see mla.py self.scale = qk_head_dim ** -0.5.
NUM_HEADS = Cfg.NUM_ATTENTION_HEADS  # 64 (global Q heads; per device num_heads // tp)
KV_LORA_RANK = Cfg.KV_LORA_RANK  # 512 = ring_mla head_dim_v
QK_HEAD_DIM = Cfg.QK_NOPE_HEAD_DIM + Cfg.QK_ROPE_HEAD_DIM  # 128 + 64 = 192 (drives the scale only)
# Tuned 640-chunk SDPA config (MLA_SDPA_CONFIG[640] in mla_config.py): per-device seq 640 on 8x4.
RING_MLA_Q_CHUNK = 32
RING_MLA_K_CHUNK = 640
# KV cache per-device seq length, decoupled from the Q chunk. Mirrors test_mla_chunked_prefill
# "production-50k+5k" (iters_isl=[5120]*11): seq_len_cache = 11 * chunk_size_global = 56320 global =
# 7040 per device on 8x4. The cache is sized for the whole 50k+5k prefill; the FIRST chunk
# (kv_actual_isl=0) writes only its 5120-token chunk (640/device, block-cyclic slab 0) and the rest
# stays pad.
SEQ_CACHE_LOCAL = 7040

# (id, kind, dim, per-device-input feature size on the gathered/scattered dim)
#   rs: feat is the FULL width each device holds; output is feat // tp.
#   ag dim=3: feat is the PER-DEVICE width; output is feat * tp.
#   ag dim=1: feat is the channel width (576); the gather is over the head dim (dim1, 1 -> tp).
MLA_CCL_OPS = [
    ("q_a_proj_rs", "rs", 3, Q_LORA_RANK),  # mla.py:718  all-reduce part 1
    ("q_ag", "ag", 3, Q_LORA_RANK // TP_FACTOR),  # mla.py:729  all-reduce part 2
    ("kv_ag", "ag", 1, KVPE_DIM),  # mla.py:797  gather-then-reduce
    ("o_proj_rs", "rs", 3, HIDDEN_SIZE),  # mla.py:912  output reduce-scatter
]

# Fabric/topology variants, shared by all CCL tests in this file. Each entry is
# (device_params, ttnn.Topology); device_params is consumed by the mesh_device fixture (indirect).
DEVICE_PARAMS_TOPOLOGY = [
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        ttnn.Topology.Linear,
    ),
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        ttnn.Topology.Ring,
    ),
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        ttnn.Topology.Linear,
    ),
]
DEVICE_PARAMS_TOPOLOGY_IDS = ["line", "ring", "fabric2d"]


def _make_global_semaphores(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


def _run_mla_ccl(mesh_device, kind, dim, feat, topology, num_iters=1, pcc_threshold=0.999):
    tp_axis = 1
    sp, tp = list(mesh_device.shape)
    # Production TP is 4; tp>TP_FACTOR (e.g. 1x8) is allowed for CCL connectivity/perf experiments.
    # The golden uses the live `tp`, so the op stays self-consistent as long as the scattered dim is
    # tile-aligned after the split.
    if kind == "rs":
        assert (feat // tp) % 32 == 0, f"rs scatter width {feat}//{tp} must be tile-aligned"
    num_links = 2 if is_blackhole() else 1

    # --- sub-device + semaphores (same scaffolding the model's TT_CCL uses) ---
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    barrier_sem = ttnn.create_global_semaphore(mesh_device, ccl_crs, 0)

    # --- inputs: each of the tp devices holds an independent [1,1,S,feat] tensor, laid out on
    #     dim1 so a tp-shard of dim1 hands each device its own slice. SP replicates (independent
    #     copies of the same per-device op). ---
    torch_in = torch.randn(1, tp, SEQ_LOCAL, feat, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, 1]),
    )

    try:
        for i in range(num_iters):
            logger.info(f"{kind} dim={dim} feat={feat}: iteration {i + 1}/{num_iters}")
            if kind == "ag":
                mdgs = _make_global_semaphores(mesh_device, ccl_crs, 2)
                tt_out = ttnn.experimental.all_gather_async(
                    tt_in,
                    dim=dim,
                    multi_device_global_semaphore=mdgs,
                    barrier_semaphore=barrier_sem,
                    num_links=num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=topology,
                    cluster_axis=tp_axis,
                )
            else:  # "rs"
                mdgs = _make_global_semaphores(mesh_device, ccl_crs, 3)
                tt_out = ttnn.experimental.reduce_scatter_minimal_async(
                    tt_in,
                    persistent_output_buffers=None,
                    dim=dim,
                    multi_device_global_semaphore=mdgs,
                    barrier_semaphore=barrier_sem,
                    num_links=num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=topology,
                    cluster_axis=tp_axis,
                )
            ttnn.synchronize_device(mesh_device)
            out_torch = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, dim)),
            )[
                0:1
            ]  # SP replicas are identical; keep the first

            # --- golden ---
            if kind == "rs":
                # reduce over the tp devices (dim1), then the concat-over-tp readback on `dim`
                # reassembles the full scattered sum.
                golden = torch_in.to(torch.float32).sum(dim=1, keepdim=True)
                out_torch = out_torch.to(torch.float32)
            elif dim == 3:
                # all_gather along width: every device ends up with the full concat; tp copies are
                # identical, so take the first one back out of the concat readback.
                golden = torch.cat([torch_in[:, d : d + 1] for d in range(tp)], dim=3)
                out_torch = out_torch[:, :, :, : feat * tp]
            else:  # ag dim == 1 (head gather): output is the tp slices stacked on dim1
                golden = torch_in
                out_torch = out_torch[:, :tp]

            logger.info(f"{kind} dim={dim} feat={feat}: in {list(torch_in.shape)} -> out {list(out_torch.shape)}")
            passed, msg = comp_pcc(out_torch, golden, pcc_threshold)
            logger.info(f"PCC: {msg}")
            assert passed, f"{kind} dim={dim} feat={feat} iter {i + 1}/{num_iters} FAILED: {msg}"
    finally:
        mesh_device.reset_sub_device_stall_group()


@pytest.mark.parametrize("ccl_id, kind, dim, feat", MLA_CCL_OPS, ids=[c[0] for c in MLA_CCL_OPS])
@pytest.mark.parametrize(
    "device_params, topology", DEVICE_PARAMS_TOPOLOGY, indirect=["device_params"], ids=DEVICE_PARAMS_TOPOLOGY_IDS
)
@pytest.mark.parametrize(
    "mesh_device", [(1, 4), (1, 8), (2, 4), (8, 4)], ids=["1x4", "1x8", "2x4", "8x4"], indirect=True
)
@pytest.mark.parametrize("num_iters", [1], ids=lambda n: f"iters{n}")
@pytest.mark.timeout(0)
def test_mla_ccl(mesh_device, device_params, topology, ccl_id, kind, dim, feat, num_iters):
    """Each chunked-MLA all-gather / reduce-scatter at its per-device shape (seq_local=640),
    reproducing the 8x4 per-device load on a 2x4 mesh. Runs the op `num_iters` times in a loop."""
    _run_mla_ccl(mesh_device, kind, dim, feat, topology, num_iters=num_iters)


# ---------------------------------------------------------------------------
# Ring-attention all-gather CCL (the gather inside ring_mla / ring_joint_sdpa)
# ---------------------------------------------------------------------------
# ring_mla (mla.py:627, chunked prefill) and ring_joint_scaled_dot_product_attention (mla.py:857,
# single-shot) internally ring-all-gather the K/V over the sequence dim across the SP axis (the
# ring-parallel axis; sp_axis=0, tp_axis=1 in mla.py), landing the full prefix in a persistent output
# buffer on every ring device. ring_mla gathers the single combined latent KV (kvpe, width
# KV_LORA_RANK + QK_ROPE_HEAD_DIM = 576; mla.py TT_CCL.get_mla_chunked_kv_buffer), replicated over TP.
#
# This exercises that gather in isolation via ttnn.experimental.ring_attention_all_gather_async
# (same op the ring attention kernels use; see tests/nightly/t3000/ccl/test_ring_attention_all_gather.py)
# with the model's sub-device / semaphore scaffolding and per-device shape (seq_local=640). The gather
# is on the SP axis, so unlike test_mla_ccl the ring length scales with the mesh's SP dim (1/1/2/8 on
# 1x4 / 1x8 / 2x4 / 8x4) rather than being fixed at the production value.
RING_SP_AXIS = 0  # mla.py sp_axis: the ring-parallel axis the K/V gather runs on
RING_SEQUENCE_INDEX = 2  # gather dim (sequence)
RING_HEAD_INDEX = 1


def _run_mla_ring_attention_ag(
    mesh_device, topology, num_iters=1, kvpe_dim=KVPE_DIM, dtype=ttnn.bfloat8_b, pcc_threshold=0.999
):
    sp, tp = list(mesh_device.shape)
    rp_axis = RING_SP_AXIS
    num_links = 2 if is_blackhole() else 1

    # Per-device gather slice stays SEQ_LOCAL (the 8x4 chunk/sp load); the full gathered sequence is
    # SEQ_LOCAL * sp. ring_mla's gathered KV is [1, 1, seq, kvpe_dim], replicated over TP (head=1).
    seq_global = SEQ_LOCAL * sp
    seq_per_device = SEQ_LOCAL
    ag_output_shape = [1, 1, seq_global, kvpe_dim]

    # --- sub-device + semaphores (same scaffolding the model's TT_CCL uses) ---
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # --- input: full (gathered) KV; shard the sequence over the SP/ring axis, replicate over TP. ---
    torch_kv = torch.rand(ag_output_shape).bfloat16()
    input_dims = [None, None]
    input_dims[rp_axis] = RING_SEQUENCE_INDEX  # shard sequence across the ring; TP replicated
    tt_in = ttnn.from_torch(
        torch_kv,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=input_dims),
    )

    output_dims = [None, None]
    output_dims[rp_axis] = RING_SEQUENCE_INDEX  # concat the ring shards back along the sequence
    output_dims[1 - rp_axis] = RING_HEAD_INDEX  # TP replicas land on the (size-1) head dim

    try:
        for it in range(num_iters):
            logger.info(f"ring_mla AG kvpe={kvpe_dim} seq={seq_global}: iteration {it + 1}/{num_iters}")
            # persistent output buffer holds the FULL gather on every device (replicated over the mesh).
            persistent_out = ttnn.from_torch(
                torch.zeros(ag_output_shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
            )
            mdgs = _make_global_semaphores(mesh_device, ccl_crs, 2)
            tt_out = ttnn.experimental.ring_attention_all_gather_async(
                [tt_in],
                persistent_output_buffer=[persistent_out],
                dim=RING_SEQUENCE_INDEX,
                multi_device_global_semaphore=mdgs,
                cluster_axis=rp_axis,
                mesh_device=mesh_device,
                num_links=num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
            ttnn.synchronize_device(mesh_device)

            # readback: concat the ring (SP) shards along the sequence, collapse TP replicas (head=1).
            tt_ag_out = ttnn.to_torch(
                tt_out[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=output_dims),
            )[:, :1]
            ring_chunks = torch.chunk(tt_ag_out, sp, dim=RING_SEQUENCE_INDEX)
            for ring_idx, tt_ring in enumerate(ring_chunks):
                # AG does not write a device's own local slice, so zero that window on both sides
                # before comparing the rest.
                tt_check = tt_ring.clone()
                torch.narrow(tt_check, RING_SEQUENCE_INDEX, ring_idx * seq_per_device, seq_per_device).zero_()
                golden = torch_kv.clone()
                torch.narrow(golden, RING_SEQUENCE_INDEX, ring_idx * seq_per_device, seq_per_device).zero_()

                passed, msg = comp_pcc(tt_check, golden, pcc_threshold)
                logger.info(f"ring AG iter {it + 1} ring {ring_idx}: PCC {msg}")
                assert passed, f"ring_mla AG iter {it + 1}/{num_iters} ring {ring_idx} FAILED: {msg}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "device_params, topology", DEVICE_PARAMS_TOPOLOGY, indirect=["device_params"], ids=DEVICE_PARAMS_TOPOLOGY_IDS
)
@pytest.mark.parametrize(
    "mesh_device", [(1, 4), (1, 8), (2, 4), (8, 4)], ids=["1x4", "1x8", "2x4", "8x4"], indirect=True
)
@pytest.mark.parametrize("num_iters", [1, 5, 10, 50], ids=lambda n: f"iters{n}")
@pytest.mark.timeout(0)
def test_mla_ring_attention_ccl(mesh_device, device_params, topology, num_iters):
    """Ring all-gather CCL inside chunked-prefill ring_mla (mla.py:627), in isolation, at the
    per-device kvpe shape (seq_local=640). Gather runs on the SP axis. Runs the op `num_iters` times
    in a loop."""
    _run_mla_ring_attention_ag(mesh_device, topology, num_iters=num_iters)


# ---------------------------------------------------------------------------
# Full ring_mla op (chunked-prefill attention, mla.py:627)
# ---------------------------------------------------------------------------
# ring_mla is the chunked-prefill attention op: it ring-all-gathers the latent K/V over the SP axis
# (the CCL exercised in isolation above) and runs flash attention against the gathered prefix,
# materializing V (= first head_dim_v cols of the latent KV) in-op. This runs the whole op in a loop
# at MLA's production per-device shape (seq_local=640, 64 heads, latent width 576, d_v 512), mirroring
# the mla.py call site. Adapted from tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
# (run_ring_mla_sdpa) but on the MLA mesh convention (sp_axis=0, tp_axis=1) and is_balanced=False
# (mla.py:581 asserts chunked prefill is unbalanced).


def _fa_rand(*shape):
    """Flash-attention-style random tensor: mostly N(0,1) with rare large spikes (matches the
    nightly ring-SDPA reference inputs)."""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def _torch_sdpa_reference(q, k, v, scale, is_causal=True):
    """Causal SDPA golden, chunked over heads so the [H, S, S] score matrix never materializes at
    full width. The single latent K/V head is broadcast across all Q heads (MLA shares one KV head)."""
    B, H, S, _ = q.shape
    Dv = v.shape[-1]
    out = torch.empty(B, H, S, Dv, dtype=q.dtype)
    head_chunk = 8
    for h0 in range(0, H, head_chunk):
        h1 = min(h0 + head_chunk, H)
        kh = k.expand(B, h1 - h0, S, k.shape[-1]) if k.shape[1] == 1 else k[:, h0:h1]
        vh = v.expand(B, h1 - h0, S, Dv) if v.shape[1] == 1 else v[:, h0:h1]
        out[:, h0:h1] = torch.nn.functional.scaled_dot_product_attention(
            q[:, h0:h1], kh, vh, is_causal=is_causal, scale=scale
        )
    return out


def _run_ring_mla(mesh_device, topology, num_iters=1, pcc_threshold=0.99):
    sp_axis, tp_axis = 0, 1  # mla.py convention; mesh is (sp, tp)
    sp, tp = list(mesh_device.shape)
    if sp < 2:
        pytest.skip(f"ring_mla needs SP>=2 for a ring; got sp={sp}")
    num_links = 2 if is_blackhole() else 1

    nhq = NUM_HEADS  # global Q heads; tp-sharded to num_heads//tp per device
    nhk = 1  # single shared latent K/V head
    d_q = d_k = KVPE_DIM  # 576: Q and K share the latent width
    d_v = KV_LORA_RANK  # 512: V is the first d_v cols of the latent KV
    scale = QK_HEAD_DIM**-0.5  # mla.py: scale on the original qk_head_dim, not the latent width

    # First chunk of the prefill: Q is the chunk (chunk_size_global = SEQ_LOCAL*sp, 640/device); the KV
    # cache is sized for the whole 50k+5k prefill (SEQ_CACHE_LOCAL*sp, 7040/device) but only this
    # chunk is written. kv_actual_isl=0 -> chunk-aligned (no rotation), logical_n = chunk window.
    chunk_global = SEQ_LOCAL * sp
    seq_cache = SEQ_CACHE_LOCAL * sp
    kv_actual_isl = 0
    logical_n = kv_actual_isl + chunk_global
    cache_batch = 1
    kv_cache_batch_idx = 0

    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_compute_grid = (grid.x - 1, grid.y)  # mla.py ring_sdpa_compute_grid
    ccl_core_grid_offset = (grid.x - 1, 0)  # mla.py TT_CCL.ring_attention_ccl_core_grid_offset

    # --- sub-device + semaphores (the model's ring-attention scaffolding) ---
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    ccl_sems = _make_global_semaphores(mesh_device, ccl_crs, 2)  # ring attention uses 2 sems

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=RING_MLA_Q_CHUNK,
        k_chunk_size=RING_MLA_K_CHUNK,
        exp_approx_mode=False,
    )

    # --- inputs ---
    torch.manual_seed(1234)
    Q = _fa_rand(1, nhq, chunk_global, d_q)  # the chunk's queries
    KV_chunk = _fa_rand(1, nhk, chunk_global, d_k)  # the chunk's latent K/V (natural token order)
    V_chunk = KV_chunk[:, :, :, :d_v]

    # Place the chunk into the oversized cache block-cyclically: with kv_actual_isl=0 every chunk token
    # lands in slab 0, and device d holds the contiguous token range [d*SEQ_LOCAL, (d+1)*SEQ_LOCAL).
    # Sharding the cache seq contiguously over SP then hands device d exactly that range at its slab 0
    # (cache rows 0..SEQ_LOCAL-1); the remaining SEQ_CACHE_LOCAL-SEQ_LOCAL rows per device stay pad.
    KV_cache = torch.zeros(cache_batch, nhk, seq_cache, d_k, dtype=KV_chunk.dtype)
    for d in range(sp):
        KV_cache[kv_cache_batch_idx, :, d * SEQ_CACHE_LOCAL : d * SEQ_CACHE_LOCAL + SEQ_LOCAL, :] = KV_chunk[
            0, :, d * SEQ_LOCAL : (d + 1) * SEQ_LOCAL, :
        ]

    q_shard_dims = [None, None]
    q_shard_dims[sp_axis] = 2  # chunk seq across the ring
    q_shard_dims[tp_axis] = 1  # heads across TP
    kv_shard_dims = [None, None]
    kv_shard_dims[sp_axis] = 2  # cache seq across the ring; single head replicated over TP

    tt_Q = ttnn.from_torch(
        Q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=q_shard_dims),
    )
    tt_KV = ttnn.from_torch(
        KV_cache,
        dtype=ttnn.bfloat8_b,  # mla.py writes the bf8 latent KV to the cache
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=kv_shard_dims),
    )
    # Gathered-KV scratch buffer, sized to the full cache and replicated over the whole mesh
    # (mla.py _chunked_kv_buf = get_mla_chunked_kv_buffer(cache_batch=1, seq_len=seq_len_cache, ...)).
    persistent_kv = ttnn.from_torch(
        torch.zeros(cache_batch, nhk, seq_cache, d_k),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
    )

    try:
        reference_output = None
        tt_out_torch = None
        for it in range(num_iters):
            logger.info(
                f"ring_mla nhq={nhq} chunk={chunk_global} cache={seq_cache} (per-dev {SEQ_LOCAL}/{SEQ_CACHE_LOCAL}) "
                f"kv_actual_isl={kv_actual_isl} logical_n={logical_n}: iteration {it + 1}/{num_iters}"
            )
            tt_out, _ = ttnn.transformer.ring_mla(
                tt_Q,
                tt_KV,
                persistent_output_buffer_kv=persistent_kv,
                head_dim_v=d_v,
                logical_n=logical_n,
                is_balanced=False,
                program_config=program_config,
                scale=scale,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=ccl_sems,
                num_links=num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=topology,
                subdevice_id=worker_sub_device_id,
                ccl_core_grid_offset=ccl_core_grid_offset,
                use_column_major_ccl=True,
                kv_cache_batch_idx=kv_cache_batch_idx,
                kv_actual_isl=kv_actual_isl,
            )
            # readback: output is the Q chunk; concat seq over SP (dim2) and heads over TP (dim1).
            tt_out_torch = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(q_shard_dims[0], q_shard_dims[1])
                ),
            )[:, :, :chunk_global, :d_v]

            # Across iterations the op must be deterministic (same inputs, fixed buffer address).
            if reference_output is None:
                reference_output = tt_out_torch
            elif not torch.equal(reference_output, tt_out_torch):
                max_diff = (reference_output - tt_out_torch).abs().max().item()
                pytest.fail(f"ring_mla iter {it + 1}/{num_iters} differs from iter 1, max diff={max_diff}")
        ttnn.synchronize_device(mesh_device)

        # First chunk (kv_actual_isl=0): plain causal SDPA of the chunk against itself.
        gt = _torch_sdpa_reference(Q, KV_chunk, V_chunk, scale=scale, is_causal=True)
        passed, msg = comp_pcc(gt, tt_out_torch, pcc_threshold)
        logger.info(f"ring_mla PCC: {msg}")
        assert passed, f"ring_mla FAILED (after {num_iters} iters): {msg}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "device_params, topology", DEVICE_PARAMS_TOPOLOGY, indirect=["device_params"], ids=DEVICE_PARAMS_TOPOLOGY_IDS
)
@pytest.mark.parametrize(
    "mesh_device", [(1, 4), (1, 8), (2, 4), (8, 4)], ids=["1x4", "1x8", "2x4", "8x4"], indirect=True
)
@pytest.mark.parametrize("num_iters", [1, 5, 10, 50], ids=lambda n: f"iters{n}")
@pytest.mark.timeout(0)
def test_ring_mla(mesh_device, device_params, topology, num_iters):
    """Full chunked-prefill ring_mla op (mla.py:627) for the FIRST chunk of the production-50k+5k
    scenario: Q chunk 640/device (chunk_size_global = 640*sp), KV cache oversized to 7040/device
    (the whole 50k+5k prefill window), kv_actual_isl=0. Ring runs on the SP axis (skipped when sp<2).
    Runs the op `num_iters` times in a loop; checks iteration-to-iteration determinism and a torch
    SDPA PCC golden."""
    _run_ring_mla(mesh_device, topology, num_iters=num_iters)
