# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.6 Multi-Latent Attention (MLA) chunked-prefill test against a torch oracle.

For each total ISL in {5K, 10K, ..., 55K}, the test compares:
  - mla_full_attention      : one-shot causal MLA over the full sequence (torch, the oracle)
  - mla_chunked_prefill     : iterative MLA in 5K chunks where each chunk's SDPA runs on
                              device via ttnn.transformer.ring_joint_scaled_dot_product_attention
                              (q_start_idx + balanced K/V layout); Q/K/V projection, KVPE
                              cache materialization, kv_b2 expansion, and o_proj stay on torch.

Kimi K2.6 attention dims are taken from moonshotai/Kimi-K2.6 config.json (text_config);
the attention stack matches the DeepSeek V3 MLA family, so we reuse
DeepseekV3Attention with Kimi-K2.6 hyperparameters and random weights.

The device SDPA shape mirrors mla_100k in tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py:
d_q=d_k=kv_lora_rank+qk_rope_head_dim=576 (latent K head dim), d_v=v_head_dim=128 per head.
K stays nhk=1 (broadcast across nhq) but V is pre-projected per-head via kv_b2 because the
op requires NQH==NVH. Mathematically equivalent to the absorbed form used by the torch oracle
(``attn @ V_per_head == (attn @ V_latent) @ kv_b2``) — moves the kv_b2 contraction to the
right of softmax so the SDPA itself shrinks to d_v=128.
"""

from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from ttnn.operations.ccl import Topology

import ttnn
from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    apply_rotary_pos_emb,
)
from tests.nightly.sdpa_perf_utils import MeshConfig
from tests.ttnn.utils_for_testing import assert_with_pcc

MESH_CONFIG = MeshConfig.detect()

# Kimi K2.6 attention dims (text_config slice of moonshotai/Kimi-K2.6 config.json).
KIMI_K26_TEXT_CONFIG = dict(
    hidden_size=7168,
    num_attention_heads=64,
    num_key_value_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rms_norm_eps=1e-5,
    rope_theta=50000.0,
    max_position_embeddings=262144,
    rope_scaling={
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    num_hidden_layers=1,
    vocab_size=163840,
)

CHUNK_SIZE = 5 * 1024
MAX_TOTAL_SEQ = 55 * 1024

# CPU SDPA materializes Q@K^T in fp32 — chunk both axes to keep peak working set bounded.
HEAD_CHUNK = 8
SEQ_CHUNK = 2048


def build_kimi_k26_attention(seed: int = 42) -> DeepseekV3Attention:
    """Instantiate a DeepseekV3Attention sized for Kimi K2.6 with deterministic random weights."""
    config = DeepseekV3Config(**KIMI_K26_TEXT_CONFIG)
    torch.manual_seed(seed)
    attn = DeepseekV3Attention(config, layer_idx=0)

    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=config.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, DeepseekV3RMSNorm) and hasattr(m, "weight"):
            nn.init.ones_(m.weight)

    attn.apply(init)
    return attn.eval().to(torch.bfloat16)


def _project_qkv_for_chunk(
    attn: DeepseekV3Attention,
    hidden_chunk: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Q (absorbed into latent space) and KVPE for one chunk.

    Args:
        hidden_chunk: [bsz, chunk_len, hidden_size]
        position_ids: [bsz, chunk_len] absolute positions for RoPE

    Returns:
        q_latent:   [bsz, num_heads, chunk_len, kv_lora_rank + qk_rope_head_dim]
        kvpe_chunk: [bsz, 1,         chunk_len, kv_lora_rank + qk_rope_head_dim]
    """
    bsz, chunk_len, _ = hidden_chunk.shape

    # Q LoRA + absorption into latent K space (so attention is over [kv_lora_rank + qk_rope_head_dim]).
    q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_chunk)))
    q = q.view(bsz, chunk_len, attn.num_heads, attn.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

    kv_b1 = attn.kv_b_proj.weight.view(attn.num_heads, -1, attn.kv_lora_rank)[:, : attn.qk_nope_head_dim]
    q_nope = torch.matmul(q_nope, kv_b1)

    # KV LoRA → latent k_nope + rope k_pe (shared across heads).
    compressed = attn.kv_a_proj_with_mqa(hidden_chunk)
    compressed, k_pe = torch.split(compressed, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, 1, chunk_len, attn.qk_rope_head_dim)
    k_nope = attn.kv_a_layernorm(compressed).view(bsz, 1, chunk_len, attn.kv_lora_rank)

    # RoPE: cos/sin sized to cover the largest absolute position touched by this chunk.
    max_pos = int(position_ids.max().item()) + 1
    cos, sin = attn.rotary_emb(k_nope, seq_len=max_pos, meta_style=True)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, meta_style=True)

    q_latent = k_pe.new_empty(bsz, attn.num_heads, chunk_len, attn.kv_lora_rank + attn.qk_rope_head_dim)
    q_latent[:, :, :, : attn.kv_lora_rank] = q_nope
    q_latent[:, :, :, attn.kv_lora_rank :] = q_pe

    kvpe_chunk = k_pe.new_empty(bsz, 1, chunk_len, attn.kv_lora_rank + attn.qk_rope_head_dim)
    kvpe_chunk[:, :, :, : attn.kv_lora_rank] = k_nope
    kvpe_chunk[:, :, :, attn.kv_lora_rank :] = k_pe
    return q_latent, kvpe_chunk


def _chunked_sdpa_and_oproj(
    attn: DeepseekV3Attention,
    q_latent: torch.Tensor,
    kvpe_cache: torch.Tensor,
    abs_q_start: int,
) -> torch.Tensor:
    """
    Run causal SDPA for Q[abs_q_start : abs_q_start + q_len) against K[0 : abs_q_start + q_len),
    then expand V from latent to v_head_dim and apply o_proj. Head- and Q-seq-chunked
    internally to keep CPU SDPA's working set bounded.

    Args:
        q_latent:    [bsz, num_heads, q_len, kv_lora_rank + qk_rope_head_dim]
        kvpe_cache:  [bsz, 1, k_len, kv_lora_rank + qk_rope_head_dim], with k_len >= abs_q_start + q_len
        abs_q_start: absolute position of q_latent[..., 0, :] in the full sequence

    Returns: [bsz, q_len, num_heads * v_head_dim] after o_proj.
    """
    bsz, num_heads, q_len, _ = q_latent.shape
    k_total = abs_q_start + q_len
    assert kvpe_cache.shape[-2] >= k_total, "KVPE cache must cover up to last Q position"
    value_cache = kvpe_cache[..., : attn.kv_lora_rank]

    out_latent = torch.empty(bsz, num_heads, q_len, attn.kv_lora_rank, dtype=q_latent.dtype, device=q_latent.device)
    # PyTorch SDPA's is_causal=True aligns Q at top-left of the K window (mask[i,j] = j<=i),
    # which is only correct when abs_q_start == 0. For cached/chunked prefill we need the
    # absolute-position causal mask: Q at absolute position p attends K positions 0..p.
    for h0 in range(0, num_heads, HEAD_CHUNK):
        h1 = min(h0 + HEAD_CHUNK, num_heads)
        for s0 in range(0, q_len, SEQ_CHUNK):
            s1 = min(s0 + SEQ_CHUNK, q_len)
            k_upto = abs_q_start + s1  # exclusive
            q_h = q_latent[:, h0:h1, s0:s1, :]
            k_h = kvpe_cache[:, :, :k_upto, :].expand(bsz, h1 - h0, -1, -1)
            v_h = value_cache[:, :, :k_upto, :].expand(bsz, h1 - h0, -1, -1)
            q_abs = torch.arange(abs_q_start + s0, abs_q_start + s1, device=q_h.device)
            k_abs = torch.arange(k_upto, device=q_h.device)
            attn_mask = k_abs.unsqueeze(0) <= q_abs.unsqueeze(1)  # [Lq, Lk] bool
            out_latent[:, h0:h1, s0:s1, :] = F.scaled_dot_product_attention(
                q_h, k_h, v_h, attn_mask=attn_mask, scale=attn.softmax_scale
            )

    return _oproj_from_latent(attn, out_latent)


def _oproj_from_latent(attn: DeepseekV3Attention, out_latent: torch.Tensor) -> torch.Tensor:
    """Expand latent SDPA output per-head via kv_b2 and apply o_proj.

    out_latent: [bsz, num_heads, q_len, kv_lora_rank]
    Returns:    [bsz, q_len, num_heads * v_head_dim] after o_proj.
    """
    bsz, _, q_len, _ = out_latent.shape
    kv_b2 = attn.kv_b_proj.weight.view(attn.num_heads, -1, attn.kv_lora_rank)[:, -attn.v_head_dim :].transpose(1, 2)
    out = torch.matmul(out_latent, kv_b2)
    out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, attn.num_heads * attn.v_head_dim)
    return attn.o_proj(out)


def _oproj_from_per_head(attn: DeepseekV3Attention, out_per_head: torch.Tensor) -> torch.Tensor:
    """Reshape per-head SDPA output and apply o_proj.

    out_per_head: [bsz, num_heads, q_len, v_head_dim]
    Returns:      [bsz, q_len, num_heads * v_head_dim] after o_proj.
    """
    bsz, _, q_len, _ = out_per_head.shape
    out = out_per_head.transpose(1, 2).contiguous().reshape(bsz, q_len, attn.num_heads * attn.v_head_dim)
    return attn.o_proj(out)


def _kvpe_to_v_per_head(attn: DeepseekV3Attention, kvpe: torch.Tensor) -> torch.Tensor:
    """Project the latent K-nope block of KVPE into per-head V.

    kvpe:    [bsz, 1, k_len, kv_lora_rank + qk_rope_head_dim]
    Returns: [bsz, num_heads, k_len, v_head_dim]
    """
    bsz, _, k_len, _ = kvpe.shape
    k_nope = kvpe[..., : attn.kv_lora_rank]  # [bsz, 1, k_len, kv_lora_rank]
    kv_b2 = attn.kv_b_proj.weight.view(attn.num_heads, -1, attn.kv_lora_rank)[:, -attn.v_head_dim :].transpose(1, 2)
    # Per-head matmul: broadcast k_nope across heads, contract over kv_lora_rank.
    k_nope_bcast = k_nope.expand(bsz, attn.num_heads, k_len, attn.kv_lora_rank)
    return torch.matmul(k_nope_bcast, kv_b2)


def _create_global_semaphores(mesh_device, cores, initial_value=0):
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


def _open_mesh_for_chunked_prefill(mesh_config: MeshConfig):
    """Open the mesh, set fabric, wire the worker sub-device + semaphores.

    Mirrors run_ring_joint_sdpa_chunked from tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
    so the ring SDPA op sees the same setup the chunked-prefill path was developed against.
    """
    sp_size = mesh_config.sp_size
    use_ring = sp_size > 2
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if use_ring else ttnn.FabricConfig.FABRIC_1D
    topology = Topology.Ring if use_ring else Topology.Linear

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    mesh_shape = ttnn.MeshShape(mesh_config.tp_size, sp_size)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    full_compute_grid = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    semaphores = _create_global_semaphores(mesh_device, ccl_sub_device_crs)
    return mesh_device, worker_sub_device_id, semaphores, topology


def _to_balanced_growing(
    src_full: torch.Tensor,
    last_uploaded_chunk: int,
    sp_size: int,
    chunk_size: int,
    slab_rows: int,
) -> torch.Tensor:
    """Permute src_full into a balanced per-device layout for the populated prefix.

    Balanced layout: device d's local K/V holds, for each chunk c, the slab covering global
    rows [c*chunk_size + d*slab_rows, c*chunk_size + (d+1)*slab_rows) — the slab whose
    global Q positions sit on device d.

    Returns a tensor of seq length (last_uploaded_chunk + 1) * chunk_size — only the
    populated slabs, no trailing padding. Contiguous dim-2 split across sp_size devices
    yields the intended per-device layout. The K shape grows by one chunk's worth of rows
    on every call; the kernel derives kv_local_padded_N from input_k's shape directly.
    """
    n_populated = last_uploaded_chunk + 1
    K_local_curr = n_populated * slab_rows
    populated_len = n_populated * chunk_size
    b_, nh_, _, d_ = src_full.shape
    perm = torch.zeros(b_, nh_, populated_len, d_, dtype=src_full.dtype, device=src_full.device)
    for dev in range(sp_size):
        for c in range(n_populated):
            local_start = dev * K_local_curr + c * slab_rows
            global_start = c * chunk_size + dev * slab_rows
            perm[:, :, local_start : local_start + slab_rows, :] = src_full[
                :, :, global_start : global_start + slab_rows, :
            ]
    return perm


def mla_full_attention(attn: DeepseekV3Attention, hidden_states: torch.Tensor) -> torch.Tensor:
    """One-shot causal MLA over the full sequence — the per-token golden output."""
    bsz, total_seq, _ = hidden_states.shape
    position_ids = torch.arange(total_seq, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
    q_latent, kvpe_full = _project_qkv_for_chunk(attn, hidden_states, position_ids)
    return _chunked_sdpa_and_oproj(attn, q_latent, kvpe_full, abs_q_start=0)


# Tile sizes for the ring SDPA call. Match the mla_100k defaults from
# tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py — known-good for d_q=576.
DEVICE_SDPA_Q_CHUNK = 160
DEVICE_SDPA_K_CHUNK = 320


def mla_chunked_prefill(
    attn: DeepseekV3Attention,
    hidden_states: torch.Tensor,
    chunk_size: int,
    mesh_config: MeshConfig = MESH_CONFIG,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative chunked prefill where each chunk's SDPA runs on device via
    ttnn.transformer.ring_joint_scaled_dot_product_attention with q_start_idx and a
    balanced K/V cache layout. Torch handles the Q/KVPE projection, the KVPE cache,
    the kv_b2 latent->per-head V expansion, and o_proj.

    Concatenated outputs match the one-shot MLA path (mla_full_attention) up to
    bfloat16/bfloat8_b quantization and ring-SDPA ordering tolerances.

    Args:
        hidden_states: [bsz, total_seq, hidden_size]
        chunk_size:    new-ISL per step. Must be divisible by sp_size and total_seq
                       must be a multiple of chunk_size. slab_rows=chunk_size/sp_size
                       must be tile-aligned (multiple of 32).

    Returns:
        output:     [bsz, total_seq, hidden_size]
        kvpe_cache: [bsz, 1, total_seq, kv_lora_rank + qk_rope_head_dim]
    """
    bsz, total_seq, _ = hidden_states.shape
    nhq = attn.num_heads
    cache_dim = attn.kv_lora_rank + attn.qk_rope_head_dim  # 576: full latent K head dim
    # V is pre-projected per-head via kv_b2 (non-absorbed form). The op requires NQH==NVH,
    # so V can't be the broadcast latent block — push the kv_b2 contraction to the right of
    # softmax, see module docstring.
    v_dim = attn.v_head_dim  # 128 per head

    sp_size = mesh_config.sp_size
    tp_size = mesh_config.tp_size

    if sp_size < 2:
        pytest.skip(f"Ring joint chunked prefill requires sp_size >= 2, got {sp_size}")

    assert total_seq % chunk_size == 0, f"total_seq {total_seq} must be a multiple of chunk_size {chunk_size}"
    assert chunk_size % sp_size == 0, f"chunk_size {chunk_size} must divide sp_size {sp_size}"
    n_chunks = total_seq // chunk_size
    slab_rows = chunk_size // sp_size
    assert slab_rows % 32 == 0, f"slab_rows {slab_rows} not tile-aligned (TILE_HEIGHT=32)"
    assert nhq % tp_size == 0, f"nhq={nhq} must be divisible by tp_size={tp_size}"

    kvpe_cache = torch.zeros(bsz, 1, total_seq, cache_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    mesh_device, worker_sub_device_id, semaphores, topology = _open_mesh_for_chunked_prefill(mesh_config)

    try:
        sp_axis = 1
        tp_axis = 0
        sdpa_compute_grid = (mesh_config.sdpa_cols, mesh_config.grid_rows)
        ccl_column = mesh_config.ccl_column
        num_links = 2

        # Q and V: TP-shard heads, SP-shard sequence (nhv==nhq with per-head V).
        sdpa_qv_shard_dims = [None, None]
        sdpa_qv_shard_dims[sp_axis] = 2
        if tp_size > 1:
            sdpa_qv_shard_dims[tp_axis] = 1

        # K has nhk=1 → no TP shard on heads; SP-shard sequence (balanced layout).
        sdpa_k_shard_dims = [None, None]
        sdpa_k_shard_dims[sp_axis] = 2

        # Joint Q/V: empty seq, replicated across SP; TP-shard heads only when tp>1.
        sdpa_joint_qv_shard_dims = [None, None]
        if tp_size > 1:
            sdpa_joint_qv_shard_dims[tp_axis] = 1
        # Joint K: nhk=1, never TP-sharded.
        sdpa_joint_k_shard_dims = [None, None]

        # Persistent K output: nhk=1, fully replicated.
        persistent_k_shard_dims = [None, None]
        # Persistent V output: nhv=nhq, TP-shard heads when tp>1, SP-replicated.
        persistent_v_shard_dims = [None, None]
        if tp_size > 1:
            persistent_v_shard_dims[tp_axis] = 1

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=DEVICE_SDPA_Q_CHUNK,
            k_chunk_size=DEVICE_SDPA_K_CHUNK,
            exp_approx_mode=False,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        q_dtype = ttnn.bfloat16
        kv_dtype = ttnn.bfloat8_b  # matches mla_100k convention

        # Empty joint tensors (wan-style: joint_seq_len=0 means the joint output is ignored).
        joint_seq_len = 0
        joint_Q = torch.zeros(bsz, nhq, joint_seq_len, cache_dim, dtype=torch.bfloat16)
        joint_K = torch.zeros(bsz, 1, joint_seq_len, cache_dim, dtype=torch.bfloat16)
        joint_V = torch.zeros(bsz, nhq, joint_seq_len, v_dim, dtype=torch.bfloat16)

        tt_joint_Q = ttnn.from_torch(
            joint_Q,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_qv_shard_dims
            ),
        )
        tt_joint_K = ttnn.from_torch(
            joint_K,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_k_shard_dims
            ),
        )
        tt_joint_V = ttnn.from_torch(
            joint_V,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_qv_shard_dims
            ),
        )

        main_row_dim = sdpa_qv_shard_dims[0] if sdpa_qv_shard_dims[0] is not None else -1
        main_col_dim = sdpa_qv_shard_dims[1] if sdpa_qv_shard_dims[1] is not None else -1

        outs = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size

            chunk = hidden_states[:, start:end, :]
            position_ids = torch.arange(start, end, dtype=torch.long, device=chunk.device).unsqueeze(0).expand(bsz, -1)
            q_latent, kvpe_chunk = _project_qkv_for_chunk(attn, chunk, position_ids)
            kvpe_cache[:, :, start:end, :] = kvpe_chunk

            # K stays in latent form (kvpe, d=576, nhk=1). V is per-head pre-projected via
            # kv_b2 from the k_nope subblock (d=v_head_dim=128, nhv=nhq) because the SDPA op
            # requires nhv==nhq.
            v_per_head_cache = _kvpe_to_v_per_head(attn, kvpe_cache)  # [bsz, nhq, total_seq, v_dim]

            K_balanced = _to_balanced_growing(kvpe_cache, i, sp_size, chunk_size, slab_rows)
            V_balanced = _to_balanced_growing(v_per_head_cache, i, sp_size, chunk_size, slab_rows)

            tt_Q = ttnn.from_torch(
                q_latent,
                dtype=q_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_qv_shard_dims
                ),
            )
            tt_K = ttnn.from_torch(
                K_balanced,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_shard_dims
                ),
            )
            tt_V = ttnn.from_torch(
                V_balanced,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_qv_shard_dims
                ),
            )

            # AllGather output buffer must match the post-gather K/V size for this chunk:
            # N_global == N_local_kv * sp_size == (i+1) * chunk_size.
            persistent_output_buffer_k = ttnn.from_torch(
                torch.zeros(bsz, 1, end, cache_dim),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_shard_dims
                ),
            )
            persistent_output_buffer_v = ttnn.from_torch(
                torch.zeros(bsz, nhq, end, v_dim),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_v_shard_dims
                ),
            )

            tt_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_output_buffer_k=persistent_output_buffer_k,
                persistent_output_buffer_v=persistent_output_buffer_v,
                joint_strategy="rear",
                logical_n=end,
                # Chunk 0 is N_local_q == N_local_kv: the op treats it as non-chunked, so
                # is_causal=True drives the standard causal mask. Chunks 1+ have
                # N_local_q < N_local_kv and trigger the chunked-prefill path implicitly.
                is_causal=(i == 0),
                is_balanced=False,
                scale=attn.softmax_scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=semaphores,
                num_links=num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=topology,
                subdevice_id=worker_sub_device_id,
                ccl_core_grid_offset=(ccl_column, 0),
                use_column_major_ccl=True,
            )

            out_per_head = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )
            # Slice seq (chunk_size). Depth (v_head_dim=128) is tile-aligned and unpadded.
            out_per_head = out_per_head[:, :, :chunk_size, :v_dim]

            outs.append(_oproj_from_per_head(attn, out_per_head))

        logger.info(
            f"Chunked-prefill device SDPA done: n_chunks={n_chunks}, sp_size={sp_size}, "
            f"tp_size={tp_size}, num_program_cache_entries={mesh_device.num_program_cache_entries()}"
        )
        return torch.cat(outs, dim=1), kvpe_cache

    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "total_seq",
    list(range(CHUNK_SIZE, MAX_TOTAL_SEQ + 1, CHUNK_SIZE)),
    ids=lambda s: f"seq{s // 1024}k",
)
def test_kimi_k26_mla_chunked_prefill(total_seq: int) -> None:
    if MESH_CONFIG.sp_size < 2:
        pytest.skip(f"Ring joint chunked prefill requires sp_size >= 2, got {MESH_CONFIG.sp_size}")

    bsz = 1
    hidden_size = KIMI_K26_TEXT_CONFIG["hidden_size"]

    attn = build_kimi_k26_attention(seed=42)

    torch.manual_seed(0)
    hidden_states = torch.randn(bsz, total_seq, hidden_size, dtype=torch.bfloat16)

    logger.info(
        f"Kimi K2.6 MLA chunked-prefill: total_seq={total_seq} "
        f"({total_seq // 1024}K), chunk_size={CHUNK_SIZE} ({CHUNK_SIZE // 1024}K), "
        f"n_chunks={total_seq // CHUNK_SIZE}, sp_size={MESH_CONFIG.sp_size}, tp_size={MESH_CONFIG.tp_size}"
    )

    with torch.no_grad():
        ref_output = mla_full_attention(attn, hidden_states)
        chunked_output, _ = mla_chunked_prefill(attn, hidden_states, CHUNK_SIZE)

    _, pcc_msg = assert_with_pcc(ref_output, chunked_output, 0.99)
    logger.info(f"PCC(full vs chunked) at seq{total_seq // 1024}k: {pcc_msg}")
