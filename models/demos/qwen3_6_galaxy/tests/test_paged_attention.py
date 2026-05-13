# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for paged KV cache in Qwen3.6-27B on BH GLX 8×4 mesh — Task T13.

Tests are written RED-first (before implementation) and run GREEN after
paged attention is implemented in the attention/decoder/model blocks.

All tests require hardware (8×4 BH GLX mesh).

Run all tests:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    mkdir -p /tmp/qwen36_logs
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_paged_attention.py -x -s -v \\
        2>&1 | tee /tmp/qwen36_logs/t13_all.log

Individual tests:
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_paged_attention.py \\
        -k test_paged_kv_cache_pcc_matches_non_paged_on_8x4 -x -s -v \\
        2>&1 | tee /tmp/qwen36_logs/t13_test1.log

Logging:
    All runs: 2>&1 | tee /tmp/qwen36_logs/t13_<step>.log
"""

import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_LAYER_IDX = 3  # full_attention layer (pattern [lin,lin,lin,full]×16 → index 3)
_B = 1
_T_PREFILL = 32
_H = 5120
_N_Q = 24
_N_KV = 4
_HEAD_DIM = 256

# Paged cache constants matching the paged_attention_config in model args
_BLOCK_SIZE = 64  # tokens per page block
_MAX_SEQ_LEN_FOR_PAGED = 512  # max seq len for these unit tests
_MAX_BATCH = 1  # max batch size for these unit tests

# PCC thresholds
_PCC_THRESH_MATCH = 0.999  # paged vs non-paged (same math, very tight)
_PCC_THRESH_REF = 0.99  # paged vs CPU reference

_SELF_ATTN_PREFIX = f"model.language_model.layers.{_LAYER_IDX}.self_attn"


# ---------------------------------------------------------------------------
# Fixture: full 8×4 BH GLX mesh
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the full 8×4 fabric mesh with FABRIC_1D_RING topology."""
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_attn_weights():
    """Load layer-3 self-attention weights from safetensors."""
    from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_tensors

    keys = [
        f"{_SELF_ATTN_PREFIX}.q_proj.weight",
        f"{_SELF_ATTN_PREFIX}.k_proj.weight",
        f"{_SELF_ATTN_PREFIX}.v_proj.weight",
        f"{_SELF_ATTN_PREFIX}.o_proj.weight",
        f"{_SELF_ATTN_PREFIX}.q_norm.weight",
        f"{_SELF_ATTN_PREFIX}.k_norm.weight",
    ]
    raw = load_qwen36_tensors(keys)
    return {
        "q_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.q_proj.weight"].float(),
        "k_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.k_proj.weight"].float(),
        "v_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.v_proj.weight"].float(),
        "o_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.o_proj.weight"].float(),
        "q_norm.weight": raw[f"{_SELF_ATTN_PREFIX}.q_norm.weight"].float(),
        "k_norm.weight": raw[f"{_SELF_ATTN_PREFIX}.k_norm.weight"].float(),
    }


def _make_random_hidden_state(T: int, seed: int = 42) -> torch.Tensor:
    """Random bfloat16 hidden state [B, T, H]."""
    torch.manual_seed(seed)
    return torch.randn(_B, T, _H, dtype=torch.bfloat16)


def _send_to_device(t: torch.Tensor, mesh_device, dtype=None):
    """Send a torch tensor to device, replicated across all 32 chips."""
    import ttnn

    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _gather_replicated(tt_tensor, mesh_device):
    """Gather a replicated TTNN tensor back to host (first device's data)."""
    import ttnn

    all_devices = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return all_devices[0:1]  # [1, ...]


def _build_rope_cos_sin(T: int):
    """Build MRoPE cos/sin for text-only inference at positions [0, T)."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=_HEAD_DIM,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    return cos, sin  # float32, [1, T, 64]


def _make_page_table(batch_size: int, seq_len: int, block_size: int, max_num_blocks: int) -> torch.Tensor:
    """Build a sequential page table for testing.

    Each sequence gets sequential physical block indices starting at
    batch_idx * ceil(seq_len / block_size). Non-overlapping allocations
    guarantee user isolation.

    Returns:
        torch.int32 [batch_size, max_blocks_per_seq]
    """
    import math

    max_blocks_per_seq = math.ceil(seq_len / block_size)
    page_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32)
    for b in range(batch_size):
        for blk in range(max_blocks_per_seq):
            page_table[b, blk] = b * max_blocks_per_seq + blk
    return page_table


def _build_paged_kv_cache(mesh_device, n_kv_per_col: int, head_dim: int, block_size: int, max_num_blocks: int):
    """Allocate a paged KV cache on-device.

    Paged cache shape: [max_num_blocks, n_kv_per_col, block_size, head_dim]
    Sharded across 4 mesh cols on dim=1 (n_kv), same as non-paged.

    Returns (k_cache_paged, v_cache_paged) as TTNN device tensors.
    """
    import ttnn

    cluster_shape = list(mesh_device.shape)  # [8, 4]
    n_kv = n_kv_per_col * cluster_shape[1]  # total KV heads

    k_zeros = torch.zeros(max_num_blocks, n_kv, block_size, head_dim)
    v_zeros = torch.zeros(max_num_blocks, n_kv, block_size, head_dim)

    col_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=cluster_shape)
    caches = []
    for t in [k_zeros, v_zeros]:
        caches.append(
            ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_shard,
            )
        )
    return caches[0], caches[1]


def _send_page_table_to_device(page_table: torch.Tensor, mesh_device):
    """Send an int32 page_table [batch, max_blocks_per_seq] to device (replicated)."""
    import ttnn

    return ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_attn_with_config(
    mesh_device,
    sd: dict,
    use_paged_kv_cache: bool,
    kv_cache_max_seq_len: int = 512,
    block_size: int = _BLOCK_SIZE,
    max_num_blocks: int = None,
):
    """Build a TtQwen36GatedAttention with or without paged KV cache.

    When use_paged_kv_cache=True, allocates paged cache internally.
    Returns the attention instance.
    """
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_device)
    attn = TtQwen36GatedAttention(
        mesh_device=mesh_device,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
        kv_cache_max_seq_len=kv_cache_max_seq_len,
        use_paged_kv_cache=use_paged_kv_cache,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
    )
    return attn, args


# ---------------------------------------------------------------------------
# Test 1: paged vs non-paged PCC match > 0.999 (same math, different cache layout)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_paged_kv_cache_pcc_matches_non_paged_on_8x4(mesh_8x4):
    """Paged and non-paged attention produce PCC > 0.999 on identical inputs.

    Both instances:
    - Use same layer weights
    - Prefill with same T=32 tokens (seeds the KV cache)
    - Run decode at same position

    The paged path uses paged_fill_cache (prefill) + paged_fused_update_cache
    (decode) + paged_scaled_dot_product_attention_decode (SDPA).
    The non-paged path uses fill_cache / update_cache / regular SDPA.

    Since both compute identical mathematical operations (just with different
    physical memory layouts for KV storage), PCC > 0.999 is expected.
    """
    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup

    print("\n[test1] Loading weights...")
    sd = _load_attn_weights()
    block_size = _BLOCK_SIZE
    max_seq = _MAX_SEQ_LEN_FOR_PAGED
    import math

    max_num_blocks = math.ceil(max_seq / block_size) * _MAX_BATCH

    # Build both attention instances with same weights
    attn_nonpaged, args = _build_attn_with_config(
        mesh_8x4,
        sd,
        use_paged_kv_cache=False,
        kv_cache_max_seq_len=max_seq,
    )
    attn_paged, _ = _build_attn_with_config(
        mesh_8x4,
        sd,
        use_paged_kv_cache=True,
        kv_cache_max_seq_len=max_seq,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
    )

    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_B, max_seq_len=max_seq)

    # ----- Prefill -----
    T_pre = _T_PREFILL
    x_pre = _make_random_hidden_state(T_pre, seed=100)
    cos_pre, sin_pre = rope.get_cos_sin_for_prefill(seq_len=T_pre)
    x_pre_tt = _send_to_device(x_pre, mesh_8x4)

    # Build page table for paged prefill: user 0 uses blocks [0, 1, ..., T_pre/block_size)
    page_table = _make_page_table(_B, T_pre, block_size, max_num_blocks)
    pt_tt = _send_page_table_to_device(page_table, mesh_8x4)

    # Non-paged prefill (fills internal cache)
    out_pre_np = attn_nonpaged.forward_prefill(x_pre_tt, rot_mats=(cos_pre, sin_pre), kv_cache=None, user_id=0)
    out_pre_np.deallocate(True)

    # Paged prefill (fills paged cache using page_table)
    out_pre_p = attn_paged.forward_prefill(
        x_pre_tt, rot_mats=(cos_pre, sin_pre), kv_cache=None, user_id=0, page_table=pt_tt
    )
    out_pre_p.deallocate(True)
    x_pre_tt.deallocate(True)
    cos_pre.deallocate(True)
    sin_pre.deallocate(True)

    # ----- Decode step -----
    cur_pos = T_pre
    x_dec = _make_random_hidden_state(1, seed=101)
    cos_dec, sin_dec = rope.get_cos_sin_for_decode(cur_pos)
    x_dec_tt = _send_to_device(x_dec, mesh_8x4)

    cur_pos_tensor = ttnn.from_torch(
        torch.tensor([cur_pos] * args.max_batch_size, dtype=torch.int32),
        device=mesh_8x4,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )

    # Non-paged decode
    out_np_tt = attn_nonpaged.forward_decode(
        x_dec_tt,
        current_pos=cur_pos_tensor,
        rot_mats=(cos_dec, sin_dec),
        page_table=None,
        kv_cache=None,
    )
    out_np_host = _gather_replicated(out_np_tt, mesh_8x4)
    out_np_tt.deallocate(True)

    # Paged decode (page_table tells where to read KV blocks from)
    out_p_tt = attn_paged.forward_decode(
        x_dec_tt,
        current_pos=cur_pos_tensor,
        rot_mats=(cos_dec, sin_dec),
        page_table=pt_tt,
        kv_cache=None,
    )
    out_p_host = _gather_replicated(out_p_tt, mesh_8x4)
    out_p_tt.deallocate(True)

    x_dec_tt.deallocate(True)
    cos_dec.deallocate(True)
    sin_dec.deallocate(True)
    pt_tt.deallocate(True)

    passing, pcc_msg = comp_pcc(out_np_host.bfloat16(), out_p_host.bfloat16(), pcc=_PCC_THRESH_MATCH)
    print(f"\n[test1] paged vs non-paged decode PCC: {pcc_msg}")
    assert passing, (
        f"[test1] FAILED — paged and non-paged outputs differ more than expected. "
        f"PCC={pcc_msg} < {_PCC_THRESH_MATCH}"
    )


# ---------------------------------------------------------------------------
# Test 2: fill isolation — user 0 KV cache is unaffected by user 1's fill
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_paged_fill_cache_isolation(mesh_8x4):
    """Paged fill_cache isolates user 0 and user 1 via disjoint page allocations.

    Steps:
    1. Allocate a paged KV cache large enough for 2 users.
    2. Fill user 0's KV cache with a known random tensor.
    3. Fill user 1's KV cache with a DIFFERENT random tensor using a DIFFERENT
       set of physical blocks (disjoint page_table rows).
    4. Verify: the values we read back from user 0's pages match what we wrote
       for user 0 (not corrupted by user 1's fill).

    This tests the physical isolation guarantee of the paged cache layout.
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    block_size = _BLOCK_SIZE
    max_seq = _MAX_SEQ_LEN_FOR_PAGED
    import math

    max_blocks_per_seq = math.ceil(max_seq / block_size)
    n_batch = 2
    max_num_blocks = max_blocks_per_seq * n_batch  # each user gets their own blocks

    args = TtQwen36ModelArgs(mesh_8x4)
    # n_kv_per_col for cache allocation
    n_cols = list(mesh_8x4.shape)[1]  # 4
    n_kv_per_col = args.n_kv_heads // n_cols  # 1

    # Allocate paged KV cache
    keys_cache, values_cache = _build_paged_kv_cache(mesh_8x4, n_kv_per_col, _HEAD_DIM, block_size, max_num_blocks)

    # User 0 page table: blocks [0, 1, ..., max_blocks_per_seq-1]
    # User 1 page table: blocks [max_blocks_per_seq, ..., 2*max_blocks_per_seq-1]
    pt_u0 = torch.zeros(1, max_blocks_per_seq, dtype=torch.int32)
    for i in range(max_blocks_per_seq):
        pt_u0[0, i] = i
    pt_u1 = torch.zeros(1, max_blocks_per_seq, dtype=torch.int32)
    for i in range(max_blocks_per_seq):
        pt_u1[0, i] = max_blocks_per_seq + i

    pt_u0_tt = _send_page_table_to_device(pt_u0, mesh_8x4)
    pt_u1_tt = _send_page_table_to_device(pt_u1, mesh_8x4)

    # Build known random K/V for user 0 and user 1
    T = _T_PREFILL  # fill T tokens
    torch.manual_seed(200)
    # k_u0 shape expected by paged_fill_cache: [1, n_kv_per_col, T, head_dim]
    k_u0 = torch.randn(1, n_kv_per_col, T, _HEAD_DIM, dtype=torch.bfloat16)
    torch.manual_seed(201)
    k_u1 = torch.randn(1, n_kv_per_col, T, _HEAD_DIM, dtype=torch.bfloat16)

    # k_u0/k_u1 shape: [1, n_kv_per_col=1, T, hd]
    # We need [1, n_kv_heads=4, T, hd] for paged_fill_cache input (so col-shard splits to [1, 1, T, hd] per col).
    # Expand: [1, 1, T, hd] → [1, 4, T, hd] by repeating the KV head.
    col_shard = ttnn.ShardTensor2dMesh(mesh_8x4, dims=(None, 1), mesh_shape=list(mesh_8x4.shape))
    k_u0_tt = ttnn.from_torch(
        k_u0.expand(1, args.n_kv_heads, T, _HEAD_DIM),
        device=mesh_8x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=col_shard,
    )
    k_u1_tt = ttnn.from_torch(
        k_u1.expand(1, args.n_kv_heads, T, _HEAD_DIM),
        device=mesh_8x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=col_shard,
    )

    # Fill user 0
    ttnn.experimental.paged_fill_cache(keys_cache, k_u0_tt, pt_u0_tt, batch_idx=0)
    # Fill user 1 (uses completely different physical blocks)
    ttnn.experimental.paged_fill_cache(keys_cache, k_u1_tt, pt_u1_tt, batch_idx=0)

    # Read back user 0's blocks and verify they match k_u0 (not corrupted by u1)
    # Extract the paged cache to host
    cluster_shape = list(mesh_8x4.shape)
    cache_host = ttnn.to_torch(
        keys_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_8x4, dims=(0, 1), mesh_shape=cluster_shape),
    )
    # cache_host shape: [max_num_blocks*8, n_kv*4, block_size, head_dim]
    # (8 rows x 4 cols stacked by ConcatMesh2dToTensor)
    # Focus on user 0's physical blocks (indices 0..max_blocks_per_seq-1, row 0)
    n_blocks_u0 = math.ceil(T / block_size)
    # Reconstruct what user 0 wrote: concatenate the blocks they own
    read_blocks = []
    for blk_idx in range(n_blocks_u0):
        phys_blk = pt_u0[0, blk_idx].item()
        tokens_in_blk = min(block_size, T - blk_idx * block_size)
        # From first mesh row (row 0), first device's blocks
        read_blocks.append(cache_host[phys_blk, 0, :tokens_in_blk, :])  # [tokens, head_dim]

    read_u0 = torch.cat(read_blocks, dim=0)  # [T, head_dim]
    written_u0 = k_u0[0, 0, :T, :]  # [T, head_dim] from first KV head

    # PCC between what was written and what was read back
    from models.common.utility_functions import comp_pcc

    passing, pcc_msg = comp_pcc(written_u0.bfloat16(), read_u0.bfloat16(), pcc=_PCC_THRESH_REF)
    print(f"\n[test2] user-0 cache isolation PCC: {pcc_msg}")
    assert passing, (
        f"[test2] FAILED — user 0 cache was corrupted by user 1's fill. " f"PCC={pcc_msg} < {_PCC_THRESH_REF}"
    )

    # Cleanup
    k_u0_tt.deallocate(True)
    k_u1_tt.deallocate(True)
    pt_u0_tt.deallocate(True)
    pt_u1_tt.deallocate(True)
    keys_cache.deallocate(True)
    values_cache.deallocate(True)


# ---------------------------------------------------------------------------
# Test 3: paged decode step PCC > 0.99 vs CPU reference
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_paged_decode_step_pcc(mesh_8x4):
    """Paged attention decode step PCC > 0.99 vs CPU reference.

    Steps:
    1. Build paged-attention model (use_paged_kv_cache=True)
    2. Prefill T=32 tokens (populates paged KV cache)
    3. Run decode step at position T=32 with page_table
    4. Compare output PCC > 0.99 against CPU reference GatedAttention
    """
    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup

    print("\n[test3] Loading weights...")
    sd = _load_attn_weights()
    block_size = _BLOCK_SIZE
    max_seq = _MAX_SEQ_LEN_FOR_PAGED
    import math

    max_num_blocks = math.ceil(max_seq / block_size) * _MAX_BATCH

    attn_paged, args = _build_attn_with_config(
        mesh_8x4,
        sd,
        use_paged_kv_cache=True,
        kv_cache_max_seq_len=max_seq,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
    )

    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_B, max_seq_len=max_seq)

    # ----- Prefill -----
    T_pre = _T_PREFILL
    x_pre = _make_random_hidden_state(T_pre, seed=300)
    cos_pre, sin_pre = rope.get_cos_sin_for_prefill(seq_len=T_pre)
    x_pre_tt = _send_to_device(x_pre, mesh_8x4)

    page_table = _make_page_table(_B, T_pre, block_size, max_num_blocks)
    pt_tt = _send_page_table_to_device(page_table, mesh_8x4)

    out_pre = attn_paged.forward_prefill(
        x_pre_tt, rot_mats=(cos_pre, sin_pre), kv_cache=None, user_id=0, page_table=pt_tt
    )
    out_pre.deallocate(True)
    x_pre_tt.deallocate(True)
    cos_pre.deallocate(True)
    sin_pre.deallocate(True)

    # ----- Decode step -----
    cur_pos = T_pre
    x_dec = _make_random_hidden_state(1, seed=301)
    cos_dec, sin_dec = rope.get_cos_sin_for_decode(cur_pos)
    x_dec_tt = _send_to_device(x_dec, mesh_8x4)

    cur_pos_tensor = ttnn.from_torch(
        torch.tensor([cur_pos] * args.max_batch_size, dtype=torch.int32),
        device=mesh_8x4,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )

    out_p_tt = attn_paged.forward_decode(
        x_dec_tt,
        current_pos=cur_pos_tensor,
        rot_mats=(cos_dec, sin_dec),
        page_table=pt_tt,
        kv_cache=None,
    )
    out_p_host = _gather_replicated(out_p_tt, mesh_8x4)
    out_p_tt.deallocate(True)
    x_dec_tt.deallocate(True)
    cos_dec.deallocate(True)
    sin_dec.deallocate(True)
    pt_tt.deallocate(True)

    # ----- CPU reference -----
    cfg_path = _SNAPSHOT_DIR / "config.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    with torch.no_grad():
        ref_attn = GatedAttention(config)
        ref_attn.eval()
        for key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(ref_attn, key).weight.data.copy_(sd[f"{key}.weight"])
        ref_attn.q_norm.weight.data.copy_(sd["q_norm.weight"])
        ref_attn.k_norm.weight.data.copy_(sd["k_norm.weight"])

        # Prefill reference
        cos_pre_ref, sin_pre_ref = _build_rope_cos_sin(T_pre)

        mask_pre = torch.zeros(1, 1, T_pre, T_pre)
        mask_pre = mask_pre.masked_fill(torch.triu(torch.ones(T_pre, T_pre), diagonal=1).bool(), float("-inf"))
        _, (k_cache_ref, v_cache_ref) = ref_attn(
            x_pre.float(), cos_pre_ref, sin_pre_ref, kv_cache=None, attention_mask=mask_pre
        )

        # Decode at cur_pos
        cos_all_ref, sin_all_ref = _build_rope_cos_sin(cur_pos + 1)
        cos_dec_ref = cos_all_ref[:, cur_pos : cur_pos + 1, :]
        sin_dec_ref = sin_all_ref[:, cur_pos : cur_pos + 1, :]
        x_dec_float = x_dec.float()
        ref_dec_out, _ = ref_attn(
            x_dec_float, cos_dec_ref, sin_dec_ref, kv_cache=(k_cache_ref, v_cache_ref), attention_mask=None
        )

    passing, pcc_msg = comp_pcc(ref_dec_out.bfloat16(), out_p_host.bfloat16(), pcc=_PCC_THRESH_REF)
    print(f"\n[test3] paged decode step PCC vs CPU ref: {pcc_msg}")
    assert passing, (
        f"[test3] FAILED — paged decode output differs from CPU reference. " f"PCC={pcc_msg} < {_PCC_THRESH_REF}"
    )


# ---------------------------------------------------------------------------
# Test 4: full 64-layer model with paged attention — Paris generation
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_paged_64layer_paris_generation(mesh_8x4):
    """Full 64-layer Qwen3.6-27B with paged attention generates ' Paris'.

    Same Paris generation test as test_full_model.py but with
    use_paged_kv_cache=True enabled in the model configuration.

    This tests end-to-end paged attention through the full model stack.
    """
    # We follow the weight loading pattern from test_full_model.py
    from safetensors.torch import load_file as load_st

    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    def _load_layer(layer_idx):
        pfx = f"model.language_model.layers.{layer_idx}"
        keys_needed = [k for k in weight_map if k.startswith(pfx + ".")]
        files_needed = sorted({weight_map[k] for k in keys_needed})
        raw = {}
        for fn in files_needed:
            shard = load_st(str(_SNAPSHOT_DIR / fn))
            for k in keys_needed:
                if k in shard:
                    raw[k] = shard[k].float()
        result = {}
        for k, v in raw.items():
            short = k[len(pfx) + 1 :]
            result[short] = v
        return result

    def _load_global_weights():
        # Exact key names as they appear in the weight map (confirmed via index.json):
        #   model.language_model.embed_tokens.weight
        #   model.language_model.norm.weight
        #   lm_head.weight  (NOT model.language_model.lm_head.weight)
        global_keys = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        ]
        files = sorted({weight_map[k] for k in global_keys if k in weight_map})
        raw = {}
        for fn in files:
            shard = load_st(str(_SNAPSHOT_DIR / fn))
            for k in global_keys:
                if k in shard:
                    raw[k] = shard[k].float()
        return {
            "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
            "norm.weight": raw["model.language_model.norm.weight"],
            "output.weight": raw["lm_head.weight"],
        }

    print("\n[test4] Loading 64-layer weights for paged Paris test...")
    global_wts = _load_global_weights()
    layers_wts = [_load_layer(i) for i in range(64)]

    # Build args with paged attention config
    args = TtQwen36ModelArgs(
        mesh_8x4,
        max_seq_len=512,
        use_paged_kv_cache=True,
        block_size=_BLOCK_SIZE,
    )

    model = TtQwen36Transformer(
        mesh_device=mesh_8x4,
        args=args,
        global_weights=global_wts,
        layers_weights=layers_wts,
    )

    # Tokenize "The capital of France is" using the actual Qwen3.6 tokenizer.
    # Same approach as test_full_model.py (tokenizer, not hardcoded IDs).
    from transformers import AutoTokenizer

    tokenizer_for_input = AutoTokenizer.from_pretrained(str(_SNAPSHOT_DIR), trust_remote_code=True)
    prompt = "The capital of France is"
    prompt_ids = tokenizer_for_input(prompt, return_tensors="pt").input_ids  # [1, T]
    T_prompt = prompt_ids.shape[1]
    print(f"[test4] Prompt '{prompt}' tokenized as: {prompt_ids.tolist()}")

    # Pad to nearest multiple of 32 for TTNN tile alignment (same as test_full_model.py)
    import math

    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=prompt_ids.dtype)
        prompt_ids_padded = torch.cat([prompt_ids, pad], dim=1)
    else:
        prompt_ids_padded = prompt_ids

    # Build page_table for prefill (user 0, sequential block allocation over T_padded)
    max_blocks_per_seq = math.ceil(512 / _BLOCK_SIZE)
    page_table = _make_page_table(1, T_padded, _BLOCK_SIZE, max_blocks_per_seq)
    pt_tt = _send_page_table_to_device(page_table, mesh_8x4)

    print(f"[test4] Running paged prefill for {T_prompt}-token Paris prompt (padded to {T_padded})...")
    logits, kv_caches, dn_states, conv_states = model.forward_prefill(
        prompt_ids_padded,
        return_caches=True,
        page_table=pt_tt,
    )

    # Use tokenizer to decode and check for "paris" (case-insensitive), same as test_full_model.py
    config_path = _SNAPSHOT_DIR / "config.json"
    with open(config_path) as f:
        _cfg = json.load(f)
    vocab_size = _cfg.get("vocab_size", logits.shape[-1])
    last_logits = logits[0, T_prompt - 1, :vocab_size]
    next_token = last_logits.argmax().item()
    decoded = tokenizer_for_input.decode([next_token])
    print(f"[test4] Next token id: {next_token}, decoded: '{decoded}'")
    assert "paris" in decoded.lower(), (
        f"[test4] FAILED — paged model predicted token {next_token} ('{decoded}'), "
        f"expected token containing 'paris'. "
        f"Paged attention may be corrupting KV cache during full-model forward."
    )
    print(f"[test4] PASSED — paged 64-layer model correctly generates token {next_token} ('{decoded}')")

    # Cleanup
    pt_tt.deallocate(True)
    for kv in kv_caches:
        if kv is not None:
            for t in kv:
                t.deallocate(True)
