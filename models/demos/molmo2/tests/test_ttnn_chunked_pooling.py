# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for the proposed TTNN image pooling path:
  - ttnn.embedding with uint32 indices for gather (no bfloat16 precision loss)
  - device-side masked mean (ttnn.sum / ttnn.clamp / ttnn.div — no D2H)
  - attn mask built from valid_mask converted to ROW_MAJOR before reshape/slice
  - column-parallel wq/wk/wv (ShardTensorToMesh dim=3), row-parallel wo
  - manual matmul attention (not SDPA — matches other branch)
  - ttnn.all_reduce after wo for TP

Compares against CPU reference image_pooling_2d from functional.py.
Does NOT modify any production code.
"""

import sys
from pathlib import Path

import pytest
import torch

import ttnn

GOLDEN_DIR = Path(__file__).parent.parent / "reference" / "golden"
HF_PATH = Path(
    "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
).expanduser()


# ---------------------------------------------------------------------------
# Fixtures (reuse from test_tt_text_decoder where possible)
# ---------------------------------------------------------------------------

_MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}

import os

_shape = _MESH_SHAPE.get(os.environ.get("MESH_DEVICE"), (1, 8))


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _shape
    is_single = rows * cols == 1
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    yield device
    ttnn.close_mesh_device(device)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def state_dict():
    from transformers import AutoModelForImageTextToText

    sys.path.insert(0, str(HF_PATH))
    hf = AutoModelForImageTextToText.from_pretrained(
        str(HF_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    sd = hf.state_dict()
    del hf
    return sd


@pytest.fixture(scope="module")
def molmo2_cfg(mesh_device):
    from models.demos.molmo2.tt.model_config import Molmo2Config

    return Molmo2Config(mesh_device=mesh_device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_HEADS = 16
HEAD_DIM = 72
PADDED_HEAD_DIM = 96  # ceil(72/32)*32
POOL_DIM = 2304  # vit_hidden * 2
HIDDEN_DIM = 1152


def _pad_w(w, num_heads=NUM_HEADS, head_dim=HEAD_DIM, padded=PADDED_HEAD_DIM):
    w = w.reshape(num_heads, head_dim, -1)
    w = torch.nn.functional.pad(w, (0, 0, 0, padded - head_dim))
    return w.reshape(num_heads * padded, -1)


def _pad_b(b, num_heads=NUM_HEADS, head_dim=HEAD_DIM, padded=PADDED_HEAD_DIM):
    b = b.reshape(num_heads, head_dim)
    b = torch.nn.functional.pad(b, (0, padded - head_dim))
    return b.reshape(-1)


def _upload(t, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mapper=None):
    if mapper is None:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


# ---------------------------------------------------------------------------
# Core TTNN pooling helper (shared by both tests)
# ---------------------------------------------------------------------------

N_PATCHES = 729  # 27×27 patches per crop


def _make_pool_weights(state_dict, mesh_device):
    """Load and shard pooling weights onto device. Returns dict of TTNN tensors."""
    num_devices = mesh_device.get_num_devices()
    col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    bias_col = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    pfx = "model.vision_backbone.image_pooling_2d"

    def load_qkv(key):
        w = _pad_w(state_dict[f"{pfx}.{key}.weight"])
        b = _pad_b(state_dict[f"{pfx}.{key}.bias"])
        return (
            ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_mapper,
            ),
            ttnn.as_tensor(
                b.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=bias_col,
            ),
        )

    wq, bq = load_qkv("wq")
    wk, bk = load_qkv("wk")
    wv, bv = load_qkv("wv")

    wo_raw = state_dict[f"{pfx}.wo.weight"]
    wo_r = wo_raw.reshape(-1, NUM_HEADS, HEAD_DIM)
    wo_r = torch.nn.functional.pad(wo_r, (0, PADDED_HEAD_DIM - HEAD_DIM))
    wo_raw = wo_r.reshape(-1, NUM_HEADS * PADDED_HEAD_DIM)
    wo = ttnn.as_tensor(
        wo_raw.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=row_mapper,
    )
    bo = ttnn.as_tensor(
        state_dict[f"{pfx}.wo.bias"].to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )

    return {
        "wq": wq,
        "bq": bq,
        "wk": wk,
        "bk": bk,
        "wv": wv,
        "bv": bv,
        "wo": wo,
        "bo": bo,
        "num_devices": num_devices,
        "n_local_heads": NUM_HEADS // num_devices,
    }


def _pool_one_chunk(feat_tt, chunk_idx_cpu, chunk_n_out, k_pool, weights, cfg, mesh_device):
    """
    Run one chunk of TTNN pooling.

    Args:
        feat_tt: full 2D feature table [total_patches, POOL_DIM] ROW_MAJOR on device
        chunk_idx_cpu: [chunk_n_out, k_pool] int64 CPU — GLOBAL indices, no -1s in this test
        chunk_n_out: number of output windows in this chunk
        k_pool: pool window size
        weights: dict from _make_pool_weights
        cfg: Molmo2Config (for compute kernel configs)
        mesh_device: mesh device

    Returns:
        pooled_cpu: [chunk_n_out, HIDDEN_DIM] float32 on CPU
    """
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    n_local_heads = weights["n_local_heads"]
    cfg_hifi2 = cfg.compute_kernel_config_hifi2
    cfg_hifi4 = cfg.compute_kernel_config_hifi4

    # Upload indices as uint32 — max index can be ~280k, bfloat16 only accurate to ~256
    idx_flat = chunk_idx_cpu.reshape(-1).clamp(min=0).to(torch.int32)
    idx_tt = ttnn.from_torch(
        idx_flat.reshape(1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    # All positions valid in this test (no -1s)
    valid_cpu = torch.ones(chunk_n_out * k_pool, dtype=torch.bfloat16)
    valid_tt = ttnn.from_torch(
        valid_cpu.reshape(1, 1, -1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    # Gather via ttnn.embedding with global uint32 indices
    gathered = ttnn.embedding(idx_tt, feat_tt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(idx_tt)
    gathered = ttnn.reshape(gathered, [1, 1, chunk_n_out * k_pool, POOL_DIM])
    gathered = ttnn.mul(gathered, valid_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    to_pool = ttnn.reshape(gathered, [1, chunk_n_out, k_pool, POOL_DIM])
    ttnn.deallocate(gathered)

    # Masked mean query — fully on device
    query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)
    vm = ttnn.reshape(valid_tt, [1, chunk_n_out, k_pool, 1])
    denom = ttnn.clamp(ttnn.sum(vm, dim=2, keepdim=True), min=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(vm)
    query = ttnn.div(query_sum, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(query_sum)
    ttnn.deallocate(denom)

    # Attn mask: ROW_MAJOR reshape avoids tile-padding artefacts on k_pool=9
    valid_rm = ttnn.to_layout(valid_tt, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(valid_tt)
    valid_4d = ttnn.to_layout(ttnn.reshape(valid_rm, [chunk_n_out, 1, 1, k_pool]), ttnn.TILE_LAYOUT)
    ttnn.deallocate(valid_rm)
    zeros_buf = ttnn.zeros_like(valid_4d)
    neg_inf_buf = _upload(torch.full((1, 1, 1, 1), float("-inf")), mesh_device, mapper=mapper)
    threshold = ttnn.full_like(valid_4d, 0.5)
    cond = ttnn.gt(valid_4d, threshold, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_mask = ttnn.where(cond, zeros_buf, neg_inf_buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in (valid_4d, zeros_buf, neg_inf_buf, threshold, cond):
        ttnn.deallocate(t)

    # QKV projections (column-parallel TP=8)
    q = ttnn.add(
        ttnn.linear(query, weights["wq"], compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        weights["bq"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k = ttnn.add(
        ttnn.linear(to_pool, weights["wk"], compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        weights["bk"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v = ttnn.add(
        ttnn.linear(to_pool, weights["wv"], compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        weights["bv"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(to_pool)

    # Reshape to [chunk_n_out, n_local_heads, 1|k_pool, padded_head_dim]
    q = ttnn.permute(ttnn.reshape(q, [chunk_n_out, 1, n_local_heads, PADDED_HEAD_DIM]), (0, 2, 1, 3))
    k = ttnn.permute(ttnn.reshape(k, [chunk_n_out, k_pool, n_local_heads, PADDED_HEAD_DIM]), (0, 2, 1, 3))
    v = ttnn.permute(ttnn.reshape(v, [chunk_n_out, k_pool, n_local_heads, PADDED_HEAD_DIM]), (0, 2, 1, 3))

    # Manual matmul attention
    k_t = ttnn.permute(k, (0, 1, 3, 2))
    attn_w = ttnn.mul(
        ttnn.matmul(q, k_t, compute_kernel_config=cfg_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        HEAD_DIM**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(k_t)
    attn_w = ttnn.add(attn_w, attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_mask)
    attn_p = ttnn.softmax(attn_w, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_w)
    attn_out = ttnn.matmul(attn_p, v, compute_kernel_config=cfg_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in (attn_p, q, k, v):
        ttnn.deallocate(t)

    attn_out = ttnn.reshape(ttnn.permute(attn_out, (0, 2, 1, 3)), [1, chunk_n_out, 1, n_local_heads * PADDED_HEAD_DIM])

    # Row-parallel wo + all_reduce + bias
    out = ttnn.linear(attn_out, weights["wo"], compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_out)
    if weights["num_devices"] > 1:
        out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.add(out, weights["bo"], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().squeeze(0).squeeze(1)
    ttnn.deallocate(out)
    return out_cpu  # [chunk_n_out, HIDDEN_DIM]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ttnn_chunked_pooling_pcc(mesh_device, state_dict, molmo2_cfg):
    """
    TTNN image pooling with uint32 gather + device masked mean + ROW_MAJOR mask slicing.
    PCC must be > 0.99 against CPU reference.
    """
    from models.demos.molmo2.reference.functional import image_pooling_2d as _pool_ref

    def pcc(a, b):
        a = a.to(torch.float32).flatten()
        b = b.to(torch.float32).flatten()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    # ---- Load golden inputs ----
    va = torch.load(GOLDEN_DIR / "vision_adapter.pt", weights_only=False)
    fp = torch.load(GOLDEN_DIR / "full_prefill.pt", weights_only=False)
    image_features_4d = va["image_features_4d"].float()  # [1, 9, 729, 2304]
    pool_idx = fp["inputs"]["image_token_pooling"].unsqueeze(0)  # [1, 1316, 4]

    B = pool_idx.shape[0]
    N_out = pool_idx.shape[1]
    k_pool = pool_idx.shape[2]
    num_devices = mesh_device.get_num_devices()
    n_local_heads = NUM_HEADS // num_devices  # 2 per device on T3K

    pfx = "model.vision_backbone.image_pooling_2d"

    # ---- CPU reference ----
    ref_out = _pool_ref(
        image_features_4d,
        pool_idx,
        state_dict[f"{pfx}.wq.weight"].float(),
        state_dict[f"{pfx}.wq.bias"].float(),
        state_dict[f"{pfx}.wk.weight"].float(),
        state_dict[f"{pfx}.wk.bias"].float(),
        state_dict[f"{pfx}.wv.weight"].float(),
        state_dict[f"{pfx}.wv.bias"].float(),
        state_dict[f"{pfx}.wo.weight"].float(),
        state_dict[f"{pfx}.wo.bias"].float(),
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )  # [1, 1316, 1152]

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    bias_col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)

    # ------------------------------------------------------------------ #
    # 1. Upload ViT features as 2D ROW_MAJOR embedding table
    #    [9*729, 2304] — indices reference this table
    # ------------------------------------------------------------------ #
    feat_2d = image_features_4d.reshape(-1, POOL_DIM).to(torch.bfloat16)
    feat_tt = ttnn.from_torch(
        feat_2d,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    # ------------------------------------------------------------------ #
    # 2. Upload indices as uint32 (not bfloat16 — max index 6560 > 256)
    # ------------------------------------------------------------------ #
    idx_flat = pool_idx.reshape(-1)
    valid_cpu = (idx_flat >= 0).float()
    idx_clipped = idx_flat.clamp(min=0).to(torch.int32)

    idx_tt = ttnn.from_torch(
        idx_clipped.reshape(1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    valid_tt = ttnn.from_torch(
        valid_cpu.reshape(1, 1, -1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    # ------------------------------------------------------------------ #
    # 3. Gather on device via ttnn.embedding
    # ------------------------------------------------------------------ #
    gathered = ttnn.embedding(idx_tt, feat_tt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(feat_tt)
    ttnn.deallocate(idx_tt)

    gathered = ttnn.reshape(gathered, [1, 1, B * N_out * k_pool, POOL_DIM])
    gathered = ttnn.mul(gathered, valid_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    to_pool = ttnn.reshape(gathered, [1, B * N_out, k_pool, POOL_DIM])
    ttnn.deallocate(gathered)

    # ------------------------------------------------------------------ #
    # 4. Masked mean query — fully on device (no D2H)
    # ------------------------------------------------------------------ #
    query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)  # [1, N_out, 1, POOL_DIM]
    vm = ttnn.reshape(valid_tt, [1, B * N_out, k_pool, 1])
    denom = ttnn.sum(vm, dim=2, keepdim=True)  # [1, N_out, 1, 1]
    ttnn.deallocate(vm)
    denom = ttnn.clamp(denom, min=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    query = ttnn.div(query_sum, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(query_sum)
    ttnn.deallocate(denom)

    # ------------------------------------------------------------------ #
    # 5. Attn mask: convert to ROW_MAJOR first, then reshape to [N_out,1,1,k_pool]
    #    Avoids tile-padding artefacts when k_pool=4 is not tile-aligned
    # ------------------------------------------------------------------ #
    valid_rm = ttnn.to_layout(valid_tt, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(valid_tt)
    valid_4d = ttnn.reshape(valid_rm, [B * N_out, 1, 1, k_pool])
    ttnn.deallocate(valid_rm)
    valid_4d = ttnn.to_layout(valid_4d, ttnn.TILE_LAYOUT)

    zeros_buf = ttnn.zeros_like(valid_4d)
    neg_inf_buf = _upload(torch.full((1, 1, 1, 1), float("-inf")), mesh_device, mapper=mapper)
    threshold = ttnn.full_like(valid_4d, 0.5)
    cond = ttnn.gt(valid_4d, threshold, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_mask = ttnn.where(cond, zeros_buf, neg_inf_buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(valid_4d)
    ttnn.deallocate(zeros_buf)
    ttnn.deallocate(neg_inf_buf)
    ttnn.deallocate(threshold)
    ttnn.deallocate(cond)

    # ------------------------------------------------------------------ #
    # 6. Load weights: column-parallel wq/wk/wv, row-parallel wo
    # ------------------------------------------------------------------ #
    cfg_hifi2 = molmo2_cfg.compute_kernel_config_hifi2
    cfg_hifi4 = molmo2_cfg.compute_kernel_config_hifi4

    def load_qkv_proj(key):
        w = state_dict[f"{pfx}.{key}.weight"]
        b = state_dict[f"{pfx}.{key}.bias"]
        w = _pad_w(w)  # [num_heads*padded_head_dim, input_dim]
        b = _pad_b(b)  # [num_heads*padded_head_dim]
        w_tt = ttnn.as_tensor(
            w.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_mapper,
        )
        b_tt = ttnn.as_tensor(
            b.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=bias_col_mapper,
        )
        return w_tt, b_tt

    wq_tt, bq_tt = load_qkv_proj("wq")
    wk_tt, bk_tt = load_qkv_proj("wk")
    wv_tt, bv_tt = load_qkv_proj("wv")

    # wo: row-parallel [hidden_dim, num_heads*padded_head_dim]
    wo = state_dict[f"{pfx}.wo.weight"]  # [hidden_dim, num_heads*head_dim]
    bo = state_dict[f"{pfx}.wo.bias"]
    # pad input (head) dim
    wo_r = wo.reshape(-1, NUM_HEADS, HEAD_DIM)
    wo_r = torch.nn.functional.pad(wo_r, (0, PADDED_HEAD_DIM - HEAD_DIM))
    wo = wo_r.reshape(-1, NUM_HEADS * PADDED_HEAD_DIM)  # [1152, 1536]
    wo_tt = ttnn.as_tensor(
        wo.T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=row_mapper,
    )
    bo_tt = _upload(bo, mesh_device, mapper=mapper)

    # ------------------------------------------------------------------ #
    # 7. Cross-attention: Q=[1,N_out,1,POOL_DIM], KV=[1,N_out,k_pool,POOL_DIM]
    # ------------------------------------------------------------------ #
    q = ttnn.linear(query, wq_tt, compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = ttnn.add(q, bq_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,N_out,1, n_local_heads*padded]
    k = ttnn.linear(to_pool, wk_tt, compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.add(k, bk_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,N_out,k_pool, n_local_heads*padded]
    v = ttnn.linear(to_pool, wv_tt, compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.add(v, bv_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(query)
    ttnn.deallocate(to_pool)

    # Reshape: [N_out, seq, n_local_heads, padded_head_dim] → permute → [N_out, n_local_heads, seq, padded]
    q = ttnn.reshape(q, [N_out, 1, n_local_heads, PADDED_HEAD_DIM])
    q = ttnn.permute(q, (0, 2, 1, 3))  # [N_out, n_local_heads, 1,      padded]
    k = ttnn.reshape(k, [N_out, k_pool, n_local_heads, PADDED_HEAD_DIM])
    k = ttnn.permute(k, (0, 2, 1, 3))  # [N_out, n_local_heads, k_pool, padded]
    v = ttnn.reshape(v, [N_out, k_pool, n_local_heads, PADDED_HEAD_DIM])
    v = ttnn.permute(v, (0, 2, 1, 3))

    k_t = ttnn.permute(k, (0, 1, 3, 2))  # [N_out, n_local_heads, padded, k_pool]
    attn_w = ttnn.matmul(q, k_t, compute_kernel_config=cfg_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(k_t)
    attn_w = ttnn.mul(attn_w, HEAD_DIM**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # attn_mask [N_out, 1, 1, k_pool] broadcasts to [N_out, n_local_heads, 1, k_pool]
    attn_w = ttnn.add(attn_w, attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_mask)

    attn_p = ttnn.softmax(attn_w, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_w)
    attn_out = ttnn.matmul(attn_p, v, compute_kernel_config=cfg_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_p)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # [N_out, n_local_heads, 1, padded] → [1, N_out, 1, n_local_heads*padded]
    attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
    attn_out = ttnn.reshape(attn_out, [1, N_out, 1, n_local_heads * PADDED_HEAD_DIM])

    # ------------------------------------------------------------------ #
    # 8. Output projection (row-parallel) + all_reduce + bias
    # ------------------------------------------------------------------ #
    out = ttnn.linear(attn_out, wo_tt, compute_kernel_config=cfg_hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_out)

    if num_devices > 1:
        out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out = ttnn.add(out, bo_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ------------------------------------------------------------------ #
    # 9. PCC
    # ------------------------------------------------------------------ #
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)
    out_cpu = out_cpu.squeeze(2)  # [1, N_out, 1152]

    pcc_val = pcc(out_cpu[0], ref_out[0].float())
    print(f"\n  TTNN chunked pooling PCC = {pcc_val:.6f}")
    print(f"  out_cpu  mean={out_cpu.mean():.4f}  std={out_cpu.std():.4f}")
    print(f"  ref_out  mean={ref_out.float().mean():.4f}  std={ref_out.float().std():.4f}")
    assert pcc_val > 0.99, f"PCC {pcc_val:.6f} < 0.99"


def test_ttnn_chunked_pooling_384frames(mesh_device, state_dict, molmo2_cfg):
    """
    384-frame scale test for chunked TTNN pooling.

    Uses synthetic random features [1, 384, 729, 2304] with LOCAL indices
    (each frame's windows reference only that frame's patches).
    Max index = 383*729 + 80*9 + 8 = 280,375 — far above bfloat16's ~256 integer limit,
    so uint32 indexing is essential.

    Strategy:
      - Upload full [384*729, 2304] feature table to device once (stays there all chunks)
      - Process 8 frames at a time (648 windows/chunk, 48 chunks) via _pool_one_chunk
      - CPU reference computed only for first 8 frames (fast) for PCC check
      - Full 48-chunk run verifies no OOM and correct output shape
    """
    from models.demos.molmo2.reference.functional import image_pooling_2d as _pool_ref

    def pcc(a, b):
        a = a.to(torch.float32).flatten()
        b = b.to(torch.float32).flatten()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    N_FRAMES = 384
    N_WINDOWS_PER_FRAME = 81  # 729 patches / k_pool=9
    k_pool = 9
    CHUNK_FRAMES = 8  # 8*81 = 648 windows/chunk, 48 chunks total

    # Synthetic features — random bfloat16 (avoids 2.5 GB float32 allocation)
    torch.manual_seed(42)
    image_features = torch.randn(1, N_FRAMES, N_PATCHES, POOL_DIM, dtype=torch.bfloat16)

    # Local indices: window j of frame i gathers patches [i*729+j*9 .. i*729+j*9+8]
    # Max index = 383*729 + 80*9 + 8 = 280,375  (bfloat16 limit ≈ 256 → would corrupt)
    pool_idx = torch.zeros(1, N_FRAMES * N_WINDOWS_PER_FRAME, k_pool, dtype=torch.int64)
    for i in range(N_FRAMES):
        for j in range(N_WINDOWS_PER_FRAME):
            base = i * N_PATCHES + j * k_pool
            pool_idx[0, i * N_WINDOWS_PER_FRAME + j] = torch.arange(base, base + k_pool)

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # ---- Upload full feature table once — stays on device for all chunks ----
    total_patches = N_FRAMES * N_PATCHES  # 279,936
    feat_2d = image_features.reshape(total_patches, POOL_DIM)  # bfloat16, [279936, 2304]
    feat_tt = ttnn.from_torch(
        feat_2d,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    # Load weights once (shared across all chunks)
    weights = _make_pool_weights(state_dict, mesh_device)

    # ---- CPU reference: first CHUNK_FRAMES only (fast) ----
    pfx = "model.vision_backbone.image_pooling_2d"
    ref_out_chunk0 = _pool_ref(
        image_features[:, :CHUNK_FRAMES].float(),
        pool_idx[:, : CHUNK_FRAMES * N_WINDOWS_PER_FRAME],
        state_dict[f"{pfx}.wq.weight"].float(),
        state_dict[f"{pfx}.wq.bias"].float(),
        state_dict[f"{pfx}.wk.weight"].float(),
        state_dict[f"{pfx}.wk.bias"].float(),
        state_dict[f"{pfx}.wv.weight"].float(),
        state_dict[f"{pfx}.wv.bias"].float(),
        state_dict[f"{pfx}.wo.weight"].float(),
        state_dict[f"{pfx}.wo.bias"].float(),
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )  # [1, 648, 1152]

    # ---- TTNN chunked: all 384 frames, 8 at a time ----
    all_outputs = []
    for chunk_start in range(0, N_FRAMES, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, N_FRAMES)
        c_frames = chunk_end - chunk_start
        c_n_out = c_frames * N_WINDOWS_PER_FRAME
        c_idx = pool_idx[0, chunk_start * N_WINDOWS_PER_FRAME : chunk_end * N_WINDOWS_PER_FRAME]
        chunk_out = _pool_one_chunk(feat_tt, c_idx, c_n_out, k_pool, weights, molmo2_cfg, mesh_device)
        all_outputs.append(chunk_out)

    ttnn.deallocate(feat_tt)
    ttnn_out = torch.cat(all_outputs, dim=0)  # [31104, 1152]

    # ---- Checks ----
    assert ttnn_out.shape == (N_FRAMES * N_WINDOWS_PER_FRAME, HIDDEN_DIM), f"Shape mismatch: {ttnn_out.shape}"

    first_chunk_ttnn = ttnn_out[: CHUNK_FRAMES * N_WINDOWS_PER_FRAME]
    pcc_val = pcc(first_chunk_ttnn, ref_out_chunk0[0].float())

    print(f"\n  384-frame chunked pooling:")
    print(f"    PCC (first {CHUNK_FRAMES}-frame chunk vs CPU ref) = {pcc_val:.6f}")
    print(f"    Total output shape: {tuple(ttnn_out.shape)}")
    print(f"    Max pool_idx: {pool_idx.max().item():,}  (bfloat16 safe limit: ~256)")
    print(f"    Chunks processed: {len(all_outputs)}")

    assert pcc_val > 0.99, f"PCC {pcc_val:.6f} < 0.99"
