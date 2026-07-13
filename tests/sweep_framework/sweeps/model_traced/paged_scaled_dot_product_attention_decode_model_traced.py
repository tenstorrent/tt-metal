# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    create_mesh_device,
    create_tensor_on_mesh,
    dispatch_axis_for_grid,
    get_mesh_shape,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
    program_config_grid_bounds,
    reconcile_golden_to_actual,
)

# The device is opened per-vector (not by the fixture) so each vector can use the
# dispatch axis its traced SDPAProgramConfig grid needs: some grids touch x=7
# (need ROW), others touch y=9 / use sub_core_grids up to y=9 (need COL), and no
# single per-suite axis serves both. The device is cached and only reopened when
# the required axis changes between consecutive vectors, so agnostic runs reuse it.
_CUR_DEVICE = None
_CUR_AXIS = "__uninit__"


def _ensure_vector_device(axis):
    global _CUR_DEVICE, _CUR_AXIS
    if _CUR_DEVICE is None or axis != _CUR_AXIS:
        _close_vector_device()
        _CUR_DEVICE = create_mesh_device(get_model_traced_mesh_shape(), dispatch_core_axis=axis)
        _CUR_AXIS = axis
    return _CUR_DEVICE


def _close_vector_device():
    global _CUR_DEVICE, _CUR_AXIS
    if _CUR_DEVICE is not None:
        try:
            ttnn.close_mesh_device(_CUR_DEVICE)
        except Exception:
            # best-effort teardown — a failed device close must not mask the real test result
            pass
    _CUR_DEVICE = None
    _CUR_AXIS = "__uninit__"


def _vector_dispatch_axis(kwargs):
    pc = kwargs.get("program_config")
    pc_val = pc.get("value", "") if isinstance(pc, dict) else str(pc or "")
    return dispatch_axis_for_grid(*program_config_grid_bounds(pc_val))


from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

TIMEOUT = 300

# NOTE:
# -----
# For most ops, the model_traced suite uses real traced configurations from
# production models plus a PyTorch/TTNN golden.  For paged SDPA decode the
# correctness oracle is substantially more complex (see
# tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention_decode.py),
# and we do not yet have a lightweight reference that matches all traced cases.
# Until such a golden is implemented, we deliberately *do not* enable the
# model_traced suite for this op to avoid claiming coverage we do not have.
#
# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("ttnn.transformer.paged_scaled_dot_product_attention_decode")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 8, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_e_dtype": [ttnn.bfloat16],
        "input_e_layout": [ttnn.TILE_LAYOUT],
        "input_e_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    # Device is opened per-vector in run() (see _ensure_vector_device) so each
    # vector can pick the dispatch axis its program_config grid needs.
    yield (None, "wormhole_b0")
    _close_vector_device()


def _paged_sdpa_input_shard_axis_and_factor(placement_dict):
    if not isinstance(placement_dict, dict):
        return None, 1
    plac_raw = placement_dict.get("placement")
    dist_raw = placement_dict.get("distribution_shape")
    if plac_raw is None or dist_raw is None:
        return None, 1
    if isinstance(plac_raw, (list, tuple)):
        plac_items = [str(x).strip().strip("'") for x in plac_raw]
    else:
        s_inner = str(plac_raw).strip()
        if s_inner.startswith("[") and s_inner.endswith("]"):
            s_inner = s_inner[1:-1]
        plac_items = [x.strip().strip("'") for x in s_inner.split(",") if x.strip()]
    if isinstance(dist_raw, (list, tuple)):
        dist_items = [int(x) for x in dist_raw]
    else:
        d_inner = str(dist_raw).strip()
        if d_inner.startswith("[") and d_inner.endswith("]"):
            d_inner = d_inner[1:-1]
        dist_items = [int(x.strip()) for x in d_inner.split(",") if x.strip()]
    axis = None
    factor = 1
    for entry, n in zip(plac_items, dist_items):
        if entry.startswith("PlacementShard("):
            try:
                d = int(entry[len("PlacementShard(") : -1])
            except ValueError:
                continue
            axis = d
            factor *= n
    return axis, factor


def _paged_sdpa_decode_chip_attn(
    q_chip, k_cache_chip, v_cache_chip, page_table, cur_pos, num_users, page_size, sliding_window_size=None
):
    """Compute per-chip paged attention output: (B, H_q, num_users, D_chip).
    q_chip:       (B, H_q, num_users, D_chip)
    k_cache_chip: (num_pages, H_kv, page_size, D_chip)
    v_cache_chip: (num_pages, H_kv, page_size, D_chip)
    page_table:   (B_pt, max_pages) int  — shared across users (rows: user idx mod B_pt)
    cur_pos:      shape varies; we use cur_pos[user_idx % cur_pos.numel()]
    """
    B, H_q, U, D = q_chip.shape
    num_pages, H_kv, _, _ = k_cache_chip.shape
    out = torch.zeros((B, H_q, num_users, D), dtype=torch.float32)
    pt = page_table.long().view(page_table.shape[0], -1) if page_table.ndim >= 2 else page_table.long().view(1, -1)
    cp = cur_pos.long().view(-1)
    cp_max = num_pages * page_size - 1
    for u in range(num_users):
        cp_u = int(cp[u % cp.numel()].item()) if cp.numel() > 0 else 0
        if cp_u < 0:
            continue  # nothing to attend to
        cp_u = min(cp_u, cp_max)
        n_active = cp_u + 1
        n_pages_active = (n_active + page_size - 1) // page_size
        pt_row = pt[u % pt.shape[0]]
        # Clamp page indices into [0, num_pages-1] so torch indexing stays valid
        # for randomly-generated inputs.
        pages = pt_row[:n_pages_active].clamp_(0, num_pages - 1)
        k_pages = k_cache_chip[pages]  # (n_pages_active, H_kv, page_size, D)
        v_pages = v_cache_chip[pages]
        # Concat along sequence axis (page_size) → (H_kv, n_pages_active*page_size, D)
        k_seq = k_pages.permute(1, 0, 2, 3).reshape(H_kv, -1, D)
        v_seq = v_pages.permute(1, 0, 2, 3).reshape(H_kv, -1, D)
        # Sliding-window: ttnn paged_sdpa_decode with sliding_window_size=W
        # only attends to the last W positions before cur_pos.
        if sliding_window_size is not None and sliding_window_size > 0:
            start = max(0, n_active - int(sliding_window_size))
        else:
            start = 0
        k_seq = k_seq[:, start:n_active, :]
        v_seq = v_seq[:, start:n_active, :]
        # GQA broadcast: K/V repeat to H_q heads
        if H_kv < H_q:
            rep = H_q // H_kv
            k_seq = k_seq.repeat_interleave(rep, dim=0)
            v_seq = v_seq.repeat_interleave(rep, dim=0)
        q_u = q_chip[0, :, u, :]  # (H_q, D)
        # Multi-head attention: when H_q != H_kv, handle GQA/MQA
        H_k = k_seq.shape[0]
        H_q_local = q_u.shape[0]
        if H_q_local < H_k:
            # MQA: each Q head attends to all K/V heads independently
            # Expand Q to match K, compute attention, then average per Q head group
            q_expanded = q_u.repeat_interleave(H_k // max(H_q_local, 1), dim=0)[:H_k]
            scores = torch.einsum("hd,htd->ht", q_expanded.float(), k_seq.float()) / (D**0.5)
            attn = torch.softmax(scores, dim=-1)
            out_expanded = torch.einsum("ht,htd->hd", attn, v_seq.float())
            # Reduce back: average groups of H_k/H_q heads
            group = H_k // max(H_q_local, 1)
            out_u = out_expanded.view(H_q_local, group, D).mean(dim=1)
        else:
            scores = torch.einsum("hd,htd->ht", q_u.float(), k_seq.float()) / (D**0.5)
            attn = torch.softmax(scores, dim=-1)
            out_u = torch.einsum("ht,htd->hd", attn, v_seq.float())
        out[0, :, u, :] = out_u
    return out


def _paged_sdpa_decode_golden(
    torch_q,
    torch_k_cache,
    torch_v_cache,
    page_table,
    cur_pos,
    num_users,
    padded_users,
    factor,
    sliding_window_size=None,
):
    """Slice Q/K/V on dim -1 by `factor`, run per-chip paged attention, concat on -1.
    Returns tensor with shape (B, H_q, padded_users, D_global)."""
    B, H_q, U, D = torch_q.shape
    num_pages, H_kv, page_size, _ = torch_k_cache.shape
    if factor > 1:
        q_chunks = torch.chunk(torch_q, factor, dim=-1)
        k_chunks = torch.chunk(torch_k_cache, factor, dim=-1)
        v_chunks = torch.chunk(torch_v_cache, factor, dim=-1)
    else:
        q_chunks = (torch_q,)
        k_chunks = (torch_k_cache,)
        v_chunks = (torch_v_cache,)
    per_chip = [
        _paged_sdpa_decode_chip_attn(q, k, v, page_table, cur_pos, num_users, page_size, sliding_window_size)
        for q, k, v in zip(q_chunks, k_chunks, v_chunks)
    ]
    out = torch.cat(per_chip, dim=-1)  # (B, H_q, num_users, D_global)
    if padded_users != num_users:
        padded = torch.zeros((B, H_q, padded_users, out.shape[-1]), dtype=out.dtype)
        padded[:, :, :num_users, :] = out
        return padded
    return out


def _batch_paged_golden(q_heads, k_chip, v_chip, page_row, pos, block, scale, sliding_window=None):
    """Causal paged attention for ONE batch from device-resident shards.

    q_heads [NQH, D]; k_chip/v_chip [num_blocks, n_kv_heads, block, D]; page_row
    maps logical->physical pages; pos = most-recent cache index (inclusive).
    Returns [NQH, D]. GQA/MQA: q head h attends KV head h // (NQH // n_kv_heads),
    so the golden must NOT collapse to KV head 0 (that ignores heads 1..n_kv-1
    and yields a wrong result for multi-KV-head configs).
    """
    d = q_heads.shape[-1]
    nqh = q_heads.shape[0]
    nkvh = k_chip.shape[1]
    n_active = int(pos) + 1
    n_blocks = (n_active + block - 1) // block
    pages = page_row[:n_blocks].long().clamp_(0, k_chip.shape[0] - 1)
    rep = max(1, nqh // max(1, nkvh))
    out = torch.empty((nqh, d), dtype=torch.float32)
    for h in range(nqh):
        kvh = min(h // rep, nkvh - 1)
        k_seq = k_chip[pages, kvh].reshape(-1, d)[:n_active].float()
        v_seq = v_chip[pages, kvh].reshape(-1, d)[:n_active].float()
        # Sliding-window (local) attention: the query at position pos attends to
        # only the last `sliding_window` tokens (gemma sliding layers). Without
        # this the golden does full attention -> wrong vs the windowed device op.
        if sliding_window and n_active > int(sliding_window):
            k_seq = k_seq[-int(sliding_window) :]
            v_seq = v_seq[-int(sliding_window) :]
        w = torch.softmax((q_heads[h].float() @ k_seq.t()) * scale, dim=-1)
        out[h] = w @ v_seq
    return out


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    input_e_dtype=None,
    input_e_layout=None,
    input_e_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Open (or reuse) a mesh device whose dispatch axis matches this vector's
    # traced program_config grid. fixture yielded None; we own the device here.
    device = _ensure_vector_device(_vector_dispatch_axis(kwargs))

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    input_c_tensor_placement = kwargs.get("input_c_tensor_placement", None)
    input_d_tensor_placement = kwargs.get("input_d_tensor_placement", None)
    input_e_tensor_placement = kwargs.get("input_e_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Handle dict input_a_shape from traced configurations (multi-input)
    if isinstance(input_a_shape, dict):
        # Traced configuration with multiple inputs
        shape_a = input_a_shape.get("input_a", input_a_shape.get("self"))
        shape_b = input_a_shape.get("input_b", input_a_shape.get("other"))
        shape_c = input_a_shape.get("input_c")
        shape_d = input_a_shape.get("input_d")
        shape_e = input_a_shape.get("input_e")
    else:
        # V2 vectors store separate input_b_shape/input_c_shape/etc. columns;
        # pull them from kwargs when present, falling back to shape_a only
        # when V2 didn't provide a per-tensor shape (e.g., sample suite).
        if isinstance(input_a_shape, (tuple, list)):
            shape_a = tuple(input_a_shape)
        else:
            shape_a = input_a_shape

        def _shape_from_kwargs(k, default):
            v = kwargs.get(k)
            if v is None or v == "__ABSENT__":
                return default
            return tuple(v) if isinstance(v, (list, tuple)) else v

        shape_b = _shape_from_kwargs("input_b_shape", shape_a)
        shape_c = _shape_from_kwargs("input_c_shape", shape_b)
        shape_d = _shape_from_kwargs("input_d_shape", shape_a)
        shape_e = _shape_from_kwargs("input_e_shape", shape_a)

    # Use provided params directly - these are optional (None is fine if not in V2 JSON)
    dtype_a = input_a_dtype
    dtype_b = input_b_dtype
    dtype_c = input_c_dtype
    dtype_d = input_d_dtype
    dtype_e = input_e_dtype

    # The trace stores the cur_pos tensor under named kwargs cur_pos_tensor_*
    # (shape/dtype/layout/memory_config/tensor_placement) rather than as
    # input_e. Pull those if present and override; the kernel requires INT32.
    cur_pos_info = extract_named_tensor_kwargs(kwargs, "cur_pos_tensor")
    if cur_pos_info:
        if cur_pos_info.get("shape") is not None:
            shape_e = tuple(cur_pos_info["shape"])
        if cur_pos_info.get("dtype") is not None:
            dtype_e = cur_pos_info["dtype"]
        if cur_pos_info.get("memory_config") is not None:
            input_e_memory_config = cur_pos_info["memory_config"]
        if cur_pos_info.get("tensor_placement") is not None:
            input_e_tensor_placement = cur_pos_info["tensor_placement"]
    if dtype_e is None:
        dtype_e = ttnn.int32

    page_table_info = extract_named_tensor_kwargs(kwargs, "page_table_tensor")
    if page_table_info:
        if page_table_info.get("shape") is not None:
            shape_d = tuple(page_table_info["shape"])
        if page_table_info.get("dtype") is not None:
            dtype_d = page_table_info["dtype"]
        if page_table_info.get("memory_config") is not None:
            input_d_memory_config = page_table_info["memory_config"]
        if page_table_info.get("tensor_placement") is not None:
            input_d_tensor_placement = page_table_info["tensor_placement"]
    if dtype_d is None:
        dtype_d = ttnn.int32

    # Extract attention_sink as a named tensor kwarg if present in V2 vector.
    # The model's gpt-oss decode passes attention_sink=weights.decode_sinks;
    # the master trace records it as a sharded tensor (placement Shard(-2)).
    sink_info = extract_named_tensor_kwargs(kwargs, "attention_sink")
    sink_shape = sink_info.get("shape") if sink_info else None
    sink_dtype = sink_info.get("dtype") if sink_info else None
    sink_layout = sink_info.get("layout") if sink_info else ttnn.TILE_LAYOUT
    sink_mem_config = sink_info.get("memory_config") if sink_info else None
    sink_placement = sink_info.get("tensor_placement") if sink_info else None

    layout_a = input_a_layout
    layout_b = input_b_layout
    layout_c = input_c_layout

    mem_config_a = input_a_memory_config
    mem_config_b = input_b_memory_config
    mem_config_c = input_c_memory_config
    mem_config_d = input_d_memory_config
    mem_config_e = input_e_memory_config
    # Create input tensors
    torch_input_a = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_a)(shape_a)
    torch_input_b = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_b)(shape_b)
    torch_input_c = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_c)(shape_c)
    # page_table and cur_pos are INT32 indices, not float values.  Generating them
    # via float-random-then-cast collapses to mostly 0/-1 (degenerate goldens
    # where cur_pos<0 short-circuits to zeros, masking real correctness).
    # Derive proper ranges from the K-cache shape: dim 0 = num_pages, dim 2 = page_size.
    _num_pages = int(shape_b[0]) if len(shape_b) >= 4 else 1
    _page_size = int(shape_b[2]) if len(shape_b) >= 4 else 1
    _max_pages_per_user = int(shape_d[-1]) if len(shape_d) >= 1 else _num_pages
    _seq_len_max = max(2, min(_num_pages, _max_pages_per_user) * _page_size)
    # page_table is a paged-KV virtual->physical block map and MUST be a valid
    # (1-to-1) mapping into [0, num_pages). Filling it with arbitrary randint
    # values makes the decode kernel dereference garbage physical blocks -> a NoC
    # transaction that never completes -> device HANG (this is exactly why the
    # BS=32 sharded-page-table configs 83ebb7ca / c8c59223 hung). Build
    # b sequences x max_pages_per_user UNIQUE physical blocks, then stack one
    # identical copy per core (sharded page tables replicate the (b,mbps) map
    # across the leading dim; ordinary page tables are just (b,mbps)). cur_pos is
    # one bounded position per user, replicated the same way.
    _b = int(shape_a[1]) if len(shape_a) >= 2 else 1
    _mbps = int(_max_pages_per_user)
    try:
        _n_virt = _b * _mbps
        if 0 < _n_virt <= _num_pages:
            _valid_map = torch.randperm(_num_pages)[:_n_virt].reshape(_b, _mbps).to(torch.int32)
        else:
            _valid_map = torch.randint(0, max(_num_pages, 1), (_b, _mbps), dtype=torch.int32)
        _pt_rows = int(shape_d[0]) if len(shape_d) >= 1 else _b
        _rep_pt = max(1, _pt_rows // max(_b, 1))
        torch_input_d = _valid_map.repeat(_rep_pt, 1)[:_pt_rows].reshape(tuple(shape_d))
        _cp_b = torch.randint(1, _seq_len_max, (_b,), dtype=torch.int32)
        if int(shape_e[-1]) == _b:
            torch_input_e = _cp_b.reshape((1,) * (len(shape_e) - 1) + (_b,)).expand(tuple(shape_e)).contiguous()
        else:
            torch_input_e = torch.randint(1, _seq_len_max, tuple(shape_e), dtype=torch.int32)
    except Exception:
        # Best-effort fallback: at least keep values within the valid range.
        torch_input_d = torch.randint(0, max(_num_pages, 1), tuple(shape_d), dtype=torch.int32)
        torch_input_e = torch.randint(1, _seq_len_max, tuple(shape_e), dtype=torch.int32)

    if len(shape_a) == 4:
        try:
            B_q, num_users, H_q, D = shape_a
            num_pg, H_kv, pg_size, _ = shape_b
            cur_p = torch_input_e.long().view(-1)
            pt_full = torch_input_d.long()
            if pt_full.ndim >= 2:
                pt_full = pt_full.view(pt_full.shape[0], -1)
            else:
                pt_full = pt_full.view(1, -1)
            _scale = kwargs.get("scale", D**-0.5)
            if _scale == "__ABSENT__" or _scale is None:
                _scale = D**-0.5
            _scale = float(_scale)
            _k_chunk = 256
            _pc = kwargs.get("program_config")
            if isinstance(_pc, dict):
                import re as _re_pc

                _kcm = _re_pc.search(r"k_chunk_size=(\d+)", str(_pc.get("value", "")))
                if _kcm:
                    _k_chunk = int(_kcm.group(1))

            golden_out = torch.zeros(B_q, num_users, H_q, D, dtype=torch.float32)
            for u in range(num_users):
                cp_u = int(cur_p[u % cur_p.numel()].item()) if cur_p.numel() > 0 else 0
                cp_u = min(max(cp_u, 0), num_pg * pg_size - 1)
                n_active = cp_u + 1
                padded_len = n_active
                if _k_chunk > 0:
                    padded_len = ((n_active + _k_chunk - 1) // _k_chunk) * _k_chunk
                n_pages_active = (padded_len + pg_size - 1) // pg_size
                pt_row = pt_full[u % pt_full.shape[0]]
                max_pages = pt_row.shape[0]
                n_pages_active = min(n_pages_active, max_pages)
                padded_len = min(padded_len, n_pages_active * pg_size)
                pages = pt_row[:n_pages_active].clamp(0, num_pg - 1)
                k_pages = torch_input_b[pages].float()
                v_pages = torch_input_c[pages].float()
                k_seq = k_pages.permute(1, 0, 2, 3).reshape(H_kv, -1, D)[:, :padded_len, :]
                v_seq = v_pages.permute(1, 0, 2, 3).reshape(H_kv, -1, D)[:, :padded_len, :]
                K_exp = torch.cat([k_seq[i : i + 1].repeat(H_q // H_kv, 1, 1) for i in range(H_kv)], dim=0).unsqueeze(0)
                V_exp = torch.cat([v_seq[i : i + 1].repeat(H_q // H_kv, 1, 1) for i in range(H_kv)], dim=0).unsqueeze(0)
                q_u = torch_input_a[0, u : u + 1, :H_q, :].permute(1, 0, 2).unsqueeze(0).float()
                mask = torch.zeros(1, H_q, 1, padded_len)
                mask[:, :, :, cp_u + 1 :] = float("-inf")
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q_u, K_exp, V_exp, attn_mask=mask, scale=_scale, is_causal=False
                )
                golden_out[0, u, :, :] = attn_out.squeeze(2)
            torch_output_tensor = golden_out.to(torch_input_a.dtype)
        except Exception:
            torch_output_tensor = torch_input_a.clone()
    else:
        torch_output_tensor = torch_input_a.clone()

    # Convert to TTNN tensors
    def _is_sharded_memory_config(mc):
        if mc is None:
            return False
        try:
            return getattr(mc, "is_sharded", lambda: False)()
        except Exception:
            return False

    # --- paged_sdpa decode Q padding fix --------------------------------------
    # The decode kernel reads padded_num_heads (= nearest_pow_2(nearest_n(H_q,32)))
    # rows per core. The master trace records Q as ROW_MAJOR with shard height =
    # H_q (e.g. 8), so only user 0 aligns and the rest read padding (PCC ~0.06),
    # and the resulting placement also trips \"not on_dispatch_core\". Rebuild Q
    # as TILE with shard height = padded_num_heads so all users align, then take
    # the direct from_torch path (which pads heads up to the shard height).
    _q_pad_override = False
    try:
        if len(shape_a) == 4 and _is_sharded_memory_config(mem_config_a):
            _spec = mem_config_a.shard_spec
            _hq = int(shape_a[2])

            def _nn(x, n):
                return ((x + n - 1) // n) * n

            def _np2(x):
                p = 1
                while p < x:
                    p *= 2
                return p

            _padded = _np2(_nn(_hq, 32))
            if int(_spec.shape[0]) < _padded:
                _new_spec = ttnn.ShardSpec(_spec.grid, [_padded, int(_spec.shape[1])], _spec.orientation)
                mem_config_a = ttnn.MemoryConfig(mem_config_a.memory_layout, mem_config_a.buffer_type, _new_spec)
                layout_a = ttnn.TILE_LAYOUT
                _q_pad_override = True
    except Exception:
        _q_pad_override = False
    # --------------------------------------------------------------------------

    if is_mesh_device and input_a_tensor_placement:
        # When the destination memory_config is sharded, ttnn.from_torch
        # promotes the per-chip logical shape to the shard height (e.g. 8 -> 32).
        # The master trace records the production logical shape (8), so we
        # create the tensor in DRAM first and then to_memory_config into the
        # sharded L1 layout to preserve the logical shape.
        if _is_sharded_memory_config(mem_config_a) and not _q_pad_override:
            tensor_a_dram = create_tensor_on_mesh(
                torch_input_a, device, dtype_a, layout_a, ttnn.DRAM_MEMORY_CONFIG, input_a_tensor_placement
            )
            tensor_a = ttnn.to_memory_config(tensor_a_dram, mem_config_a)
            try:
                tensor_a_dram.deallocate(True)
            except Exception:
                # Best-effort cleanup of the intermediate DRAM tensor — if
                # deallocate raises (already-freed, mesh teardown), the sweep
                # should continue to verify correctness on tensor_a.
                pass
        else:
            tensor_a = create_tensor_on_mesh(
                torch_input_a, device, dtype_a, layout_a, mem_config_a, input_a_tensor_placement
            )
        tensor_b = create_tensor_on_mesh(
            torch_input_b, device, dtype_b, layout_b, mem_config_b, input_b_tensor_placement
        )
        tensor_c = create_tensor_on_mesh(
            torch_input_c, device, dtype_c, layout_c, mem_config_c, input_c_tensor_placement
        )
        tensor_d = create_tensor_on_mesh(
            torch_input_d,
            device,
            dtype_d,
            ttnn.ROW_MAJOR_LAYOUT,
            mem_config_d,
            input_d_tensor_placement,
        )
        tensor_e = create_tensor_on_mesh(
            torch_input_e,
            device,
            dtype_e,
            ttnn.ROW_MAJOR_LAYOUT,
            mem_config_e,
            input_e_tensor_placement,
        )
    else:
        tensor_a = ttnn.from_torch(
            torch_input_a,
            dtype=dtype_a,
            layout=layout_a,
            device=device,
            memory_config=mem_config_a,
        )
        tensor_b = ttnn.from_torch(
            torch_input_b,
            dtype=dtype_b,
            layout=layout_b,
            device=device,
            memory_config=mem_config_b,
        )
        tensor_c = ttnn.from_torch(
            torch_input_c,
            dtype=dtype_c,
            layout=layout_c,
            device=device,
            memory_config=mem_config_c,
        )
        tensor_d = ttnn.from_torch(
            torch_input_d,
            dtype=dtype_d,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mem_config_d,
        )
        tensor_e = ttnn.from_torch(
            torch_input_e,
            dtype=dtype_e,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mem_config_e,
        )

    start_time = start_measuring_time()
    # paged_scaled_dot_product_attention_decode signature:
    # (input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, *, is_causal=True, attn_mask=None, cur_pos_tensor=None, ...)
    # So tensor_a=Q, tensor_b=K, tensor_c=V, tensor_d=page_table, tensor_e=cur_pos
    #
    # The master trace records page_table_tensor as a NAMED kwarg (the model
    # called it by name), so the sweep must also pass it by name to produce a
    # matching trace.  Only pass is_causal when the master config includes it
    # (most traces omit it, relying on the default).
    is_causal = op_kwargs.pop("is_causal", None)
    if is_causal is not None:
        op_kwargs["is_causal"] = is_causal

    # Pass sliding_window_size even when None — master records it whenever the
    # model explicitly passed it (gpt-oss decode does, with config.sliding_window
    # potentially being None). build_op_kwargs strips None values, so use
    # __absent_keys__ to disambiguate "explicit None" from "key absent".
    absent_keys = kwargs.get("__absent_keys__", set())
    if "sliding_window_size" not in absent_keys and "sliding_window_size" in kwargs:
        op_kwargs["sliding_window_size"] = kwargs.get("sliding_window_size")

    # Inject the attention_sink tensor when V2 specifies it. Master records the
    # sink as a separate kwarg with its own placement; we mirror that here.
    sink_tensor = None
    if sink_shape is not None:
        torch_sink = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32),
            sink_dtype if sink_dtype is not None else ttnn.bfloat16,
        )(tuple(sink_shape))
        if is_mesh_device and sink_placement:
            sink_tensor = create_tensor_on_mesh(
                torch_sink,
                device,
                sink_dtype if sink_dtype is not None else ttnn.bfloat16,
                sink_layout if sink_layout is not None else ttnn.TILE_LAYOUT,
                sink_mem_config,
                sink_placement,
            )
        else:
            sink_tensor = ttnn.from_torch(
                torch_sink,
                dtype=sink_dtype if sink_dtype is not None else ttnn.bfloat16,
                layout=sink_layout if sink_layout is not None else ttnn.TILE_LAYOUT,
                device=device,
                memory_config=sink_mem_config,
            )
        op_kwargs["attention_sink"] = sink_tensor

    # build_op_kwargs strips program_config; parse from raw kwargs
    if "program_config" not in op_kwargs:
        traced_pc = kwargs.get("program_config")
        if isinstance(traced_pc, dict) and traced_pc.get("type") == "SDPAProgramConfig":
            import re

            val = traced_pc.get("value", "")
            # Grid is recorded either as "(x=8,y=8)" or "8-9" (a grid SIZE).
            gm = re.search(r"compute_with_storage_grid_size=\(x=(\d+),y=(\d+)\)", val) or re.search(
                r"compute_with_storage_grid_size=(\d+)-(\d+)", val
            )
            qm = re.search(r"q_chunk_size=(\d+)", val)
            km = re.search(r"k_chunk_size=(\d+)", val)
            em = re.search(r"exp_approx_mode=(\w+)", val)
            mcm = re.search(r"max_cores_per_head_batch=(\d+)", val)
            # sub_core_grids: {[x1-y1 - x2-y2], ...} — the explicit kernel
            # placement that keeps the op off dispatch cores. Must be preserved;
            # dropping it caused "not on_dispatch_core". std::nullopt when absent.
            sub_core_grids = None
            if "sub_core_grids=std::nullopt" not in val:
                ranges = re.findall(r"\[(\d+)-(\d+)\s*-\s*(\d+)-(\d+)\]", val)
                if ranges:
                    sub_core_grids = ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(ttnn.CoreCoord(int(a), int(b)), ttnn.CoreCoord(int(c), int(d)))
                            for a, b, c, d in ranges
                        }
                    )
            if gm:
                pc_kwargs = dict(
                    compute_with_storage_grid_size=ttnn.CoreCoord(int(gm.group(1)), int(gm.group(2))),
                    q_chunk_size=int(qm.group(1)) if qm else 0,
                    k_chunk_size=int(km.group(1)) if km else 0,
                )
                if em:
                    pc_kwargs["exp_approx_mode"] = em.group(1).lower() == "true"
                if mcm:
                    pc_kwargs["max_cores_per_head_batch"] = int(mcm.group(1))
                if sub_core_grids is not None:
                    pc_kwargs["sub_core_grids"] = sub_core_grids
                op_kwargs["program_config"] = ttnn.SDPAProgramConfig(**pc_kwargs)
        elif traced_pc is not None and traced_pc != "__ABSENT__" and not isinstance(traced_pc, dict):
            op_kwargs["program_config"] = traced_pc

    # Pass memory_config from V2 vector when present (master records it).
    v2_memory_config = kwargs.get("memory_config")
    if v2_memory_config is not None and v2_memory_config != "__ABSENT__":
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        op_kwargs.setdefault("memory_config", parse_dict_value("memory_config", v2_memory_config))

    ttnn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tensor_a,  # Q
        tensor_b,  # K
        tensor_c,  # V
        page_table_tensor=tensor_d,
        cur_pos_tensor=tensor_e,
        **op_kwargs,
    )
    e2e_perf = stop_measuring_time(start_time)

    # Per-chip paged-attention golden from DEVICE-RESIDENT inputs. The op pads
    # query heads to 32 and packs each batch into those head-rows, so each chip's
    # output [1, X, Y, D] holds b_eff = X*Y//32 active batches (batch k's valid
    # heads [0:Y] at dim1 slot k*(32//Y)). Build the golden from that chip's own
    # Q/K/V/cur_pos/page_table (read back) so it's robust to mesh-shard ordering
    # and matches what the chip actually computed. (cf. the standalone repro.)
    _scale = op_kwargs.get("scale")
    out_dts = ttnn.get_device_tensors(ttnn_output) if is_mesh_device else [ttnn_output]
    q_dts = ttnn.get_device_tensors(tensor_a) if is_mesh_device else [tensor_a]
    k_dts = ttnn.get_device_tensors(tensor_b) if is_mesh_device else [tensor_b]
    v_dts = ttnn.get_device_tensors(tensor_c) if is_mesh_device else [tensor_c]
    pt_dts = ttnn.get_device_tensors(tensor_d) if is_mesh_device else [tensor_d]
    cp_dts = ttnn.get_device_tensors(tensor_e) if is_mesh_device else [tensor_e]

    o0 = ttnn.to_torch(out_dts[0])

    X, Y, D = o0.shape[1], o0.shape[2], o0.shape[3]
    PADDED = 32
    NH = Y
    # Two output layouts:
    #  - [1, B, NH, D]: dim1 is the batch (one row per batch), dim2 the query
    #    heads. Detected when dim1 == #batches (cur_pos count). Iterate batches
    #    directly (stride 1) — the head-packing formula below mis-strides this
    #    whenever NH < 32 and B > 1 (it processed B*NH//32 rows at stride 32//NH,
    #    comparing batch k's golden against output row (32//NH)*k -> ~0 PCC).
    #  - head-packed: NH padded to 32 with 32//NH batches per 32-row block.
    # The per-chip batch is the LAST dim of cur_pos, not its element count:
    # sharded cur_pos is (num_cores, batch) so .numel() (=cores*batch) inflated
    # _nbatch and forced the head-packed branch (b_eff=2, stride=4) -> ~0.43 PCC.
    _cp0 = ttnn.to_torch(cp_dts[0])
    _nbatch = int(_cp0.shape[-1]) if _cp0.ndim >= 1 else int(_cp0.numel())
    if X == _nbatch:
        b_eff = X
        stride = 1
    else:
        b_eff = max(1, (X * Y) // PADDED)
        stride = max(1, PADDED // Y)
    if _scale in (None, "__ABSENT__"):
        _scale = float(D) ** -0.5
    _scale = float(_scale)
    block = ttnn.to_torch(k_dts[0]).shape[2]

    all_g, all_d = [], []
    for i in range(len(out_dts)):
        ot = ttnn.to_torch(out_dts[i])
        qc = ttnn.to_torch(q_dts[i]).float().reshape(-1, NH, D)
        kc = ttnn.to_torch(k_dts[i]).float()
        vc = ttnn.to_torch(v_dts[i]).float()
        cp = ttnn.to_torch(cp_dts[i]).reshape(-1)
        pt = ttnn.to_torch(pt_dts[i])
        pt = pt.reshape(pt.shape[-2], -1) if pt.ndim >= 2 else pt.reshape(1, -1)
        for k in range(b_eff):
            _sw = op_kwargs.get("sliding_window_size")
            g = _batch_paged_golden(
                qc[k % qc.shape[0]],
                kc,
                vc,
                pt[k % pt.shape[0]],
                int(cp[k % cp.numel()].item()),
                block,
                _scale,
                sliding_window=_sw,
            )
            all_g.append(g)
            all_d.append(ot[0, stride * k, :NH, :].float())
    pcc = check_with_pcc(torch.stack(all_g), torch.stack(all_d), 0.99)
    return [pcc, e2e_perf]
