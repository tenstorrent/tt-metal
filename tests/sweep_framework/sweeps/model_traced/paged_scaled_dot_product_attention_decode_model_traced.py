# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs

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
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)
    del device


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
        # scores: (H_q, n_active)
        scores = torch.einsum("hd,htd->ht", q_u.float(), k_seq.float()) / (D**0.5)
        attn = torch.softmax(scores, dim=-1)
        out_u = torch.einsum("ht,htd->hd", attn, v_seq.float())  # (H_q, D)
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
    _seq_len_max = max(2, _num_pages * _page_size)
    torch_input_d = torch.randint(0, max(_num_pages, 1), tuple(shape_d), dtype=torch.int32)
    torch_input_e = torch.randint(1, _seq_len_max, tuple(shape_e), dtype=torch.int32)

    # Real paged-SDPA-decode golden. Per user u: gather K/V from page_table[u]
    # pages, trim to cur_pos[u]+1 positions, compute SDPA. Q/K_cache/V_cache are
    # Shard(-1) at runtime, so we slice on -1 and run per-chip then concat —
    # matching what mesh_tensor_to_torch reassembles.
    if len(shape_a) == 4:
        # ttnn paged_sdpa_decode returns logical shape (B, H_q, num_users, D);
        # mesh_tensor_to_torch strips the tile-pad, so do not pad the golden.
        # Trace-validation mode: every chip receives the FULL per-chip Q/K/V via
        # replicate_with_topology and runs paged-SDPA on them. Pass factor=1 so
        # the golden does NOT chunk; reconcile_golden_to_actual handles the
        # shard-axis tile of the gathered output.
        _sliding_window = kwargs.get("sliding_window_size")
        if _sliding_window == "__ABSENT__":
            _sliding_window = None
        torch_output_tensor = _paged_sdpa_decode_golden(
            torch_input_a,
            torch_input_b,
            torch_input_c,
            torch_input_d,
            torch_input_e,
            num_users=shape_a[2],
            padded_users=shape_a[2],
            factor=1,
            sliding_window_size=_sliding_window,
        ).to(torch_input_a.dtype)
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

    if is_mesh_device and input_a_tensor_placement:
        # When the destination memory_config is sharded, ttnn.from_torch
        # promotes the per-chip logical shape to the shard height (e.g. 8 -> 32).
        # The master trace records the production logical shape (8), so we
        # create the tensor in DRAM first and then to_memory_config into the
        # sharded L1 layout to preserve the logical shape.
        if _is_sharded_memory_config(mem_config_a):
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

    # Pass memory_config from V2 vector when present (master records it).
    v2_memory_config = kwargs.get("memory_config")
    if v2_memory_config is not None and v2_memory_config != "__ABSENT__":
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        op_kwargs.setdefault("memory_config", parse_dict_value("memory_config", v2_memory_config))

    output_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tensor_a,  # Q
        tensor_b,  # K
        tensor_c,  # V
        page_table_tensor=tensor_d,
        cur_pos_tensor=tensor_e,
        **op_kwargs,
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, input_b_tensor_placement
        )
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
