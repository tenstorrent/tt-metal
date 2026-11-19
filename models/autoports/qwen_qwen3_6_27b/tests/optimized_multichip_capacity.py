# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only stage-5 TP4 inventory, without TTNN/torch imports or file writes.

This is a source-derived allocation model, not a measured allocator peak or a
DRAM transaction counter. Import and call plan with final effective policies;
the CLI illustrates integrated1 plus inherited stage-4 geometry. See the audit.
"""

import json
import math

TILE_BYTES = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048, "float32": 4096}


def align(n, multiple):
    return (n + multiple - 1) // multiple * multiple


def tensor_bytes(shape, dtype="bfloat16", layout="TILE"):
    """Eight-bank DRAM allocation, including last-bank and row alignment."""
    if layout == "TILE":
        pages = math.prod(shape[:-2]) * (align(shape[-2], 32) // 32) * (align(shape[-1], 32) // 32)
        page_bytes = TILE_BYTES[dtype]
    else:
        pages = math.prod(shape[:-1])
        page_bytes = shape[-1] * {"bfloat16": 2, "float32": 4, "uint32": 4, "int32": 4}[dtype]
    return align(pages, 8) * align(page_bytes, 64)


def matrices(kind, policy=None):
    p = {"cores": 8, "readers": 2, "packed_mlp": True, "narrow_gdn_correction": True, **(policy or {})}
    shapes = {"mlp.down_proj.weight": (4352, 5120)}
    if p["packed_mlp"]:
        shapes["mlp.gate_up.weight"] = (5120, 8704)
    else:
        shapes.update({f"mlp.{role}_proj.weight": (5120, 4352) for role in ("gate", "up")})
    if kind == "linear_attention":
        shapes.update(
            {
                "packed_gdn": (5120, 4160),
                "packed_gdn_residual": (5120, 2624 if p["narrow_gdn_correction"] else 4160),
                "linear_attn.out_proj.weight": (1536, 5120),
            }
        )
    elif kind == "full_attention":
        shapes["self_attn.o_proj.weight"] = (1536, 5120)
        shapes.update(
            {"packed_qkvg": (5120, 3584)}
            if p.get("packed_qkvg")
            else {"packed_qkv": (5120, 2048), "gate": (5120, 1536)}
        )
    else:
        raise ValueError(kind)
    joint = p.get("packed_correction_options") if kind == "linear_attention" else None
    if joint is not None:
        if joint.get("release_separate", True):
            del shapes["packed_gdn"], shapes["packed_gdn_residual"]
        shapes["packed_gdn_correction_joint"] = (5120, 6784)
    result = {}
    for name, (k, n) in shapes.items():
        opts = {**p, **p.get("geometry", {}).get(name, {})}
        if name == "packed_gdn_correction_joint":
            opts = {
                **p,
                **p.get("geometry", {}).get("packed_gdn", {}),
                **{key: value for key, value in joint.items() if key in ("cores", "readers", "block")},
            }
        if k % (32 * opts["cores"]):
            raise ValueError(f"Invalid activation K/core geometry: {name}")
        width = (
            align(n, 32 * math.lcm(8 * opts["readers"], opts["cores"])) if p.get("dram_sharded", True) else align(n, 32)
        )
        role = "mlp" if name.startswith("mlp.") else "attention"
        dtype = (
            "bfloat8_b"
            if name.endswith("_residual") or name == "packed_gdn_correction_joint"
            else p.get(role + "_dtype", "bfloat8_b")
        )
        result[name] = {
            "logical_shape": [k, n],
            "physical_shape": [k, width],
            "dtype": dtype,
            "bytes": tensor_bytes((k, width), dtype),
            "active_decode": not (joint is not None and name in ("packed_gdn", "packed_gdn_residual")),
        }
    return result


def layer_inventory(kind, policy=None):
    p = policy or {}
    matrix = matrices(kind, p)
    primary = sum(v["bytes"] for v in matrix.values())
    row_copies = (
        sum(
            tensor_bytes(v["logical_shape"], v["dtype"])
            for k, v in matrix.items()
            if k.endswith(("down_proj.weight", "out_proj.weight", "o_proj.weight"))
        )
        if p.get("fused_mmrs", True)
        else 0
    )
    swiglu = tensor_bytes((5120, 8704), p.get("mlp_dtype", "bfloat8_b")) if p.get("minimal_swiglu", "prefill") else 0
    return {
        "matrices": matrix,
        "decode_matrix_read_once_bytes": sum(v["bytes"] for v in matrix.values() if v["active_decode"]),
        "mmrs_copy_bytes": row_copies,
        "swiglu_copy_bytes": swiglu,
        "resident_matrix_bytes": primary + row_copies + swiglu,
    }


def plan(
    batch=1,
    context=262144,
    *,
    table_columns=None,
    physical_pages=None,
    gdn_policy=None,
    full_policy=None,
    ccl_rs_signatures_per_layer=1,
    ccl_ag_signatures_per_layer=None,
    ccl_stats_signatures_per_layer=0,
):
    """Plan independent layers' state and weights plus future embedding/head.

    table_columns is BEFORE create_paged_state's >=16 column rounding.
    physical_pages is max(original_page_table)+1, not the number of live pages.
    ccl_rs_signatures_per_layer counts warmed short-M reduce signatures with
    identical BF16/width5120 buffers; use zero when persistent_ccl is disabled.
    Additional dtypes/shapes require their own inventory, not this shortcut.
    AG count defaults to RS count. Supply stats AG count explicitly if warmed;
    these outputs add DRAM capacity only with ccl_decode_memory='DRAM'. Counts
    are common to both layer kinds; calculate differing inventories separately.
    """
    if not (1 <= batch <= 32 and 1 <= context <= 262144):
        raise ValueError("Outside audited batch/context range")
    gp, fp = gdn_policy or {}, full_policy or {}
    layers = {k: layer_inventory(k, p) for k, p in (("linear_attention", gp), ("full_attention", fp))}
    columns = table_columns if table_columns is not None else (context + 31) // 32
    pages = physical_pages if physical_pages is not None else batch * columns
    padded_columns = align(columns, 16) if columns >= 16 else columns
    pages += batch * (padded_columns - columns)
    kv = 16 * 2 * tensor_bytes((pages, 1, 32, 256), "bfloat8_b")
    state = 48 * tensor_bytes((batch, 12, 128, 128))
    conv = 48 * tensor_bytes((batch, 3, 2560), layout="ROW_MAJOR")
    # All tensors allocated in from_state_dict, excluding matrices and states.
    aux_specs = [
        (128, (1, 1, 5120), "float32", "TILE"),
        (128, (1, 1, 1280), "float32", "TILE"),
        (32, (1, 1, 256), "float32", "TILE"),
        (48, (1, 1, 128), "bfloat16", "TILE"),
        (144, (1, 1, 12), "float32", "TILE"),
        (192, (1, 1, 2560), "bfloat16", "TILE"),
        (192, (1, 1, 2560), "bfloat16", "ROW_MAJOR"),
        (48, (1, 12, 2560), "bfloat16", "ROW_MAJOR"),
        (144, (1, 1, 32, 32), "float32", "TILE"),
        (48, (1, 1, 32, 96), "float32", "TILE"),
        (48 * 513, (1,), "uint32", "ROW_MAJOR"),
        (16, (batch, padded_columns), "int32", "ROW_MAJOR"),
    ]
    auxiliary = sum(count * tensor_bytes(shape, dtype, layout) for count, shape, dtype, layout in aux_specs)
    future = tensor_bytes((62080, 5120)) + tensor_bytes((5120, 62464)) + tensor_bytes((1, 1, 5120), "float32")
    weights = (
        48 * layers["linear_attention"]["resident_matrix_bytes"]
        + 16 * layers["full_attention"]["resident_matrix_bytes"]
    )
    ccl_dram = sum(
        count * ccl_rs_signatures_per_layer * tensor_bytes((2, 1, 32, 5120), p.get("ccl_dtype", "bfloat16"))
        for count, p in ((48, gp), (16, fp))
        if p.get("persistent_ccl", False)
    )
    ag_count = ccl_rs_signatures_per_layer if ccl_ag_signatures_per_layer is None else ccl_ag_signatures_per_layer
    ccl_dram += sum(
        count
        * (
            ccl_rs_signatures_per_layer * tensor_bytes((1, 1, 32, 1280), p.get("ccl_dtype", "bfloat16"))
            + ag_count * tensor_bytes((1, 1, 32, 5120), p.get("ccl_dtype", "bfloat16"))
            + ccl_stats_signatures_per_layer * tensor_bytes((1, 1, 32, 128))
        )
        for count, p in ((48, gp), (16, fp))
        if p.get("persistent_ccl", False) and p.get("ccl_decode_memory") == "DRAM"
    )
    # create_paged_state owns one immutable row-index tensor for each warmed
    # B>1 signature. This plan models the single requested batch signature.
    cache_indices = (
        16 * tensor_bytes((batch,), "int32", "ROW_MAJOR") if batch > 1 and fp.get("batched_prefill_cache", False) else 0
    )
    from ..tt.fused_decode_norm import capacity as norm_capacity

    norm_inventory = {
        kind: norm_capacity(
            p.get("distributed_decode_norm", "off"), activation_dram=p.get("ccl_decode_memory") == "DRAM"
        )
        for kind, p in (("linear_attention", gp), ("full_attention", fp))
    }
    norm_dram = sum(
        count * norm_inventory[kind]["dram_bytes_bound"]
        for kind, count in (("linear_attention", 48), ("full_attention", 16))
    )
    resident = weights + future + kv + state + conv + auxiliary + ccl_dram + cache_indices + norm_dram
    hidden = tensor_bytes((batch, context, 1280))
    rope_pair = 2 * tensor_bytes((1, 1, context, 64))
    known_transients = 3 * hidden + rope_pair + tensor_bytes((batch, 12, 128, 128), "float32") + 32 * 1024**2
    return {
        "batch": batch,
        "context": context,
        "layers": layers,
        "table_columns": padded_columns,
        "physical_pages_per_layer": pages,
        "resident_64_layer_matrices_bytes": weights,
        "future_bf16_embedding_head_and_norm_bytes": future,
        "kv_bytes": kv,
        "recurrent_bytes": state,
        "convolution_history_bytes": conv,
        "auxiliary_bytes": auxiliary,
        "persistent_ccl_dram_bytes": ccl_dram,
        "prefill_batch_index_bytes": cache_indices,
        "fused_decode_norm_inventory_per_layer": norm_inventory,
        "fused_decode_norm_persistent_dram_bytes_bound": norm_dram,
        "resident_bytes": resident,
        "reserve_bytes": 4 * 1024**3,
        "known_transient_components_bytes": known_transients,
        "reserve_remainder_bytes": 4 * 1024**3 - known_transients,
        "estimated_total_bytes": resident + 4 * 1024**3,
        "nominal_32gib_margin_bytes": 32 * 1024**3 - resident - 4 * 1024**3,
        "status": "source-derived plan; not a device peak or full-model acceptance",
    }


def decode_lower_bound(kind, batch, visible_lengths, policy=None, bandwidth=512e9):
    """Physical stored matrix tiles once + unavoidable state/KV reads.

    visible_lengths includes the token being decoded, independently per user.
    No allocator slack, duplicate resident copies, or CCL intermediate capacity
    is treated as traffic. Excludes cache updates and intermediate movements.
    """
    if len(visible_lengths) != batch:
        raise ValueError("Provide one visible KV length for each logical user")
    weight = layer_inventory(kind, policy)["decode_matrix_read_once_bytes"]
    recurrent = 2 * batch * 12 * 128 * 128 * 2 if kind == "linear_attention" else 0
    kv = 2 * 8 * 1088 * sum((n + 31) // 32 for n in visible_lengths) if kind == "full_attention" else 0
    total = weight + recurrent + kv
    return {
        "matrix_read_once_bytes": weight,
        "recurrent_read_write_bytes": recurrent,
        "visible_kv_tile_read_bytes": kv,
        "lower_bound_bytes": total,
        "roofline_ms": total / bandwidth * 1000,
    }


def recurrent_kernel_traffic(
    batch, *, cores=110, grouped_qk=True, variant="cumulative_exp_scalar", g_bytes=4, beta_bytes=2
):
    """Exact source-requested payload for raw packed BF16 local12 decode.

    Covers only this kernel, not convolution, projections, or NoC/bus protocol
    overhead. State dtype is BF16 and output FP32. All inputs/outputs are DRAM.
    This actual-path counter is deliberately separate from the roofline.
    """
    if batch < 1 or cores < 1 or g_bytes not in (2, 4) or beta_bytes not in (2, 4):
        raise ValueError("Invalid recurrent kernel parameters")
    heads = 12 * batch
    ncores = min(heads, cores)
    grouped = grouped_qk and batch >= 3
    grouped_reads = variant == "grouped_reads" or variant.startswith("cumulative")
    cached_gate = variant == "gate_cache" or variant.startswith("cumulative")
    start = qk_count = gate_count = 0
    for index in range(ncores):
        count = heads // ncores + (index < heads % ncores)
        qk_count += sum(
            not grouped_reads or not grouped or h == start or h % 3 == 0 for h in range(start, start + count)
        )
        gate_count += sum(not cached_gate or h == start or h % 12 == 0 for h in range(start, start + count))
        start += count
    parts = {
        "state_read_bytes": heads * 16 * 2048,
        "state_write_bytes": heads * 16 * 2048,
        "qk_read_bytes": qk_count * 2 * 256,
        "v_read_bytes": heads * 256,
        "g_beta_read_bytes": gate_count * 16 * (g_bytes + beta_bytes),
        "output_write_bytes_including_padding": align(heads, 32) * 512,
    }
    return {
        **parts,
        "total_requested_dram_payload_bytes": sum(parts.values()),
        "qk_group_reads": qk_count,
        "gate_vector_reads": gate_count,
    }


if __name__ == "__main__":
    gp = {"persistent_ccl": True}
    fp = {"persistent_ccl": True, "packed_qkvg": True}
    print(
        json.dumps(
            {
                "full_context": plan(gdn_policy=gp, full_policy=fp),
                "short_cases": [
                    {
                        "batch": b,
                        "length": s,
                        "capacity": plan(b, s, table_columns=(s + 33 + 31) // 32, gdn_policy=gp, full_policy=fp),
                        "decode": {
                            k: decode_lower_bound(k, b, [s + 1] * b, p)
                            for k, p in (("linear_attention", gp), ("full_attention", fp))
                        },
                    }
                    for b in (1, 32)
                    for s in (33, 513)
                ],
            },
            indent=2,
        )
    )
