# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# HAND-AUTHORED (not generated). Scaled-down variants of graph-capture cases
# whose real geometry needs more Tensix cores than the 2-node Quasar emulator
# has, so the captured case is unrunnable there.
#
# The captured nlp_concat_heads_decode case shards its output across
# num_heads=32 cores (compute_output_specs: num_cores_to_corerangeset(num_heads,
# grid)), so on the 1-core emulator it aborts before the op runs:
#   "Target number of cores 32 is greater than total number of available cores 1"
#
# These variants drop num_heads to 1 so the whole op fits on a single core. The
# kernel path (and its Quasar uplift — the borrowed q_out DFB -> LocalTensorAccessor
# conversion) is the same; only the head fan-out shrinks.
#
# The same treatment is applied to the other GREEN-uplifted ops whose captured
# cases are prefill-sized or span 32 cores (typecast, sharded_to_interleaved,
# nlp_concat_heads) and to the paged SDPA-decode case, whose captured
# SDPAProgramConfig names an 8x8 grid the emulator does not have. Shapes shrink,
# dtypes / layouts / memory configs / kwargs stay as captured. Run with:
#   pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_emu_small_grid.py -m emulator
#
# bfloat8_b is replaced by bfloat16 (typecast targets float32): Quasar has no Bfp8_b
# (is_supported_quasar lists the MxFp8 formats instead) and ValidateProgramSpec
# rejects any DFB carrying it before the program is built — confirmed on the
# emulator with the captured bf8 typecast / sharded_to_interleaved / paged-cache
# cases. Every bf8 tensor in the llama32_1b capture (KV cache, MLP + lm_head
# weights, SDPA activations) is therefore unrunnable on Quasar as captured.
#
# ``_call`` (optional) names the callable when it differs from ``op`` — the paged
# decode case keeps the mainline op name so graph_case's data generators
# (page table / cur_pos) still key on it, but runs the experimental.quasar fork,
# exactly as test_paged_scaled_dot_product_attention_decode.py does.
# ---------------------------------------------------------------------------
"""Single-core emulator smoke variants of otherwise-too-wide graph_ops cases."""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

# nlp_concat_heads_decode with a single head: input is one height-sharded shard on
# core (0,0); output is width-sharded across num_heads (=1) cores, i.e. core (0,0).
_NLP_CONCAT_HEADS_DECODE_1HEAD = {
    "id": "1head_32x64_bf16_hs-l1",
    "op": "ttnn.experimental.nlp_concat_heads_decode",
    "count": 1,
    "args": [
        {
            "k": "t",
            "shape": [1, 1, 32, 64],
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
            },
        },
    ],
    "kwargs": {
        "num_heads": {"k": "lit", "v": 1},
    },
    # Output spec unobserved (None) -> graph_case checks count + finiteness and lets
    # the op pin down its own placement, so we don't have to hand-derive the shard grid.
    "outs": [None],
}

_DRAM_INT = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}
_L1_INT = {"layout": "INTERLEAVED", "buffer": "L1", "shard": None}


def _hs_l1_1core(shard_shape):
    return {
        "layout": "HEIGHT_SHARDED",
        "buffer": "L1",
        "shard": {"grid": [[0, 0, 0, 0]], "shape": list(shard_shape), "orientation": "ROW_MAJOR"},
    }


# typecast: captured [1,8,1024,64] / [1,32,1024,64] bf16 -> bf8 (prefill-sized).
# One tile-row keeps the TILE-interleaved TypecastProgramFactory path; the target
# dtype is float32 because bf8 does not exist on Quasar (see header).
_TYPECAST_1TILE = {
    "id": "typecast_32x64_bf16_to_fp32_int-dram",
    "op": "ttnn.typecast",
    "count": 1,
    "args": [{"k": "t", "shape": [1, 1, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _DRAM_INT}],
    "kwargs": {"dtype": {"k": "dtype", "v": "FLOAT32"}},
    "outs": [{"k": "t", "shape": [1, 1, 32, 64], "dtype": "FLOAT32", "layout": "TILE", "mem": _DRAM_INT}],
}

# sharded_to_interleaved: captured width-shards [1,1,32,8192] bf8 over 32 cores
# (shard [32,256]). One shard of the same [32,256] geometry on core (0,0), bf16.
_S2I_1SHARD = {
    "id": "s2i_32x256_bf16_ws-l1_1core",
    "op": "ttnn.sharded_to_interleaved",
    "count": 1,
    "args": [
        {
            "k": "t",
            "shape": [1, 1, 32, 256],
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 256], "orientation": "ROW_MAJOR"},
            },
        }
    ],
    "kwargs": {"memory_config": dict(_L1_INT, k="mem")},
    "outs": [None],
}

# nlp_concat_heads: captured [1,32,1024,64] -> [1,1,1024,2048] (prefill). Four heads
# of one tile-row each: [1,4,32,64] -> [1,1,32,256]; same interleaved reader/writer.
_NLP_CONCAT_HEADS_4HEADS = {
    "id": "nlp_concat_heads_4x32x64_bf16_int-dram",
    "op": "ttnn.experimental.nlp_concat_heads",
    "count": 1,
    "args": [{"k": "t", "shape": [1, 4, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _DRAM_INT}],
    "kwargs": {"memory_config": dict(_DRAM_INT, k="mem")},
    "outs": [{"k": "t", "shape": [1, 1, 32, 256], "dtype": "BFLOAT16", "layout": "TILE", "mem": _DRAM_INT}],
}

# paged SDPA decode: the captured geometry (32 q heads, 8 kv heads, 128 pages x 32
# rows caches, cur_pos 8) fits the emulator as-is; only the captured 8x8 compute
# grid does not. compute_with_storage_grid_size -> [1,1]; caches bf8 -> bf16.
_PAGED_SDPA_DECODE_1CORE = {
    "id": "paged_sdpa_decode_32x64_bf16_hs-l1_grid1x1_bf16kv",
    "op": "ttnn.transformer.paged_scaled_dot_product_attention_decode",
    "_call": "ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode",
    "count": 1,
    "args": [
        {"k": "t", "shape": [1, 1, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _hs_l1_1core([32, 64])},
        {"k": "t", "shape": [128, 8, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _DRAM_INT},
        {"k": "t", "shape": [128, 8, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _DRAM_INT},
    ],
    "kwargs": {
        "page_table_tensor": {"k": "t", "shape": [1, 128], "dtype": "INT32", "layout": "ROW_MAJOR", "mem": _DRAM_INT},
        "cur_pos_tensor": {"k": "t", "shape": [1], "dtype": "INT32", "layout": "ROW_MAJOR", "mem": _DRAM_INT},
        "scale": {"k": "lit", "v": 0.125},
        "sliding_window_size": {"k": "lit", "v": None},
        "program_config": {
            "kind": "SDPAProgramConfig",
            "fields": {
                "compute_with_storage_grid_size": [1, 1],
                "sub_core_grids": None,
                "q_chunk_size": 0,
                "k_chunk_size": 0,
                "exp_approx_mode": True,
                "max_cores_per_head_batch": 16,
            },
            "k": "cfg",
        },
        "memory_config": dict(_DRAM_INT, k="mem"),
    },
    "outs": [{"k": "t", "shape": [1, 1, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _DRAM_INT}],
}

# paged_update_cache: captured case fits the emulator except for its bf8 cache
# ([128,8,32,64]); same geometry with a bf16 cache. The update (arg 1) is the
# height-sharded single-core [1,1,8,64] bf16 slab exactly as captured.
_PAGED_UPDATE_CACHE_BF16 = {
    "id": "paged_update_cache_128x8x32x64_bf16_int-dram",
    "op": "ttnn.experimental.paged_update_cache",
    "count": 1,
    "args": [
        {"k": "t", "shape": [128, 8, 32, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _DRAM_INT},
        {"k": "t", "shape": [1, 1, 8, 64], "dtype": "BFLOAT16", "layout": "TILE", "mem": _hs_l1_1core([32, 64])},
    ],
    "kwargs": {
        "update_idxs_tensor": {"k": "t", "shape": [1], "dtype": "INT32", "layout": "ROW_MAJOR", "mem": _DRAM_INT},
        "page_table": {"k": "t", "shape": [1, 128], "dtype": "INT32", "layout": "ROW_MAJOR", "mem": _DRAM_INT},
    },
    "outs": [None],
}

CASES = [
    _NLP_CONCAT_HEADS_DECODE_1HEAD,
    _PAGED_UPDATE_CACHE_BF16,
    _TYPECAST_1TILE,
    _S2I_1SHARD,
    _NLP_CONCAT_HEADS_4HEADS,
    _PAGED_SDPA_DECODE_1CORE,
]


def _resolve(dotted):
    """'ttnn.experimental.foo' -> the ttnn callable."""
    obj = ttnn
    for name in dotted.split(".")[1:]:
        obj = getattr(obj, name)
    return obj


# Known Quasar blockers, confirmed on emu-quasar-1x3 (2026-09-03). WH/BH run these
# cases normally (they pass); on Quasar they xfail, strictly — a fix shows up as a failure
# telling you to drop the entry.
_XFAIL = {
    _PAGED_SDPA_DECODE_1CORE["id"]: (
        "sdpa_decode quasar fork: DFB 'out_o' sets allow_instance_multi_binding (tree-reduction "
        "writer P+C / compute P+C); ValidateProgramSpec rejects the flag on Gen2 (program_spec.cpp:1288)"
    ),
}


def _run_expecting_quasar_blocker(fn, mesh_device, reason):
    """Quasar-only strict xfail: WH/BH must pass; on Quasar the known blocker must still fire."""
    if not G._is_quasar(mesh_device):
        fn()
        return
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — the blocker surfaces as a TT_FATAL RuntimeError
        pytest.xfail(f"{reason}: {str(exc).splitlines()[0][:160]}")
    pytest.fail(f"XPASS on Quasar — blocker fixed, drop the _XFAIL entry: {reason}")


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_emu_small_grid(ttnn_mesh_device, reset_seeds, case):
    op = _resolve(case.get("_call", case["op"]))
    run = lambda: G.run_case(op, case, ttnn_mesh_device, op_name=case["op"])  # noqa: E731
    if case["id"] in _XFAIL:
        _run_expecting_quasar_blocker(run, ttnn_mesh_device, _XFAIL[case["id"]])
    else:
        run()
