# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``edge_trim`` (QWEN38_EDGE_TRIM=1, default off): trims of the decode step outside the 48 layer blocks (embedding, PLE,
LM-head epilogue).  Options (``QWEN38_EDGE_TRIM_OPTS`` = comma list replacing the default set):

  candidate_row  the sampling candidate row from greedy_tail's scan (registry kernel ``candidate_row``, served at TP=8):
                 replaces the per-shard ttnn.topk chain (9 programs, ~210 us).  Values are the chain's bits; ids of
                 EQUAL values may come in another order (the scan takes the lowest id first, topk's network does
                 not), and at a shard's 32nd value a tie may keep a different id.  The greedy token is unaffected.
  embed_reduce   the token embedding's all_reduce + mesh_partition (reduce_scatter 22.8 + all_gather 13.9 + partition
                 1.7 us) as ONE row_reduce v2 program: seven of the eight partials are exact zeros, so the reduce-scatter
                 of the row is the partition of the all-reduce, bit for bit.
  ple_gather     the PLE's embedding / key / query all_gathers (hidden-width bf16 rows) on row_gather
  ple_kv         the PLE key and value projections as one DRAM-sharded matmul
"""

from __future__ import annotations

import os

ENV = "QWEN38_EDGE_TRIM"
OPTS_ENV = "QWEN38_EDGE_TRIM_OPTS"
ALL_OPTS = ("candidate_row", "embed_reduce", "ple_gather", "ple_kv")
DEFAULT_OPTS = (
    "embed_reduce",
    "ple_kv",
)  # bitwise, clear unit gains (x48); ple_gather (marginal) and candidate_row (tie order of ids) opt-in


def options(environ=os.environ) -> tuple[str, ...]:
    if environ.get(ENV, "1") != "1":  # default on (x48)
        return ()
    listed = environ.get(OPTS_ENV)
    opts = DEFAULT_OPTS if listed is None else tuple(o for o in listed.split(",") if o)
    unknown = set(opts) - set(ALL_OPTS)
    if unknown:
        raise ValueError(f"{OPTS_ENV}: unknown options {sorted(unknown)}")
    return opts


def on(name: str, environ=os.environ) -> bool:
    return name in options(environ)


def hidden_gather(x, *, memory_config):
    """``ttnn.all_gather(x, dim=3, cluster_axis=1)`` of a hidden-width bf16 TILE row block: with 'ple_gather' on and the
    contract admitted and one valid row (a 4-row block measured no faster), ``row_gather`` (only the valid rows cross the fabric; consumers are row independent), else the
    stock all_gather."""

    import ttnn

    if on("ple_gather"):
        from .. import row_gather  # noqa: PLC0415

        if int(x.shape[2]) == 1 and row_gather.admits(
            x, cluster_axis=1, memory_config=memory_config
        ):  # 4 rows: no gain
            return row_gather.row_gather(x, cluster_axis=1, memory_config=memory_config)
    return ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=memory_config)


KV_CORES = int(
    os.environ.get("QWEN38_EDGE_TRIM_KV_CORES", "8")
)  # activation storage cores of the merged matmul (8: 10 K tiles each, in0_block_w 5)
_KV: dict[int, tuple] = {}


def build_kv(module):
    """The PLE key and value weights (per device ``[2560, 1280]`` and ``[2560, 320]``, DRAM interleaved) as ONE DRAM-sharded
    bf16 weight ``[2560, 1600]`` (device d's columns ``[key_d | value_d]``), with the matmul's activation and program
    configs, built once at construction (before any trace capture) from the resident tensors; nothing is cached on disk
    and the originals stay."""

    import torch

    import ttnn

    from ... import decode_matmul as dm
    from ...contracts import MESH_SHAPE

    mesh = module.mesh_device
    key, value = module.weights.key, module.weights.value
    parts = []
    for k_dev, v_dev in zip(ttnn.get_device_tensors(key), ttnn.get_device_tensors(value)):
        parts.append(torch.cat([ttnn.to_torch(k_dev), ttnn.to_torch(v_dev)], dim=3))
    host = torch.cat(parts, dim=3).contiguous()  # [1, 1, 2560, TP * 1600]
    k_dim, n_dim = int(host.shape[2]), int(host.shape[3]) // len(parts)
    weight = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dm.dram_sharded_weight_memory_config(mesh, k_dim, n_dim),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, 3)),
    )
    act_config, program_config = dm.dram_sharded_matmul_configs(mesh, k_dim, n_dim, num_cores=KV_CORES)
    weight.update_tensor_topology(key.tensor_topology())
    _KV[id(module)] = (weight, act_config, program_config, int(ttnn.get_device_tensors(key)[0].shape[3]))
    return _KV[id(module)]


def ple_kv(module, full_embedding):
    """``(key_flat, value)`` of ``Qwen38TTNNPLE._project``'s two linears as one matmul on the merged weight: each output
    column is its own dot product over the same K tiles in the same order, so the columns are the two linears' bits."""

    import ttnn

    weight, act_config, program_config, key_width = _KV[id(module)]
    activation = ttnn.to_memory_config(full_embedding, act_config)
    out = ttnn.linear(
        activation,
        weight,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=program_config,
        compute_kernel_config=module.compute_config,
    )
    ttnn.deallocate(activation)
    n_dim = int(out.shape[3])
    key_flat = ttnn.slice(out, (0, 0, 0, 0), (1, 1, 1, key_width), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    value = ttnn.slice(out, (0, 0, 0, key_width), (1, 1, 1, n_dim), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)
    key_flat.update_tensor_topology(module.weights.key.tensor_topology())
    value.update_tensor_topology(module.weights.value.tensor_topology())
    return key_flat, value
