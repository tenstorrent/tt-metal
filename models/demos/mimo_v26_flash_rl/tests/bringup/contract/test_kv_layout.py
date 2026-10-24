# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, "KV cache" (serving_contract.md): the device model's state is the served KV cache.

For every served layer and state tensor (key after RoPE, value after the value scale), the tensor
``state.kv_tensors(layer)[name]`` must be the cache tt-d-gen's KV Manager reads through the table:
    dtype       the served KV dtype (serving_answers.yaml kv_dtype, else the spec's dataformats.kv_cache)
    storage     TILE, DRAM, ND-sharded with shard [1, 1, 32, width], ROUND_ROBIN_1D over every DRAM bank (one 32-token
                record of one head contiguous in one bank; never DRAM-interleaved)
    width       key 192 on every layer, value 128 (MiMo: head_dim / v_head_dim, swa_head_dim / swa_v_head_dim)
    per chip    a whole number of regions of (KV heads / TP) x (max_seq / SP) rows: full-attention layers 1 head,
                sliding layers 2 heads per chip on 8x4, every position kept (the sliding cache is not window-bounded)
and ``state.formats()`` names the same dtype for key and value. Gates M.1 (assemble): the ladder runs on this cache.
"""

import math

import pytest
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

from models.demos.mimo_v26_flash_rl.tests.bringup.contract import _serving as S


def _problems(ttnn, mesh, s, state) -> list[str]:
    cfg, geo = S.hf_config(s), S.geometry(s)
    want_dtype = getattr(ttnn, S.kv_dtype(s))
    banks = mesh.dram_grid_size().x
    rows = geo["max_seq"] // geo["sp"]
    out = []
    formats = state.formats()
    for name in ("key", "value"):
        if formats.get(name) != want_dtype:
            out.append(f"state.formats()[{name!r}] is {formats.get(name)}, the served KV dtype is {want_dtype}")
    for layer in geo["layers"]:
        tensors = state.kv_tensors(layer)
        heads = S.kv_heads(cfg, layer)
        if heads % geo["tp"]:
            out.append(f"layer {layer}: {heads} KV heads do not split over TP {geo['tp']}")
            continue
        local = heads // geo["tp"]
        for name, width in S.widths(cfg, layer).items():
            where = f"layer {layer} {name}"
            t = tensors.get(name)
            if t is None:
                out.append(f"{where}: state.kv_tensors({layer}) has no {name!r}")
                continue
            if t.dtype != want_dtype:
                out.append(
                    f"{where}: dtype {t.dtype}, want {want_dtype} (one cache format: the ladder runs on the "
                    "served one)"
                )
            if t.layout != ttnn.TILE_LAYOUT:
                out.append(f"{where}: layout {t.layout}, want TILE (the KV Manager decodes 32x32 tiles)")
            mc = t.memory_config()
            nd = mc.nd_shard_spec
            if mc.buffer_type != ttnn.BufferType.DRAM or nd is None:
                out.append(
                    f"{where}: memory config {mc} is not an ND-sharded DRAM buffer (a DRAM-interleaved cache "
                    "spreads one 32-token record over several banks; a table record is one bank)"
                )
            else:
                if list(nd.shard_shape) != [1, 1, S.RECORD, width]:
                    out.append(f"{where}: ND shard {list(nd.shard_shape)}, want [1, 1, {S.RECORD}, {width}]")
                if nd.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
                    out.append(f"{where}: shard distribution {nd.shard_distribution_strategy}, want ROUND_ROBIN_1D")
                if nd.grid.num_cores() != banks:
                    out.append(f"{where}: shards over {nd.grid.num_cores()} DRAM banks, want all {banks}")
            shape = list(t.shape)
            if shape[-1] != width:
                out.append(f"{where}: row width {shape[-1]}, want {width}")
            region = local * rows
            n_rows = math.prod(shape[:-1])
            if n_rows % region or n_rows < region:
                out.append(
                    f"{where}: per-chip shape {shape} holds {n_rows} rows, not a whole number of regions of "
                    f"{local} head(s) x {rows} positions (max_seq {geo['max_seq']} / SP {geo['sp']})"
                )
    return out


@mesh_parametrize
@device_timeout(spec())
def test_kv_layout(mesh_device):
    if mesh_device is None:
        pytest.skip("no mesh (CPU proof mode): the served cache is a device property")
    import ttnn

    s = spec()
    geo = S.geometry(s)
    try:
        model = s.hooks().device_model(mesh_device, s, geo["layers"], lm_head=False)
    except NotImplementedError as e:
        pytest.fail(f"not built: hooks.device_model ({e})", pytrace=False)
    state = model.new_state(geo["max_seq"])
    missing = [f"state.{m}()" for m in ("formats", "kv_tensors") if not callable(getattr(state, m, None))]
    if missing:
        pytest.fail(
            "not built: the device model's state lacks " + ", ".join(missing) + " (serving_contract.md, 'KV cache')",
            pytrace=False,
        )
    problems = _problems(ttnn, mesh_device, s, state)
    for p in problems:
        print(f"FAIL {p}")
    print(f"kv layout: {len(geo['layers'])} layers checked, {len(problems)} problems")
    assert not problems, f"{len(problems)} KV layout problems, first: {problems[0]}"
