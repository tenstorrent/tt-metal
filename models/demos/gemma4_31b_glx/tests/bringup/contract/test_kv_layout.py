# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""serving_contract.md, "KV cache": the device model's state IS the served cache.

On layers 0-5 (five sliding layers and global layer 5) of the hooks' device model:
  - state.formats() names every cache tensor with the served dtype (the kv_dtype answer, default bfloat8_b);
  - state.cache(layer) -> {name: ttnn.Tensor} gives the device tensors: global layers one packed 640-wide cache
    (K and V views overlap), sliding layers K and V, 256 wide;
  - every one is a DRAM ND-sharded tensor, shard [1, 1, 32, width], ROUND_ROBIN_1D, ROW_MAJOR orientation, over all
    mesh.dram_grid_size().x DRAM banks: one 32-token table record is contiguous in one bank (tt-d-gen
    migration_strategy_builder.cpp: one entry is one address + size). Never DRAM-interleaved;
  - TILE layout, shape [slots * layers_in_tensor, local heads, max_seq / sp, width] with heads per chip: 1 global,
    4 sliding (TP 4), max_seq / sp rows per chip (SP 8 block-cyclic, 640 rows per chunk per chip).
"""

import pytest
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
LAYERS = list(range(6))


@device_timeout(S)
@mesh_parametrize
def test_kv_layout(mesh_device):
    if mesh_device is None:
        pytest.skip("CPU proof mode: no mesh")
    import ttnn
    from models.demos.gemma4_31b_glx.tests.bringup.contract import serving_helpers as H

    chunk, max_seq, sp = H.geometry(S)
    try:
        model = S.hooks().device_model(mesh_device, S, LAYERS, lm_head=False)
    except NotImplementedError as e:
        H.not_built(f"hooks.device_model: {e}")
    state = model.new_state(max_seq)
    for attr in ("formats", "cache"):
        if not callable(getattr(state, attr, None)):
            H.not_built(f"state.{attr}() (serving_contract.md, 'KV cache')")
    want_dtype = getattr(ttnn, H.kv_dtype_name(S))
    banks = mesh_device.dram_grid_size().x
    fails = []
    fmts = state.formats()
    for name, dt in fmts.items():
        if dt != want_dtype:
            fails.append(f"state.formats()[{name!r}] = {dt}, served {want_dtype}")
    for layer in LAYERS:
        tensors = state.cache(layer)
        width = H.GLOBAL_PACKED if H.is_global(layer) else H.SLIDING_DIM
        heads = H.GLOBAL_HEADS // S.mesh[1] if H.is_global(layer) else H.SLIDING_HEADS // S.mesh[1]
        if H.is_global(layer) and len(tensors) != 1:
            fails.append(f"layer {layer} (global): {len(tensors)} cache tensors; one packed 640-wide cache expected")
        if not H.is_global(layer) and len(tensors) != 2:
            fails.append(f"layer {layer} (sliding): {len(tensors)} cache tensors; K and V expected")
        for name, t in tensors.items():
            tag = f"layer {layer} {name}"
            if t.dtype != want_dtype:
                fails.append(f"{tag}: dtype {t.dtype}, served {want_dtype}")
            if t.layout != ttnn.TILE_LAYOUT:
                fails.append(f"{tag}: layout {t.layout}, TILE expected (kv_dump_compare.py decodes tiles)")
            shape = list(t.shape)
            if shape[-1] != width or shape[-2] != max_seq // sp or shape[-3] != heads:
                fails.append(f"{tag}: per-chip shape {shape}, want [.., {heads}, {max_seq // sp}, {width}]")
            mc = t.memory_config()
            if mc.buffer_type != ttnn.BufferType.DRAM:
                fails.append(f"{tag}: buffer {mc.buffer_type}, DRAM expected")
            nd = getattr(mc, "nd_shard_spec", None)
            if nd is None:
                fails.append(f"{tag}: not ND-sharded ({mc.memory_layout}); a record must sit in one DRAM bank")
                continue
            if list(nd.shard_shape) != [1, 1, 32, width]:
                fails.append(f"{tag}: shard {list(nd.shard_shape)}, want [1, 1, 32, {width}]")
            if nd.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
                fails.append(f"{tag}: distribution {nd.shard_distribution_strategy}, ROUND_ROBIN_1D expected")
            if nd.orientation != ttnn.ShardOrientation.ROW_MAJOR:
                fails.append(f"{tag}: orientation {nd.orientation}, ROW_MAJOR expected")
            if nd.grid.num_cores() != banks:
                fails.append(f"{tag}: grid of {nd.grid.num_cores()} banks, all {banks} DRAM banks expected")
    if hasattr(state, "free"):
        state.free()
    for f in fails:
        print(f"FAIL kv_layout: {f}")
    assert not fails, f"{len(fails)} KV layout checks failed: {fails[:4]}"
