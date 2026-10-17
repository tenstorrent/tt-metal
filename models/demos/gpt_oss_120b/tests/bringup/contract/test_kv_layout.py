# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, KV cache (serving_contract.md "## KV cache"): the device model's state IS the served cache.

Written by SC.1 from tt-d-gen's code; frozen with every task whose gate runs it (gates: assemble).

The KV Manager reads one table record = one address and a size in ONE DRAM bank, 32 tokens of one head
(kv_manager/src/control_plane/services/migration_strategy_builder.cpp:147-165), so every layer's K and V cache must be
an ND-sharded DRAM tensor with shard [1, 1, 32, width], ROUND_ROBIN_1D over every DRAM bank of the chip, never
DRAM-interleaved; in the served dtype (serving_answers.yaml kv_dtype); and full length on every layer: a sliding layer
(window 128) keeps all max_seq positions too, since the table must hold a record for every (slot, layer, position).

The device model interface this checks (testing/harness.py plus the contract's hook):
    model = hooks.device_model(mesh, spec, layers, lm_head=False); state = model.new_state(max_seq)
    state.formats() -> {"key": ttnn dtype, "value": ttnn dtype}
    state.cache_tensors(layer) -> {"key": ttnn.Tensor, "value": ttnn.Tensor}   the device caches backing the layer
"""

import os

import bringup_fw
import pytest
from bringup_fw.core import metrics
from bringup_fw.testing import serving as SV
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

# the run's own framework ($BRINGUP_FW_DIR, set by env.sh), never a copy the workspace shadows it with
assert not os.environ.get("BRINGUP_FW_DIR") or os.path.realpath(bringup_fw.__file__).startswith(
    os.path.realpath(os.environ["BRINGUP_FW_DIR"]) + os.sep
), f"bringup_fw imported from {bringup_fw.__file__}, not $BRINGUP_FW_DIR"
S = spec()
MAX_SEQ = int(S.get("target.seq"))
HEAD_DIM = 64
RECORD_TOKENS = 32
KV_DTYPES = {"bfp8": "bfloat8_b", "bfloat8_b": "bfloat8_b", "bf16": "bfloat16", "bfloat16": "bfloat16"}


def record_width() -> int:
    """Columns of one 32-token record: one head (per_head, the default) or every head a chip holds (per_chip)."""
    layout = str(SV.answer(S, "kv_table_configs") or "per_head")
    n_kv = int(S.get("checkpoint.config.num_key_value_heads"))
    return HEAD_DIM * (n_kv // S.mesh[1] if layout == "per_chip" else 1)


@device_timeout(S)
@mesh_parametrize
def test_kv_layout(mesh_device):
    if mesh_device is None:  # the framework's CPU proof modes open no mesh
        return
    import ttnn

    want = getattr(ttnn, KV_DTYPES[str(SV.answer(S, "kv_dtype") or "bfp8").lower()])
    try:
        model = S.hooks().device_model(mesh_device, S, S.layers(), lm_head=False)
    except NotImplementedError as e:
        pytest.fail(f"not built: hooks.device_model ({e})", pytrace=False)
    state = model.new_state(MAX_SEQ)
    missing = [h for h in ("formats", "cache_tensors") if not callable(getattr(state, h, None))]
    if missing:
        pytest.fail(
            f"not built: the device model's state has no {', '.join(f'{m}()' for m in missing)} "
            "(serving_contract.md, KV cache)",
            pytrace=False,
        )

    sp, banks, width = S.mesh[0], mesh_device.dram_grid_size().x, record_width()
    fails = []
    formats = state.formats()
    for name in ("key", "value"):
        if formats.get(name) != want:
            fails.append(f"state.formats()[{name!r}] = {formats.get(name)}, the served KV dtype is {want}")
    for layer in S.layers():
        caches = state.cache_tensors(layer)
        for name in ("key", "value"):
            t = caches.get(name)
            where = f"layer {layer} {name}"
            if t is None:
                fails.append(f"{where}: state.cache_tensors({layer}) has no {name!r}")
                continue
            mc = t.memory_config()
            nd = mc.nd_shard_spec
            if mc.buffer_type != ttnn.BufferType.DRAM:
                fails.append(f"{where}: buffer {mc.buffer_type}, the served cache lives in DRAM")
            if nd is None:
                fails.append(
                    f"{where}: {mc.memory_layout} with no ND shard spec; a DRAM-interleaved cache spreads one "
                    "32-token record over pages in many banks, and a table entry is one address in ONE bank"
                )
            else:
                if list(nd.shard_shape) != [1, 1, RECORD_TOKENS, width]:
                    fails.append(f"{where}: shard {list(nd.shard_shape)}, want [1, 1, {RECORD_TOKENS}, {width}]")
                if nd.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
                    fails.append(f"{where}: shards {nd.shard_distribution_strategy}, want ROUND_ROBIN_1D")
                if nd.grid.num_cores() != banks:
                    fails.append(f"{where}: shards over {nd.grid.num_cores()} DRAM banks, want all {banks}")
            if t.dtype != want:
                fails.append(f"{where}: dtype {t.dtype}, want the served {want}")
            if t.layout != ttnn.TILE_LAYOUT:
                fails.append(f"{where}: layout {t.layout}, want TILE (the KV Manager decodes bfp8 tiles)")
            shape = list(t.shape)
            if shape[-1] != width or shape[-2] != MAX_SEQ // sp:
                fails.append(
                    f"{where}: per-chip shape {shape}, want [..., {MAX_SEQ // sp}, {width}]: every layer, sliding "
                    f"ones too, holds all {MAX_SEQ} positions ({MAX_SEQ // sp} per SP chip)"
                )
    metrics.record("kv_layout_failures", len(fails))
    for f in fails:
        print(f"FAIL kv layout: {f}")
    assert not fails, f"{len(fails)} KV layout problems, first: {fails[0]}"
