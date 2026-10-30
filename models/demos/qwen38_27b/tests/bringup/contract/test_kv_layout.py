# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served KV cache is the device model's state (serving_contract.md, "KV cache"; gates assemble).

The KV Manager reads one 32-token record of one (slot, layer, head) from ONE DRAM bank
(tt-d-gen kv_manager/src/control_plane/services/migration_strategy_builder.cpp:147-165), so every attention layer's
K and V cache is ND-sharded in DRAM with shard [1, 1, 32, head_dim], ROUND_ROBIN_1D over all DRAM banks, in the served
KV dtype (serving_answers.yaml kv_dtype), TILE layout (the layout tt-d-gen's kv_dump_compare.py decodes). The GDN
state (conv tail, recurrent matrix) keeps the precision the owner chose (gdn_state_dtype).

Device model hooks used: device_model(mesh, spec, layers, lm_head=False).new_state(max_seq) -> state with
state.formats() -> {name: ttnn dtype} and state.cache_tensors(layer) -> {name: ttnn.Tensor}.
Metrics: kv_layout_failures.
"""

from __future__ import annotations

import pytest
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
RECORD_TOKENS = 32
HEAD_DIM = 256  # Qwen3.8 full attention head_dim (config.json text_config.head_dim)
DTYPES = {
    "bfp8": "bfloat8_b",
    "bfloat8_b": "bfloat8_b",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}


def served_dtypes() -> dict[str, str]:
    """{state name: ttnn dtype name} the owner's answers ask for."""
    from bringup_fw.testing import serving as SV

    kv = SV.answer(S, "kv_dtype") or S.get("dataformats.kv_cache.default") or "bfp8"
    gdn = SV.answer(S, "gdn_state_dtype") or {"recurrent": "fp32", "conv": "bf16"}
    if not isinstance(gdn, dict):  # "recurrent=fp32,conv=bf16"
        gdn = dict(kv_.split("=") for kv_ in str(gdn).replace(" ", "").split(",") if "=" in kv_)
    out = {"key": DTYPES[str(kv).lower()], "value": DTYPES[str(kv).lower()]}
    out.update({k: DTYPES[str(v).lower()] for k, v in gdn.items() if str(v).lower() in DTYPES})
    return out


def attention_layers() -> list[int]:
    fixed = set(S.state_fixed)
    return [i for i in S.layers() if set(S.state_names(i)) - fixed]


@device_timeout(S)
@mesh_parametrize
def test_kv_layout(mesh_device):
    if mesh_device is None:
        pytest.skip("no mesh (BRINGUP_IMPL=reference): the served layout is the device model's")
    from bringup_fw.core import metrics

    import ttnn

    try:
        model = S.hooks().device_model(mesh_device, S, S.layers(), lm_head=False)
    except NotImplementedError as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    state = model.new_state(int(S.get("target.seq")))
    missing = [h for h in ("formats", "cache_tensors") if not callable(getattr(state, h, None))]
    if missing:
        pytest.fail(
            f"not built: the device model's state lacks {missing} (serving_contract.md, 'KV cache')", pytrace=False
        )

    want = served_dtypes()
    fails = []
    formats = {k: str(v).split(".")[-1].lower() for k, v in state.formats().items()}
    for name in sorted({n for i in S.layers() for n in S.state_names(i)}):
        if name not in formats:
            fails.append(f"state.formats() has no entry for {name!r}")
        elif name in want and formats[name] != want[name]:
            fails.append(f"state.formats()[{name!r}] is {formats[name]}, the served format is {want[name]}")

    banks = mesh_device.dram_grid_size().x
    for layer in attention_layers():
        tensors = state.cache_tensors(layer)
        for name in ("key", "value"):
            t = tensors.get(name)
            where = f"layer {layer} {name}"
            if t is None:
                fails.append(f"{where}: state.cache_tensors() has no {name!r} tensor")
                continue
            mc = t.memory_config()
            nd = mc.nd_shard_spec
            if mc.buffer_type != ttnn.BufferType.DRAM:
                fails.append(f"{where}: buffer {mc.buffer_type}, not DRAM")
            if nd is None:
                fails.append(
                    f"{where}: not ND-sharded ({mc.memory_layout}); a 32-token record must be contiguous in "
                    "one DRAM bank, never DRAM-interleaved"
                )
                continue
            shard = [int(x) for x in nd.shard_shape]
            if shard != [1, 1, RECORD_TOKENS, HEAD_DIM]:
                fails.append(f"{where}: shard {shard}, want [1, 1, {RECORD_TOKENS}, {HEAD_DIM}]")
            if nd.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
                fails.append(f"{where}: distribution {nd.shard_distribution_strategy}, want ROUND_ROBIN_1D")
            if nd.grid.num_cores() != banks:
                fails.append(f"{where}: sharded over {nd.grid.num_cores()} DRAM banks, want all {banks}")
            if str(t.dtype).split(".")[-1].lower() != want[name]:
                fails.append(f"{where}: dtype {t.dtype}, the served format is {want[name]}")
            if t.layout != ttnn.TILE_LAYOUT:
                fails.append(f"{where}: layout {t.layout}, want TILE (what tt-d-gen's kv_dump_compare decodes)")
    if hasattr(state, "free"):
        state.free()
    metrics.record("kv_layout_failures", len(fails))
    for f in fails:
        print(f"FAIL kv layout: {f}")
    assert not fails, f"{len(fails)} KV layout problems, first: {fails[0]}"
