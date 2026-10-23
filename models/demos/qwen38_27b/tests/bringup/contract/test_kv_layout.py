# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, KV cache (serving_contract.md "KV cache"): the device model's state is the served cache.

Full-attention key / value: the kv_dtype answer, TILE, DRAM, ND-sharded with shard [1, 1, 32, head_dim] ROUND_ROBIN_1D
over every DRAM bank (each 32-token record contiguous in one bank, as the KV Manager reads it), never interleaved.
Linear-attention conv / recurrent: the fixed_state_dtype answer (default recurrent fp32, conv bf16).

Hooks used: device_model(mesh, spec, layers, lm_head=False), model.new_state(max_seq), state.formats() ->
{name: ttnn dtype}, state.device_tensors(layer) -> {name: ttnn.Tensor}.
"""

from __future__ import annotations

from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

from models.demos.qwen38_27b.tests.bringup.contract import _common as C

S = spec()
RECORD = 32


def _layout_fails(name: str, t, head_dim: int, n_banks: int | None) -> list[str]:
    import ttnn

    fails = []
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM:
        fails.append(f"{name}: buffer {mc.buffer_type}, want DRAM")
    nd = mc.nd_shard_spec
    if nd is None:
        fails.append(f"{name}: not ND-sharded ({mc.memory_layout}); one record must be contiguous in one DRAM bank")
        return fails
    shape = list(nd.shard_shape)
    if shape[-2:] != [RECORD, head_dim] or any(d != 1 for d in shape[:-2]):
        fails.append(f"{name}: shard shape {shape}, want [1, 1, {RECORD}, {head_dim}]")
    if nd.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
        fails.append(f"{name}: shard distribution {nd.shard_distribution_strategy}, want ROUND_ROBIN_1D")
    if n_banks is not None and nd.grid.num_cores() != n_banks:
        fails.append(f"{name}: shards over {nd.grid.num_cores()} DRAM banks, want all {n_banks}")
    if t.layout != ttnn.TILE_LAYOUT:
        fails.append(f"{name}: layout {t.layout}, want TILE")
    return fails


@device_timeout(S)
@mesh_parametrize
def test_kv_layout(mesh_device):
    if mesh_device is None:
        return  # CPU proof mode: no device model to inspect
    import ttnn

    s = S
    layers = s.layers()
    model = C.device_model(s, mesh_device, layers)
    C.require(model, "new_state")
    state = model.new_state(int(s.get("target.seq")))
    C.require(state, "formats", "device_tensors", what="device model state")
    kv_want = getattr(ttnn, C.KV_TTNN[str(C.answer(s, "kv_dtype", "bfp8")).lower()])
    fx = C.answer(s, "fixed_state_dtype", {"recurrent": "fp32", "conv": "bf16"})
    fx = fx if isinstance(fx, dict) else {"recurrent": fx, "conv": fx}
    fixed_want = {k: getattr(ttnn, C.STATE_TTNN[str(v).lower()]) for k, v in fx.items()}
    try:
        n_banks = int(mesh_device.dram_grid_size().x)
    except Exception:  # noqa: BLE001  (older ttnn: the bank count is then not checked)
        n_banks = None

    fails = []
    formats = state.formats()
    for name in ("key", "value"):
        if formats.get(name) != kv_want:
            fails.append(f"state.formats()[{name!r}] = {formats.get(name)}, want {kv_want} (kv_dtype answer)")
    for name, want in fixed_want.items():
        if formats.get(name) != want:
            fails.append(f"state.formats()[{name!r}] = {formats.get(name)}, want {want} (fixed_state_dtype answer)")
    head_dim = int(C.text_config(s)["head_dim"])
    for i in C.attn_layers(s):
        tensors = state.device_tensors(i)
        for name in ("key", "value"):
            if name not in tensors:
                fails.append(f"layer {i}: state.device_tensors has no {name!r}")
                continue
            t = tensors[name]
            if t.dtype != kv_want:
                fails.append(f"layer {i} {name}: dtype {t.dtype}, want {kv_want}")
            fails += [f"layer {i} " + f for f in _layout_fails(name, t, head_dim, n_banks)]
    for i in C.linear_layers(s):
        tensors = state.device_tensors(i)
        for name, want in fixed_want.items():
            if name not in tensors:
                fails.append(f"layer {i}: state.device_tensors has no {name!r}")
            elif tensors[name].dtype != want:
                fails.append(f"layer {i} {name}: dtype {tensors[name].dtype}, want {want}")
    if hasattr(state, "free"):
        state.free()
    C.report("kv_layout", fails)
