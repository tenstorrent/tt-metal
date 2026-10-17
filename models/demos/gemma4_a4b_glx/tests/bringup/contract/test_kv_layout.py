# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served KV cache layout, on the device model's state (a sliding and a global layer):

- every tensor holding a layer's KV (state.cache_tensors(layer) -> {name: ttnn.Tensor}) is in DRAM, ND-sharded with
  shard [1, 1, 32, width], ROW_MAJOR, ROUND_ROBIN_1D over every DRAM bank: one 32-token record of one head is
  contiguous in one bank, which is what one KV table entry (an address and a size) can describe;
- its dtype is the served KV dtype (state.formats(), the kv_dtype answer, bfp8 by default);
- it is zero when allocated (pad rows past actual_end stay zero until written).
"""

import pytest
from bringup_fw.testing.contract import engine_env
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
engine_env(S)
pytestmark = device_timeout(S)
LAYERS = (4, 5)  # sliding (8 KV heads x 256), global (2 KV heads x 512, K = V)
RECORD = 32


def _not_built(why):
    pytest.fail(f"not built: {why}", pytrace=False)


@mesh_parametrize
def test_kv_layout(mesh_device):
    if mesh_device is None:
        return
    import torch

    import ttnn

    seq = S.rung(S.data["ladder"][0]["name"])["seq"]
    model = S.hooks().device_model(mesh_device, S, list(LAYERS), lm_head=False)
    state = model.new_state(seq)
    if not callable(getattr(state, "cache_tensors", None)):
        _not_built("state.cache_tensors(layer) -> {name: ttnn.Tensor} (the tensors the KV table points into)")
    served = set(state.formats().values())
    if served != {ttnn.bfloat8_b}:
        print(f"NOTE served KV formats {served} (the default answer is bfp8)")
    banks = mesh_device.dram_grid_size().x
    want_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    failed = []
    for layer in LAYERS:
        tensors = state.cache_tensors(layer)
        if not tensors:
            failed.append(f"layer {layer}: no cache tensors")
        for name, t in tensors.items():
            where = f"layer {layer} {name}"
            mc = t.memory_config()
            nd = getattr(mc, "nd_shard_spec", None)
            if mc.buffer_type != ttnn.BufferType.DRAM:
                failed.append(f"{where}: buffer {mc.buffer_type}, want DRAM")
            if nd is None:
                failed.append(
                    f"{where}: {mc.memory_layout} with no ND shard spec (DRAM-interleaved scatters a record "
                    "over banks; one table entry cannot describe it)"
                )
                continue
            shape = tuple(nd.shard_shape)
            if len(shape) != 4 or shape[:3] != (1, 1, RECORD) or shape[3] % 32:
                failed.append(f"{where}: shard {shape}, want [1, 1, {RECORD}, width] (width a multiple of 32)")
            if nd.grid != want_grid:
                failed.append(f"{where}: shard grid {nd.grid}, want all {banks} DRAM banks")
            if nd.orientation != ttnn.ShardOrientation.ROW_MAJOR:
                failed.append(f"{where}: orientation {nd.orientation}, want ROW_MAJOR")
            if nd.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
                failed.append(f"{where}: {nd.shard_distribution_strategy}, want ROUND_ROBIN_1D")
            if t.dtype not in served:
                failed.append(f"{where}: dtype {t.dtype} is not the served KV format {served}")
            if t.layout != ttnn.TILE_LAYOUT:
                failed.append(f"{where}: layout {t.layout}, want TILE")
            nz = [i for i, d in enumerate(ttnn.get_device_tensors(t)) if torch.count_nonzero(ttnn.to_torch(d))]
            if nz:
                failed.append(f"{where}: not zero when allocated on chips {nz[:4]}")
            print(f"{where}: shape {tuple(t.shape)} {t.dtype} shard {shape}")
    assert not failed, "\n".join(failed)
