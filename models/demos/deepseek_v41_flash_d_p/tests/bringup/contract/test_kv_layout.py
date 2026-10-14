# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, layout test: the allocated caches and the exported KV table, checked on the CPU
(serving_contract.md "KV cache", "Adapter and table").

The adapter allocates the served layers' caches (caches.served_prefix: the whole model in a full run, layers 0-3 in
this partial one) for 2 slots at chunk 2048 / max_seq 4096 and exports the table the way the runner's migration path
does (kv_migration_stages -> allgather_kv_stage_layouts -> build_kv_chunk_table(..., first_layer_idx, num_my_layers,
stage_layouts)). No chunk runs. Pass:
  - every served cache (KvCaches.by_config, one tensor per table config): bf16 ROW_MAJOR in DRAM, ND-sharded with
    shard [1, 1, rows, width] (32 token positions per shard: 32 rows, 16 for a ratio-2 compressed cache),
    ROUND_ROBIN_1D over every DRAM bank: one record = one shard = one address and size in one bank
  - the table (caches.table_rules): the launch harness's tables.read_table accepts it; the configs are the served ones
    in order, each covering exactly its layers, every (slot, layer, position) record present with the config's record
    size; every record's offset 64-byte aligned and inside its bank's allocation of that cache; no two records overlap
    on one chip and bank
"""

import pytest

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)
N_SLOTS = 2


@mesh_parametrize
def test_kv_layout(mesh_device, tmp_path):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.common.prefill.runners.migration import get_num_dram_banks

    R.self_check()
    layers = C.served_prefix()
    try:
        _, runtime, kv = C.build(mesh_device, layers, N_SLOTS)
    except C.NotBuilt as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    banks = get_num_dram_banks(mesh_device)
    fails, limits = [], {}
    for cfg in C.served_configs(layers):
        f, lim = C.layout_failures(kv.by_config[cfg["name"]], cfg, banks)
        fails += f
        if lim:
            limits[cfg["name"]] = lim
            print(f"[kv layout] {cfg['name']}: layers {cfg['layers']}, bank allocation [{lim[0]:#x}, {lim[1]:#x})")
    path = C.export_table(runtime, kv, mesh_device, str(tmp_path / "table.pb"), len(layers))
    fails += [f"table: {x}" for x in C.table_rules(path, layers, N_SLOTS, limits)]
    print(f"[kv layout] layers {layers}, {N_SLOTS} slots, {banks} DRAM banks: {len(fails)} failures")
    assert not fails, "\n".join(fails[:40])
