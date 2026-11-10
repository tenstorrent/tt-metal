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
  - INDEX_K_SERVED=fp8_row192 (P.39, owner 2026-11-06): the index keys' encoding, byte for byte. Slot 0's index-key
    cache of every KV source is loaded (state.load_host: the model's own device write path, the same encode as a
    chunk's) with the golden's keys of [0, MAX_SEQ - 40) as bf16 (a last record with pad rows), and every record of
    the source and of the re-index layers serving its keys is read through the table: the rows of complete groups
    equal the stock ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(round_scale_to_power_of_two=False) of
    the same keys packed [e4m3 | fp32 scale | zeros], every byte (the writer tt-blaze's test checks against); pad rows
    all zero. Against tt-blaze's torch recipe (caches.blaze_rows, _quantize_scaled_keys): the scales byte-equal, every
    value within one e4m3 step (the stock cast lands one step higher at ~1 % of values and keeps e4m3 subnormals the
    recipe flushes; counted and printed)
    INDEX_K_COPY=bfp8 (P.41, owner-authorized test change 2026-11-07): the served rows are built from the keys' bfp8
    copy, so the recipe and the stock cast are applied to the keys the rows are built from: the golden's bf16 keys
    rounded to bfloat8_b by tt-metal's host rule (dequantized; ckv_bfp4_ttnn/tests/reference.round_values, which
    ttnn.bringup.bfp_round equals for bf16 inputs). Every other check, tolerance and limit unchanged.
    INDEX_K_SERVED_FROM=bf16 (P.44, owner-authorized test change 2026-11-10): the served rows are encoded from the
    chunk's bf16 keys again, so the reference input is the golden's keys unrounded (P.39's byte-exact check); under
    INDEX_K_SERVED_FROM=bfp8 it stays the bfp8-rounded keys above.
  - P.42 (owner-authorized contract change 2026-11-07): WINDOW_KV_KEEP=snapshot128 makes 0_window_kv a Snapshot
    config, [slots x layers, 1, 128, 512] (4 records a (slot, layer), every chip the same bytes: the table's device
    group is the whole mesh, no record past position 128); REINDEX_SERVE=alias gives the re-index layers no slab of
    their own: their 4_index_k_r1 entries equal layer 20's (caches.table_rules), so the encoding check above reads
    layer 20's bytes through them
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
    if any(c.get("fp8") for c in C.served_configs(layers)):
        fails += _encoding_failures(mesh_device, runtime, kv, path, tmp_path, layers)
    print(f"[kv layout] layers {layers}, {N_SLOTS} slots, {banks} DRAM banks: {len(fails)} failures")
    assert not fails, "\n".join(fails[:40])


def _raw(t):
    """The bytes of mesh chip (0, 0)'s shard of a replicated device tensor (its host buffer: to_torch / to_dtype of
    FP8_E4M3 flush e4m3 subnormals)."""
    import torch

    import ttnn

    host = t.cpu()
    buf = host.host_buffer()
    return torch.from_dlpack(buf.get_shard(ttnn.MeshCoordinate(0, 0))).view(torch.uint8).flatten().clone()


def _recipe_failures(got, recipe, what) -> tuple[list[str], int]:
    """Served rows vs Blaze's torch recipe: scales byte-equal, every value within one e4m3 step. (failures, values
    that differ)."""
    import torch

    g, w = torch.from_numpy(got), torch.from_numpy(recipe)
    out = []
    if not torch.equal(g[:, C.FP8_VALUES : C.FP8_VALUES + 4], w[:, C.FP8_VALUES : C.FP8_VALUES + 4]):
        out.append(f"{what}: scales differ from Blaze's recipe")
    gv = g[:, : C.FP8_VALUES].contiguous().view(torch.float8_e4m3fn).double()
    wv = w[:, : C.FP8_VALUES].contiguous().view(torch.float8_e4m3fn).double()
    step = torch.where(
        wv.abs() >= 2**-6, 2.0 ** (torch.floor(torch.log2(wv.abs().clamp_min(2**-6))) - 3), 2.0**-6
    )
    far = int(((gv - wv).abs() > step).sum())
    if far:
        out.append(f"{what}: {far} values beyond one e4m3 step of Blaze's recipe")
    return out, int((gv != wv).sum())


def _stock_rows(mesh, keys):
    """The stock per_token_cast_to_fp8 (round_scale_to_power_of_two=False) of bf16 keys [n, 128] on the device, packed
    [e4m3 | fp32 scale | zeros] on the host: uint8 [n, 192]. TILE input in blocks of 1024 rows (32 cores each: the
    cast's TILE path runs one core per 32 x 128 block, agent rule 18)."""
    import numpy as np
    import torch

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.cores import from_torch

    out = []
    for a in range(0, keys.shape[0], 1024):
        blk = keys[a : a + 1024].to(torch.bfloat16)
        x = from_torch(
            blk.reshape(1, 1, -1, 128),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        e4m3, sc = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False)
        n = blk.shape[0]
        v = _raw(e4m3)[: n * 128].reshape(n, 128)  # (raw bytes: subnormals kept)
        s = _raw(sc)[: n * 4].reshape(n, 4)
        for t in (x, e4m3, sc):
            ttnn.deallocate(t)
        raw = torch.zeros(n, C.FP8_ROW_BYTES, dtype=torch.uint8)
        raw[:, : C.FP8_VALUES] = v
        raw[:, C.FP8_VALUES : C.FP8_VALUES + 4] = s
        raw[((v & 0x7F) == 0).all(dim=-1)] = 0  # (an all-zero key: a pad row)
        out.append(raw.numpy())
    return np.concatenate(out)


def _encoding_failures(mesh, runtime, kv, path, tmp_path, layers) -> list[str]:
    """P.39: slot 0's served index keys, written by the model's device path from the golden's keys, vs Blaze's recipe
    and the stock cast, byte for byte, through the table (the source's and its re-index layers' records)."""
    import numpy as np

    import ttnn

    gk = C.GoldenKV()
    reader = C.TableReader(path, C.device_map(mesh, str(tmp_path / "device_map.json")))
    st = runtime.state(kv, 0)
    n_tok = R.MAX_SEQ - 40  # (a last record with pad rows)
    fails, checked = [], []
    for cfg in C.served_configs(layers):
        if not cfg.get("fp8"):
            continue
        r = cfg["ratio"]
        for src in [i for i in cfg["layers"] if i not in cfg["of"]]:
            keys = gk.g.state(src)["index_k"][: n_tok // r].float()
            st.compress[src].index_k.load_host(keys, 0)
            ttnn.synchronize_device(mesh)
            from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

            k8_source = settings.get("INDEX_K_SERVED_FROM") == "bfp8"  # P.44: bf16 = the golden keys as they are
            if st.compress[src].index_k.k8 is not None and k8_source:  # INDEX_K_COPY=bfp8 (P.41): the bfp8 keys
                from ttnn.bringup.ckv_bfp4_ttnn.tests.reference import round_values

                keys = round_values(keys.bfloat16().float(), mbits=7)
            recipe = C.blaze_rows(keys)
            stock = _stock_rows(mesh, keys)
            for layer in [src] + sorted(j for j, s in cfg["of"].items() if s == src):
                recs = reader.records(cfg["name"], 0, [layer], 0, n_tok)
                got, pad = C.valid_and_pad(recs, cfg, layer, 0, n_tok, raw_rows=True)
                what = f"encoding: {cfg['name']} layer {layer} (keys of layer {src}, {got.shape[0]} rows)"
                if got.shape != stock.shape or got.shape != recipe.shape:
                    fails.append(f"{what}: shape {got.shape} vs per_token_cast_to_fp8 {stock.shape}")
                    continue
                bad = np.nonzero((got != stock).any(axis=1))[0]
                if bad.size:
                    i = int(bad[0])
                    cols = np.nonzero(got[i] != stock[i])[0]
                    fails.append(
                        f"{what}: {bad.size} rows differ from per_token_cast_to_fp8, first row {i} at bytes "
                        f"{cols[:8].tolist()}"
                    )
                f, nd = _recipe_failures(got, recipe, what)
                fails += f
                fails += C.fp8_row_failures(got, pad, cfg, layer, "encoding")
                checked.append((cfg["name"], layer, int(got.shape[0]), int(pad.shape[0]), f"recipe-diff {nd}"))
    print(f"[kv layout] fp8 index-key encoding checked byte for byte (config, layer, rows, pad rows): {checked}")
    if not checked:
        fails.append("encoding: no fp8 index-key cache checked")
    return fails
