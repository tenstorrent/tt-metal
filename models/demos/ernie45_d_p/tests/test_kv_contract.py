# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.15: KV cache in the prefill-server contract layout.

Runs the full model chunked 2k->2k with contract-KV writes enabled (bf8, [users*layers, 1, seq, 128] per chip,
DRAM ROUND_ROBIN_1D), builds the KV chunk address table exactly as the server consumes it, and reads EVERY
32-token chunk of every (layer, head, K/V) back through `table.read_device_chunk` (the migration read path).
Checks: table structure, protobuf round-trip, table readback == device tensor bytes, and PCC vs the golden
(K in native interleaved order: no HF->Meta permutation for ERNIE).
"""

import os
import tempfile

import torch

import ttnn
from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.reference.ernie_ref import pcc
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import Golden
from models.demos.ernie45_d_p.tt.kv_contract import ErnieContractKV
from models.demos.ernie45_d_p.tt.model import TtErnieModel

TASK = "P2.15"
BLOCK = 32
D = ttnn.experimental.disaggregation


@mesh_1x4
def test_kv_contract(mesh_device, cfg, loader, record):
    G = Golden(4096, 2048)
    task = record.task
    L, n_kv, hd, seq, chunk = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, G.seq, G.chunk
    tokens = G.tokens()
    model = TtErnieModel(mesh_device, loader, cfg, lm_head=False)
    cache = model.new_cache(seq)
    ckv = ErnieContractKV(mesh_device, num_layers=L, max_seq=seq)
    for c in range(seq // chunk):
        h = model.prefill_chunk(tokens[c * chunk : (c + 1) * chunk], c * chunk, cache, contract_kv=ckv)
        ttnn.deallocate(h)
    ttnn.synchronize_device(mesh_device)

    failed = []

    def check(name, ok):
        print(f"{'ok  ' if ok else 'FAIL'} contract: {name}")
        if not ok:
            failed.append(name)

    k_t, v_t = ckv.cache.k, ckv.cache.v
    check("per-chip K/V shape [users*layers, 1, seq, head_dim]", list(k_t.shape) == [L, 1, seq, hd] == list(v_t.shape))
    check("cache dtype bfloat8_b", k_t.dtype == ttnn.bfloat8_b and v_t.dtype == ttnn.bfloat8_b)
    check("K and V are distinct allocations", k_t.buffer_address() != v_t.buffer_address())

    table = ckv.address_table(seq_len=seq, chunk_size=chunk, num_kv_heads=n_kv)
    check("2 * num_kv_heads configs (K heads then V heads)", table.num_configs() == 2 * n_kv)
    check("entries = configs * layers * seq/32", table.total_entries() == 2 * n_kv * L * (seq // BLOCK))
    check(
        "zero-padded config names in id order",
        [table.config_name(i) for i in range(2 * n_kv)] == [f"{i:02d}" for i in range(2 * n_kv)],
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "kv_table.pb")
        D.export_to_protobuf_file(table, path)
        t2 = D.import_from_protobuf_file(path)
        same = all(
            table.lookup(l, p, 0, cid).noc_addr == t2.lookup(l, p, 0, cid).noc_addr
            for l in (0, L // 2, L - 1)
            for p in (0, seq // 2, seq - BLOCK)
            for cid in range(2 * n_kv)
        )
        check("protobuf export/import round-trip preserves lookups", same and t2.num_configs() == table.num_configs())

    # Device tensor as the ground truth for "what the table must point at": per chip c -> head c.
    dev_k = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(k_t)]  # each [L, 1, seq, hd]
    dev_v = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(v_t)]
    worst, byte_mismatch = 1.0, 0
    for layer in range(L):
        gk, gv = G.kv(layer)
        for kind, base, dev, gold in (("k", 0, dev_k, gk), ("v", n_kv, dev_v, gv)):
            rows = []
            for h in range(n_kv):
                chunks = [
                    ttnn.to_torch(
                        D.tensor_from_bfp8_bytes(
                            table.read_device_chunk(layer=layer, position=p, slot=0, config_id=base + h),
                            [1, 1, BLOCK, hd],
                        )
                    ).float()[0, 0]
                    for p in range(0, seq, BLOCK)
                ]
                got = torch.cat(chunks, dim=0)  # [seq, hd]
                byte_mismatch += int(not torch.equal(got, dev[h][layer, 0]))
                rows.append(got)
            got_all = torch.stack(rows)  # [n_kv, seq, hd]
            p = pcc(got_all, gold)
            metrics.record(task, f"pcc_kv_contract_{kind}_L{layer:02d}", p)
            worst = min(worst, p)
    check("table readback == device cache contents for every (layer, head, K/V)", byte_mismatch == 0)
    metrics.record(task, "pcc_kv_contract_min", worst)
    metrics.record(task, "contract_checks_failed", len(failed))
    print(f"worst contract-KV pcc vs golden: {worst:.6f}; failed checks: {failed}")
    assert not failed and worst >= 0.97
