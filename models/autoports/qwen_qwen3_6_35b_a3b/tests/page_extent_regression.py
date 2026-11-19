# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Causal read-only page-table padding for small and nonaligned capacities."""

import json
import os
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.decoder_factory import Decoder as FunctionalDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def main():
    output = (
        Path(__file__).parents[1]
        / "doc"
        / {
            "functional": "functional_decoder",
            "fused": "fused_decoder",
            "optimized": "optimized_decoder",
        }[os.environ.get("QWEN_DECODER", "functional")]
        / "page_extent_regression.json"
    )
    record(output)
    torch.set_num_threads(8)
    torch.manual_seed(572)
    c, sd = config(), weights(3)
    hf = layer(sd, 3)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        dec = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=3, mesh_device=mesh)
        for capacity in (1, 31, 32, 33, 255, 256, 257, 511, 512, 513):
            pages = (capacity + 31) // 32
            order = torch.randperm(pages)
            state = dec.new_kv_state(context_length=capacity, page_order=order)
            table = ttnn.to_torch(state["page_table"]).flatten()
            assert torch.equal(table[:pages].long(), order)
            assert table.numel() % 8 == 0 and torch.all(table[pages:] == order[-1])
            x = torch.randn(1, capacity, 2048, dtype=torch.bfloat16) * 0.05
            expected = hf_forward(hf, x, DynamicCache(config=c), 0)
            tx = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            td = ttnn.from_torch(x[:, -1:], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            plan = dec.prepare_prefill(seq_len=capacity)
            positions = dec.prepare_positions(start_pos=capacity - 1, seq_len=1)
            with device_only():
                out = dec.prefill_forward(tx, state=state, plan=plan)
                decoded = dec.decode_forward(td, state=state, positions=positions)
            got = ttnn.to_torch(out).reshape(x.shape)
            got_d = ttnn.to_torch(decoded).reshape(1, 1, 2048)
            for name in ("k", "v"):
                host_cache = ttnn.to_torch(state[name])
                assert torch.count_nonzero(host_cache[pages:]) == 0
                if capacity % 32:
                    assert torch.count_nonzero(host_cache[order[-1], :, capacity % 32 :, :]) == 0
            row = dict(
                capacity=capacity,
                owned_pages=pages,
                physical_pages=state["k"].shape[0],
                table_entries=table.numel(),
                prefill_pcc=pcc(expected, got),
                decode_pcc=pcc(expected[:, -1:], got_d),
                unowned_pages_untouched=True,
                future_rows_untouched=True,
            )
            assert min(row["prefill_pcc"], row["decode_pcc"]) >= 0.995, row
            rows.append(row)
            print("PAGE_EXTENT", row, flush=True)
            del state, tx, td, out, decoded, plan, positions
    finally:
        ttnn.close_mesh_device(mesh)
    output.write_text(json.dumps(dict(results=rows, padding="read-only aliases of final owned page"), indent=2) + "\n")


if __name__ == "__main__":
    main()
