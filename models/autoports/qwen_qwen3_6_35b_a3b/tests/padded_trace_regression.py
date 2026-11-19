# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One decode trace across padded page tables and nonzero causal-tail guards."""

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
        / "padded_trace_regression.json"
    )
    record(output)
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    torch.set_num_threads(8)
    torch.manual_seed(934)
    c, sd = config(), weights(3)
    hf = layer(sd, 3)
    requests = []
    for order in ([2, 5], [1, 6]):
        x = torch.randn(1, 33, 2048, dtype=torch.bfloat16) * 0.05
        ref = hf_forward(hf, x, DynamicCache(config=c), 0)
        requests.append(dict(order=order, x=x, ref=ref))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        dec = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=3, mesh_device=mesh)
        state = dec.new_kv_state(context_length=33, page_order=torch.tensor(requests[0]["order"]))
        plan = dec.prepare_prefill(seq_len=32)
        pos = dec.prepare_positions(start_pos=32, seq_len=1)

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        tx, td = upload(requests[0]["x"][:, :32]), upload(requests[0]["x"][:, 32:])

        def refresh(dest, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            host = ttnn.from_torch(t, dtype=dtype, layout=layout)
            ttnn.copy_host_to_device_tensor(host, dest)

        with device_only():
            warm_p = dec.prefill_forward(tx, state=state, plan=plan)
            warm_d = dec.decode_forward(td, state=state, positions=pos)
        del warm_p, warm_d
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = dec.decode_forward(td, state=state, positions=pos)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(out)
        rows, outputs = [], []
        # Repeat request zero after changing all page-table entries and inputs.
        for idx in (0, 1, 0):
            r = requests[idx]
            pair = []
            for sentinel in (0.0, 64.0):
                initial = torch.full((8, 2, 32, 256), -16.0, dtype=torch.bfloat16)
                initial[r["order"]] = 0
                initial[r["order"][-1], :, 1:] = sentinel
                refresh(state["k"], initial)
                refresh(state["v"], -initial)
                pages = torch.tensor([r["order"] + [r["order"][-1]] * 6], dtype=torch.int32)
                refresh(state["page_table"], pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                refresh(tx, r["x"][:, :32])
                refresh(td, r["x"][:, 32:])
                with device_only():
                    prefill = dec.prefill_forward(tx, state=state, plan=plan)
                got_p = ttnn.to_torch(prefill).reshape(1, 32, 2048).clone()
                del prefill
                with device_only():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                got_d = ttnn.to_torch(out).reshape(1, 1, 2048).clone()
                for name, sign in (("k", 1), ("v", -1)):
                    host = ttnn.to_torch(state[name])
                    unowned = [i for i in range(8) if i not in r["order"]]
                    assert torch.equal(host[unowned], sign * initial[unowned])
                    assert torch.equal(host[r["order"][-1], :, 1:], sign * initial[r["order"][-1], :, 1:])
                row = dict(
                    request=idx,
                    sentinel=sentinel,
                    page_order=r["order"],
                    position=32,
                    prefill_pcc=pcc(r["ref"][:, :32], got_p),
                    decode_pcc=pcc(r["ref"][:, 32:], got_d),
                    guards_untouched=True,
                )
                assert min(row["prefill_pcc"], row["decode_pcc"]) >= 0.995, row
                rows.append(row)
                pair.append((got_p, got_d))
                print("PADDED_TRACE", row, flush=True)
            assert all(torch.equal(a, b) for a, b in zip(pair[0], pair[1]))
            outputs.append(pair[0][1])
        assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
        ttnn.release_trace(mesh, trace)
        output.write_text(
            json.dumps(
                dict(
                    results=rows,
                    captures=1,
                    table_entries=8,
                    logical_capacity=33,
                    sentinel_masked_bit_equal=True,
                    repeated_request_bit_equal=True,
                    changed_request_changed_output=True,
                    tracking=True,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
