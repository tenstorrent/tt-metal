# SPDX-License-Identifier: Apache-2.0
"""Locate prefill corruption relative to capture and logical output assembly."""

import json

import torch

import ttnn

from .multichip_coverage import OUT, Harness
from .run_decoder import pcc

if __name__ == "__main__":
    torch.set_num_threads(8)
    h = Harness(0, capacity=8704)
    rows = []
    try:
        pages = h.pages(1890)
        for length in (8192, 8193):
            inp = h.inp(h.input(length)[None])
            plan = h.model.prepare_prefill(page_table_host=pages, seq_len=length)
            for repeat in range(3):
                out = h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)
                for rank, t in enumerate(ttnn.get_device_tensors(out)):
                    actual = ttnn.to_torch(t).float()
                    reference = h.reference[f"prefill_{length}"].float()
                    row = dict(
                        length=length,
                        repeat=repeat,
                        rank=rank,
                        pcc=pcc(reference, actual),
                        max_abs=float(actual.abs().max()),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                del out
        (OUT / "prefill_before_capture.json").write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        h.close()
