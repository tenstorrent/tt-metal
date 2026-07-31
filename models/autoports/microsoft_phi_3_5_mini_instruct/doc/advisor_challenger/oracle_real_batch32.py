#!/usr/bin/env python3
"""Real-weight HF-vs-TTNN decode oracle for the shipped batch-32 prerequisite."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tests import test_optimized_decoder as test_helpers
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


BATCH = 32
SNAPSHOT = Path(os.environ.get("HF_HOME", "/huggingface")) / (
    "hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/"
    "2fe192450127e6a83f7441aef6e3ca586c338b77"
)
OUT = Path(__file__).with_name("oracle_real_batch32.json")


def main() -> None:
    test_helpers.HF_CACHE = SNAPSHOT
    cfg = test_helpers._hf_config()
    state = test_helpers._real_layer0_state_dict()
    hf_layer = test_helpers._make_hf_layer(cfg, state)

    generator = torch.Generator().manual_seed(4102)
    hidden_host = (torch.randn(BATCH, 1, cfg.hidden_size, generator=generator) * 0.1).to(torch.bfloat16)
    position_ids_host = torch.zeros((BATCH, 1), dtype=torch.long)
    with torch.no_grad():
        reference = hf_layer(hidden_host, position_ids=position_ids_host, use_cache=False)[0]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh,
            max_position_embeddings=64,
            batch=BATCH,
        )
        kv_cache = OptimizedDecoder.allocate_paged_kv_cache(
            hf_config=cfg,
            mesh_device=mesh,
            max_batch_size=BATCH,
            max_seq_len=32,
            block_size=32,
        )
        hidden = ttnn.Tensor(
            hidden_host.reshape(1, 1, BATCH, cfg.hidden_size), ttnn.bfloat16
        ).to(ttnn.TILE_LAYOUT).to(mesh)
        current_host = torch.zeros(BATCH, dtype=torch.int32)
        current_pos = ttnn.Tensor(current_host, ttnn.int32).to(mesh)
        position_ids = ttnn.Tensor(current_host.to(torch.uint32), ttnn.uint32).to(mesh)
        page_table = ttnn.Tensor(torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1), ttnn.int32).to(mesh)

        output = decoder.decode_forward(
            hidden,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=1,
        )
        ttnn.synchronize_device(mesh)
        actual = ttnn.to_torch(output).reshape(BATCH, 1, cfg.hidden_size)
    finally:
        ttnn.close_mesh_device(mesh)

    passed, message = comp_pcc(reference.float(), actual.float(), pcc=0.995)
    pcc = float(message.rsplit("PCC: ", 1)[1])
    record = {
        "decode_batch": BATCH,
        "oracle_weights": "real",
        "reference": "Hugging Face Phi3DecoderLayer",
        "snapshot": str(SNAPSHOT),
        "pcc_threshold": 0.995,
        "pcc": pcc,
        "passed": bool(passed),
        "message": message,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not passed:
        raise SystemExit("real-weight batch-32 oracle failed")


if __name__ == "__main__":
    main()
