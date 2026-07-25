# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bounded single-process DRAM-capacity probe for the functional decoder."""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _synthetic_state_dict,
    _to_torch,
    _to_tt,
)
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import FunctionalDecoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_len", type=int)
    args = parser.parse_args()

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        config = _config()
        state = _synthetic_state_dict(config)
        decoder = FunctionalDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER_IDX,
            mesh_device=mesh_device,
            max_cache_len=args.seq_len,
        )
        hidden = torch.zeros(1, args.seq_len, config.hidden_size, dtype=torch.bfloat16)
        key_cache, value_cache = decoder.create_kv_cache()
        output = decoder.prefill_forward(
            _to_tt(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
        )
        actual = _to_torch(output)
        assert tuple(actual.shape) == tuple(hidden.shape)
        print(f"CAPACITY_PROBE_PASS seq_len={args.seq_len} output_shape={tuple(actual.shape)}")
    finally:
        ttnn.close_mesh_device(mesh_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
