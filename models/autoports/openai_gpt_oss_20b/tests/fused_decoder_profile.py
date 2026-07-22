# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One-iteration profiler driver for Stage 02 fused decoder reports."""

from __future__ import annotations

import argparse

import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    _config,
    _empty_caches,
    _synthetic_state,
    _tt_tensor,
)
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import EMITTED_PREFILL_SEQUENCE, REPRESENTATIVE_LAYER
from models.autoports.openai_gpt_oss_20b.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("prefill", "decode"), required=True)
    args = parser.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128_000_000)
    try:
        config = _config()
        decoder = FusedDecoder.from_state_dict(
            _synthetic_state(config),
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh,
        )
        generator = torch.Generator().manual_seed(5501)
        prefill_input = _tt_tensor(
            torch.randn(
                (1, 1, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            ),
            mesh,
        )
        decode_input = _tt_tensor(
            torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh
        )
        key_cache, value_cache = _empty_caches(config, mesh)
        warm = decoder.prefill_forward(prefill_input, key_cache, value_cache)
        ttnn.synchronize_device(mesh)
        warm.deallocate(True)
        if args.path == "prefill":
            signpost("FUSED_PREFILL")
            output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
            ttnn.synchronize_device(mesh)
            signpost("FUSED_PREFILL_END")
        else:
            warm = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
            ttnn.synchronize_device(mesh)
            warm.deallocate(True)
            signpost("FUSED_DECODE")
            output = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
            ttnn.synchronize_device(mesh)
            signpost("FUSED_DECODE_END")
        output.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
