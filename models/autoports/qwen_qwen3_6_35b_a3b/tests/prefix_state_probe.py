# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: isolate prefill-state error by substituting a canonical seed.

This is not an acceptance test or production fallback. All subsequent decode
steps use the current fused decoder. Its result cannot satisfy a stage gate.
"""
import argparse
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_decoder import run
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    oracle = torch.load(args.oracle_cache, weights_only=False)
    parameters = dict(oracle["key"]["parameters"])
    required = ("layer", "seed", "prompt_length", "decode_length", "continuation_length")
    assert set(parameters) == set(required), "Unexpected oracle parameter contract"
    assert all(isinstance(parameters[k], int) for k in required)
    layer_idx = parameters["layer"]
    cfg = config()
    assert cfg.layer_types[layer_idx] == "linear_attention", "Canonical recurrent seed requires a linear layer"
    assert oracle["prefix"].shape[1] == parameters["prompt_length"]
    assert oracle["tokens"].shape[1] == parameters["decode_length"]
    assert oracle["continuation"].shape[1] == parameters["continuation_length"]
    hf = layer(weights(layer_idx), layer_idx)
    cache = DynamicCache(config=cfg)
    hf_forward(hf, oracle["prefix"], cache, 0)
    reference_state = {
        "conv": cache.layers[layer_idx].conv_states[:, :, -3:].transpose(1, 2).contiguous(),
        "recurrent": cache.layers[layer_idx].recurrent_states.float().contiguous(),
    }
    del hf, cache, oracle
    original_plan, original_prefill = FusedDecoder.prepare_prefill, FusedDecoder.prefill_forward

    def prepare(self, **kwargs):
        plan = original_plan(self, **kwargs)
        if kwargs["seq_len"] == parameters["prompt_length"] and kwargs.get("start_pos", 0) == 0:
            self.diagnostic_seed = {
                k: ttnn.from_torch(
                    v,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16 if k == "conv" else ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                )
                for k, v in reference_state.items()
            }
        return plan

    def prefill(self, x, **kwargs):
        out = original_prefill(self, x, **kwargs)
        if hasattr(self, "diagnostic_seed"):
            for k, v in self.diagnostic_seed.items():
                ttnn.copy(v, kwargs["state"][k])
            del self.diagnostic_seed
        return out

    FusedDecoder.prepare_prefill, FusedDecoder.prefill_forward = prepare, prefill
    print("DIAGNOSTIC_ONLY_CANONICAL_PREFIX: not acceptance evidence", parameters, flush=True)
    try:
        run(SimpleNamespace(**parameters, repeats=1, oracle_cache=args.oracle_cache, output=args.output))
    finally:
        FusedDecoder.prepare_prefill, FusedDecoder.prefill_forward = original_plan, original_prefill
