# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Official-weight HF-vs-TTNN decode check for Qwen3.6 layer 0."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import (
    MODEL_ID,
    MODEL_REVISION,
    FunctionalDecoder,
    _to_device,
)
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder
from models.common.utility_functions import comp_pcc

LAYER = 0
SNAPSHOT = Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION
SHARDS = (
    "model-00001-of-00015.safetensors",
    "model-00006-of-00015.safetensors",
    "model-00008-of-00015.safetensors",
)


def _real_state():
    prefix = f"model.language_model.layers.{LAYER}."
    state = {}
    for shard_name in SHARDS:
        shard = SNAPSHOT / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"Required official shard is missing: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)})
    return state


def _hf_layer(config, state):
    prefix = f"model.language_model.layers.{LAYER}."
    local = {key.removeprefix(prefix): value for key, value in state.items()}
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, LAYER)
    missing, unexpected = layer.load_state_dict(local, strict=True, assign=True)
    assert not missing and not unexpected
    return layer.eval()


@torch.no_grad()
def run(optimized=False, candidate="default", batch=1):
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION).text_config
    state = _real_state()
    hf_layer = _hf_layer(config, state)
    hidden = (torch.randn(batch, 1, config.hidden_size) * 0.2).bfloat16()
    reference = hf_layer(
        hidden,
        position_embeddings=(None, None),
        attention_mask=None,
        past_key_values=DynamicCache(config=config),
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder_cls = OptimizedDecoder if optimized else FunctionalDecoder
        decoder = decoder_cls.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER,
            mesh_device=mesh,
            batch=batch,
            max_context=64,
            page_size=64,
            **({"candidate": candidate} if optimized else {}),
        )
        hidden_tt = _to_device(hidden.reshape(1, 1, batch, config.hidden_size), mesh_device=mesh)
        page_table = _to_device(
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        positions = _to_device(
            torch.zeros(batch, dtype=torch.uint32),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        output = decoder.decode_forward(
            hidden_states=hidden_tt,
            page_table=page_table,
            current_positions=positions,
        )
        ttnn.synchronize_device(mesh)
        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape_as(reference)
        passed, message = comp_pcc(reference.float(), actual.float(), 0.995)
        print(
            "LINEAR_ATTENTION_REAL_WEIGHT_DECODE_PCC",
            f"path={'optimized' if optimized else 'functional'}",
            f"candidate={candidate if optimized else 'functional'}",
            f"batch={batch}",
            message,
        )
        assert passed, message
        return {
            "kind": "linear_attention",
            "path": "optimized" if optimized else "functional",
            "candidate": candidate if optimized else "functional",
            "batch": batch,
            "reference": "hf",
            "passed": bool(passed),
            "pcc": float(message),
        }
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="default")
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    result = run(args.optimized, args.candidate, args.batch)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
