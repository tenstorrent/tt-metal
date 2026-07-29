# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Official-weight HF-vs-TTNN decode check for Qwen3.6 full-attention layer 3."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, MODEL_REVISION, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc

LAYER = 3
SNAPSHOT = Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION


def _real_state():
    """Load every layer-3 tensor named by the checkpoint index.

    Full-attention layer 3 spans four shards.  Deriving the shard set from the
    index avoids silently reusing the linear-layer shard subset.
    """

    prefix = f"model.language_model.layers.{LAYER}."
    with (SNAPSHOT / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    shard_names = sorted({shard for key, shard in weight_map.items() if key.startswith(prefix)})
    if not shard_names:
        raise KeyError(f"No checkpoint tensors found for {prefix}")

    state = {}
    for shard_name in shard_names:
        shard = SNAPSHOT / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"Required official shard is missing: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)})
    expected = {key for key in weight_map if key.startswith(prefix)}
    assert state.keys() == expected, f"Missing layer tensors: {sorted(expected - state.keys())}"
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
def run(decoder_kind="optimized", candidate="default"):
    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FALLBACK_AUDIT", f"throw_exception_on_fallback={ttnn.CONFIG.throw_exception_on_fallback}")
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION).text_config
    config._attn_implementation = "eager"
    state = _real_state()
    hf_layer = _hf_layer(config, state)
    hidden = (torch.randn(1, 1, config.hidden_size) * 0.2).bfloat16()
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)

    # The TTNN cache is empty and current_positions is zero.  The equivalent HF
    # contract is uncached one-token attention.  Passing a DynamicCache for
    # isolated layer 3 creates a sparse-cache-layer setup that is not part of
    # the decoder-layer contract and previously made this control misleading.
    reference = hf_layer(
        hidden,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=None,
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder_cls = {"fused": FusedDecoder, "optimized": OptimizedDecoder}[decoder_kind]
        kwargs = {"candidate": candidate} if decoder_kind == "optimized" else {}
        print("DECODER_PATH", decoder_cls.__module__, decoder_cls.__name__)
        decoder = decoder_cls.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER,
            mesh_device=mesh,
            batch=1,
            max_context=64,
            page_size=64,
            **kwargs,
        )
        hidden_tt = _to_device(hidden.unsqueeze(0), mesh_device=mesh)
        page_table = _to_device(
            torch.tensor([[0]], dtype=torch.int32),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        positions = _to_device(
            torch.tensor([0], dtype=torch.uint32),
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
        print("FULL_ATTENTION_REAL_WEIGHT_DECODE_PCC", message)
        assert passed, message
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", choices=("fused", "optimized"), default="optimized")
    parser.add_argument("--candidate", default="default")
    args = parser.parse_args()
    run(args.decoder, args.candidate)
