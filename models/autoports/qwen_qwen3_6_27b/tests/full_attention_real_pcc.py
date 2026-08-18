# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Official-weight HF-vs-optimized TTNN decode check for layer 3."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import (
    MODEL_ID,
    default_snapshot,
    MODEL_REVISION,
    FunctionalDecoder,
    _to_device,
)
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder
from models.common.utility_functions import comp_pcc

LAYER = 3
SNAPSHOT = default_snapshot()


def _real_state():
    prefix = f"model.language_model.layers.{LAYER}."
    with (SNAPSHOT / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith(prefix)})
    state = {}
    for shard_name in shards:
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
def run(candidate, batch=1, functional=False, compare_functional=False, probe_qkv=False, probe_only=False):
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).text_config
    config._attn_implementation = "eager"
    state = _real_state()
    hf_layer = _hf_layer(config, state)
    hidden = (torch.randn(batch, 1, config.hidden_size) * 0.2).bfloat16()
    positions_cpu = torch.zeros((batch, 1), dtype=torch.long)
    position_ids = positions_cpu.unsqueeze(0).expand(3, -1, -1)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    reference = hf_layer(
        hidden,
        position_embeddings=rotary(hidden, position_ids),
        position_ids=positions_cpu,
        attention_mask=None,
        past_key_values=DynamicCache(config=config),
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        if compare_functional:
            functional_decoder = FunctionalDecoder.from_state_dict(
                state,
                hf_config=config,
                layer_idx=LAYER,
                mesh_device=mesh,
                batch=batch,
                max_context=64,
                page_size=64,
            )
            functional_hidden = _to_device(hidden.reshape(1, 1, batch, config.hidden_size), mesh_device=mesh)
            functional_page_table = _to_device(
                torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
                mesh_device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            )
            functional_positions = _to_device(
                torch.zeros(batch, dtype=torch.uint32),
                mesh_device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            functional_output = functional_decoder.decode_forward(
                hidden_states=functional_hidden,
                page_table=functional_page_table,
                current_positions=functional_positions,
            )
            ttnn.synchronize_device(mesh)
            reference = ttnn.to_torch(ttnn.get_device_tensors(functional_output)[0]).reshape_as(reference)

        decoder_cls = FunctionalDecoder if functional else OptimizedDecoder
        decoder = decoder_cls.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER,
            mesh_device=mesh,
            batch=batch,
            max_context=64,
            page_size=64,
            **({} if functional else {"candidate": candidate}),
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
        if probe_qkv:
            assert not functional
            q_width = config.num_attention_heads * config.head_dim
            kv_width = config.num_key_value_heads * config.head_dim
            norm = decoder._rms_norm_decode(
                ttnn.to_memory_config(hidden_tt, decoder.decode_residual_memory_config),
                "input_norm",
            )
            packed = decoder._decode_linear(
                norm,
                "qkv_gate_decode",
                k=config.hidden_size,
                n=2 * q_width + 2 * kv_width,
                in0_block_w=(decoder.policy.qkv_decode_in0_block_w or decoder.policy.decode_in0_block_w),
            )
            norm_host = ttnn.to_torch(ttnn.get_device_tensors(norm)[0]).reshape(-1, config.hidden_size)[0]
            actual_packed = ttnn.to_torch(ttnn.get_device_tensors(packed)[0]).reshape(-1, 2 * q_width + 2 * kv_width)[0]
            prefix = f"model.language_model.layers.{LAYER}."
            q_and_gate = (
                state[prefix + "self_attn.q_proj.weight"]
                .to(torch.bfloat16)
                .transpose(-2, -1)
                .reshape(config.hidden_size, config.num_attention_heads, 2 * config.head_dim)
            )
            q = q_and_gate[..., : config.head_dim].reshape(config.hidden_size, q_width)
            gate = q_and_gate[..., config.head_dim :].reshape(config.hidden_size, q_width)
            k = state[prefix + "self_attn.k_proj.weight"].to(torch.bfloat16).transpose(-2, -1)
            v = state[prefix + "self_attn.v_proj.weight"].to(torch.bfloat16).transpose(-2, -1)
            expected_packed = norm_host @ torch.cat([q, k, v, gate], dim=-1)
            _, probe_message = comp_pcc(expected_packed.float(), actual_packed.float(), 0.0)
            print("FULL_ATTENTION_REAL_QKV_PROJECTION_PCC", probe_message)
            actual_qkv, actual_gate = ttnn.split(packed, (q_width + 2 * kv_width, q_width), dim=-1)
            actual_qkv_host = ttnn.to_torch(ttnn.get_device_tensors(actual_qkv)[0]).reshape(-1, q_width + 2 * kv_width)[
                0
            ]
            actual_gate_host = ttnn.to_torch(ttnn.get_device_tensors(actual_gate)[0]).reshape(-1, q_width)[0]
            _, qkv_split_message = comp_pcc(
                expected_packed[: q_width + 2 * kv_width].float(),
                actual_qkv_host.float(),
                0.0,
            )
            _, gate_split_message = comp_pcc(
                expected_packed[q_width + 2 * kv_width :].float(),
                actual_gate_host.float(),
                0.0,
            )
            print("FULL_ATTENTION_REAL_QKV_SPLIT_PCC", qkv_split_message)
            print("FULL_ATTENTION_REAL_GATE_SPLIT_PCC", gate_split_message)
            _, _, actual_v = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.to_memory_config(actual_qkv, ttnn.L1_MEMORY_CONFIG),
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                memory_config=decoder.decode_attention_memory_config,
            )
            actual_v_host = ttnn.to_torch(ttnn.get_device_tensors(actual_v)[0])
            expected_v = expected_packed[q_width + kv_width : q_width + 2 * kv_width].reshape(
                config.num_key_value_heads, config.head_dim
            )
            print("FULL_ATTENTION_REAL_V_HEADS_SHAPE", tuple(actual_v_host.shape))
            selected_v = actual_v_host[0, 0, : config.num_key_value_heads, :]
            _, v_heads_message = comp_pcc(expected_v.float(), selected_v.float(), 0.0)
            print(
                "FULL_ATTENTION_REAL_V_HEADS_PCC",
                v_heads_message,
            )
            if probe_only:
                return None

        output = decoder.decode_forward(
            hidden_states=hidden_tt,
            page_table=page_table,
            current_positions=positions,
        )
        ttnn.synchronize_device(mesh)
        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape_as(reference)
        passed, message = comp_pcc(reference.float(), actual.float(), 0.995)
        print(
            "FULL_ATTENTION_REAL_WEIGHT_DECODE_PCC",
            f"path={'functional' if functional else 'optimized'}",
            f"candidate={'functional' if functional else candidate}",
            f"batch={batch}",
            f"reference={'functional' if compare_functional else 'hf'}",
            message,
        )
        assert passed, message
        return {
            "kind": "full_attention",
            "path": "functional" if functional else "optimized",
            "candidate": "functional" if functional else candidate,
            "batch": batch,
            "reference": "functional" if compare_functional else "hf",
            "passed": bool(passed),
            "pcc": float(message),
        }
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="default")
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--compare-functional", action="store_true")
    parser.add_argument("--probe-qkv", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    result = run(
        args.candidate,
        args.batch,
        args.functional,
        args.compare_functional,
        args.probe_qkv or args.probe_only,
        args.probe_only,
    )
    if args.result_json is not None and result is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
