# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Nonzero full-shape HF-vs-TTNN check for representative full attention."""

import argparse
import math
import time

import torch
from tracy import signpost
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, FunctionalDecoder, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder
from models.common.utility_functions import comp_pcc

LAYER = 3


def _state(config):
    prefix = f"model.language_model.layers.{LAYER}."
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim

    def diagonal(rows, columns, scale):
        value = torch.zeros(rows, columns, dtype=torch.bfloat16)
        count = min(rows, columns)
        value[torch.arange(count), torch.arange(count)] = scale
        return value

    return {
        prefix + "input_layernorm.weight": torch.linspace(-0.02, 0.02, hidden).bfloat16(),
        prefix + "post_attention_layernorm.weight": torch.linspace(0.01, -0.01, hidden).bfloat16(),
        prefix + "self_attn.q_proj.weight": diagonal(2 * q_width, hidden, 0.25),
        prefix + "self_attn.k_proj.weight": diagonal(kv_width, hidden, 0.2),
        prefix + "self_attn.v_proj.weight": diagonal(kv_width, hidden, 0.15),
        prefix + "self_attn.o_proj.weight": diagonal(hidden, q_width, 0.2),
        prefix + "self_attn.q_norm.weight": torch.linspace(-0.01, 0.01, config.head_dim).bfloat16(),
        prefix + "self_attn.k_norm.weight": torch.linspace(0.01, -0.01, config.head_dim).bfloat16(),
        prefix + "mlp.gate_proj.weight": diagonal(intermediate, hidden, 0.1),
        prefix + "mlp.up_proj.weight": diagonal(intermediate, hidden, 0.08),
        prefix + "mlp.down_proj.weight": diagonal(hidden, intermediate, 0.12),
    }


def _hf_layer(config, state):
    prefix = f"model.language_model.layers.{LAYER}."
    local = {key.removeprefix(prefix): value for key, value in state.items()}
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, LAYER)
    missing, unexpected = layer.load_state_dict(local, strict=True, assign=True)
    assert not missing and not unexpected
    return layer.eval()


@torch.no_grad()
def run(
    mode,
    sequence,
    capacity_only=False,
    optimized=False,
    candidate="default",
    batch=1,
    iterations=0,
):
    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FALLBACK_AUDIT", f"throw_exception_on_fallback={ttnn.CONFIG.throw_exception_on_fallback}")
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    config._attn_implementation = "eager"
    state = _state(config)
    logical_sequence = 1 if mode == "decode" else sequence
    hidden = (torch.randn(batch, logical_sequence, config.hidden_size) * 0.2).bfloat16()
    positions_cpu = torch.arange(logical_sequence, dtype=torch.long).reshape(1, -1).expand(batch, -1)
    reference = None
    if not capacity_only:
        hf_layer = _hf_layer(config, state)
        position_ids = positions_cpu.unsqueeze(0).expand(3, -1, -1)
        rotary = Qwen3_5TextRotaryEmbedding(config)
        position_embeddings = rotary(hidden, position_ids)
        attention_mask = torch.full(
            (1, 1, logical_sequence, logical_sequence),
            torch.finfo(torch.bfloat16).min,
            dtype=torch.bfloat16,
        )
        attention_mask = torch.triu(attention_mask, diagonal=1)
        reference = hf_layer(
            hidden,
            position_embeddings=position_embeddings,
            position_ids=positions_cpu,
            attention_mask=attention_mask,
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
            max_context=max(64, logical_sequence),
            page_size=64,
            **({"candidate": candidate} if optimized else {}),
        )
        hidden_tt = _to_device(hidden.unsqueeze(0), mesh_device=mesh)
        pages_per_user = math.ceil(max(64, logical_sequence) / 64)
        page_table = _to_device(
            torch.arange(batch * pages_per_user, dtype=torch.int32).reshape(batch, pages_per_user),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        positions_host = (
            torch.zeros(batch, dtype=torch.uint32)
            if mode == "decode"
            else torch.arange(logical_sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1).expand(batch, -1)
        )
        positions = _to_device(
            positions_host,
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        if mode == "decode":
            output = decoder.decode_forward(
                hidden_states=hidden_tt,
                page_table=page_table,
                current_positions=positions,
            )
        else:
            output = decoder.prefill_forward(
                hidden_states=hidden_tt,
                page_table=page_table,
                current_positions=positions,
            )
        ttnn.synchronize_device(mesh)
        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).squeeze(0)
        if capacity_only:
            key_cache = ttnn.to_torch(ttnn.get_device_tensors(decoder.caches["key"])[0])
            print(
                "FULL_ATTENTION_CAPACITY",
                f"sequence={logical_sequence}",
                f"output_shape={tuple(actual.shape)}",
                f"output_nonzero={torch.count_nonzero(actual).item()}",
                f"key_cache_nonzero={torch.count_nonzero(key_cache).item()}",
            )
            assert tuple(actual.shape) == (1, logical_sequence, config.hidden_size)
            assert torch.count_nonzero(actual).item() > 0
            assert torch.count_nonzero(key_cache).item() > 0
            return
        passed, message = comp_pcc(reference.float(), actual.float(), 0.995)
        print(
            f"FULL_ATTENTION_SYNTHETIC_PCC mode={mode} sequence={logical_sequence}",
            f"path={'optimized' if optimized else 'functional'}",
            f"candidate={candidate if optimized else 'functional'}",
            message,
        )
        assert passed, message
        if iterations:
            elapsed = []
            forward = decoder.decode_forward if mode == "decode" else decoder.prefill_forward
            forward(hidden_states=hidden_tt, page_table=page_table, current_positions=positions)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL" if mode == "prefill" else "PERF_DECODE")
            for _ in range(iterations):
                started = time.perf_counter()
                forward(hidden_states=hidden_tt, page_table=page_table, current_positions=positions)
                ttnn.synchronize_device(mesh)
                elapsed.append((time.perf_counter() - started) * 1000)
            signpost("PERF_PREFILL_END" if mode == "prefill" else "PERF_DECODE_END")
            print(
                "FULL_ATTENTION_SYNTHETIC_LATENCY",
                f"mode={mode}",
                f"batch={batch}",
                f"sequence={logical_sequence}",
                f"median_ms={torch.tensor(elapsed).median().item():.6f}",
                f"min_ms={min(elapsed):.6f}",
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--sequence", type=int, default=33)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="default")
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--iterations", type=int, default=0)
    args = parser.parse_args()
    run(
        args.mode,
        args.sequence,
        args.capacity_only,
        args.optimized,
        args.candidate,
        args.batch,
        args.iterations,
    )
