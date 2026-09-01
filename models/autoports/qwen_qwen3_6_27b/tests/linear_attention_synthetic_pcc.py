# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Nonzero full-shape HF-vs-TTNN check for representative gated delta net."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from tracy import signpost
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, FunctionalDecoder, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder
from models.common.utility_functions import comp_pcc

LAYER = 0


def _state(config):
    prefix = f"model.language_model.layers.{LAYER}."
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    value_width = config.linear_num_value_heads * config.linear_value_head_dim
    conv_width = 2 * key_width + value_width

    def diagonal(rows, columns, scale):
        value = torch.zeros(rows, columns, dtype=torch.bfloat16)
        count = min(rows, columns)
        value[torch.arange(count), torch.arange(count)] = scale
        return value

    conv = torch.zeros(conv_width, 1, config.linear_conv_kernel_dim, dtype=torch.bfloat16)
    conv[:, 0, -1] = 0.5
    return {
        prefix + "input_layernorm.weight": torch.linspace(-0.02, 0.02, hidden).bfloat16(),
        prefix + "post_attention_layernorm.weight": torch.linspace(0.01, -0.01, hidden).bfloat16(),
        prefix + "linear_attn.in_proj_qkv.weight": diagonal(conv_width, hidden, 0.2),
        prefix + "linear_attn.in_proj_z.weight": diagonal(value_width, hidden, 0.15),
        prefix + "linear_attn.in_proj_b.weight": diagonal(config.linear_num_value_heads, hidden, 0.1),
        prefix + "linear_attn.in_proj_a.weight": diagonal(config.linear_num_value_heads, hidden, 0.08),
        prefix + "linear_attn.conv1d.weight": conv,
        prefix + "linear_attn.dt_bias": torch.full((config.linear_num_value_heads,), 0.1, dtype=torch.bfloat16),
        prefix + "linear_attn.A_log": torch.full((config.linear_num_value_heads,), math.log(0.5), dtype=torch.float32),
        prefix + "linear_attn.norm.weight": torch.linspace(0.98, 1.02, config.linear_value_head_dim).bfloat16(),
        prefix + "linear_attn.out_proj.weight": diagonal(hidden, value_width, 0.2),
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
    result_json=None,
):
    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FALLBACK_AUDIT", f"throw_exception_on_fallback={ttnn.CONFIG.throw_exception_on_fallback}")
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    state = _state(config)
    logical_sequence = 1 if mode == "decode" else sequence
    hidden = (torch.randn(batch, logical_sequence, config.hidden_size) * 0.2).bfloat16()
    reference = None
    if not capacity_only:
        hf_layer = _hf_layer(config, state)
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
            max_context=max(64, logical_sequence),
            page_size=64,
            **({"candidate": candidate} if optimized else {}),
        )
        # Decode's contract is [1, 1, batch, hidden]; prefill's is
        # [1, batch, sequence, hidden]. `hidden` is [batch, sequence, hidden],
        # so unsqueeze(0) is only right for prefill. At batch 1 the two shapes
        # coincide, which is why this went unnoticed; at batch 32 the decode
        # tensor became [1, 32, 1, hidden], whose tile-padded height is 1024,
        # and the width-sharded decode memory config rejected it with
        # "Shard height 32 must match physical height 1024".
        if mode == "decode":
            hidden_tt = _to_device(hidden.reshape(1, 1, batch, -1), mesh_device=mesh)
        else:
            hidden_tt = _to_device(hidden.unsqueeze(0), mesh_device=mesh)
        unused_page_table = _to_device(
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        position_values = (
            torch.zeros(batch, dtype=torch.uint32)
            if mode == "decode"
            else torch.arange(logical_sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1).expand(batch, -1)
        )
        positions = _to_device(
            position_values,
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        if mode == "decode":
            output = decoder.decode_forward(
                hidden_states=hidden_tt,
                page_table=unused_page_table,
                current_positions=positions,
            )
        else:
            output = decoder.prefill_forward(
                hidden_states=hidden_tt,
                page_table=unused_page_table,
                current_positions=positions,
            )
        ttnn.synchronize_device(mesh)
        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).squeeze(0)
        if capacity_only:
            recurrent = ttnn.to_torch(ttnn.get_device_tensors(decoder.caches["recurrent"])[0])
            result = {
                "mode": mode,
                "sequence": logical_sequence,
                "batch": batch,
                "path": "optimized" if optimized else "functional",
                "candidate": candidate if optimized else "functional",
                "output_shape": list(actual.shape),
                "output_nonzero": torch.count_nonzero(actual).item(),
                "recurrent_nonzero": torch.count_nonzero(recurrent).item(),
                "recurrent_dtype": str(decoder.caches["recurrent"].dtype),
            }
            print(
                "LINEAR_ATTENTION_CAPACITY",
                f"sequence={logical_sequence}",
                f"output_shape={tuple(actual.shape)}",
                f"output_nonzero={torch.count_nonzero(actual).item()}",
                f"recurrent_nonzero={torch.count_nonzero(recurrent).item()}",
                f"recurrent_dtype={decoder.caches['recurrent'].dtype}",
            )
            assert tuple(actual.shape) == (1, logical_sequence, config.hidden_size)
            assert torch.count_nonzero(actual).item() > 0
            assert torch.count_nonzero(recurrent).item() > 0
            if result_json is not None:
                Path(result_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return
        passed, message = comp_pcc(reference.float(), actual.float(), 0.995)
        print(
            f"LINEAR_ATTENTION_SYNTHETIC_PCC mode={mode} sequence={logical_sequence}",
            f"path={'optimized' if optimized else 'functional'}",
            f"candidate={candidate if optimized else 'functional'}",
            message,
        )
        assert passed, message
        if iterations:
            elapsed = []
            forward = decoder.decode_forward if mode == "decode" else decoder.prefill_forward
            forward(hidden_states=hidden_tt, page_table=unused_page_table, current_positions=positions)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL" if mode == "prefill" else "PERF_DECODE")
            for _ in range(iterations):
                started = time.perf_counter()
                forward(
                    hidden_states=hidden_tt,
                    page_table=unused_page_table,
                    current_positions=positions,
                )
                ttnn.synchronize_device(mesh)
                elapsed.append((time.perf_counter() - started) * 1000)
            signpost("PERF_PREFILL_END" if mode == "prefill" else "PERF_DECODE_END")
            print(
                "LINEAR_ATTENTION_SYNTHETIC_LATENCY",
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
    parser.add_argument("--sequence", type=int, default=5)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="default")
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    run(
        args.mode,
        args.sequence,
        args.capacity_only,
        args.optimized,
        args.candidate,
        args.batch,
        args.iterations,
        args.result_json,
    )
