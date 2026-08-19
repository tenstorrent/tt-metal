# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe a width-sharded residual contract through one real multichip decoder.

This is an optimization-stage probe, not part of the shipped model path.  It
keeps the completed multichip decoder weights and submodules, then runs a
candidate decode sequence with:

* input residual converted to per-device width-sharded L1;
* both layer RMSNorms producing width-sharded L1 output;
* token mixer and MoE called with the sharded activations;
* residual adds producing width-sharded L1 output.

The purpose is to reject or validate the lower-movement residual family without
measuring only an immediate restore to the old replicated DRAM boundary.
"""

from __future__ import annotations

import argparse
import statistics
import time
import traceback

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import _state_for_perf, _target_text_config
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_multichip_decoder import (
    _mesh_first_to_torch,
    _prepare_multichip_decode_after_prefill,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder
from models.common.utility_functions import comp_pcc


def _width_sharded_mem_config(mesh_device, hidden_size: int) -> ttnn.MemoryConfig:
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        (32, hidden_size // 8),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _rms_norm_width_sharded(x: ttnn.Tensor, weight: ttnn.Tensor, eps: float, mem_config: ttnn.MemoryConfig):
    return ttnn.rms_norm(
        x,
        epsilon=eps,
        weight=weight,
        memory_config=mem_config,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
            subblock_w=4,
            block_h=1,
            block_w=8,
            inplace=False,
        ),
    )


def _as_width_sharded(tensor: ttnn.Tensor, mem_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    if "WIDTH_SHARDED" in str(tensor.memory_config()) and "BufferType::L1" in str(tensor.memory_config()):
        return tensor
    return ttnn.to_memory_config(tensor, mem_config)


def _timed_median_ms(mesh_device, fn, iterations: int) -> tuple[float, ttnn.Tensor]:
    out = fn()
    ttnn.synchronize_device(mesh_device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), out


def _candidate_decode_width_sharded(
    layer: MultichipDecoder,
    hidden_states: ttnn.Tensor,
    kwargs: dict,
    mem_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    residual = _as_width_sharded(hidden_states, mem_config)
    print(f"candidate_input_mem={residual.memory_config()}")

    hidden_states = _rms_norm_width_sharded(residual, layer.input_layernorm_weight, layer.cfg.rms_norm_eps, mem_config)
    print(f"candidate_after_input_rms_mem={hidden_states.memory_config()}")

    if layer.layer_type == "full_attention":
        mixer_out = layer.token_mixer.decode_forward(
            hidden_states,
            position_embeddings=kwargs["position_embeddings"],
            kv_cache=kwargs["kv_cache"],
            page_table=kwargs["page_table"],
            current_pos=kwargs["current_pos"],
        )
    else:
        mixer_out, _ = layer.token_mixer.decode_forward(hidden_states, linear_state=kwargs["linear_state"])
    print(f"candidate_mixer_out_mem={mixer_out.memory_config()}")

    mixer_out = _as_width_sharded(mixer_out, mem_config)
    hidden_states = ttnn.add(residual, mixer_out, memory_config=mem_config)
    print(f"candidate_after_attention_residual_mem={hidden_states.memory_config()}")

    residual = hidden_states
    hidden_states = _rms_norm_width_sharded(
        hidden_states, layer.post_attention_layernorm_weight, layer.cfg.rms_norm_eps, mem_config
    )
    print(f"candidate_after_post_rms_mem={hidden_states.memory_config()}")

    mlp_out = layer.mlp(hidden_states)
    print(f"candidate_mlp_out_mem={mlp_out.memory_config()}")

    mlp_out = _as_width_sharded(mlp_out, mem_config)
    hidden_states = ttnn.add(residual, mlp_out, memory_config=mem_config)
    print(f"candidate_final_mem={hidden_states.memory_config()}")
    return hidden_states


def _run_layer(mesh_device, layer_idx: int, seq_len: int, iterations: int) -> None:
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    layer = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    decode_input, kwargs = _prepare_multichip_decode_after_prefill(mesh_device, cfg, layer, layer_idx, seq_len)
    mem_config = _width_sharded_mem_config(mesh_device, cfg.hidden_size)

    print(f"layer_idx={layer_idx} layer_type={cfg.layer_types[layer_idx]} seq_len={seq_len}")
    print(f"decode_input_shape={tuple(decode_input.shape)} decode_input_mem={decode_input.memory_config()}")
    print(f"width_sharded_mem={mem_config}")

    baseline_ms, baseline_out = _timed_median_ms(
        mesh_device, lambda: layer.decode_forward(decode_input, **kwargs).hidden_states, iterations
    )
    print(f"baseline_default_decode_ms_median={baseline_ms:.6f}")
    print(f"baseline_default_output_mem={baseline_out.memory_config()}")

    try:
        candidate_ms, candidate_out = _timed_median_ms(
            mesh_device,
            lambda: _candidate_decode_width_sharded(layer, decode_input, kwargs, mem_config),
            iterations,
        )
    except Exception as exc:  # noqa: BLE001 - this is an evidence probe.
        print("candidate_status=failed")
        print(f"candidate_failure_type={type(exc).__name__}")
        print(f"candidate_failure={exc}")
        traceback.print_exc()
        return

    expected = _mesh_first_to_torch(baseline_out)
    actual = _mesh_first_to_torch(candidate_out)
    ok, pcc = comp_pcc(expected.float(), actual.float(), pcc=0.995)
    max_abs = torch.max(torch.abs(expected.float() - actual.float())).item()
    print("candidate_status=ran")
    print(f"candidate_width_sharded_decode_ms_median={candidate_ms:.6f}")
    print(f"candidate_pcc_ok={ok} {pcc}")
    print(f"candidate_max_abs={max_abs:.8f}")
    print(f"candidate_speedup_vs_default={baseline_ms / candidate_ms:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), trace_region_size=32_000_000)
    try:
        for layer_idx, seq_len in ((0, 5), (3, 33)):
            _run_layer(mesh_device, layer_idx, seq_len, args.iterations)
    finally:
        ttnn.synchronize_device(mesh_device)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
