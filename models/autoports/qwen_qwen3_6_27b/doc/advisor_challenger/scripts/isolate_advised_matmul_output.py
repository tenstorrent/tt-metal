"""Isolate the advised matmul output shard under the incumbent DRAM-sharded program."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("linear_attention", "full_attention"), required=True)
    ap.add_argument("--policy", required=True)
    args = ap.parse_args()
    os.environ["CHALLENGER_LAYER_KIND"] = args.kind
    harness_path = Path(__file__).with_name("harness.py")
    spec = importlib.util.spec_from_file_location("challenger_harness", harness_path)
    harness = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(harness)
    policy = json.load(open(args.policy))["shipped_policy"]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = harness.build(mesh, policy)
        decoder = state["decoder"]
        if args.kind == "linear_attention":
            key_width = int(decoder.hf_config.linear_num_key_heads) * int(decoder.hf_config.linear_key_head_dim)
            value_width = int(decoder.hf_config.linear_num_value_heads) * int(decoder.hf_config.linear_value_head_dim)
            value_heads = int(decoder.hf_config.linear_num_value_heads)
            n = 2 * key_width + value_width + value_width + 2 * value_heads
            output = decoder._decode_linear(
                state["hidden"], "linear_packed_decode", k=decoder.hidden_size, n=n,
                in0_block_w=decoder.policy.linear_packed_in0_block_w,
                compute_kernel_config=decoder.linear_input_compute_kernel_config,
            )
            expected_cores, expected_width = 103, 160
        else:
            q_width = decoder.num_heads * decoder.head_dim
            kv_width = decoder.num_kv_heads * decoder.head_dim
            output = decoder._decode_linear(
                state["hidden"], "qkv_gate_decode", k=decoder.hidden_size,
                n=2 * q_width + 2 * kv_width,
                in0_block_w=(decoder.policy.qkv_decode_in0_block_w or decoder.policy.decode_in0_block_w),
            )
            expected_cores, expected_width = 90, 160
        ttnn.synchronize_device(mesh)
        spec_out = output.memory_config().shard_spec
        actual_cores = spec_out.grid.num_cores()
        actual_shape = list(spec_out.shape)
        print(json.dumps({
            "kind": args.kind,
            "expected_advised_cores": expected_cores,
            "expected_advised_shard_shape": [32, expected_width],
            "actual_cores": actual_cores,
            "actual_shard_shape": actual_shape,
            "advisor_config_executed": actual_cores == expected_cores and actual_shape == [32, expected_width],
        }, sort_keys=True))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
