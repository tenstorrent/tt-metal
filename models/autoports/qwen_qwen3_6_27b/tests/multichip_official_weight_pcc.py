# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Official-weight TP4 decode gate against the optimized single-chip baseline.

This deliberately compares TTNN with TTNN.  The optimized-decoder stage owns
the separate HF gates; this stage isolates mesh sharding, collectives, and
per-rank cache ownership using the exact same official tensors and inputs.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, MODEL_REVISION, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc

LINEAR_LAYER = 0
FULL_LAYER = 3
SNAPSHOT = Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION


def load_official_layers():
    """Load only the checkpoint shards/tensors needed by layers 0 and 3."""
    prefixes = tuple(f"model.language_model.layers.{layer}." for layer in (LINEAR_LAYER, FULL_LAYER))
    with (SNAPSHOT / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith(prefixes)})
    state = {}
    for shard_name in shards:
        shard = SNAPSHOT / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"Required official shard is missing: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefixes)})
    return state, shards


def upload(tensor, mesh, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def replicas(tensor):
    return [ttnn.to_torch(value) for value in ttnn.get_device_tensors(tensor)]


def local_conv(global_conv, rank):
    return torch.cat(
        [
            global_conv[..., rank * 512 : (rank + 1) * 512, :],
            global_conv[..., 2048 + rank * 512 : 2048 + (rank + 1) * 512, :],
            global_conv[..., 4096 + rank * 1536 : 4096 + (rank + 1) * 1536, :],
        ],
        dim=-2,
    )


def pcc(actual, expected, threshold):
    passed, message = comp_pcc(expected.float(), actual.float(), threshold)
    return bool(passed), message


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--candidate", default="default", help="TP4 candidate; reference remains optimized default")
    parser.add_argument(
        "--allow-pcc-failure", action="store_true", help="Retain a rejected-candidate artifact without failing"
    )
    parsed = parser.parse_args()

    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).text_config
    state, shards = load_official_layers()
    batch = parsed.batch
    max_context = 128
    pages_per_user = math.ceil(max_context / 64)
    hidden = (torch.randn(batch, 1, config.hidden_size) * 0.2).bfloat16()
    input_host = hidden.reshape(1, 1, batch, config.hidden_size)
    page_host = torch.arange(batch * pages_per_user, dtype=torch.int32).reshape(batch, pages_per_user).flip(0)
    positions_host = (torch.arange(batch, dtype=torch.int64) % 64).to(torch.uint32)

    expected = {}
    expected_caches = {}
    one = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        one_x = _to_device(input_host, mesh_device=one)
        one_page = _to_device(page_host, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
        one_positions = _to_device(positions_host, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
        for kind, layer in (("linear_attention", LINEAR_LAYER), ("full_attention", FULL_LAYER)):
            decoder = OptimizedDecoder.from_state_dict(
                state,
                hf_config=config,
                layer_idx=layer,
                mesh_device=one,
                batch=batch,
                max_context=max_context,
                candidate="default",
            )
            output = decoder.decode_forward(hidden_states=one_x, page_table=one_page, current_positions=one_positions)
            ttnn.synchronize_device(one)
            expected[kind] = host(output)
            cache_names = ("conv", "recurrent") if kind == "linear_attention" else ("key", "value")
            expected_caches[kind] = {name: host(decoder.caches[name]) for name in cache_names}
    finally:
        ttnn.close_mesh_device(one)

    results = {}
    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000)
    try:
        mesh_x = upload(input_host, mesh)
        mesh_page = upload(page_host, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        mesh_positions = upload(positions_host, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for kind, layer in (("linear_attention", LINEAR_LAYER), ("full_attention", FULL_LAYER)):
            decoder = MultichipDecoder.from_state_dict(
                state,
                hf_config=config,
                layer_idx=layer,
                mesh_device=mesh,
                batch=batch,
                max_context=max_context,
                candidate=parsed.candidate,
            )
            output = decoder.decode_forward(
                hidden_states=mesh_x, page_table=mesh_page, current_positions=mesh_positions
            )
            ttnn.synchronize_device(mesh)
            output_replicas = replicas(output)
            output_pass, output_message = pcc(output_replicas[0], expected[kind], 0.995)
            replica_pass = all(pcc(value, output_replicas[0], 0.99999)[0] for value in output_replicas[1:])

            if kind == "linear_attention":
                conv = replicas(decoder.caches["conv"])
                recurrent = replicas(decoder.caches["recurrent"])
                cache_pass = {
                    "conv": [
                        pcc(value, local_conv(expected_caches[kind]["conv"], rank), 0.995)[0]
                        for rank, value in enumerate(conv)
                    ],
                    "recurrent": [
                        pcc(
                            value,
                            expected_caches[kind]["recurrent"][:, rank * 12 : (rank + 1) * 12],
                            0.995,
                        )[0]
                        for rank, value in enumerate(recurrent)
                    ],
                }
                cache_shapes = {
                    "conv": [tuple(value.shape) for value in conv],
                    "recurrent": [tuple(value.shape) for value in recurrent],
                }
                expected_shapes = {
                    "conv": [(1, batch, 2560, 4)] * 4,
                    "recurrent": [(batch, 12, 128, 128)] * 4,
                }
            else:
                key = replicas(decoder.caches["key"])
                value = replicas(decoder.caches["value"])
                cache_pass = {
                    # TP4 full attention intentionally uses BFP4 while the
                    # optimized baseline uses BF16, so cache PCC is diagnostic
                    # at the same policy-aware bars as the stacked smoke.
                    "key": [
                        pcc(cache, expected_caches[kind]["key"][:, rank : rank + 1], 0.95)[0]
                        for rank, cache in enumerate(key)
                    ],
                    "value": [
                        pcc(cache, expected_caches[kind]["value"][:, rank : rank + 1], 0.99)[0]
                        for rank, cache in enumerate(value)
                    ],
                }
                cache_shapes = {
                    "key": [tuple(cache.shape) for cache in key],
                    "value": [tuple(cache.shape) for cache in value],
                }
                local_shape = (batch * pages_per_user, 1, 64, 256)
                expected_shapes = {"key": [local_shape] * 4, "value": [local_shape] * 4}

            policy = {
                "requested_candidate": parsed.candidate,
                "effective_candidate": decoder.candidate,
                "attention_weight_dtype": str(decoder.policy.attention_weight_dtype),
                "cache_dtype": str(decoder.policy.cache_dtype),
                "ccl_dtype": "bfloat16",
            }
            print(
                "MULTICHIP_OFFICIAL_WEIGHT",
                f"kind={kind}",
                f"batch={batch}",
                output_message,
                f"replicas_equal={replica_pass}",
                f"cache_pcc={cache_pass}",
                f"cache_shapes={cache_shapes}",
                f"policy={policy}",
                f"page_table={page_host.tolist()}",
                f"positions={positions_host.tolist()}",
                "reference=optimized_single_chip",
                "fallback_audit=True",
            )
            if not parsed.allow_pcc_failure:
                assert output_pass, output_message
            assert replica_pass
            if not parsed.allow_pcc_failure:
                assert all(all(values) for values in cache_pass.values())
            assert cache_shapes == expected_shapes
            results[kind] = {
                "output_pass": output_pass,
                "output_pcc": output_message,
                "replicas_equal": replica_pass,
                "cache_pass": cache_pass,
                "cache_shapes": cache_shapes,
                "policy": policy,
            }
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    result = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpoint_shards": shards,
        "batch": batch,
        "max_context": max_context,
        "mesh": [1, 4],
        "reference": "OptimizedDecoder 1x1, candidate=default",
        "results": results,
    }
    if parsed.result_json is not None:
        parsed.result_json.parent.mkdir(parents=True, exist_ok=True)
        parsed.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
