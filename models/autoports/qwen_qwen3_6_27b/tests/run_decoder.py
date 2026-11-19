# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layer-only HF comparison. All torch use is outside TT forward boundaries."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tracy import signpost
from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance

MODEL = "Qwen/Qwen3.6-27B"
REVISION = "8f33b98937c6031bb49e713dee3e8e185aafd26f"


def load_layer(index):
    root = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{index}."
    weights = {}
    for shard in sorted({v for k, v in mapping.items() if k.startswith(prefix)}):
        with safe_open(root / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(prefix):
                    weights[name[len(prefix) :]] = f.get_tensor(name)
    assert weights, prefix
    return weights


def pcc(a, b):
    return torch.corrcoef(torch.stack((a.double().flatten(), b.double().flatten())))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--profile-drain", action="store_true", help="Drain profiler between passes, outside signposts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-file", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        **json.loads((Path(__file__).parents[1] / "doc/hf_config.json").read_text())["text_config"]
    )
    config._attn_implementation = "sdpa"
    if args.synthetic:
        kind = config.layer_types[args.layer]
        stats = json.loads((Path(__file__).parents[1] / f"doc/functional_decoder/{kind}_weight_stats.json").read_text())
        weights = {k: (torch.randn(v["shape"]) * v["std"] + v["mean"]).to(torch.bfloat16) for k, v in stats.items()}
    else:
        weights = load_layer(args.layer)
    stats = {
        k: {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "mean": v.float().mean().item(),
            "std": v.float().std().item(),
        }
        for k, v in weights.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".weights.json").write_text(json.dumps(stats, indent=2) + "\n")
    ref = Qwen3_5DecoderLayer(config, args.layer).eval().to(torch.bfloat16)
    ref.load_state_dict(weights, strict=True)
    cache = DynamicCache(config=config)
    rope = Qwen3_5TextRotaryEmbedding(config)
    x = torch.randn(args.batch, args.length, 5120, dtype=torch.bfloat16) * 0.1
    d = torch.randn(args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
    d2 = torch.randn(args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
    if args.activation_file:
        captured = torch.load(args.activation_file, map_location="cpu", weights_only=True)
        x, d, d2 = (captured[k] for k in ("x", "d", "d2"))
        assert tuple(x.shape) == (args.batch, args.length, 5120)
    pos = torch.arange(args.length).expand(args.batch, -1)
    hf_states = {}

    def snapshot_hf_state(name):
        if config.layer_types[args.layer] == "linear_attention":
            cached = cache.layers[args.layer]
            hf_states[name] = [cached.conv_states.transpose(1, 2)[:, -3:].clone(), cached.recurrent_states.clone()]

    with torch.no_grad():
        expected = ref(x, position_embeddings=rope(x, pos), past_key_values=cache)
        snapshot_hf_state("prefill")
        expected_decode = ref(
            d, position_embeddings=rope(d, torch.full((args.batch, 1), args.length)), past_key_values=cache
        )
        snapshot_hf_state("decode")
        expected_decode2 = ref(
            d2, position_embeddings=rope(d2, torch.full((args.batch, 1), args.length + 1)), past_key_values=cache
        )
    snapshot_hf_state("decode2")
    print("HF_REFERENCE_READY", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)

    def drain_profile():
        if args.profile_drain:
            ttnn.ReadDeviceProfiler(mesh)

    try:
        layer = decoder_class().from_state_dict(
            weights, **decoder_kwargs(), hf_config=config, layer_idx=args.layer, mesh_device=mesh
        )
        if layer.kind == "linear_attention":
            state = layer.create_recurrent_state(args.batch)
            pk, dk = {}, {}
        else:
            # Reserve both follow-on decode tokens, including a page crossing.
            pages = (args.length + 2 + 31) // 32
            state = layer.create_paged_state(torch.randperm(args.batch * pages).reshape(args.batch, pages))
            pk = {"start_pos": 0, "rope": tuple(layer._upload(t.unsqueeze(1)) for t in rope(x, pos))}
            dk = {
                "current_pos": ttnn.from_torch(
                    torch.full((args.batch,), args.length, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "rope": tuple(layer._upload(t.unsqueeze(1)) for t in rope(d, torch.full((args.batch, 1), args.length))),
            }
        tx, td = layer._upload(x), layer._upload(d)
        state_tensors = [state.conv, state.recurrent] if layer.kind == "linear_attention" else [state.key, state.value]
        zero = [ttnn.from_torch(torch.zeros(tuple(t.shape)), dtype=t.dtype, layout=t.layout) for t in state_tensors]
        equivalence = None
        reference_implementation = "functional"
        read_state = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        hf_state_pcc = {}

        def compare_hf_state(name):
            if name in hf_states:
                hf_state_pcc[name] = [pcc(a, read_state(b)) for a, b in zip(hf_states[name], state_tensors)]

        if os.environ.get("QWEN_DECODER") in ("fused", "optimized"):
            from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder

            reference_class = FunctionalDecoder
            if os.environ.get("QWEN_DECODER") == "optimized":
                # See AUTODEBUG_functional_state_oracle.md: the stage-1 polynomial
                # inverse is not a valid state oracle for correlated text inputs.
                from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder

                reference_class = FusedDecoder
                reference_implementation = "fused"
            control = reference_class.from_state_dict(weights, hf_config=config, layer_idx=args.layer, mesh_device=mesh)
            control_state = (
                control.create_recurrent_state(args.batch)
                if layer.kind == "linear_attention"
                else control.create_paged_state(
                    ttnn.to_torch(state.page_table, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                )
            )
            read = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            control_prefill = read(control.prefill_forward(tx, state=control_state, **pk))
            drain_profile()
            control_decode = read(control.decode_forward(td, state=control_state, **dk))
            drain_profile()
            fused_prefill = read(layer.prefill_forward(tx, state=state, **pk))
            drain_profile()
            fused_decode = read(layer.decode_forward(td, state=state, **dk))
            drain_profile()
            control_buffers = (
                [control_state.conv, control_state.recurrent]
                if layer.kind == "linear_attention"
                else [control_state.key, control_state.value]
            )
            equivalence = {
                "prefill_pcc": pcc(control_prefill, fused_prefill),
                "decode_pcc": pcc(control_decode, fused_decode),
                "prefill_contribution_pcc": pcc(control_prefill.float() - x.float(), fused_prefill.float() - x.float()),
                "decode_contribution_pcc": pcc(control_decode.float() - d.float(), fused_decode.float() - d.float()),
                "state_pcc": [pcc(read(a), read(b)) for a, b in zip(control_buffers, state_tensors)],
            }
            del control, control_state, control_buffers
            for host, device in zip(zero, state_tensors):
                ttnn.copy_host_to_device_tensor(host, device)
        warm = layer.prefill_forward(tx, state=state, **pk)
        ttnn.synchronize_device(mesh)
        warm_host = ttnn.to_torch(warm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        del warm
        drain_profile()
        for host, device in zip(zero, state_tensors):
            ttnn.copy_host_to_device_tensor(host, device)
        ttnn.synchronize_device(mesh)
        prefill_start = time.perf_counter()
        signpost("PERF_PREFILL")
        with (
            patch.object(ttnn, "from_torch", side_effect=AssertionError("runtime from_torch")),
            patch.object(ttnn, "to_torch", side_effect=AssertionError("runtime to_torch")),
        ):
            actual = layer.prefill_forward(tx, state=state, **pk)
        ttnn.synchronize_device(mesh)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000
        signpost("PERF_PREFILL_END")
        drain_profile()
        actual = ttnn.to_torch(actual, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        compare_hf_state("prefill")
        print("PREFILL_PCC", pcc(expected, actual), flush=True)
        state_tensors = [state.conv, state.recurrent] if layer.kind == "linear_attention" else [state.key, state.value]
        saved = [
            ttnn.from_torch(
                ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)), dtype=t.dtype, layout=t.layout
            )
            for t in state_tensors
        ]

        def restore():
            for host, device in zip(saved, state_tensors):
                ttnn.copy_host_to_device_tensor(host, device)

        actual_d = layer.decode_forward(td, state=state, **dk)
        ttnn.synchronize_device(mesh)
        del actual_d
        drain_profile()
        restore()
        d2_host = ttnn.from_torch(d2, dtype=td.dtype, layout=td.layout)
        if layer.kind != "linear_attention":
            pos2_host = ttnn.from_torch(
                torch.full((args.batch,), args.length + 1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            rope2_host = tuple(
                ttnn.from_torch(t.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                for t in rope(d2, torch.full((args.batch, 1), args.length + 1))
            )
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        with (
            patch.object(ttnn, "from_torch", side_effect=AssertionError("runtime from_torch")),
            patch.object(ttnn, "to_torch", side_effect=AssertionError("runtime to_torch")),
        ):
            traced_output = layer.decode_forward(td, state=state, **dk)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        drain_profile()
        repeated = []
        elapsed = []
        for iteration in range(args.repeats):
            restore()
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            if iteration == args.repeats - 1:
                signpost("PERF_DECODE")
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            if iteration == args.repeats - 1:
                signpost("PERF_DECODE_END")
            elapsed.append((time.perf_counter() - start) * 1000)
            drain_profile()
            repeated.append(ttnn.to_torch(traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)))
        actual_d = repeated[-1]
        compare_hf_state("decode")
        ttnn.copy_host_to_device_tensor(d2_host, td)
        if layer.kind != "linear_attention":
            ttnn.copy_host_to_device_tensor(pos2_host, dk["current_pos"])
            for host, device in zip(rope2_host, dk["rope"]):
                ttnn.copy_host_to_device_tensor(host, device)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        actual_d2 = ttnn.to_torch(traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        compare_hf_state("decode2")
        drain_profile()
        positive_control = None
        if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1":
            unsafe = ttnn.empty_like(td)
            try:
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            except RuntimeError as exc:
                positive_control = "device buffer" in str(exc).lower() and "corrupted on replay" in str(exc).lower()
                print("ALLOCATION_POSITIVE_CONTROL", str(exc)[:300], flush=True)
            else:
                positive_control = False
            del unsafe
            assert positive_control
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.release_trace(mesh, trace_id)
        result = {
            "provenance": provenance(),
            reference_implementation + "_equivalence": equivalence,
            "equivalence_reference": reference_implementation,
            "equivalence_reference_sha256": hashlib.sha256(
                (Path(__file__).parents[1] / ("tt/" + reference_implementation + "_decoder.py")).read_bytes()
            ).hexdigest(),
            "hf_state_pcc": hf_state_pcc,
            "activation_file": str(args.activation_file) if args.activation_file else None,
            "activation_sha256": (
                hashlib.sha256(args.activation_file.read_bytes()).hexdigest() if args.activation_file else None
            ),
            "weights": "synthetic_from_real_stats" if args.synthetic else "real_checkpoint",
            "model": MODEL,
            "revision": REVISION,
            "layer": args.layer,
            "length": args.length,
            "batch": args.batch,
            "prefill_pcc": pcc(expected, actual),
            "decode_pcc": pcc(expected_decode, actual_d),
            "prefill_per_user_pcc": [pcc(expected[b], actual[b]) for b in range(args.batch)],
            "decode_per_user_pcc": [pcc(expected_decode[b], actual_d[b]) for b in range(args.batch)],
            "decode_traced": True,
            "repeated_bit_equal": all(torch.equal(repeated[0], t) for t in repeated[1:]),
            "prefill_repeated_bit_equal": torch.equal(warm_host, actual),
            "trace_replay_wall_ms": elapsed,
            "warmed_prefill_wall_ms": prefill_ms,
            "updated_input_pcc": pcc(expected_decode2, actual_d2),
            "updated_input_changed_output": not torch.equal(actual_d, actual_d2),
            "allocation_positive_control": positive_control,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        if equivalence is not None:
            assert min(v for k, v in equivalence.items() if k != "state_pcc") >= 0.995, equivalence
            assert min(equivalence["state_pcc"]) >= 0.995, equivalence
        if os.environ.get("QWEN_DECODER") in ("fused", "optimized"):
            assert all(value >= 0.995 for row in hf_state_pcc.values() for value in row), hf_state_pcc
        assert result["prefill_pcc"] >= 0.995 and result["decode_pcc"] >= 0.995
        assert result["updated_input_pcc"] >= 0.995 and result["updated_input_changed_output"]
        assert result["repeated_bit_equal"] and result["prefill_repeated_bit_equal"]
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
