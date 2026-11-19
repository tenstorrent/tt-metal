# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-input singleton/chunk64 linear-attention diagnostic; never a model gate.

Run serially under the device owner's timeout. Each case uploads independent
copies of the same captured state. Normal decode and production source are not
modified. --validate-inputs-only never imports TTNN or opens a device.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def metrics(torch, left, right):
    a, b = left.double().flatten(), right.double().flatten()
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    delta, ac, bc = a - b, a - a.mean(), b - b.mean()
    denom = ac.norm() * bc.norm()
    equal = torch.equal(left.flatten(), right.flatten())
    return dict(
        bit_equal=equal,
        pcc=1.0 if equal else float((ac @ bc) / denom) if denom > 0 else None,
        max_abs=float(delta.abs().max()),
        rms_difference=float(delta.square().mean().sqrt()),
        changed_elements=int(torch.count_nonzero(delta)),
        elements=a.numel(),
    )


def load_inputs(torch, directory, layer_index, position):
    metadata = json.loads((directory / "capture_metadata.json").read_text())
    stats = json.loads((directory / "statistics.json").read_text())
    assert stats["precision_config"] == "baseline" and stats["gate_modified"] is False
    assert stats["start"] == stats["target"] == position == 2559
    captures = torch.load(directory / "captures.pt", map_location="cpu", weights_only=True)
    matches = [
        (key, meta)
        for key, meta in metadata.items()
        if meta["phase"] == "suffix"
        and meta["layer"] == layer_index
        and meta["event"] == "normalized.input_layernorm.weight"
        and position in meta.get("positions", [])
    ]
    assert len(matches) == 1
    key, meta = matches[0]
    assert meta["context"]["public_length"] == 1 and meta["context"]["decode"] is True
    x = captures[key][meta["positions"].index(position)].reshape(1, 1, 2048).clone()
    assert x.dtype == torch.bfloat16 and torch.isfinite(x).all()
    states, sources = {}, {}
    for boundary, event in (("fp32_carry", "before_bf16_cast"), ("bf16_boundary", "after_bf16_cast")):
        states[boundary], sources[boundary] = {}, {}
        for name, dtype, shape in (
            ("conv", torch.bfloat16, (1, 3, 2048)),
            ("recurrent", torch.float32, (1, 8, 128, 128)),
        ):
            state_key = f"suffix/l{layer_index:02}/state.public_begin.{event}/{name}"
            parts = captures[state_key]
            assert len(parts) == 4 and all(
                tuple(v.shape) == shape and v.dtype == dtype and torch.isfinite(v).all() for v in parts
            )
            states[boundary][name] = [v.clone() for v in parts]
            sources[boundary][name] = state_key
    for index in range(4):
        assert torch.equal(states["fp32_carry"]["conv"][index], states["bf16_boundary"]["conv"][index])
        assert torch.equal(
            states["fp32_carry"]["recurrent"][index].bfloat16().float(), states["bf16_boundary"]["recurrent"][index]
        )
        prefix = captures[f"prefix/l{layer_index:02}/state.last_attention.after/recurrent"][index]
        assert torch.equal(prefix, states["fp32_carry"]["recurrent"][index])
    mismatches = [
        path
        for path, expected in stats["source_sha256"].items()
        if Path(path).parent == ROOT / "tt" and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
    ]
    assert not mismatches, ("Captured runtime source changed", mismatches)
    return x, states, dict(normalized_key=key, normalized_metadata=meta, state_sources=sources), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--position", type=int, default=2559)
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(8)
    x_cpu, states, provenance, localization = load_inputs(torch, args.captures_dir, args.layer, args.position)
    if args.validate_inputs_only:
        print("INPUTS_VALID", json.dumps(provenance), flush=True)
        return
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1" and not os.environ.get("TT_METAL_WATCHER")
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    tensor_path = args.output_dir / "operands_outputs.pt"
    assert not result_path.exists() and not tensor_path.exists(), "Use a fresh output directory"
    sources = [
        Path(__file__),
        args.captures_dir / "capture_metadata.json",
        args.captures_dir / "statistics.json",
        *sorted((ROOT / "tt").glob("*.py")),
    ]
    result = dict(
        status="running",
        command=sys.argv,
        started=time.time(),
        mesh=[1, 4],
        layer=args.layer,
        position=args.position,
        scope="Same normalized input and independently cloned prefix state; no gate/performance claim or ordinary-decode modification",
        gate_modified=False,
        production_runtime_changed=False,
        inputs=provenance,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        public_boundary="Compare exact captured FP32-before-cast and BF16-rounded-FP32-after-cast states; cache_boundary=False in every call, with no second cast",
        boundary_rounding_per_rank=[
            metrics(torch, before, after)
            for before, after in zip(states["fp32_carry"]["recurrent"], states["bf16_boundary"]["recurrent"])
        ],
        cases=[],
        comparisons=[],
        limitations=[
            "Final outputs are per-rank attention projection partials before the model collective/residual.",
            "Raw singleton and chunk recurrence also differ in decay evaluation; recorded g/decay and coefficients expose this, without claiming identical internal arithmetic.",
            "No64-vs63+1 control: available normalized capture retains selected rows, not the complete final64 real operands.",
        ],
        tensors=str(tensor_path),
        original_gate=localization.get("gate"),
    )
    saved = {"input/normalized": x_cpu, "input/states": states}
    mesh = layer = None

    def save():
        torch.save(saved, tensor_path)
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        import ttnn
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_REVISION, Checkpoint
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import layer_precision, resolve_precision

        def host(tensor):
            values = [ttnn.to_torch(part).detach().clone() for part in ttnn.get_device_tensors(tensor)]
            assert len(values) == 4 and all(torch.isfinite(v).all() for v in values)
            return values

        def upload(value, dtype, *, dim=None, layout=None):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                device=mesh,
                layout=ttnn.TILE_LAYOUT if layout is None else layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if dim is None else ttnn.ShardTensorToMesh(mesh, dim=dim),
            )

        checkpoint = Checkpoint()
        assert checkpoint.config.layer_types[args.layer] == "linear_attention"
        result["checkpoint"] = dict(path=str(checkpoint.path), revision=HF_REVISION)
        sd = checkpoint.select(f"model.language_model.layers.{args.layer}.")
        result["attention_weight_sha256"] = {
            k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for k, v in sd.items()
            if k.startswith("linear_attn.")
        }
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        layer = MultichipDecoder.from_state_dict(
            sd,
            hf_config=checkpoint.config,
            layer_idx=args.layer,
            mesh_device=mesh,
            full_model_optimized=True,
            precision_config=layer_precision(resolve_precision("baseline"), args.layer),
        )
        del sd
        result["runtime_precision"] = layer.precision_summary()
        result["execution_geometry"] = layer.execution_geometry
        original_step, original_chunk, original_linear = layer._recurrent_step, layer.chunk64_state, layer.linear
        outputs = {}
        for boundary in ("bf16_boundary", "fp32_carry"):
            for branch, decode, projection_k in (
                ("singleton_k512", True, 512),
                ("singleton_k1024", True, 1024),
                ("chunk_n1_k1024", False, 1024),
            ):
                label = boundary + "/" + branch
                print("LINEAR_SINGLETON_CASE", label, flush=True)
                state = {
                    "recurrent": upload(torch.cat(states[boundary]["recurrent"], dim=1), ttnn.float32, dim=1),
                    "conv": upload(
                        torch.cat(states[boundary]["conv"], dim=2), ttnn.bfloat16, dim=2, layout=ttnn.ROW_MAJOR_LAYOUT
                    ),
                }
                for name in state:
                    assert all(torch.equal(a, b) for a, b in zip(host(state[name]), states[boundary][name]))
                x = upload(x_cpu, ttnn.bfloat16)
                observed = {}

                def step(q, k, v, g, beta, initial, **kwargs):
                    assert "raw_recurrence" not in observed and kwargs.get("round_state") is False
                    for name, value in (
                        ("q", q),
                        ("k", k),
                        ("v", v),
                        ("g", g),
                        ("beta", beta),
                        ("initial_state", initial),
                        ("decay", kwargs.get("decay_value")),
                    ):
                        if value is not None:
                            observed[name] = host(value)
                    out, new = original_step(q, k, v, g, beta, initial, **kwargs)
                    observed["raw_recurrence"], observed["returned_state"] = host(out), host(new)
                    return out, new

                def chunk(q, k, v, g, beta, initial):
                    assert "raw_recurrence" not in observed and q.shape[-2] == 1
                    for name, value in (
                        ("q", q),
                        ("k", k),
                        ("v", v),
                        ("g", g),
                        ("beta", beta),
                        ("initial_state", initial),
                    ):
                        observed[name] = host(value)
                    out, new, raw = original_chunk(q, k, v, g, beta, initial)
                    observed["raw_recurrence"], observed["returned_state"] = host(raw), host(new)
                    observed["chunk_return_bf16"] = host(out)
                    return out, new, raw

                def linear(value, name, *positional, **kwargs):
                    out = original_linear(value, name, *positional, **kwargs)
                    if name == "attention_combined":
                        observed["combined_projection"] = host(out)
                    if name == "linear_attn.out_proj.weight":
                        observed["gated_before_projection"] = host(value)
                    return out

                with (
                    patch.object(layer, "_recurrent_step", step),
                    patch.object(layer, "chunk64_state", chunk),
                    patch.object(layer, "linear", linear),
                ):
                    final = layer._linear_attention(
                        x, state, decode=decode, cache_boundary=False, projection_k=projection_k
                    )
                assert all(
                    key in observed
                    for key in ("raw_recurrence", "returned_state", "combined_projection", "gated_before_projection")
                )
                observed["final_attention"] = host(final)
                observed["new_recurrent"], observed["new_conv"] = host(state["recurrent"]), host(state["conv"])
                # Both engines return exactly8 local heads x128 values for one real row.
                observed["raw_recurrence_canonical"] = [
                    v.reshape(-1, 128)[:8].reshape(1, 1, 8, 128).clone() for v in observed["raw_recurrence"]
                ]
                saved[label] = observed
                outputs[label] = observed
                result["cases"].append(
                    dict(
                        label=label,
                        decode=decode,
                        projection_k=projection_k,
                        cache_boundary=False,
                        recurrent_path="_recurrent_step" if decode else "Chunk64OutputGraph(n=1)",
                        raw_shapes=[list(v.shape) for v in observed["raw_recurrence"]],
                        saved_keys=sorted(observed),
                    )
                )
                save()
                for tensor in (x, final, *state.values()):
                    ttnn.deallocate(tensor)
        for left, right in itertools.combinations(outputs, 2):
            result["comparisons"].append(
                dict(
                    left=left,
                    right=right,
                    per_rank={
                        name: [metrics(torch, a, b) for a, b in zip(outputs[left][name], outputs[right][name])]
                        for name in (
                            "combined_projection",
                            "raw_recurrence_canonical",
                            "new_recurrent",
                            "new_conv",
                            "gated_before_projection",
                            "final_attention",
                        )
                    },
                )
            )
        ttnn.synchronize_device(mesh)
        result["status"] = "completed_diagnostic_not_gate"
    except BaseException:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.time() - result["started"]
        save()
        if mesh is not None:
            import ttnn

            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
            save()


if __name__ == "__main__":
    main()
