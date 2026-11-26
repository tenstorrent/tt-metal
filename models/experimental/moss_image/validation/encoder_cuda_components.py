# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused native CUDA Qwen3.5 GDN diagnostics using explicit captured inputs.

This is component replay, not integrated generation. Hooks observe native math;
additional derived norm diagnostics never replace native activations.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from models.experimental.moss_image.validation.cuda_reference import Capture
from models.experimental.moss_image.tt.checkpoint import resolve_checkpoint, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--branch", choices=["positive", "negative"], default="positive")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Native diagnostic requires CUDA; CPU fallback forbidden")
    torch.cuda.set_device(0)
    checkpoint = resolve_checkpoint("Base", args.checkpoint, local_files_only=True)
    source = json.loads((args.capture / "manifest.json").read_text())
    cap = Capture(args.output_dir, budget_gib=2)
    cap.manifest.update(
        {
            "scope": "Native CUDA GDN component replay",
            "activation_injection": True,
            "input_injection": "Native CUDA captured layer inputs, explicitly for isolated numerical diagnosis",
            "capture_manifest_sha256": sha256(args.capture / "manifest.json"),
            "checkpoint": str(checkpoint),
            "branch": args.branch,
            "transformers_source_sha256": sha256(Path(native.__file__)),
            "helper_sha256": sha256(Path(__file__)),
            "environment": {
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
            },
            "results": {},
        }
    )
    cap.flush()
    index = json.loads((checkpoint / "text_encoder/model.safetensors.index.json").read_text())["weight_map"]
    config = AutoConfig.from_pretrained(
        checkpoint / "text_encoder", trust_remote_code=False, local_files_only=True
    ).text_config
    verified = set()

    def weight(name):
        shard = checkpoint / "text_encoder" / index[name]
        if shard.name not in verified:
            expected = source["checkpoint"]["files"]["text_encoder/" + shard.name]["sha256"]
            if sha256(shard) != expected:
                raise ValueError(f"Checkpoint hash mismatch: {shard.name}")
            verified.add(shard.name)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def captured(key):
        record = source["artifacts"][key]
        path = (args.capture / record["path"]).resolve()
        if not path.is_relative_to(args.capture.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError(f"Capture integrity mismatch: {key}")
        return torch.load(path, weights_only=True, map_location="cpu")

    inputs = captured(f"text/{args.branch}/inputs")
    cap.save("tokens", inputs)
    original_conv = native.causal_conv1d_fn
    original_fconv = native.F.conv1d
    original_scan = native.torch_chunk_gated_delta_rule
    original_l2 = native.l2norm
    active = {"layer": None, "conv": False, "norm": 0}

    def save(key, value):
        return cap.save(f'layer_{active["layer"]}/{key}', value)

    def convolution(*a, **kw):
        result = original_fconv(*a, **kw)
        if active["conv"]:
            save("conv_before_silu", result[:, :, : a[0].shape[-1]])
        return result

    def causal(*a, **kw):
        active["conv"] = True
        try:
            result = original_conv(*a, **kw)
            save("conv_after_silu", result)
            return result
        finally:
            active["conv"] = False

    def l2(*a, **kw):
        result = original_l2(*a, **kw)
        name = "query_l2norm" if active["norm"] == 0 else "key_l2norm"
        save(name, {"input": a[0], "output": result})
        active["norm"] += 1
        return result

    def scan(*a, **kw):
        active["norm"] = 0
        save("scan_inputs", dict(zip(("query", "key", "value"), a[:3])) | {k: kw[k] for k in ("g", "beta")})
        save("beta", kw["beta"])
        save("decay", kw["g"])
        result = original_scan(*a, **kw)
        save("scan_output", result[0])
        return result

    native.F.conv1d = convolution
    native.causal_conv1d_fn = causal
    native.torch_chunk_gated_delta_rule = scan
    native.l2norm = l2
    start = time.perf_counter()
    try:
        with torch.no_grad():
            if 0 in args.layers:
                active["layer"] = 0
                embedding = weight("language_model.embed_tokens.weight").to(device="cuda:0", dtype=torch.bfloat16)
                embedding_out = torch.nn.functional.embedding(inputs["input_ids"].to("cuda:0"), embedding)
                save("embedding", embedding_out)
                expected_input = captured(f"text/{args.branch}/layers/language_model.layers.0")["input"]
                cap.manifest["embedding_bitwise_equal_to_native_capture"] = torch.equal(
                    embedding_out.cpu(), expected_input
                )
                if not cap.manifest["embedding_bitwise_equal_to_native_capture"]:
                    raise RuntimeError("Native CUDA embedding differs from source capture")
                del embedding, embedding_out
            for index_ in args.layers:
                if config.layer_types[index_] != "linear_attention":
                    raise ValueError("Focused diagnostic supports GDN layers only")
                active["layer"] = index_
                layer = native.Qwen3_5DecoderLayer(config, index_).to(dtype=torch.bfloat16)
                prefix = f"language_model.layers.{index_}."
                layer.load_state_dict(
                    {name[len(prefix) :]: weight(name) for name in index if name.startswith(prefix)}, strict=True
                )
                layer = layer.eval().to("cuda:0")
                handles = []
                for name, module in layer.named_modules():
                    if name not in {
                        "input_layernorm",
                        "post_attention_layernorm",
                        "linear_attn.in_proj_qkv",
                        "linear_attn.in_proj_z",
                        "linear_attn.in_proj_a",
                        "linear_attn.in_proj_b",
                        "linear_attn.out_proj",
                        "linear_attn.norm",
                        "mlp.gate_proj",
                        "mlp.up_proj",
                        "mlp.down_proj",
                        "mlp",
                    }:
                        continue

                    def hook(module, a, kw, output, name=name):
                        save(name, {"inputs": a, "kwargs": kw, "output": output})
                        if name == "post_attention_layernorm":
                            save("attention_residual", a[0])
                        if name == "linear_attn.norm":
                            x, gate = a
                            normalized = x.float() * torch.rsqrt(
                                x.float().square().mean(-1, keepdim=True) + module.variance_epsilon
                            )
                            normalized = module.weight * normalized.to(x.dtype)
                            gate_silu = torch.nn.functional.silu(gate.float())
                            save("gated_norm_before_gate", normalized)
                            save("gated_norm_gate_silu", gate_silu)
                            save("gated_norm_output", output)
                            exact = torch.equal((normalized * gate_silu).to(x.dtype), output)
                            cap.manifest["results"].setdefault(str(index_), {})[
                                "derived_gated_norm_matches_native_bitwise"
                            ] = exact
                            if not exact:
                                raise RuntimeError("Derived norm diagnostic differs from native output")

                    handles.append(module.register_forward_hook(hook, with_kwargs=True))
                reference = captured(f"text/{args.branch}/layers/language_model.layers.{index_}")
                x = reference["input"].to("cuda:0")
                save("input", x)
                output = layer(x, position_embeddings=(None, None), attention_mask=None, past_key_values=None)
                save("output", output)
                expected = reference["output"]
                actual = output.cpu()
                relative_rms = float((actual.float() - expected.float()).norm() / expected.float().norm())
                exact = torch.equal(actual, expected)
                cap.manifest["results"][str(index_)].update(
                    {
                        "layer_output_matches_original_native_capture_bitwise": exact,
                        "relative_rms_to_original_capture": relative_rms,
                    }
                )
                if not exact:
                    raise RuntimeError(f"Component replay differs from original native layer {index_}: {relative_rms}")
                for handle in handles:
                    handle.remove()
                del layer, x, output
                cap.manifest["status"] = "capturing"
                cap.flush()
        cap.manifest["elapsed_seconds_including_capture"] = time.perf_counter() - start
        cap.manifest["status"] = "complete"
        cap.flush()
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(cap.path),
                    "results": cap.manifest["results"],
                    "artifacts": len(cap.manifest["artifacts"]),
                }
            ),
            flush=True,
        )
    except BaseException as exc:
        cap.manifest["status"] = "failed"
        cap.manifest["error"] = repr(exc)
        cap.flush()
        raise
    finally:
        native.causal_conv1d_fn = original_conv
        native.F.conv1d = original_fconv
        native.torch_chunk_gated_delta_rule = original_scan
        native.l2norm = original_l2


if __name__ == "__main__":
    main()
