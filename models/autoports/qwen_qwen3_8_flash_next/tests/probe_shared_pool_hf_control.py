# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only original-HF geometry discriminator; never a waiver of the TT gate.

Real checkpoint layers0/1/3, the shared-pool probe's exact prompts and forced
42/43/44 histories. No TTNN imports, device execution, quantization or sampler.
Run only when explicitly scheduled; --self-test does not load model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

PROMPTS = ([785, 13458, 13], [785, 13, 13458, 13, 785] * 6 + [785, 13, 13458])
FORCED = (42, 43, 44)
REVISION = "812d96e10a0dd82b69d749e68693698cf6c36daa"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs_for(requests, eos, *, width=None):
    width = width or max(len(PROMPTS[r]) for r in requests)
    assert width >= max(len(PROMPTS[r]) for r in requests)
    ids = torch.full((len(requests), width), eos, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, request in enumerate(requests):
        length = len(PROMPTS[request])
        ids[row, -length:] = torch.tensor(PROMPTS[request])
        mask[row, -length:] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    return ids, mask, positions


def cache_inventory(cache):
    """Record actual cache dtype/shape, including recurrent and convolution lists."""
    inventory = {}

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            inventory[path] = {"dtype": str(value.dtype), "shape": list(value.shape), "device": str(value.device)}
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}/{key}")

    for index in (0, 1, 3):
        for key, value in vars(cache.layers[index]).items():
            visit(value, f"layer{index}/{key}")
    return inventory


def select_hf_rows(cache, mask, order, indices):
    """Native reorder supports both permutation and a strict subset for all layer types.

    The installed generic batch_select_indices does not cover linear cache
    layers; reorder_cache explicitly selects every conv/recurrent/KV/indexer state.
    """
    indices = torch.tensor(indices, dtype=torch.long)
    cache.reorder_cache(indices)
    return mask.index_select(0, indices), tuple(order[index] for index in indices.tolist())


def self_test():
    # Only small tensor preparation and the unchanged frozen scoring function.
    from .probe_shared_pool_generator import decode_geometry_comparison

    ids, mask, positions = inputs_for((1, 0), 999)
    assert ids.shape == (2, 33) and mask.sum(-1).tolist() == [33, 3]
    for row, request in enumerate((1, 0)):
        assert ids[row, mask[row].bool()].tolist() == PROMPTS[request]
        assert positions[row, mask[row].bool()].tolist() == list(range(len(PROMPTS[request])))
    torch.manual_seed(0)
    a = torch.randn(2, 128)
    assert decode_geometry_comparison(a, a)["passed"]
    assert not decode_geometry_comparison(a, -a)["passed"]
    assert inputs_for((0,), 999)[0].shape == (1, 3)
    assert inputs_for((0,), 999, width=33)[0].shape == (1, 33)
    from transformers.cache_utils import Cache, DynamicIndexedLayer, LinearAttentionLayer

    linear, indexed = LinearAttentionLayer(number_of_states=3), DynamicIndexedLayer()
    for state in range(3):
        values = torch.tensor([[state + 1.0, 2.0], [state + 11.0, 12.0]])
        linear.update_conv_state(values.clone(), state_idx=state)
        linear.update_recurrent_state(values[:, None, :].clone(), state_idx=state)
    values = torch.tensor([[[[1.0, 2.0]]], [[[11.0, 12.0]]]])
    indexed.update(values, values + 20)
    indexed.update_indexer(values[:, 0])
    indexed.indexer_position_ids = torch.tensor([[[0], [30]]])
    cache = Cache(layers=[linear, indexed])
    mask, order = select_hf_rows(cache, torch.ones(2, 1), (0, 1), [1, 0])
    assert order == (1, 0) and indexed.keys[0, 0, 0, 0] == 11
    mask, order = select_hf_rows(cache, mask, order, [1])
    assert order == (0,) and mask.shape == (1, 1)
    assert indexed.keys[0, 0, 0, 0] == 1 and indexed.values[0, 0, 0, 0] == 21
    assert indexed.indexer_keys[0, 0, 0] == 1 and indexed.indexer_position_ids.tolist() == [[[0]]]
    for state in range(3):
        assert linear.conv_states[state].shape[0] == linear.recurrent_states[state].shape[0] == 1
        assert linear.conv_states[state][0, 0] == state + 1
        assert linear.recurrent_states[state][0, 0, 0] == state + 1
    print("HF_CONTROL_SELF_TEST_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/" + REVISION),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "doc/full_model/shared_pool_hf_control_v2.json")
    parser.add_argument(
        "--tensors", type=Path, default=Path("/home/ttuser/qwen38-flash/reference/shared_pool_hf_control_v2.pt")
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.repeats < 2 or args.threads < 1:
        parser.error("At least two repeats and one CPU thread are required")
    if args.snapshot.resolve().name != REVISION:
        parser.error("Snapshot must resolve to the pinned revision " + REVISION)
    if args.output.exists() or args.tensors.exists():
        parser.error("Refusing to overwrite diagnostic evidence; select fresh output and tensor paths")

    import transformers
    from transformers import cache_utils, masking_utils
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

    from .full_model_hf_reduced import LAYERS, load_reduced
    from .probe_shared_pool_generator import decode_geometry_comparison, tensor_comparison

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    start = time.monotonic()
    report = {
        "completed": False,
        "scope": "CPU original-HF reduced-layer backend discriminator; not TT acceptance or gate waiver",
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "revision": REVISION,
        "snapshot": str(args.snapshot.resolve()),
        "layers": list(LAYERS),
        "prompts": PROMPTS,
        "forced_tokens": FORCED,
        "seed": 0,
        "logical_decode_positions": [[len(p) + step for step in range(3)] for p in PROMPTS],
        "repeats": args.repeats,
        "command": [sys.executable, *sys.argv],
        "survivor_history": "B2 prefill; B2 input42; native reorder[1,0]; B2 input43; native select[1]; B1 input44",
        "comparisons": [],
        "runs": {},
        "no_tt_gate_waiver": True,
        "source_hashes": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                ROOT / "tests/full_model_hf_reduced.py",
                ROOT / "tests/probe_ple_quantization.py",
                ROOT / "tests/probe_shared_pool_generator.py",
                Path(hf.__file__),
                Path(cache_utils.__file__),
                Path(masking_utils.__file__),
                args.snapshot / "config.json",
                args.snapshot / "model.safetensors.index.json",
            )
        },
    }
    runs = {}
    try:
        model, weight_bytes = load_reduced(args.snapshot)
        assert all(x.device.type == "cpu" for x in model.parameters())
        assert not model.training and model.config._attn_implementation == "eager"
        assert not getattr(model, "_use_kernels", False), "Control must use original CPU HF operators"
        ple = [layer.ple.ple_embedding.ngram_embedding for layer in model.model.layers if layer.ple is not None]
        assert all(not lookup.quantized for lookup in ple)
        eos = model.config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        report["backend"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version,
            "torch_build": torch.__config__.show(),
            "device": "cpu",
            "threads": args.threads,
            "interop_threads": 1,
            "attention_implementation": model.config._attn_implementation,
            "model_class": type(model).__name__,
            "cache": "original HF DynamicCache; fresh per run",
            "parameter_numel_by_dtype": dict(
                Counter(
                    {
                        dtype: sum(p.numel() for p in model.parameters() if str(p.dtype) == dtype)
                        for dtype in {str(p.dtype) for p in model.parameters()}
                    }
                )
            ),
            "checkpoint_weight_bytes_mmap": weight_bytes,
            "autocast": False,
            "compile": False,
            "kernel_substitutions": False,
            "use_kernels": getattr(model, "_use_kernels", False),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "functions": {
                name: {"module": getattr(hf, name).__module__, "source": inspect.getsourcefile(getattr(hf, name))}
                for name in (
                    "torch_recurrent_gated_delta_rule",
                    "torch_chunk_gated_delta_rule",
                    "causal_conv1d_fn",
                    "causal_conv1d_update",
                )
            },
            "loader_changes": [
                "retains original HF layers0/1/3 with original cache indices",
                "checkpoint tensors mmap/assign=True at their stored dtype; no model cast or dequantization",
                "PLE lookup substitutes exact selected BF16 checkpoint rows; quantized=False",
                "rotary embedding rebuilt by original HF constructor outside meta device",
            ],
            "padding": "left padding uses EOS with explicit attention mask and logical per-row position_ids",
            "eos_padding_id": eos,
            "prefill_partition": "one original-HF prefill call; not the TT dyadic chunk schedule",
            "logits_to_keep": 1,
        }
        print("HF_CONTROL_LOADED", flush=True)
        captured = {}
        handles = []

        def hook(name):
            def save(module, inputs, output):
                captured[name] = output[:, -1].detach().clone()

            return save

        for index, layer in zip(LAYERS, model.model.layers, strict=True):
            handles.append(layer.register_forward_hook(hook(f"layer_{index}")))
        handles.append(model.model.hyper_connection_mixer.register_forward_hook(hook("terminal_hidden")))

        def run(name, requests, *, width=None, migrate=False, survivor=False):
            torch.manual_seed(0)
            order = tuple(requests)
            ids, mask, positions = inputs_for(order, eos, width=width)
            cache, outputs, records, transitions = None, [], [], []
            for step in range(4):
                if step:
                    if migrate and step == 2:
                        mask, order = select_hf_rows(cache, mask, order, [1, 0])
                        transitions.append({"before_input": 43, "api": "cache.reorder_cache([1,0])", "order": order})
                    if survivor and step == 3:
                        assert order == (1, 0), "Survivor must retain original request0 after migration"
                        mask, order = select_hf_rows(cache, mask, order, [1])
                        transitions.append({"before_input": 44, "api": "cache.reorder_cache([1])", "order": order})
                    ids = torch.full((len(order), 1), FORCED[step - 1], dtype=torch.long)
                    positions = torch.tensor([[len(PROMPTS[r]) + step - 1] for r in order])
                    mask = torch.cat([mask, torch.ones(len(order), 1, dtype=torch.long)], dim=1)
                captured.clear()
                tick = time.monotonic()
                output = model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    output_router_logits=False,
                )
                cache = output.past_key_values
                logits = output.logits[:, -1].detach().clone()
                assert torch.isfinite(logits).all(), (name, step, "nonfinite")
                qsa_positions = cache.layers[3].indexer_position_ids[0]
                for row, request in enumerate(order):
                    visible = qsa_positions[row, mask[row].bool()].tolist()
                    assert visible == list(range(len(PROMPTS[request]) + step)), (name, step, visible)
                outputs.append(
                    {
                        request: {
                            "logits": logits[row].clone(),
                            "hidden": {key: value[row].clone() for key, value in captured.items()},
                        }
                        for row, request in enumerate(order)
                    }
                )
                records.append(
                    {
                        "phase": "prefill" if step == 0 else f"forced_{FORCED[step-1]}",
                        "request_order": order,
                        "input_shape": list(ids.shape),
                        "logical_last_positions": positions[:, -1].tolist(),
                        "visible_lengths": mask.sum(-1).tolist(),
                        "physical_cache_length": mask.shape[-1],
                        "cache_class": type(cache).__name__,
                        "logits_dtype": str(logits.dtype),
                        "cache_tensor_inventory": cache_inventory(cache),
                        "transitions_so_far": list(transitions),
                        "seconds": time.monotonic() - tick,
                        "top1": logits.argmax(-1).tolist(),
                    }
                )
                print(
                    json.dumps(
                        {
                            "run": name,
                            "step": step,
                            **{key: value for key, value in records[-1].items() if key != "cache_tensor_inventory"},
                        }
                    ),
                    flush=True,
                )
            runs[name], report["runs"][name] = outputs, records

        with torch.inference_mode():
            for repeat in range(args.repeats):
                for request in range(2):
                    run(f"solo{request}/repeat{repeat}", (request,))
                    run(f"padded_solo{request}/repeat{repeat}", (request,), width=33)
                run(f"mixed/repeat{repeat}", (0, 1))
                run(f"reversed/repeat{repeat}", (1, 0))
                run(f"migrated/repeat{repeat}", (0, 1), migrate=True)
                run(f"survivor/repeat{repeat}", (0, 1), migrate=True, survivor=True)
        for handle in handles:
            handle.remove()

        def compare(label, expected, actual, requests):
            for step in range(4):
                reference = torch.stack([expected[step][r]["logits"] for r in requests])
                candidate = torch.stack([actual[step][r]["logits"] for r in requests])
                score = decode_geometry_comparison(reference, candidate)
                report["comparisons"].append(
                    {
                        "name": label,
                        "step": step,
                        "phase": "prefill" if step == 0 else f"forced_{FORCED[step-1]}",
                        "requests": requests,
                        "frozen_geometry_gate": score,
                        "hidden": {
                            f"request{r}/{key}": tensor_comparison(
                                expected[step][r]["hidden"][key], actual[step][r]["hidden"][key]
                            )
                            for r in requests
                            for key in expected[step][r]["hidden"]
                        },
                    }
                )

        for repeat in range(args.repeats):
            for family in ("solo", "padded_solo"):
                combined = [
                    {request: runs[f"{family}{request}/repeat{repeat}"][step][request] for request in range(2)}
                    for step in range(4)
                ]
                compare(
                    f"combined_B1_{family}_vs_mixed/repeat{repeat}", combined, runs[f"mixed/repeat{repeat}"], [0, 1]
                )
            for request in range(2):
                for family in ("solo", "padded_solo"):
                    compare(
                        f"B1_{family}_vs_mixed/repeat{repeat}/request{request}",
                        runs[f"{family}{request}/repeat{repeat}"],
                        runs[f"mixed/repeat{repeat}"],
                        [request],
                    )
                compare(
                    f"padding_only/repeat{repeat}/request{request}",
                    runs[f"solo{request}/repeat{repeat}"],
                    runs[f"padded_solo{request}/repeat{repeat}"],
                    [request],
                )
            for variant in ("reversed", "migrated"):
                compare(
                    f"{variant}_vs_mixed/repeat{repeat}",
                    runs[f"mixed/repeat{repeat}"],
                    runs[f"{variant}/repeat{repeat}"],
                    [0, 1],
                )
            for reference in ("solo0", "padded_solo0", "mixed", "migrated"):
                compare(
                    f"survivor_vs_{reference}/repeat{repeat}",
                    runs[f"{reference}/repeat{repeat}"],
                    runs[f"survivor/repeat{repeat}"],
                    [0],
                )
        for name in list(runs):
            if name.endswith("repeat0"):
                for repeat in range(1, args.repeats):
                    compare(
                        f"repeat/{name}/{repeat}",
                        runs[name],
                        runs[name.removesuffix("repeat0") + f"repeat{repeat}"],
                        sorted(set.intersection(*(set(step) for step in runs[name]))),
                    )
        report["completed"] = True
        report["all_recorded_frozen_comparisons_pass"] = all(
            item["frozen_geometry_gate"]["passed"] for item in report["comparisons"]
        )
        report["nonexact_comparisons"] = [
            f'{c["name"]}/step{c["step"]}' for c in report["comparisons"] if not c["frozen_geometry_gate"]["exact"]
        ]
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": report, "runs": runs}, args.tensors)
        report["tensor_artifact"] = {"path": str(args.tensors), "sha256": digest(args.tensors)}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"completed": report["completed"], "output": str(args.output), "wall_seconds": report["wall_seconds"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
