# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Any, Iterable, cast

import torch
from transformers.cache_utils import DynamicCache

from models.demos.deepseek_v3.utils.hf_model_utils import (
    add_dynamic_weight_loading_hooks,
    add_gc_hooks,
    add_model_io_logging_hooks,
    load_model_uninitialized,
    load_model_weights,
    load_tokenizer,
)

DEEPSEEK_V3_HF_MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/proj_sw/user_dev/deepseek-ai"))
DEEPSEEK_V3_CACHE_PATH = Path(os.getenv("DEEPSEEK_V3_CACHE", "/proj_sw/user_dev/deepseek-v3-cache"))


class CaptureFinished(Exception):
    pass


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A script to trace the IO of the deepseek model.")
    parser.add_argument(
        "--local_model_path", type=str, help="Path to the local model directory.", default=DEEPSEEK_V3_HF_MODEL_PATH
    )
    parser.add_argument(
        "--model_input",
        type=str,
        help="Model input to complete.",
        default=open(Path(__file__).parent / "default_generate_input.txt", "r").read().strip(),
    )
    parser.add_argument(
        "--path_to_model_io_log",
        type=str,
        help="Path to output the selected layers' IO to. This should be a directory where, for each mode ('prefill' and 'decode') and for each layer group a torch-saved list of tuples of module IO (saved_module_name, input args, input kwargs, output) is created.",
        default=DEEPSEEK_V3_CACHE_PATH / "test_io_cache",
    )
    parser.add_argument(
        "--num_decode_tokens",
        type=int,
        help="Number of tokens from model_input to run in decode mode.",
        default=32,
    )
    parser.add_argument(
        "--num_loader_threads",
        type=int,
        help="Number of threads to use for loading the model weights.",
        default=1,
    )
    parser.add_argument(
        "layer_groups",
        nargs="*",
        type=str,
        help="List of layer groups to log IO for. Can either be a torch module name or a state-dict-style layer path. The path can contain a range of indices, out of which only one will be logged. Only one layer for each torch module name will be logged. Defaults to a hardcoded layers.",
        default=[
            "model",
            "model.norm",
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.0.input_layernorm",
            "model.layers.0.mlp",
            "model.layers.0.self_attn",
            "model.layers.0.self_attn.q_a_layernorm",
            "model.layers.0.self_attn.kv_a_layernorm",
            "model.layers.0.post_attention_layernorm",
            "model.layers.3",
            "model.layers.3.mlp",
            "model.layers.3.mlp.gate",
            "model.layers.3.mlp.experts.0-255",
            "model.layers.3.mlp.shared_experts",
            "lm_head",
        ],
    )
    parser.add_argument(
        "--early_stop_layer",
        nargs="?",
        type=str,
        help="If specified, the script will stop capturing IO after this layer is reached. This can be a torch module name or a state-dict-style layer path.",
    )
    return parser


def unwrap_range(r: str) -> list[str]:
    if "-" in r:
        start, end = map(int, r.split("-"))
        return [str(i) for i in range(start, end + 1)]
    return [r]


def parse_layer_ranges(layer: str) -> list[str]:
    if "." in layer:
        base, rest = layer.split(".", 1)
        rest_parsed = parse_layer_ranges(rest)
        unwrapped_bases = unwrap_range(base)
        return [f"{b}.{r}" for b in unwrapped_bases for r in rest_parsed]
    return unwrap_range(layer)


def trim_kv_cache(kv_cache: DynamicCache, layer_idx: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache = cast(tuple[torch.Tensor, torch.Tensor], kv_cache[layer_idx])
    return (
        key_cache[:, :, :length, :],
        value_cache[:, :, :length, :],
    )


def convert_layer_log_entry(
    model: torch.nn.Module, log: tuple[str, tuple, dict[str, Any], Any]
) -> tuple[str, tuple, dict[str, Any], Any]:
    name, args, kwargs, output = log

    module = dict(model.named_modules())[name]
    if module.__class__.__name__ != "DeepseekV3Attention":
        return log

    # If the layer is an attention layer, we need to put the kv_cache back
    output_kv_cache_length = kwargs.pop("kv_cache_length")
    output_tensor, _, output_cache = output
    layer_idx: int = module.layer_idx
    _, input_length, _ = kwargs["hidden_states"].shape
    kwargs["past_key_value"] = trim_kv_cache(kwargs["past_key_value"], layer_idx, output_kv_cache_length - input_length)
    return (name, args, kwargs, (output_tensor, trim_kv_cache(output_cache, layer_idx, output_kv_cache_length)))


def save_io_logs(
    model: torch.nn.Module,
    layer_groups: Iterable[str],
    logs: list[dict[str, list[tuple[str, tuple, dict[str, Any], Any]]]],
    path_to_model_io_log: Path,
    mode: str,
):
    for layer_group in layer_groups:
        if any(not log[layer_group] for log in logs):
            print(f"Did not capture logs for some of the runs for the layer group {layer_group} in {mode}.")
        layer_group_logs: list[list[tuple[str, tuple, dict[str, Any], Any]]] = [
            [convert_layer_log_entry(model, log_entry) for log_entry in log[layer_group]] for log in logs
        ]
        torch.save(layer_group_logs, path_to_model_io_log / f"{mode}.{layer_group}.pt")


def main():
    # Parse the sysargs
    parser = create_parser()
    args = parser.parse_args()

    # Create the output directory
    path_to_model_io_log = Path(args.path_to_model_io_log)
    os.makedirs(path_to_model_io_log, exist_ok=True)

    # Load the tokenizer
    print("Loading tokenizer")
    tokenizer = load_tokenizer(args.local_model_path)
    print("Tokenizer loaded successfully")

    # Load the model with uninitialized weights
    print("Loading uninitialized model from repo version")
    model = load_model_uninitialized()
    print("Model loaded successfully")

    # Load the model weights
    print("Loading model weights")
    weights_dict = load_model_weights(args.local_model_path)
    add_dynamic_weight_loading_hooks(
        model,
        weights_dict,
        thread_pool_executor=(
            concurrent.futures.ThreadPoolExecutor(args.num_loader_threads) if args.num_loader_threads > 1 else None
        ),
    )

    # Unwrap layer groups
    print("Setting up model I/O logging hooks")
    layer_groups_map = {
        layer: layer_range for layer_range in args.layer_groups for layer in parse_layer_ranges(layer_range)
    }

    # Prepare the io logging hook
    log_dict: dict[str, list[tuple[str, tuple, dict[str, Any], Any]]] = {
        layer_group: [] for layer_group in args.layer_groups
    }

    def submodule_hook(
        model: torch.nn.Module,
        name: str,
        args: tuple,
        kwargs: dict[str, Any],
        output: Any,
        log_dict: dict[str, list[tuple[str, tuple, dict[str, Any], Any]]] = log_dict,
        early_stop_layer: str | None = args.early_stop_layer,
        layer_groups_map: dict[str, str] = layer_groups_map,
    ):
        module = dict(model.named_modules())[name]
        if module.__class__.__name__ == "DeepseekV3Attention":
            _, _, kwargs["kv_cache_length"], _ = kwargs["past_key_value"][module.layer_idx][
                0
            ].shape  # For restoring the cache length later to a proper size
        log_dict[layer_groups_map[name]].append((name, args, kwargs, output))
        if name == early_stop_layer:
            raise CaptureFinished()

    # Add the logging hooks
    add_model_io_logging_hooks(
        model, lambda _1, _2: None, submodule_hook, lambda _1: print(), list(layer_groups_map.keys())
    )

    # Add garbage collection hooks
    print("Adding garbage collection hooks")
    add_gc_hooks(model)

    # Run the model
    print("Running the model")

    model_inputs = tokenizer(args.model_input, return_tensors="pt")
    print(f"# of tokens total: {model_inputs.input_ids.shape[1]}                                      ")

    assert args.num_decode_tokens < len(
        model_inputs.input_ids[0]
    ), "num_decode_tokens must be less than the number of tokens in the input."

    kv_cache = DynamicCache()

    # Capture prefill
    with torch.no_grad():
        print("Running the prefill phase")
        try:
            model(
                input_ids=model_inputs.input_ids[:, : -args.num_decode_tokens],
                attention_mask=model_inputs.attention_mask[:, : -args.num_decode_tokens],
                past_key_values=kv_cache,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        except CaptureFinished:
            print()

    # Save the prefill logs
    save_io_logs(model, args.layer_groups, [log_dict], args.path_to_model_io_log, "prefill")
    print(f"Model I/O prefill logs saved under {args.path_to_model_io_log}")

    # Capture decode
    decode_logs = []
    for tok_idx in range(args.num_decode_tokens):
        print(f"Running the decode phase {tok_idx + 1}/{args.num_decode_tokens}")
        input_ids = model_inputs.input_ids[:, -args.num_decode_tokens + tok_idx - 1 : -args.num_decode_tokens + tok_idx]
        attention_mask = torch.full(
            (input_ids.shape[0], model_inputs.input_ids.shape[1] - args.num_decode_tokens + tok_idx + 1), float("-inf")
        )
        log_dict.update({layer_group: [] for layer_group in log_dict})
        with torch.no_grad():
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=kv_cache,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            except CaptureFinished:
                print()
        decode_logs.append(log_dict.copy())

    # Save the decode logs
    save_io_logs(model, args.layer_groups, decode_logs, args.path_to_model_io_log, "decode")
    print(f"Model I/O decode logs saved under {args.path_to_model_io_log}")


if __name__ == "__main__":
    main()
