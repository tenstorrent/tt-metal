# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Export prepared values in an ordinary, self-contained HF safetensors directory.

No model is constructed or executed here. Recipes describe standard Linear
weights [out,in]; the caller must check the deployment loader's packing layout.
"""
from collections import defaultdict
from fnmatch import fnmatchcase
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch

from .quantize import factor_hessian, gptq_search, search_linear
from .validation import validate_repacking


REPORT = "tt_bfp_quantization.json"
INDEX = "model.safetensors.index.json"
METHODS = {"round", "max-minus-one", "gptq-search"}


def _safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError("Checkpoint export requires: pip install 'tt-bfp-quant[checkpoint]'") from error
    return safe_open, save_file


def _json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _quantized_config(value):
    if isinstance(value, dict):
        return any(
            (k in {"quantization_config", "compression_config"} and v is not None) or _quantized_config(v)
            for k, v in value.items()
        )
    return isinstance(value, list) and any(_quantized_config(v) for v in value)


def _inventory(model):
    safe_open, _ = _safetensors()
    config = _json(model / "config.json")
    if _quantized_config(config):
        raise ValueError("Use an original floating-point checkpoint, not an already quantized checkpoint")
    index = _json(model / INDEX) if (model / INDEX).is_file() else None
    if index is None:
        filenames = ["model.safetensors"]
        weight_map = None
    else:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Invalid safetensors index: expected a nonempty weight_map")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
            raise ValueError("Safetensors weight_map keys and filenames must be strings")
        filenames = sorted(set(weight_map.values()))
        if (model / "model.safetensors").exists():
            raise ValueError("Ambiguous checkpoint: both single-file and sharded safetensors exist")
    inventory = {}
    for filename in filenames:
        path = Path(filename)
        if path.is_absolute() or len(path.parts) != 1 or path.suffix != ".safetensors":
            raise ValueError("Only root-level safetensors shards are supported")
        if not (model / filename).is_file():
            raise ValueError(f"Missing {filename}; export needs a local HF safetensors checkpoint")
        with safe_open(str(model / filename), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in inventory or (weight_map is not None and weight_map.get(name) != filename):
                    raise ValueError(f"Duplicate tensor or inconsistent shard index: {name}")
                view = reader.get_slice(name)
                dtype = view.get_dtype()
                if dtype.startswith("F8") or name.endswith((".qweight", ".weight_scale_inv")):
                    raise ValueError("Packed/FP8 input is unsupported; use the original floating-point checkpoint")
                inventory[name] = {"file": filename, "shape": list(view.get_shape()), "dtype": dtype}
    if weight_map is not None and set(weight_map) != set(inventory):
        raise ValueError("Safetensors index lists tensors missing from the shards")
    return config, index, inventory


def _selection(recipe, inventory):
    if not isinstance(recipe, dict) or set(recipe) != {"schema_version", "rules"} or recipe["schema_version"] != 1:
        raise ValueError("Recipe must contain schema_version: 1 and rules")
    if not isinstance(recipe["rules"], list) or not recipe["rules"]:
        raise ValueError("Recipe rules must be a nonempty list")
    selected = {}
    for rule in recipe["rules"]:
        if not isinstance(rule, dict) or not {"match", "method", "bits"} <= set(rule):
            raise ValueError("Each rule requires match, method and bits")
        if set(rule) - {"match", "method", "bits", "output_splits"}:
            raise ValueError(f"Unknown recipe fields: {set(rule) - {'match', 'method', 'bits', 'output_splits'}}")
        if (
            not isinstance(rule["method"], str)
            or rule["method"] not in METHODS
            or type(rule["bits"]) is not int
            or rule["bits"] not in (4, 8)
        ):
            raise ValueError("Use method round|max-minus-one|gptq-search and bits 4|8")
        if rule["method"] == "gptq-search" and rule["bits"] != 4:
            raise ValueError("GPTQ + search currently supports BFP4 only")
        patterns = [rule["match"]] if isinstance(rule["match"], str) else rule["match"]
        if not isinstance(patterns, list) or not patterns or any(not isinstance(p, str) or not p for p in patterns):
            raise ValueError("match must be a pattern string or a nonempty list of pattern strings")
        for pattern in patterns:
            matches = sorted(n for n in inventory if fnmatchcase(n, pattern))
            if not matches:
                raise ValueError(f"Recipe pattern matched no checkpoint tensors: {pattern}")
            for name in matches:
                if name in selected:
                    raise ValueError(f"Overlapping recipe patterns select {name} more than once")
                shape, dtype = inventory[name]["shape"], inventory[name]["dtype"]
                if len(shape) != 2 or min(shape) <= 0 or dtype not in {"F16", "BF16", "F32"}:
                    raise ValueError(f"{name}: expected a nonempty FP16/BF16/FP32 Linear matrix [out,in]")
                splits = rule.get("output_splits")
                if splits is not None and (
                    not isinstance(splits, list)
                    or any(type(n) is not int or n <= 0 for n in splits)
                    or sum(splits) != shape[0]
                ):
                    raise ValueError(f"{name}: output_splits must be positive widths summing to {shape[0]}")
                selected[name] = {
                    **inventory[name],
                    "method": rule["method"],
                    "bits": rule["bits"],
                    "output_splits": splits,
                }
    return selected


def _hessian_paths(path, selected):
    required = {name for name, item in selected.items() if item["method"] == "gptq-search"}
    if not required:
        return {}
    if path is None:
        raise ValueError("GPTQ rules require --hessians: a JSON map of exact checkpoint weight names to H.pt files")
    path = Path(path).resolve()
    mapping = _json(path)
    if not isinstance(mapping, dict) or required - set(mapping):
        raise ValueError(
            f"Hessian map must include every GPTQ weight; missing {sorted(required - set(mapping)) if isinstance(mapping, dict) else 'all'}"
        )
    result = {}
    for name in required:
        if not isinstance(mapping[name], str):
            raise ValueError(f"Hessian path for {name} must be a string")
        hpath = (path.parent / mapping[name]).resolve()
        if not hpath.is_file():
            raise ValueError(f"Missing Hessian for {name}: {hpath}")
        result[name] = hpath
    return result


def _copy_assets(source, target):
    # Copy ordinary loader assets, never optimizer states, caches or old exports.
    suffixes = {".json", ".model", ".txt", ".tiktoken", ".jinja", ".vocab", ".merges", ".py", ".md"}
    copied = []
    for file in sorted(source.iterdir()):
        if (
            not file.is_file()
            or file.name.startswith(".")
            or file.name in {REPORT, INDEX}
            or file.name.endswith(".index.json")
        ):
            continue
        if file.suffix in suffixes or file.name in {"LICENSE", "NOTICE"}:
            shutil.copyfile(file, target / file.name)
            copied.append(file.name)
    # Recent tokenizer versions can keep named chat templates here.
    templates = source / "chat_templates"
    if templates.is_dir():
        (target / "chat_templates").mkdir()
        for file in sorted(templates.glob("*.jinja")):
            if file.is_file():
                shutil.copyfile(file, target / "chat_templates" / file.name)
                copied.append(f"chat_templates/{file.name}")
    return copied


def export_checkpoint(
    model,
    output,
    recipe,
    *,
    hessians=None,
    dry_run=False,
    backend="auto",
    threads=8,
    damping=0.01,
    block_size=128,
    validate_ttnn=False,
    progress=None,
):
    """Prepare selected Linear weights, preserving all tensor names/shapes/dtypes.

    recipe is a JSON path. hessians is an optional JSON path mapping checkpoint
    weight names to single-tensor .pt files (relative to that JSON's directory).
    Output must not exist. A staging directory is renamed only after success.
    Memory is bounded by one checkpoint shard, one working matrix and one
    cached Hessian factor, plus quantizer/factorization temporaries.
    """
    from . import __version__

    started = time.monotonic()
    model, output = Path(model).resolve(), Path(output).absolute()
    resolved_output = output.resolve()
    if output.exists() or output.is_symlink() or resolved_output == model or model in resolved_output.parents:
        raise ValueError("Choose a new output directory outside the source model; exports are never overwritten")
    if type(threads) is not int or threads <= 0 or type(block_size) is not int or block_size <= 0:
        raise ValueError("threads and block_size must be positive integers")
    if not 0 < damping < float("inf"):
        raise ValueError("damping must be positive and finite")
    config, index, inventory = _inventory(model)
    recipe_data = _json(recipe)
    selected = _selection(recipe_data, inventory)
    if config.get("tie_word_embeddings") or (config.get("text_config") or {}).get("tie_word_embeddings"):
        for name in selected:
            if name.endswith(("lm_head.weight", "embed_tokens.weight", "shared.weight", "wte.weight")):
                raise ValueError(
                    "Tied embedding/output weights need a model-specific layout adapter; exclude them from this recipe"
                )
    hpaths = {} if dry_run and hessians is None else _hessian_paths(hessians, selected)
    plan = {
        "package_version": __version__,
        "source": str(model),
        "output": str(output),
        "recipe": recipe_data,
        "tensor_count": len(inventory),
        "selected_count": len(selected),
        "selected": selected,
        "layout": "linear [out,in] -> TT [in,out]",
        "dry_run": dry_run,
        "hessians_required": sorted(n for n, item in selected.items() if item["method"] == "gptq-search"),
    }
    if dry_run:
        return plan
    safe_open, save_file = _safetensors()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    records, cached_key, cached_factor = {}, None, None
    factor_seconds, reused_factors = 0.0, 0
    try:
        assets = _copy_assets(model, staging)
        shards = defaultdict(list)
        for name, item in inventory.items():
            shards[item["file"]].append(name)
        for filename, names in sorted(shards.items()):
            targets = [n for n in names if n in selected]
            if not targets:
                shutil.copyfile(model / filename, staging / filename)
                continue
            # Group shared Hessians within a shard; retain only the latest factor.
            targets.sort(key=lambda n: (str(hpaths.get(n, "")), n))
            with safe_open(str(model / filename), framework="pt", device="cpu") as reader:
                tensors = {name: reader.get_tensor(name) for name in names}
                metadata = reader.metadata()
                for name in targets:
                    spec, weight = selected[name], tensors[name]
                    if progress:
                        progress(f"Preparing {name} ({spec['method']}, BFP{spec['bits']})")
                    options = {"output_splits": spec["output_splits"], "backend": backend, "threads": threads}
                    if spec["method"] == "gptq-search":
                        key = hpaths[name]
                        cache_hit = key == cached_key
                        if not cache_hit:
                            cached_factor = None
                            h = torch.load(key, map_location="cpu", weights_only=True)
                            if not isinstance(h, torch.Tensor) or tuple(h.shape) != (weight.shape[1], weight.shape[1]):
                                raise ValueError(
                                    f"{name}: Hessian must be a single tensor matching input width {weight.shape[1]}"
                                )
                            cached_factor = factor_hessian(h, damping=damping)
                            del h
                            cached_key = key
                            factor_seconds += cached_factor.seconds
                        else:
                            reused_factors += 1
                        q, info = gptq_search(weight, factor=cached_factor, block_size=block_size, **options)
                        info.update(hessian=str(key), factor_cache_hit=cache_hit)
                    else:
                        q, info = search_linear(
                            weight,
                            bits=spec["bits"],
                            exponent_deltas=(0,) if spec["method"] == "round" else (0, -1),
                            **options,
                        )
                    validation_started = time.monotonic()
                    carrier = q.to(weight.dtype).contiguous()
                    if not torch.equal(carrier.float(), q):
                        raise ValueError(f"{name}: prepared values do not fit source dtype {weight.dtype} exactly")
                    info["validation"] = validate_repacking(
                        q, spec["bits"], output_splits=spec["output_splits"], native=validate_ttnn
                    )
                    info["validation_seconds"] = time.monotonic() - validation_started
                    info["carrier_dtype"] = str(weight.dtype)
                    records[name] = info
                    tensors[name] = carrier
                    del q, carrier, weight
                save_file(tensors, str(staging / filename), metadata=metadata)
                del tensors
        if index is not None:
            # Names, shapes, file membership and dtypes stay the same, so the
            # original weight_map and metadata.total_size remain correct.
            shutil.copyfile(model / INDEX, staging / INDEX)
        report = {
            **plan,
            "assets": assets,
            "weights": records,
            "factor_seconds": factor_seconds,
            "factor_cache_hits": reused_factors,
            "seconds": time.monotonic() - started,
            "runtime_dtype_changed": False,
        }
        (staging / REPORT).write_text(json.dumps(report, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError("Output appeared during export; refusing to replace it")
        staging.rename(output)
        return report
    finally:
        torch.set_num_threads(previous_threads)
        if staging.exists():
            shutil.rmtree(staging)
