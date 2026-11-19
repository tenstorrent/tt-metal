# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only packed PLE quantization/cache setup; never used inside forward."""

NF4_VALUES = (
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.3379152,
    0.4407098,
    0.562617,
    0.7229568,
    1.0,
)


def _round32(value):
    return (value + 31) // 32 * 32


def quantization_spec(*, quantization="absmax", group_size=160, codebook_values=None, scale_codebook_values=None):
    """Validate and canonically identify the exact FLOAT32 reconstruction levels."""
    import hashlib
    import math
    import struct

    scale_lut = scale_codebook_values is not None
    allowed_groups = (10,) if scale_lut else (16, 20, 32, 40, 80, 160)
    if quantization not in ("absmax", "lsq") or group_size not in allowed_groups:
        raise ValueError(f"Supported quantization is absmax/lsq with groups{allowed_groups}")
    values = tuple(NF4_VALUES if codebook_values is None else codebook_values)
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected16 finite codebook values")
    packed_values = struct.pack("<16f", *values)
    values = struct.unpack("<16f", packed_values)
    if not all(left < right for left, right in zip(values, values[1:])):
        raise ValueError("Codebook must have16 strictly increasing FLOAT32 levels")
    result = {
        "quantization": quantization,
        "lsq_iterations": 5 if quantization == "lsq" else 0,
        "group_size": group_size,
        "scales_per_row": 160 // group_size,
        "codebook_values": list(values),
        "codebook_sha256": hashlib.sha256(packed_values).hexdigest(),
    }
    if scale_lut:
        scale_values = tuple(scale_codebook_values)
        if len(scale_values) != 256 or not all(math.isfinite(value) and value >= 0 for value in scale_values):
            raise ValueError("Expected256 finite nonnegative scale codebook values")
        packed_scales = struct.pack("<256f", *scale_values)
        scale_values = struct.unpack("<256f", packed_scales)
        if not all(left < right for left, right in zip(scale_values, scale_values[1:])):
            raise ValueError("Scale codebook must have256 strictly increasing FLOAT32 levels")
        if any(bits & 0xFFFF for bits in struct.unpack("<256I", packed_scales)):
            raise ValueError("Scale codebook values must be exactly representable in BF16")
        result.update(
            scale_encoding="uint8_fixed_log_lut",
            scale_words_per_row=4,
            scale_codebook_values=list(scale_values),
            scale_codebook_sha256=hashlib.sha256(packed_scales).hexdigest(),
        )
    return result


def _quantize_rows(
    weight,
    *,
    quantization="absmax",
    group_size=160,
    codebook_values=None,
    scale_codebook_values=None,
    statistics=None,
):
    """Setup-only fixed4-bit quantizer, optional5-step group-local scale fit.

    Returns contiguous row-order packed words and BF16 group scales, or four
    UINT32 words each packing four8-bit group-scale IDs when a scale LUT is
    supplied. Byte0 corresponds to the first group in that word. Fitting
    uses only the current weight group; it never sees evaluation activations.
    """
    import torch

    spec = quantization_spec(
        quantization=quantization,
        group_size=group_size,
        codebook_values=codebook_values,
        scale_codebook_values=scale_codebook_values,
    )
    if weight.ndim != 2 or weight.shape[-1] != 160:
        raise ValueError("Expected checkpoint rows shaped[rows,160]")
    values = torch.tensor(spec["codebook_values"], dtype=torch.float32)
    weight = weight.float().reshape(weight.shape[0], spec["scales_per_row"], group_size)
    boundaries = (values[:-1] + values[1:]) / 2
    scales = weight.abs().amax(-1, keepdim=True).clamp_min(1e-30 if quantization == "absmax" else 1e-20)
    if quantization == "lsq":
        for _ in range(5):
            levels = values[torch.bucketize(weight / scales, boundaries)]
            denominator = levels.square().sum(-1, keepdim=True).clamp_min(1e-20)
            scales = ((weight * levels).sum(-1, keepdim=True) / denominator).clamp_min(1e-20)
    scale_payload = None
    if scale_codebook_values is not None:
        scale_table = torch.tensor(spec["scale_codebook_values"], dtype=torch.float32)
        if statistics is not None:
            group_absmax = weight.abs().amax(-1, keepdim=True)
            nonzero = group_absmax > 0
            minimum_positive = scale_table[1] if scale_table[0] == 0 else scale_table[0]
            below = (scales < minimum_positive) & nonzero
            above = scales > scale_table[-1]
            counts = {
                "scale_groups": scales.numel(),
                "zero_weight_groups": int((~nonzero).sum()),
                "below_positive_scale_lut_minimum": int(below.sum()),
                "above_scale_lut_maximum": int(above.sum()),
            }
            for key, count in counts.items():
                statistics[key] = statistics.get(key, 0) + count
            if nonzero.any():
                statistics["minimum_nonzero_fitted_scale"] = min(
                    statistics.get("minimum_nonzero_fitted_scale", float("inf")), float(scales[nonzero].min())
                )
            statistics["maximum_fitted_scale"] = max(statistics.get("maximum_fitted_scale", 0.0), float(scales.max()))
            for name, mask in (("below", below), ("above", above)):
                key = f"maximum_abs_weight_in_{name}_range_groups"
                statistics[key] = max(statistics.get(key, 0.0), float(group_absmax.masked_fill(~mask, 0).max()))
        scale_ids = torch.bucketize(scales, (scale_table[:-1] + scale_table[1:]) / 2)
        scales = scale_table[scale_ids]
        scale_ids = scale_ids.reshape(-1, 4)
        scale_payload = torch.zeros(scale_ids.shape[0], dtype=torch.int64)
        for lane in range(4):
            scale_payload |= scale_ids[:, lane] << (8 * lane)
        scale_payload = scale_payload.to(torch.uint32)
    else:
        scales = scales.bfloat16()
    divisor = scales.float().clamp_min(1e-30) if scale_payload is not None else scales.float()
    codes = torch.bucketize(weight / divisor, boundaries).reshape(-1, 8)
    packed = torch.zeros(codes.shape[0], dtype=torch.int64)
    for lane in range(8):
        packed |= codes[:, lane] << (4 * lane)
    return packed.to(torch.uint32), scales.flatten() if scale_payload is None else scale_payload


def prepare_checkpoint_cache(
    snapshot,
    cache_dir,
    *,
    chunk_rows=65536,
    quantization="absmax",
    group_size=160,
    codebook_values=None,
    scale_codebook_values=None,
):
    """Stream all checkpoint PLE shards into persistent stripe-layout memmaps.

    This is setup, never runtime. Cache files contain the entire weight domain,
    numerically ordered shard_0..shard_127. Quantization working memory is bounded
    by chunk_rows; no full BF16/FLOAT32 copy of the embedding table is made.
    Cache directory must be on persistent storage outside the source checkout.
    An atomically written manifest is the completion marker; interrupted caches
    are rebuilt. Concurrent preparation of the same cache directory is forbidden.
    """
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import torch
    from safetensors import safe_open

    snapshot, cache_dir = Path(snapshot), Path(cache_dir)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    fingerprint = hashlib.sha256((snapshot / "model.safetensors.index.json").read_bytes()).hexdigest()
    spec = quantization_spec(
        quantization=quantization,
        group_size=group_size,
        codebook_values=codebook_values,
        scale_codebook_values=scale_codebook_values,
    )
    scale_dtype = np.uint32 if scale_codebook_values is not None else np.uint16
    scale_storage_per_row = spec.get("scale_words_per_row", spec["scales_per_row"])
    legacy = (
        quantization == "absmax"
        and group_size == 160
        and spec["codebook_sha256"] == quantization_spec()["codebook_sha256"]
    )
    identity = {
        "revision": snapshot.name,
        "index_sha256": fingerprint,
        "format": (
            "u4_grouped_u8_scale_lut_stripe32_v3"
            if scale_codebook_values is not None
            else "nf4_absmax_bf16_scale_stripe32_v1"
            if legacy
            else "u4_grouped_bf16_scale_stripe32_v2"
        ),
    }
    if not legacy:
        identity.update(spec)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("Cache identity mismatch; use a separate cache directory")
        for name, dtype in (("words", np.uint32), ("scales", scale_dtype)):
            expected = 32 * manifest[f"{name}_columns"] * np.dtype(dtype).itemsize
            if (cache_dir / f"{name}.bin").stat().st_size != expected:
                raise ValueError(f"Incomplete {name} cache")
        return manifest
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    keys = sorted(
        (key for key in index if key.startswith(prefix)), key=lambda key: int(key.split("shard_")[1].split(".")[0])
    )
    if len(keys) != 128:
        raise ValueError(f"Expected all 128 PLE shards, received {len(keys)}")
    shapes = []
    for shard_number, key in enumerate(keys):
        if key != f"{prefix}shard_{shard_number}.weight":
            raise ValueError("Noncontiguous PLE shard numbering")
        with safe_open(snapshot / index[key], framework="pt") as f:
            shape = f.get_slice(key).get_shape()
        if shape != [2500012, 160]:
            raise ValueError(f"Unexpected checkpoint shape: {key}: {shape}")
        shapes.append(shape)
    rows = sum(shape[0] for shape in shapes)
    word_columns = _round32((rows * 20 + 31) // 32)
    scale_columns = _round32((rows * scale_storage_per_row + 31) // 32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    words = np.memmap(cache_dir / "words.bin", mode="w+", dtype=np.uint32, shape=(32, word_columns))
    scales = np.memmap(cache_dir / "scales.bin", mode="w+", dtype=scale_dtype, shape=(32, scale_columns))

    def write_striped(destination, values, linear_start):
        for stripe in range(32):
            offset = (stripe - linear_start) % 32
            part = values[offset::32]
            column = (linear_start + offset) // 32
            destination[stripe, column : column + len(part)] = part

    row_offset = 0
    statistics = {} if scale_codebook_values is not None else None
    for key, shape in zip(keys, shapes):
        with safe_open(snapshot / index[key], framework="pt") as f:
            source = f.get_slice(key)
            for start in range(0, shape[0], chunk_rows):
                end = min(start + chunk_rows, shape[0])
                packed, scale = _quantize_rows(
                    source[start:end, :],
                    quantization=quantization,
                    group_size=group_size,
                    codebook_values=spec["codebook_values"],
                    scale_codebook_values=scale_codebook_values,
                    statistics=statistics,
                )
                write_striped(words, packed.numpy(), (row_offset + start) * 20)
                scale_data = scale.numpy() if scale_codebook_values is not None else scale.view(torch.uint16).numpy()
                write_striped(scales, scale_data, (row_offset + start) * scale_storage_per_row)
        row_offset += shape[0]
        print(f"PACKED_PLE_SHARD_DONE {key} rows={row_offset}", flush=True)
    words.flush()
    scales.flush()
    manifest = dict(
        identity,
        rows=rows,
        words_columns=word_columns,
        scales_columns=scale_columns,
        word_bytes=int(words.nbytes),
        scale_bytes=int(scales.nbytes),
        chunk_rows=chunk_rows,
    )
    manifest.update(spec)
    if statistics is not None:
        manifest["quantization_statistics"] = statistics
    temporary = manifest_path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest
