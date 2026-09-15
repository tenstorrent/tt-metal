# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Strict CPU-only captured Q/K/V contract; no TTNN imports or implicit conversions."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import torch

SCHEMA = "sdpa-captured-inputs-v1"
SCALE = 1 / math.sqrt(128)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def primitive(value, depth=0):
    require(depth <= 8, "Metadata nesting exceeds eight levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        require(math.isfinite(value), "Metadata contains a nonfinite float")
        return
    if type(value) is list:
        for item in value:
            primitive(item, depth + 1)
        return
    if type(value) is dict:
        require(all(type(k) is str for k in value), "Metadata keys must be strings")
        for item in value.values():
            primitive(item, depth + 1)
        return
    raise ValueError("Metadata permits JSON primitives, lists and dictionaries only")


def canonical_metadata(metadata):
    primitive(metadata)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(len(encoded) <= 65536, "Metadata exceeds 64 KiB")
    return encoded


def tensor_sha256(tensor):
    require(
        type(tensor) is torch.Tensor and tensor.device.type == "cpu" and tensor.is_contiguous(),
        "Hash requires a plain contiguous CPU tensor",
    )
    # Contract is logical C-order BF16 bit patterns, little endian, not torch.save bytes.
    require(tensor.dtype is torch.bfloat16, "Hash requires BF16")
    digest = hashlib.sha256()
    values = tensor.detach().view(torch.uint16).reshape(-1)
    for offset in range(0, values.numel(), 1048576):
        digest.update(values[offset : offset + 1048576].numpy().astype("<u2", copy=False).tobytes())
    return digest.hexdigest()


def validate_artifact(artifact):
    require(
        type(artifact) is dict and set(artifact) == {"schema", "q", "k", "v", "metadata"},
        "Artifact must contain exactly schema, q, k, v, metadata",
    )
    require(artifact["schema"] == SCHEMA, "Unsupported captured-input schema")
    meta = artifact["metadata"]
    require(
        type(meta) is dict and set(meta) == {"causal", "mask", "scale", "provenance"},
        "Metadata must contain exactly causal, mask, scale, provenance",
    )
    canonical_metadata(meta)
    require(meta["causal"] is False, "Only explicitly noncausal attention is supported")
    require(meta["mask"] is None, "Masks, padding masks, attention bias and windowing are unsupported")
    require(
        type(meta["scale"]) is float and meta["scale"] == SCALE,
        "Scale must be exactly Python 1/math.sqrt(128); no custom or folded scale",
    )
    provenance = meta["provenance"]
    require(
        type(provenance) is dict and provenance.get("source_kind") in ("model", "synthetic"),
        "Provenance must explicitly identify source_kind=model or synthetic",
    )
    for name in ("model_id", "layer_id", "capture_stage"):
        require(type(provenance.get(name)) is str and bool(provenance[name]), "Provenance requires a nonempty " + name)
    tensors = [artifact[k] for k in ("q", "k", "v")]
    shape = None
    for name, tensor in zip(("q", "k", "v"), tensors):
        require(type(tensor) is torch.Tensor, name + " must be a plain torch.Tensor, not a subclass")
        require(tensor.device.type == "cpu" and tensor.layout is torch.strided, name + " must be a dense CPU tensor")
        require(tensor.dtype is torch.bfloat16, name + " must already be BF16; no implicit casting")
        require(not tensor.requires_grad and tensor.is_contiguous(), name + " must be detached and contiguous")
        require(
            tensor.ndim == 4
            and tensor.shape[0] == 1
            and tensor.shape[1] > 0
            and tensor.shape[2] > 0
            and tensor.shape[2] % 512 == 0
            and tensor.shape[3] == 128,
            name + " must have layout [1,H,N,128], H>0, N>0 a multiple of512",
        )
        require(
            shape is None or tuple(tensor.shape) == shape, "Q/K/V shapes must match: no GQA, MQA or cross-attention"
        )
        shape = tuple(tensor.shape)
        require(bool(torch.isfinite(tensor).all()), name + " contains NaN or infinity")
    return dict(
        schema=SCHEMA,
        shape=list(shape),
        dtype="bfloat16",
        layout="BHND-contiguous",
        metadata=meta,
        metadata_sha256=hashlib.sha256(canonical_metadata(meta)).hexdigest(),
        input_sha256={name: tensor_sha256(x) for name, x in zip(("q", "k", "v"), tensors)},
        input_hash_encoding="Logical C-order BF16 uint16 bit patterns, little endian",
        useful_attention_flops=4 * shape[1] * shape[2] ** 2 * shape[3],
    )


def file_digest(stream):
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1048576), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def file_identity(status):
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def load_capture(path, max_file_bytes=4 * 1024**3):
    path = Path(path).expanduser().resolve(strict=True)
    require(path.is_file(), "Capture must be a local regular file")
    require(type(max_file_bytes) is int and max_file_bytes > 0, "Invalid file-size limit")
    require(0 < path.stat().st_size <= max_file_bytes, "Capture exceeds configured serialized file-size limit")
    # Some torch builds register nested-tensor classes at import. Remove the entire
    # user-extendable allowlist while loading, then restore it without executing it.
    # This helper is for isolated evaluation processes, not concurrent deserializers.
    get_globals = getattr(torch.serialization, "get_safe_globals", None)
    previous_globals = get_globals() if get_globals is not None else []
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode), "Opened capture must be a regular file")
        require(0 < opened.st_size <= max_file_bytes, "Opened capture exceeds serialized file-size limit")
        require(file_identity(opened) == file_identity(path.stat()), "Capture path changed while opening")
        before = file_digest(stream)
        try:
            if get_globals is not None:
                torch.serialization.clear_safe_globals()
            artifact = torch.load(stream, weights_only=True, map_location="cpu")
        finally:
            if get_globals is not None:
                torch.serialization.clear_safe_globals()
                torch.serialization.add_safe_globals(previous_globals)
        require(file_digest(stream) == before, "Capture changed while loading")
        require(file_identity(os.fstat(stream.fileno())) == file_identity(opened), "Capture file changed while loading")
        require(file_identity(path.stat()) == file_identity(opened), "Capture path changed while loading")
    info = validate_artifact(artifact)
    info.update(
        artifact_path=str(path),
        artifact_sha256=before,
        artifact_bytes=opened.st_size,
        loading="torch.load(weights_only=True,map_location=cpu); no unsafe fallback",
    )
    return [artifact[k] for k in ("q", "k", "v")], info


def select_rows(length, count):
    require(
        type(length) is int and length > 0 and type(count) is int and count > 0,
        "Positive integer sequence length and sample count required",
    )
    count = min(count, length)
    # Integer arithmetic: independent of torch linspace floating precision/version.
    return [0] if count == 1 else [i * (length - 1) // (count - 1) for i in range(count)]


def reference(inputs, rows, query_batch=16, key_block=4096):
    """Original BF16 inputs, all heads/KV, stable online FP64 softmax, bounded scratch."""
    q, k, v = inputs
    require(
        rows == sorted(set(rows)) and rows and rows[0] >= 0 and rows[-1] < q.shape[2],
        "Reference rows must be sorted unique in-range indices",
    )
    require(query_batch > 0 and key_block > 0, "Positive reference block sizes required")
    out = torch.empty((1, q.shape[1], len(rows), 128), dtype=torch.float64)
    for h in range(q.shape[1]):
        for start in range(0, len(rows), query_batch):
            qr = q[0, h, rows[start : start + query_batch], :].double()
            m = torch.full((qr.shape[0], 1), -torch.inf, dtype=torch.float64)
            total = torch.zeros_like(m)
            value = torch.zeros_like(qr)
            for pos in range(0, k.shape[2], key_block):
                scores = (qr @ k[0, h, pos : pos + key_block, :].double().T) * SCALE
                new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
                correction = (m - new_m).exp()
                weights = (scores - new_m).exp()
                value = value * correction + weights @ v[0, h, pos : pos + key_block, :].double()
                total = total * correction + weights.sum(-1, keepdim=True)
                m = new_m
            out[0, h, start : start + qr.shape[0], :] = value / total
    require(bool(torch.isfinite(out).all()), "FP64 reference is nonfinite; do not report accuracy")
    return out


def metrics(actual, expected, original_v):
    require(actual.shape == expected.shape, "Metric shape mismatch")
    a, e = actual.double(), expected.double()
    require(bool(torch.isfinite(a).all() and torch.isfinite(e).all()), "Nonfinite metric input")
    delta = a - e
    norm, error_norm = float(e.norm()), float(delta.norm())
    af, ef = a.flatten(), e.flatten()
    ac, ec = af - af.mean(), ef - ef.mean()
    pcc_den = float(ac.norm() * ec.norm())
    zero = ef == 0
    zero_mismatches = int((zero & (delta.flatten() != 0)).sum())
    rel = delta.flatten()[~zero].abs() / ef[~zero].abs()
    # Center both outputs by the SAME mean original V. No regression/gain alignment.
    mean = original_v.mean(dim=2, keepdim=True, dtype=torch.float64)
    residual = e - mean
    constant = torch.equal(original_v, original_v[:, :, :1, :].expand_as(original_v))
    residual_norm = 0.0 if constant else float(residual.norm())
    gain = float(af @ ef) / norm**2 if norm else None
    result = dict(
        l2_pct=100 * error_norm / norm if norm else None,
        relative_l2_undefined=not bool(norm),
        pcc=float(ac @ ec) / pcc_den if pcc_den else None,
        gain=gain,
        max_abs=float(delta.abs().max()),
        absolute_error_rms=float(delta.square().mean().sqrt()),
        reference_rms=float(e.square().mean().sqrt()),
        zero_reference_mismatch_count=zero_mismatches,
        max_relative_error_pct=None if zero_mismatches else (100 * float(rel.max()) if rel.numel() else 0.0),
        max_relative_error_unbounded=bool(zero_mismatches),
        residual_l2_pct=100 * error_norm / residual_norm if residual_norm else None,
        residual_relative_undefined=not bool(residual_norm),
        residual_reference_rms=0.0 if constant else float(residual.square().mean().sqrt()),
        residual_definition="Same FP64 original-V mean subtracted from both outputs; no gain fitting",
        bf16_rounding_floor_l2_pct=100 * float((e.bfloat16().double() - e).norm()) / norm if norm else None,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--max-file-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, info = load_capture(args.artifact, args.max_file_bytes)
    info["sampled_query_rows"] = select_rows(info["shape"][2], args.sample_rows)
    print(json.dumps(info, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
