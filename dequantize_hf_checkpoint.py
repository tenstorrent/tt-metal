    #!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Export a dequantized MiniMax-M2 HuggingFace checkpoint.

The source checkpoint uses **block-wise FP8** quantization
(``quant_method: fp8``, ``float8_e4m3fn``, ``weight_block_size: [128, 128]``,
dynamic activations) — the same scheme DeepSeek-V3 uses. Each quantized
weight ``X.weight`` (FP8 e4m3) is paired with a companion
``X.weight_scale_inv`` (float32), one scale per 128×128 block:

    dequant[r, c] = float(weight[r, c]) * scale_inv[r // 128, c // 128]

Quantized tensors are the attention projections (q/k/v/o_proj) and all
routed-expert weights (``block_sparse_moe.experts.E.w{1,2,3}``). Everything
else — the router gate, ``e_score_correction_bias``, ``lm_head``,
``embed_tokens``, and all norms — is already bf16/f32 and passes through
unchanged. Detection is automatic: a tensor is dequantized iff it has a
``.weight_scale_inv`` companion in the index.

This writes a new safetensors checkpoint where every quantized weight is
bfloat16 and the ``weight_scale_inv`` companions are dropped.

Standalone by design: depends only on ``torch`` and ``safetensors`` (no
``ttnn``/blaze imports) so it can run on a CI host with a long timeout.

Usage (run as a file — no blaze/ttnn needed):

    python blaze/weights/minimax_m2/dequantize_hf_checkpoint.py \\
        /MLPerf/minimax_m2/model \\
        --output-model-path /MLPerf/minimax_m2/dequant \\
        --verify --hash

Options:
    --output-model-path   Output dir. Default: <model_path>/../minimax_m2_dequant
    --force               Overwrite the output directory if it exists.
    --verify              Post-dequant shape/dtype/block-grid checks.
    --hash                Write dequant_manifest.json (per-tensor SHA-256 +
                          a single checkpoint hash) for trace-sync handoff.
"""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("minimax_m2.dequant")

MODEL_INDEX_FILENAME = "model.safetensors.index.json"
SCALE_SUFFIX = ".weight_scale_inv"
BLOCK = 128  # weight_block_size from config.json: [128, 128]


def dequantize_block_fp8(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block: int = BLOCK,
) -> torch.Tensor:
    """Dequantize one block-wise FP8 weight to bfloat16.

    ``dequant[r, c] = float(weight[r, c]) * scale_inv[r // block, c // block]``

    Vectorized: the [n_row_blocks, n_col_blocks] scale grid is expanded to the
    full [rows, cols] via repeat_interleave, then cropped (handles a ragged
    final block where a dim is not a multiple of ``block``).
    """
    w = weight_fp8.to(torch.float32)
    rows, cols = w.shape
    sr, sc = scale_inv.shape
    expected = ((rows + block - 1) // block, (cols + block - 1) // block)
    if (sr, sc) != expected:
        raise ValueError(
            f"scale_inv grid {(sr, sc)} != expected {expected} for weight {tuple(w.shape)} "
            f"at block={block} (transposed or wrong block size?)."
        )
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:rows, :cols]
    return (w * s).to(torch.bfloat16).contiguous()


def _load_weight_map(model_path: Path) -> dict[str, str]:
    with (model_path / MODEL_INDEX_FILENAME).open("r", encoding="utf-8") as f:
        weight_map = dict(json.load(f)["weight_map"])
    for key, shard in weight_map.items():
        if Path(shard).name != shard:  # guard against path traversal in the index
            raise ValueError(f"Unsafe shard name for key {key!r}: {shard!r}")
    return weight_map


def _handle(handles: dict[str, Any], model_path: Path, shard: str) -> Any:
    if shard not in handles:
        handles[shard] = safe_open(model_path / shard, framework="pt", device="cpu")
    return handles[shard]


def _copy_non_weight_artifacts(source: Path, dest: Path) -> None:
    """Copy config.json, tokenizer, etc. — everything except weights + index."""
    for item in source.iterdir():
        if item.name == MODEL_INDEX_FILENAME or item.suffix == ".safetensors":
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def save_dequantized_checkpoint(
    source_model_path: Path,
    output_model_path: Path,
    *,
    overwrite: bool = False,
    verify: bool = False,
    write_hash: bool = False,
) -> Path:
    source_model_path = source_model_path.resolve()
    output_model_path = output_model_path.resolve()
    if output_model_path == source_model_path:
        raise ValueError("Output path must differ from source path.")
    if not source_model_path.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {source_model_path}")
    if output_model_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_model_path}. Use --force to overwrite."
            )
        shutil.rmtree(output_model_path)
    output_model_path.mkdir(parents=True, exist_ok=True)
    _copy_non_weight_artifacts(source_model_path, output_model_path)

    weight_map = _load_weight_map(source_model_path)
    scale_keys = {k for k in weight_map if k.endswith(SCALE_SUFFIX)}
    # A weight is quantized iff it owns a `.weight_scale_inv` companion.
    quantized = {k[: -len(".weight_scale_inv")] + ".weight" for k in scale_keys}

    # One output shard per source shard — preserves the original sharding
    # layout and bounds peak memory to a single shard at a time.
    keys_by_src_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key in scale_keys:
            continue  # scale_inv companions are consumed, never written out
        keys_by_src_shard.setdefault(shard, []).append(key)

    src_shards = sorted(keys_by_src_shard)
    total = len(src_shards)
    output_weight_map: dict[str, str] = {}
    total_size = 0
    handles: dict[str, Any] = {}

    try:
        for idx, src_shard in enumerate(src_shards, start=1):
            out_shard = f"model-{idx:05d}-of-{total:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            n_deq = 0
            for key in keys_by_src_shard[src_shard]:
                t = _handle(handles, source_model_path, weight_map[key]).get_tensor(key)
                if key in quantized:
                    scale_key = key[: -len(".weight")] + SCALE_SUFFIX
                    scale = _handle(
                        handles, source_model_path, weight_map[scale_key]
                    ).get_tensor(scale_key)
                    t = dequantize_block_fp8(t, scale)
                    n_deq += 1
                tensors[key] = t
                output_weight_map[key] = out_shard
                total_size += t.numel() * t.element_size()
            save_file(tensors, str(output_model_path / out_shard))
            log.info(
                "shard %d/%d %s: %d tensors (%d dequantized)",
                idx, total, src_shard, len(tensors), n_deq,
            )
    finally:
        for h in handles.values():
            if hasattr(h, "close"):
                h.close()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    with (output_model_path / MODEL_INDEX_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    log.info(
        "Dequantized checkpoint written to %s (%d tensors, %d shards, %.1f GiB)",
        output_model_path, len(output_weight_map), total, total_size / 2**30,
    )

    if verify:
        _verify(source_model_path, output_model_path, output_weight_map, quantized)
    if write_hash:
        _write_manifest(output_model_path, output_weight_map)
    return output_model_path


def _verify(
    source_path: Path,
    output_path: Path,
    output_weight_map: dict[str, str],
    quantized: set[str],
) -> None:
    """Shape/dtype/companion-absence + block-grid sanity checks."""
    log.info("Verifying dequantized checkpoint ...")
    errors: list[str] = []
    src_map = _load_weight_map(source_path)
    src_h: dict[str, Any] = {}
    out_h: dict[str, Any] = {}
    try:
        for key, out_shard in output_weight_map.items():
            if key.endswith(SCALE_SUFFIX):
                errors.append(f"scale_inv companion leaked into output: {key}")
                continue
            out_t = _handle(out_h, output_path, out_shard).get_tensor(key)
            src_t = _handle(src_h, source_path, src_map[key]).get_tensor(key)
            if key in quantized:
                if out_t.dtype != torch.bfloat16:
                    errors.append(f"{key}: expected bfloat16, got {out_t.dtype}")
                if tuple(out_t.shape) != tuple(src_t.shape):
                    errors.append(f"{key}: shape {tuple(src_t.shape)} → {tuple(out_t.shape)}")
                if not torch.isfinite(out_t.to(torch.float32)).all():
                    errors.append(f"{key}: non-finite values after dequant")
            else:
                if out_t.dtype != src_t.dtype:
                    errors.append(f"{key}: dtype changed {src_t.dtype} → {out_t.dtype}")
                if tuple(out_t.shape) != tuple(src_t.shape):
                    errors.append(f"{key}: shape changed {tuple(src_t.shape)} → {tuple(out_t.shape)}")
        for scale_key in (k for k in src_map if k.endswith(SCALE_SUFFIX)):
            if scale_key in output_weight_map:
                errors.append(f"Companion key present in output: {scale_key}")
    finally:
        for h in list(src_h.values()) + list(out_h.values()):
            if hasattr(h, "close"):
                h.close()
    if errors:
        for e in errors[:50]:
            log.error("  FAIL: %s", e)
        raise RuntimeError(f"Verification failed with {len(errors)} error(s).")
    log.info("Verification passed.")


def _write_manifest(output_path: Path, output_weight_map: dict[str, str]) -> None:
    """Per-tensor SHA-256 + a single rolled-up checkpoint hash.

    Exchange the ``checkpoint_sha256`` with anyone syncing on traces (e.g. the
    trace owner) to prove both sides use byte-identical dequantized weights.
    """
    log.info("Hashing dequantized tensors ...")
    per_tensor: dict[str, str] = {}
    out_h: dict[str, Any] = {}
    try:
        for key in sorted(output_weight_map):
            t = _handle(out_h, output_path, output_weight_map[key]).get_tensor(key)
            arr = t.contiguous().view(torch.uint8).numpy().tobytes()
            per_tensor[key] = hashlib.sha256(arr).hexdigest()
    finally:
        for h in out_h.values():
            if hasattr(h, "close"):
                h.close()
    rollup = hashlib.sha256()
    for key in sorted(per_tensor):
        rollup.update(key.encode())
        rollup.update(bytes.fromhex(per_tensor[key]))
    manifest = {"checkpoint_sha256": rollup.hexdigest(), "tensors": per_tensor}
    with (output_path / "dequant_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("checkpoint_sha256 = %s", manifest["checkpoint_sha256"])
    log.info("manifest written to %s", output_path / "dequant_manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dequantize a MiniMax-M2 block-FP8 HuggingFace checkpoint to bfloat16."
    )
    p.add_argument("model_path", type=Path, help="Source HF checkpoint directory.")
    p.add_argument(
        "--output-model-path", type=Path, default=None,
        help="Output directory. Defaults to <model_path>/../minimax_m2_dequant.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    p.add_argument("--verify", action="store_true", help="Run post-dequantization checks.")
    p.add_argument("--hash", action="store_true", help="Write dequant_manifest.json with SHA-256 hashes.")
    args = p.parse_args()

    model_path = args.model_path.resolve()
    output_path = (
        args.output_model_path.resolve()
        if args.output_model_path is not None
        else model_path.parent / "minimax_m2_dequant"
    )
    save_dequantized_checkpoint(
        model_path, output_path,
        overwrite=args.force, verify=args.verify, write_hash=args.hash,
    )


if __name__ == "__main__":
    main()
