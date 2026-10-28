# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Add the mean-centred MoE gate bias to an existing TTNN weight cache, host-only.

A model that sets MOE_GATE_BIAS_CENTERED (GLM-5.3) loads ``layer_{i}.gate.e_score_correction_bias_centered``
instead of ``layer_{i}.gate.e_score_correction_bias`` (see tt_moe_gate_prefill.centre_gate_bias). This writes the
centred file for every MoE layer of the checkpoint that the cache already holds a gate for, from the checkpoint's
fp32 bias, without opening a device or touching any other cache file. The uncentred files are left in place.

Each layer is checked first: the cached uncentred bias must equal bf16(checkpoint bias) exactly, so a cache built
from another checkpoint is refused rather than mixed. An existing centred file is verified, never rewritten.

    python models/demos/deepseek_v3_d_p/scripts/centre_gate_bias_cache.py \\
        --checkpoint /mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4 \\
        --cache-dir /mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/GLM-5.3-Cache-prefill/glm_5_3_bh_32dev/8x4
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import (
    GATE_BIAS_CACHE_NAME,
    GATE_BIAS_CENTERED_CACHE_NAME,
    centre_gate_bias,
)

_SUFFIX = "_dtype_BFLOAT16_layout_TILE.tensorbin"
_BIAS_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$")


def _checkpoint_biases(checkpoint: Path) -> dict[int, tuple[str, str]]:
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    return {int(m.group(1)): (key, shard) for key, shard in weight_map.items() if (m := _BIAS_KEY.match(key))}


def _as_tensor(bias: torch.Tensor, cache_stem: Path) -> ttnn.Tensor:
    # Same call as TtMoEGatePrefill._convert_and_cache_gate_weights: host, bf16, TILE, unbroadcasted.
    return ttnn.as_tensor(
        bias, device=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, cache_file_name=str(cache_stem)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="HF checkpoint dir (fp32 gate biases)")
    parser.add_argument("--cache-dir", type=Path, required=True, help="TTNN weight cache dir holding layer_*.gate.*")
    parser.add_argument("--dry-run", action="store_true", help="verify only; write nothing")
    args = parser.parse_args()

    biases = _checkpoint_biases(args.checkpoint)
    layers = sorted(i for i in biases if (args.cache_dir / f"layer_{i}.gate.{GATE_BIAS_CACHE_NAME}{_SUFFIX}").exists())
    if not layers:
        raise SystemExit(f"no layer_*.gate.{GATE_BIAS_CACHE_NAME}{_SUFFIX} in {args.cache_dir} matches the checkpoint")
    logger.info(f"{len(layers)} cached MoE gates in {args.cache_dir}: layers {layers[0]}..{layers[-1]}")

    written = verified = 0
    for i in layers:
        key, shard = biases[i]
        with safe_open(str(args.checkpoint / shard), "pt") as f:
            raw = f.get_tensor(key)
        assert raw.dtype == torch.float32, f"layer {i}: checkpoint bias is {raw.dtype}, centring needs the fp32 one"

        legacy = ttnn.to_torch(_as_tensor(raw, args.cache_dir / f"layer_{i}.gate.{GATE_BIAS_CACHE_NAME}"))
        if not torch.equal(legacy.reshape(-1).to(torch.bfloat16), raw.to(torch.bfloat16)):
            raise SystemExit(
                f"layer {i}: cached uncentred bias != bf16(checkpoint bias); wrong checkpoint for this cache"
            )

        centred = centre_gate_bias(raw)
        stem = args.cache_dir / f"layer_{i}.gate.{GATE_BIAS_CENTERED_CACHE_NAME}"
        existed = Path(f"{stem}{_SUFFIX}").exists()
        if not existed and args.dry_run:
            continue
        loaded = ttnn.to_torch(_as_tensor(centred, stem))  # writes the file on a miss
        if not torch.equal(loaded.reshape(-1).to(torch.bfloat16), centred.to(torch.bfloat16)):
            raise SystemExit(f"layer {i}: {stem}{_SUFFIX} does not hold bf16(bias - mean(bias))")
        if existed:
            verified += 1
        else:
            # The CI user is outside the cache's group: a fresh file must be world-readable.
            os.chmod(f"{stem}{_SUFFIX}", 0o664)
            written += 1
        logger.info(
            f"layer {i}: mean {raw.mean().item():.4f}, bf16 distinct {raw.to(torch.bfloat16).unique().numel()} -> "
            f"{centred.to(torch.bfloat16).unique().numel()} ({'verified' if existed else 'written'})"
        )
    logger.info(f"done: {written} written, {verified} already present and verified, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
