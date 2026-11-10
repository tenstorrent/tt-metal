# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""MINIMAX_H3_LOFI: LoFi math fidelity as a default-off option per compute-bound section.

LoFi is lossy: it trades matmul precision for speed, so it is off by default.

`MINIMAX_H3_LOFI=<comma list>` of the names in SECTIONS, or `all` (every section). Unset or empty: every helper returns
its argument itself (the same object), so the default path is exactly today's code. An unknown name raises at import.
Only the math fidelity moves: fp32_dest_acc_en, packer_l1_acc, math_approx_mode, dst_full_sync_en, throttle_level and
every dtype stay as the caller set them. Each enabled section logs its effective fidelity once.
"""

from __future__ import annotations

import os

import torch

import ttnn

SECTIONS = ("qkv", "to_out", "ff1", "ff2", "sparse_sdpa", "dense_sdpa")


def _parse(value: str) -> frozenset[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name != "all" and name not in SECTIONS]
    if unknown:
        msg = f"MINIMAX_H3_LOFI: unknown section(s) {unknown}; known: {', '.join(SECTIONS)} or all"
        raise ValueError(msg)
    return frozenset(SECTIONS) if "all" in names else frozenset(names)


ENABLED = _parse(os.environ.get("MINIMAX_H3_LOFI", ""))

WEIGHT_SECTIONS = ("qkv", "to_out", "ff1", "ff2")
WEIGHT_BITS = 5
ROUND_WEIGHTS = os.environ.get("MINIMAX_H3_LOFI_ROUND_WEIGHTS", "0") == "1"
if ROUND_WEIGHTS and not ENABLED.intersection(WEIGHT_SECTIONS):
    msg = f"MINIMAX_H3_LOFI_ROUND_WEIGHTS=1 needs one of {', '.join(WEIGHT_SECTIONS)} in MINIMAX_H3_LOFI"
    raise ValueError(msg)
_logged: set[str] = set()
_compute_configs: dict[tuple[str, int], tuple[object, object]] = {}
_program_configs: dict[tuple[str, int], tuple[object, object]] = {}


def enabled(section: str) -> bool:
    assert section in SECTIONS, section
    return section in ENABLED


def _log_once(section: str, detail: str) -> None:
    if section not in _logged:
        _logged.add(section)
        print(f"[MINIMAX_H3_LOFI] {section}: {detail} (LoFi, lossy)")


def compute_config(section: str, config):
    """`config` itself unless `section` is enabled; then a cached copy with math_fidelity=LoFi, all else kept."""
    if not enabled(section) or config is None:
        return config
    key = (section, id(config))
    hit = _compute_configs.get(key)
    if hit is None or hit[0] is not config:
        lofi = type(config)(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=config.math_approx_mode,
            fp32_dest_acc_en=config.fp32_dest_acc_en,
            packer_l1_acc=config.packer_l1_acc,
            dst_full_sync_en=config.dst_full_sync_en,
            throttle_level=config.throttle_level,
        )
        hit = _compute_configs[key] = (config, lofi)
        _log_once(
            section,
            f"math_fidelity {config.math_fidelity} -> LoFi (fp32_dest_acc_en={config.fp32_dest_acc_en}, "
            f"packer_l1_acc={config.packer_l1_acc}, math_approx_mode={config.math_approx_mode} unchanged)",
        )
    return hit[1]


def sdpa_program_config(section: str, program_config):
    """`program_config` itself unless `section` is enabled; then a cached copy with matmul_math_fidelity=LoFi.

    The SDPA ops apply `matmul_math_fidelity` to their Q.K^T and softmax.V matmuls (and the sparse op's P.1 row sum,
    which shares the P.V setup); the rest of the kernel keeps the compute config's fidelity. Part of the program hash.
    """
    if not enabled(section) or program_config is None:
        return program_config
    key = (section, id(program_config))
    hit = _program_configs.get(key)
    if hit is None or hit[0] is not program_config:
        pc = program_config
        lofi = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
            sub_core_grids=pc.sub_core_grids,
            q_chunk_size=pc.q_chunk_size,
            k_chunk_size=pc.k_chunk_size,
            exp_approx_mode=pc.exp_approx_mode,
            max_cores_per_head_batch=pc.max_cores_per_head_batch,
            k_splits=pc.k_splits,
            matmul_math_fidelity=ttnn.MathFidelity.LoFi,
            segmented_accumulation=pc.segmented_accumulation,
        )
        hit = _program_configs[key] = (program_config, lofi)
        was = "the compute config's fidelity" if pc.matmul_math_fidelity is None else str(pc.matmul_math_fidelity)
        _log_once(
            section,
            f"Q.K^T / softmax.V matmuls {was} -> LoFi via SDPAProgramConfig.matmul_math_fidelity "
            "(compute config, fp32_dest_acc_en and dtypes unchanged)",
        )
    return hit[1]


def round_significant(t: torch.Tensor, bits: int = WEIGHT_BITS) -> torch.Tensor:
    """`t` rounded to `bits` significant bits, to nearest with ties to even, in `t`'s dtype (exact for bits <= 8)."""
    x = t.to(torch.float32)
    mantissa, exponent = torch.frexp(x)
    rounded = torch.ldexp(torch.round(torch.ldexp(mantissa, torch.full_like(exponent, bits))), exponent - bits)
    return rounded.to(t.dtype)


def round_weight(section: str, weight: torch.Tensor) -> torch.Tensor:
    """`weight` itself unless MINIMAX_H3_LOFI_ROUND_WEIGHTS=1 and `section` is enabled; then rounded to WEIGHT_BITS."""
    assert section in WEIGHT_SECTIONS, section
    if not (ROUND_WEIGHTS and enabled(section)):
        return weight
    _log_once(f"{section} weights", f"rounded to {WEIGHT_BITS} significant bits (MINIMAX_H3_LOFI_ROUND_WEIGHTS)")
    return round_significant(weight)


def weight_cache_tag() -> str:
    """Suffix for the transformer weight-cache subfolder when weights are rounded (they must never load or overwrite
    another cache); empty with MINIMAX_H3_LOFI_ROUND_WEIGHTS off."""
    if not ROUND_WEIGHTS:
        return ""
    return "_lofiw5_" + "-".join(s for s in WEIGHT_SECTIONS if s in ENABLED)
