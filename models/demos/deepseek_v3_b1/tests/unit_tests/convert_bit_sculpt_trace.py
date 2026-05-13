# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a bit_sculpt debug-trace dump into the on-disk format consumed by
``run_host_io_decoder_sweep.py`` (and matching the harness's own KV-cache dump
layout).

Input layout (bit_sculpt, "Group A" trace, post-merge form):
    <bit_sculpt_dir>/
        kv_cache_layer_{i}.safetensors    # key 'kv_post_transform_layer_{i}',
                                          # shape (T_total, 576) bf16
        step_0/                           # prefill, T_new = num_prefill_tokens
            decoder_input_layer_0.safetensors        (real file)
            decoder_input_layer_{1..N}.safetensors   (symlinks to
                                          decoder_output_layer_{i-1}.safetensors)
            decoder_output_layer_{0..N}.safetensors  (real files)
        step_1/ ... step_{S}/             # decode steps, T_new = 1 each
            (same structure)

The tarballs MUST be extracted first (see REPORT.md "Extracting the decoder I/O
tarballs"). This tool only reads ``step_*/`` directories on disk.

Output layout (matches what ``run_host_io_decoder_sweep.py`` expects /
produces):
    <out_dir>/
        {prompt_name}.pt                              # hidden-state trace
            {"input":  (L, HIDDEN_SIZE) bf16,
             "output": (L, HIDDEN_SIZE) bf16}
        kv_cache_reference_{prompt_name}.pt           # optional KV reference
            (1, L, kvpe_dim) bf16  -- slot-agnostic; consumed by the
            harness's --validate-kv-cache-cross-trace gate.

with ``L = num_prefill_tokens + num_decode_steps``, ``HIDDEN_SIZE = 7168``, and
``kvpe_dim = 576`` (=512 kv_latent + 64 k_pe_roped) for DeepSeek V3.

Usage example::

    python -m models.demos.deepseek_v3_b1.tests.unit_tests.convert_bit_sculpt_trace \\
        --bit-sculpt-dir /data/asaigal/bit_sculpt/results/deepseek-r1-0528/debug_trace/cache-design-gen8192 \\
        --layer-idx 4 \\
        --prompt-name blitz_test \\
        --out-dir /data/asaigal/pipeclean_traces/blitz_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

HIDDEN_SIZE = 7168
KV_PE_DIM = 576  # 512 (kv_latent_normed) + 64 (k_pe_roped) — DeepSeek V3 MLA
DEFAULT_NUM_PREFILL_TOKENS = 65
DEFAULT_NUM_DECODE_STEPS = 8192


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_single_tensor(path: Path) -> torch.Tensor:
    """Load a safetensors file containing exactly one tensor and return it.

    The bit_sculpt symlinks (``decoder_input_layer_{i+1}.safetensors ->
    decoder_output_layer_{i}.safetensors``) cause the safetensors tensor key to
    reflect the TARGET file, not the symlink path. So we never key-by-name; we
    just take the only tensor in the file and validate the count.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected safetensors file not found: {path}")
    d = load_file(str(path))
    if len(d) != 1:
        raise ValueError(f"{path}: expected exactly 1 tensor, got {len(d)} (keys={list(d.keys())})")
    return next(iter(d.values()))


def _step_dir(bit_sculpt_dir: Path, step: int) -> Path:
    """Resolve and validate a per-step directory exists."""
    p = bit_sculpt_dir / f"step_{step}"
    if not p.is_dir():
        raise FileNotFoundError(
            f"Per-step directory missing: {p}. Did you extract the artifacts-*.tar.zst tarballs "
            f"in {bit_sculpt_dir}? See REPORT.md 'Extracting the decoder I/O tarballs'."
        )
    return p


def _load_hidden_states(
    bit_sculpt_dir: Path,
    layer_idx: int,
    num_prefill_tokens: int,
    num_decode_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-step decoder I/O for one layer into ``(L, HIDDEN_SIZE)`` tensors.

    Validates each loaded tensor's shape/dtype:
        step_0 input / output:           (num_prefill_tokens, HIDDEN_SIZE) bf16
        step_{1..S} input / output:      (1, HIDDEN_SIZE) bf16
        final concatenation:             (num_prefill_tokens + num_decode_steps, HIDDEN_SIZE) bf16
    """
    L_expected = num_prefill_tokens + num_decode_steps
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    total_steps = num_decode_steps + 1  # +1 for the prefill step (step_0)

    log_every = max(1, total_steps // 20)
    for s in range(total_steps):
        step_dir = _step_dir(bit_sculpt_dir, s)
        in_path = step_dir / f"decoder_input_layer_{layer_idx}.safetensors"
        out_path = step_dir / f"decoder_output_layer_{layer_idx}.safetensors"
        in_t = _load_single_tensor(in_path)
        out_t = _load_single_tensor(out_path)

        expected_T = num_prefill_tokens if s == 0 else 1
        for label, t in (("input", in_t), ("output", out_t)):
            if t.dtype != torch.bfloat16:
                raise ValueError(f"step_{s}/{label} layer {layer_idx}: expected bf16, got {t.dtype} ({in_path})")
            if t.ndim != 2 or t.shape != (expected_T, HIDDEN_SIZE):
                raise ValueError(
                    f"step_{s}/{label} layer {layer_idx}: expected shape ({expected_T}, {HIDDEN_SIZE}), "
                    f"got {tuple(t.shape)}"
                )
        inputs.append(in_t)
        outputs.append(out_t)

        if (s + 1) % log_every == 0 or s + 1 == total_steps:
            logger.info(f"loaded step {s + 1}/{total_steps}")

    input_tensor = torch.cat(inputs, dim=0)
    output_tensor = torch.cat(outputs, dim=0)
    if input_tensor.shape != (L_expected, HIDDEN_SIZE):
        raise ValueError(
            f"concatenated input shape mismatch: got {tuple(input_tensor.shape)}, expected ({L_expected}, {HIDDEN_SIZE})"
        )
    if output_tensor.shape != (L_expected, HIDDEN_SIZE):
        raise ValueError(
            f"concatenated output shape mismatch: got {tuple(output_tensor.shape)}, "
            f"expected ({L_expected}, {HIDDEN_SIZE})"
        )
    return input_tensor.contiguous(), output_tensor.contiguous()


def _load_kv_cache(bit_sculpt_dir: Path, layer_idx: int, L_expected: int) -> torch.Tensor:
    """Load the cumulative KV cache for one layer and return shape ``(L, KV_PE_DIM)`` bf16.

    Validates the file exists, the single-tensor-per-file contract, dtype bf16,
    and that the total cache length matches ``L_expected``
    (= ``num_prefill_tokens + num_decode_steps``).
    """
    path = bit_sculpt_dir / f"kv_cache_layer_{layer_idx}.safetensors"
    kv = _load_single_tensor(path)
    if kv.dtype != torch.bfloat16:
        raise ValueError(f"{path}: expected bf16, got {kv.dtype}")
    if kv.ndim != 2 or kv.shape != (L_expected, KV_PE_DIM):
        raise ValueError(f"{path}: expected shape ({L_expected}, {KV_PE_DIM}), got {tuple(kv.shape)}")
    return kv.contiguous()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_hidden_states(out_dir: Path, prompt_name: str, inp: torch.Tensor, out: torch.Tensor) -> Path:
    """Write the ``{prompt_name}.pt`` file in the run_host_io_decoder_sweep contract."""
    out_path = out_dir / f"{prompt_name}.pt"
    torch.save({"input": inp, "output": out}, out_path)
    logger.info(
        f"wrote {out_path} "
        f"(input shape={tuple(inp.shape)} {inp.dtype}, output shape={tuple(out.shape)} {out.dtype})"
    )
    return out_path


def _write_kv_cache(out_dir: Path, prompt_name: str, kv: torch.Tensor) -> Path:
    """Write the ``kv_cache_reference_{prompt_name}.pt`` file.

    Shape on disk is ``(1, L, KV_PE_DIM)`` bf16 — matches the per-(slot, prompt)
    slice the harness extracts from the on-device KV cache (leading head dim,
    then the position range, then the kv_pe channel range). The file is
    slot-agnostic because every replicated slot is expected to match the same
    reference; consumed by the harness's --validate-kv-cache-cross-trace gate.
    """
    kv_3d = kv.unsqueeze(0)  # (1, L, 576)
    out_path = out_dir / f"kv_cache_reference_{prompt_name}.pt"
    torch.save(kv_3d, out_path)
    logger.info(f"wrote {out_path} (shape={tuple(kv_3d.shape)} {kv_3d.dtype})")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a bit_sculpt debug-trace dump (extracted) into the on-disk "
            "format consumed by run_host_io_decoder_sweep.py. Emits one "
            "{prompt_name}.pt hidden-state trace plus an optional KV-cache "
            "reference matching the harness's own dump layout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bit-sculpt-dir",
        type=Path,
        required=True,
        help=(
            "Path to the extracted bit_sculpt trace dir (must contain "
            "kv_cache_layer_*.safetensors at root and step_*/ subdirs)."
        ),
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        required=True,
        help="Decoder layer index to convert (e.g. 4 for the first MoE layer in DeepSeek V3).",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        required=True,
        help="Output filename stem (used as --prompt for run_host_io_decoder_sweep).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "--num-prefill-tokens",
        type=int,
        default=DEFAULT_NUM_PREFILL_TOKENS,
        help="Number of prefill tokens in step_0 (T_new at step_0).",
    )
    parser.add_argument(
        "--num-decode-steps",
        type=int,
        default=DEFAULT_NUM_DECODE_STEPS,
        help="Number of decode steps (= number of step_{1..S} dirs).",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Skip KV-cache emission (hidden-state trace only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.bit_sculpt_dir.is_dir():
        raise FileNotFoundError(f"--bit-sculpt-dir does not exist or is not a directory: {args.bit_sculpt_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    L_expected = args.num_prefill_tokens + args.num_decode_steps
    logger.info(
        f"Conversion plan: bit_sculpt_dir={args.bit_sculpt_dir} layer={args.layer_idx} "
        f"prompt_name={args.prompt_name!r} out_dir={args.out_dir} "
        f"prefill={args.num_prefill_tokens} decode={args.num_decode_steps} L={L_expected} "
        f"emit_kv_cache={not args.no_kv_cache}"
    )

    inp, out = _load_hidden_states(
        bit_sculpt_dir=args.bit_sculpt_dir,
        layer_idx=args.layer_idx,
        num_prefill_tokens=args.num_prefill_tokens,
        num_decode_steps=args.num_decode_steps,
    )
    _write_hidden_states(args.out_dir, args.prompt_name, inp, out)

    if not args.no_kv_cache:
        kv = _load_kv_cache(args.bit_sculpt_dir, args.layer_idx, L_expected)
        _write_kv_cache(args.out_dir, args.prompt_name, kv)

    logger.info("conversion complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
