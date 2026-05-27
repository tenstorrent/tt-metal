#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import argparse
import sys
import typing
from pathlib import Path

from loguru import logger

if sys.version_info < (3, 11):
    try:
        import typing_extensions as _te

        for _name in ("Unpack", "Self", "Required", "NotRequired", "LiteralString", "Never"):
            if not hasattr(typing, _name) and hasattr(_te, _name):
                setattr(typing, _name, getattr(_te, _name))
    except ImportError:
        # typing_extensions is a transitive dep of transformers; if it's
        # missing, the user has bigger problems than this demo.
        logger.error(
            "typing_extensions is a transitive dep of transformers; if it's missing, the user has bigger problems than this demo."
        )
        raise ImportError(
            "typing_extensions is a transitive dep of transformers; if it's missing, the user has bigger problems than this demo."
        )

import torch

_HERE = Path(__file__).resolve().parent
_BUNDLED_CONFIG_DIR = _HERE.parent / "configs" / "minimax-m2.7"
_REFERENCE_DIR = _HERE.parent / "reference"
_REFERENCE_FILES = (
    "config.json",
    "configuration_minimax_m2.py",
    "modeling_minimax_m2.py",
)
_HF_ID = "MiniMaxAI/MiniMax-M2.7"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hf-id", default=_HF_ID, help="HuggingFace repo id (online mode).")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use the bundled config.json instead of fetching from HF.",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help=(
            "Override num_hidden_layers. "
            "Default: 2 for the forward-pass mode; "
            "the full 62 when --print-structure is set."
        ),
    )
    p.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help=(
            "Override num_local_experts. "
            "Default: 8 for the forward-pass mode; "
            "the full 256 when --print-structure is set."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="num_experts_per_tok (default: min(8, num_experts)).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Print config + size estimate, do not instantiate the model.",
    )
    p.add_argument(
        "--print-structure",
        action="store_true",
        help=(
            "Instantiate the model on the 'meta' device (no real allocations) "
            "and print the full nn.Module tree, then exit. Defaults to the full "
            "62-layer / 256-expert config; use --num-layers / --num-experts to shrink."
        ),
    )
    p.add_argument(
        "--from-reference",
        action="store_true",
        help=(
            "Build the model strictly from the local files in "
            "models/demos/minimax2.7/reference/ (config.json, "
            "configuration_minimax_m2.py, modeling_minimax_m2.py) -- no HF "
            "download, no HF cache lookup, no native transformers class. "
            "These files were pulled verbatim from the HF repo."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _humanize(n: float) -> str:
    for unit, mag in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= mag:
            return f"{n / mag:.2f}{unit}"
    return f"{n:.0f}"


def _estimate_params(cfg) -> dict:
    """Param count estimate. Approximate; ignores small biases / scales."""
    H = cfg.hidden_size
    I = cfg.intermediate_size
    L = cfg.num_hidden_layers
    E = cfg.num_local_experts
    V = cfg.vocab_size
    A = cfg.num_attention_heads
    KV = cfg.num_key_value_heads
    D = cfg.head_dim

    embed = V * H
    lm_head = V * H  # not tied
    # qkv_proj + o_proj + full-dim QK-Norm scales
    per_layer_attn = H * (A * D + 2 * KV * D) + (A * D) * H + (A + KV) * D
    # 3 SwiGLU GEMMs per expert + router gate
    per_layer_moe = E * (3 * H * I) + H * E + E
    per_layer_norms = 2 * H  # input_layernorm, post_attention_layernorm
    per_layer = per_layer_attn + per_layer_moe + per_layer_norms
    layers_total = L * per_layer
    final_norm = H
    total = embed + lm_head + layers_total + final_norm
    return {
        "embed": embed,
        "lm_head": lm_head,
        "per_layer": per_layer,
        "layers_total": layers_total,
        "total": total,
    }


def _full_size_estimate_from_bundled() -> int:
    """Param count for stock M2.7 (62 layers, 256 experts), read straight from JSON."""
    import json

    with (_BUNDLED_CONFIG_DIR / "config.json").open() as f:
        raw = json.load(f)

    class _Bag:
        pass

    cfg = _Bag()
    for k in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_local_experts",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        setattr(cfg, k, raw[k])
    return _estimate_params(cfg)["total"]


def _try_import_native():
    """Return (MiniMaxM2Config, MiniMaxM2ForCausalLM) if transformers ships them, else (None, None)."""
    try:
        from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM  # type: ignore[attr-defined]

        return MiniMaxM2Config, MiniMaxM2ForCausalLM
    except (ImportError, AttributeError):
        return None, None


def _check_reference_dir() -> None:
    missing = [f for f in _REFERENCE_FILES if not (_REFERENCE_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Reference directory {_REFERENCE_DIR} is missing files {missing}. "
            f"Pull them with the `huggingface-cli download` command in "
            f"models/demos/minimax2.7/reference/README.md, or run:\n"
            f"  curl -fsSL -o {_REFERENCE_DIR}/modeling_minimax_m2.py \\\n"
            f"    https://huggingface.co/MiniMaxAI/MiniMax-M2.7/resolve/main/modeling_minimax_m2.py\n"
            f"  curl -fsSL -o {_REFERENCE_DIR}/configuration_minimax_m2.py \\\n"
            f"    https://huggingface.co/MiniMaxAI/MiniMax-M2.7/resolve/main/configuration_minimax_m2.py"
        )


def _load_config(args: argparse.Namespace):
    """Load the M2.7 HF config without dragging in network unless required.

    Strategy (highest priority first):
      0. ``--from-reference``: load strictly from
         ``models/demos/minimax2.7/reference/`` via AutoConfig with
         ``trust_remote_code=True``. The reference/ directory contains the
         exact ``configuration_minimax_m2.py`` / ``modeling_minimax_m2.py``
         files from HF, so no network and no HF cache are touched.
      1. If transformers ships ``MiniMaxM2Config`` natively (>= 4.57),
         instantiate it from either the HF id (online) or the bundled
         ``configs/minimax-m2.7/`` directory (offline). No remote code needed.
      2. Else, go through ``AutoConfig`` with ``trust_remote_code=True`` so
         the auto_map can resolve ``configuration_minimax_m2.MiniMaxM2Config``.
         ``--offline`` is mapped to ``local_files_only=True``, which makes
         transformers use the HF cache (populated by a prior online run)
         and fail loudly if the cache is empty -- no silent network access.
    """
    from transformers import AutoConfig

    # Path 0: explicit reference/ -- highest precedence, fully local.
    if args.from_reference:
        _check_reference_dir()
        logger.info(f"Loading config from local reference dir {_REFERENCE_DIR}")
        return AutoConfig.from_pretrained(str(_REFERENCE_DIR), trust_remote_code=True)

    NativeConfig, _ = _try_import_native()

    # Path 1: native transformers class (cleanest, no remote code).
    if NativeConfig is not None:
        if args.offline:
            if not (_BUNDLED_CONFIG_DIR / "config.json").exists():
                raise FileNotFoundError(_BUNDLED_CONFIG_DIR / "config.json")
            logger.info(f"Loading bundled config via native MiniMaxM2Config from {_BUNDLED_CONFIG_DIR}")
            return NativeConfig.from_pretrained(str(_BUNDLED_CONFIG_DIR))
        logger.info(f"Loading config via native MiniMaxM2Config from {args.hf_id}")
        return NativeConfig.from_pretrained(args.hf_id)

    # Path 2: dynamic remote code via auto_map. Always uses the HF id so
    # transformers can find / cache the auto_map .py files; --offline maps
    # to local_files_only so we never silently hit the network.
    logger.info(
        f"Loading config from {args.hf_id} via AutoConfig + remote code "
        f"(local_files_only={args.offline}). Bundled config dir is unused on "
        f"this code path because transformers < 4.57 needs the auto_map .py "
        f"files (which the HF cache supplies)."
    )
    return AutoConfig.from_pretrained(
        args.hf_id,
        trust_remote_code=True,
        local_files_only=args.offline,
    )


def _apply_overrides(
    cfg,
    *,
    num_layers: int | None,
    num_experts: int | None,
    top_k: int | None,
):
    """Apply CLI overrides to a freshly-loaded HF config.

    Any override left at ``None`` keeps the corresponding stock M2.7 value
    (62 layers, 256 experts, top-8). Always pops ``quantization_config`` so
    construction doesn't try to allocate FP8 tensors and ``to_dict()`` doesn't
    crash.
    """
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers
        # All M2.7 layers are full attention; keep that invariant after a shrink.
        if hasattr(cfg, "attn_type_list"):
            cfg.attn_type_list = [1] * num_layers
    if num_experts is not None:
        cfg.num_local_experts = num_experts
    if top_k is not None:
        cfg.num_experts_per_tok = top_k

    # Random-init / meta-init: drop FP8 quantization config. It is only
    # meaningful for loaded weights, and leaving it in place causes two
    # failures:
    #   1. The HF quantizer would try to allocate FP8 tensors at __init__.
    #   2. Setting it to None crashes ``PretrainedConfig.to_dict()`` (called
    #      via ``GenerationConfig.from_model_config``) because that method
    #      does ``self.quantization_config.to_dict()`` without a None check.
    # Removing the attribute from ``__dict__`` makes ``hasattr(cfg, ...)``
    # return False so ``to_dict()`` skips the key entirely.
    cfg.__dict__.pop("quantization_config", None)
    if hasattr(cfg, "quantization_config") and cfg.quantization_config is None:
        # Class-level default in some transformers versions; fall back to an
        # empty dict, which the same to_dict() codepath handles via its
        # ``isinstance(..., dict)`` branch.
        cfg.quantization_config = {}

    return cfg


def _resolve_overrides(args: argparse.Namespace) -> tuple[int | None, int | None, int | None]:
    """Resolve --num-layers / --num-experts / --top-k into concrete values.

    For the regular forward-pass mode we apply the small-shrink defaults
    (2 layers, 8 experts) when the user didn't pass anything. For
    --print-structure we leave whatever the user passed alone -- ``None``
    means "use the stock M2.7 value" (62 / 256 / 8).
    """
    if args.print_structure:
        layers = args.num_layers  # None -> stock 62
        experts = args.num_experts  # None -> stock 256
        top_k = args.top_k  # None -> stock 8
    else:
        layers = args.num_layers if args.num_layers is not None else 2
        experts = args.num_experts if args.num_experts is not None else 8
        top_k = args.top_k if args.top_k is not None else min(8, experts)
    return layers, experts, top_k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _instantiate_model(cfg, args: argparse.Namespace, *, on_meta: bool):
    """Construct a MiniMaxM2ForCausalLM from ``cfg``.

    ``on_meta=True`` allocates parameters on the ``meta`` device (no real
    storage) — used by ``--print-structure`` to fit the full 229 B model
    in a few hundred MB so we can ``print(model)``.
    """
    _, NativeForCausalLM = _try_import_native()

    def _build():
        # On --from-reference we explicitly want the reference/ modeling file
        # to drive the model construction (not the upstream native class).
        # AutoModelForCausalLM.from_config + trust_remote_code follows the
        # cfg.auto_map back to the directory cfg was loaded from.
        if args.from_reference:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        if NativeForCausalLM is not None:
            return NativeForCausalLM(cfg)
        from transformers import AutoModelForCausalLM

        # Always allow remote code on this path; --offline only controlled
        # the *config download* step. The modeling file should already be in
        # the HF cache from a previous online run -- transformers won't try
        # to re-fetch it because the auto_map files are already present.
        return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    if on_meta:
        with torch.device("meta"):
            return _build()
    return _build()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    if args.from_reference:
        src_label = f"local reference dir {_REFERENCE_DIR}"
    elif args.offline and _try_import_native()[0] is not None:
        src_label = "bundled config (native MiniMaxM2Config)"
    elif args.offline:
        src_label = f"HF cache for {args.hf_id} (no network)"
    else:
        src_label = args.hf_id
    print(f"[1/4] Loading config ({src_label}) ...")
    try:
        cfg = _load_config(args)
    except Exception as e:
        print(f"  ERROR: failed to load config: {type(e).__name__}: {e}")
        if args.offline:
            print(
                "  hint: --offline does not hit the network. With transformers "
                ">= 4.57 it uses the bundled JSON; otherwise it expects the HF "
                "cache to already contain the auto_map files. Run once without "
                "--offline (or `huggingface-cli download MiniMaxAI/MiniMax-M2.7 "
                "configuration_minimax_m2.py modeling_minimax_m2.py config.json`) "
                "to populate the cache, then retry with --offline."
            )
        else:
            print(
                "  hint: check your network / HF auth, or run `huggingface-cli " "login` if the repo gates downloads."
            )
        return 1

    full_total = _full_size_estimate_from_bundled()
    print(
        f"      Stock M2.7: 62 layers, 256 experts, ~{_humanize(full_total)} params "
        f"(~{full_total * 2 / 1e9:.0f} GB at bf16) -- will not fit on CPU."
    )

    layers, experts, top_k = _resolve_overrides(args)

    if args.print_structure:
        # No shrink banner unless the user asked for one.
        action = (
            "preserving stock M2.7 config"
            if (layers is None and experts is None and top_k is None)
            else "applying explicit overrides"
        )
        print(
            f"[2/3] Building model on the meta device ({action}; "
            f"layers={layers if layers is not None else cfg.num_hidden_layers}, "
            f"experts={experts if experts is not None else cfg.num_local_experts}, "
            f"top_k={top_k if top_k is not None else cfg.num_experts_per_tok}) ..."
        )
    else:
        assert layers is not None and experts is not None and top_k is not None
        print(
            f"[2/4] Shrinking: layers {cfg.num_hidden_layers} -> {layers}, "
            f"experts {cfg.num_local_experts} -> {experts}, "
            f"top-k {cfg.num_experts_per_tok} -> {top_k}."
        )

    cfg = _apply_overrides(
        cfg,
        num_layers=layers,
        num_experts=experts,
        top_k=top_k,
    )

    if not args.print_structure:
        sizes = _estimate_params(cfg)
        item_size = 2 if dtype == torch.bfloat16 else 4
        print(
            f"      Shrunk params: ~{_humanize(sizes['total'])} "
            f"(embed {_humanize(sizes['embed'])}, lm_head {_humanize(sizes['lm_head'])}, "
            f"layers {_humanize(sizes['layers_total'])}). "
            f"Est. weights footprint at {args.dtype}: "
            f"~{sizes['total'] * item_size / 1e9:.2f} GB."
        )

    if args.check_only:
        print("\n--check-only set; not instantiating the model.")
        return 0

    # ---- --print-structure: meta-device load, dump tree, exit ----
    if args.print_structure:
        print("[3/3] Allocating on meta device and printing module tree ...")
        model = _instantiate_model(cfg, args, on_meta=True)
        model.eval()
        total = sum(p.numel() for p in model.parameters())
        print()
        print("=" * 80)
        print("MiniMaxM2ForCausalLM module tree (parameters live on meta device):")
        print("=" * 80)
        print(model)
        print("=" * 80)
        print(
            f"Total parameters: {_humanize(total)} ({total:,}). "
            f"Footprint at bf16 would be ~{total * 2 / 1e9:.1f} GB "
            f"(but no real storage was allocated)."
        )
        return 0

    # ---- Default path: random init + single forward pass ----
    print("[3/4] Instantiating model with random weights (this is the slow step) ...")
    model = _instantiate_model(cfg, args, on_meta=False)
    model.eval()

    if dtype != torch.float32:
        # Cast post-init: HF init code uses nn.init.normal_, which historically
        # has had bf16 issues. Allocating in fp32 then converting is the safe path.
        print(f"      Casting model to {args.dtype} ...")
        model = model.to(dtype)

    actual_params = sum(p.numel() for p in model.parameters())
    print(
        f"      Actual params: {_humanize(actual_params)} "
        f"({actual_params * dtype.itemsize / 1e9:.2f} GB on CPU at {args.dtype})."
    )

    print(f"[4/4] Running forward pass on input shape ({args.batch_size}, {args.seq_len}) ...")
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len))
    with torch.no_grad():
        out = model(input_ids)

    logits = out.logits if hasattr(out, "logits") else out[0]
    print(f"      Output logits shape: {tuple(logits.shape)}  dtype={logits.dtype}")

    last = logits[0, -1].float()
    topk = torch.topk(last, k=5)
    print("      Top-5 next-token logits for the last position (random weights -> noise):")
    for tok_id, logit in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f"        token {tok_id:>6d}  logit={logit:+.4f}")

    print("\nDone. Random-weight CPU smoke test successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
