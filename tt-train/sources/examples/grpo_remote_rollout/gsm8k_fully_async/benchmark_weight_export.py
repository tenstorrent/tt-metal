# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Single-process weight-export round-trip benchmark for ttml models.

Builds a real ttml model (Qwen3 or Llama) with real HF weights on a `[1, 1]`
mesh on chip 0, then for a configurable number of iterations, for every entry
of the model's HF-keyed dict:

  1. Times `ttnn.to_torch(t)` (D->H copy).
  2. Times `ttnn.from_torch(host, dtype=bf16, layout=TILE, device=mesh,
     memory_config=DRAM_INTERLEAVED, mesh_mapper=replicate)` (H->D copy)
     into a fresh output dict (NOT the model's parameter store).

The dict-assembly call (`qwen3_weights_ref_hf_dict` / `weights_ref_hf_dict`)
is NOT timed -- it runs once per iter outside the measurement window.

DRAM headroom: each H->D result is deallocated with `ttnn.deallocate(force=True)`
immediately after its h2d time is recorded, so peak DRAM stays at `model +
1 tensor`. Without this, a Qwen3-4B round-trip would need `model (~8 GiB) +
uploaded_dict (~8 GiB)` in a P150's ~12 GiB and would OOM. The output dict's
values become stale handles right after deallocation; the dict is still built
so the schema and record-keeping are visible.

Prints per-iter one-liners, a top-N slowest table for iter 0 (by d2h time),
and an aggregate steady-state summary.

Default model set (in run order):
  * Qwen/Qwen3-0.6B-Base
  * meta-llama/Llama-3.2-1B-Instruct
  * Qwen/Qwen3-1.7B-Base
  * Qwen/Qwen3-4B-Base

The device is closed and re-opened between models so DRAM starts empty every
run.

Usage:
    python3 tt-train/sources/examples/grpo_remote_rollout/gsm8k_fully_async/benchmark_weight_export.py
    python3 ... --models qwen3-0.6b-base,qwen3-4b-base
    python3 ... --iters 5 --warmup-iters 2
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List

import torch

# Make `utils/` importable the same way the fully-async example does.
_HERE = Path(__file__).resolve().parent
_EX_ROOT = _HERE.parent
if str(_EX_ROOT) not in sys.path:
    sys.path.insert(0, str(_EX_ROOT))

import ttml  # noqa: E402
import ttnn  # noqa: E402

# The only utils import: pure on-device dict assembler.
from utils.qwen3_overrides import qwen3_weights_ref_hf_dict  # noqa: E402


_MODELS: List[Dict[str, Any]] = [
    {
        "label": "qwen3-0.6b-base",
        "kind": "qwen3",
        "hf_repo": "Qwen/Qwen3-0.6B-Base",
        "max_seq_len": 1024,
    },
    # {
    #     "label": "llama-3.2-1b-instruct",
    #     "kind": "llama",
    #     "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
    #     # Reuse the existing tt-train Llama-3.2-1B config (arch fields only; we
    #     # override vocab_size and max_sequence_length from the tokenizer / CLI).
    #     "yaml": "model_configs/llama3_2_1B.yaml",
    #     "max_seq_len": 2048,
    # },
    {
        "label": "qwen3-1.7b-base",
        "kind": "qwen3",
        "hf_repo": "Qwen/Qwen3-1.7B-Base",
        "max_seq_len": 1024,
    },
    {
        "label": "qwen3-4b-base",
        "kind": "qwen3",
        "hf_repo": "Qwen/Qwen3-4B-Base",
        "max_seq_len": 1024,
    },
]


def _build_qwen3(hf_repo: str, max_seq_len: int):
    """Return `(model, build_dict_callable)` for a ttml Qwen3 built from an HF repo."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM
    from ttml.models import RunnerType
    from ttml.models.qwen3 import Qwen3, create_qwen3_config_from_hf
    from ttml.models.qwen3.weights import load_weights_from_hf

    hf_config = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
    qwen_config = create_qwen3_config_from_hf(hf_config, max_seq_len, runner_type=RunnerType.Default)

    model = Qwen3(qwen_config)
    tie = bool(getattr(hf_config, "tie_word_embeddings", False))

    path = snapshot_download(
        repo_id=hf_repo,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        token=os.environ.get("HF_TOKEN"),
    )
    hf_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)
    sd = hf_model.state_dict()
    del hf_model
    load_weights_from_hf(model, sd, qwen_config, tie_word_embeddings=tie)
    del sd
    gc.collect()

    def build_dict():
        return qwen3_weights_ref_hf_dict(model, tie_word_embeddings=tie)

    return model, build_dict


def _ensure_safetensors_dir(model_dir: str) -> str:
    """If a HF snapshot only ships pytorch_model*.bin, convert to a single
    model.safetensors on first use. Mirrors llama_grpo_completer._ensure_safetensors_dir."""
    from safetensors.torch import save_file

    p = Path(model_dir)
    if list(p.glob("*.safetensors")):
        return model_dir

    bin_files = sorted(p.glob("pytorch_model*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"Neither *.safetensors nor pytorch_model*.bin found in {model_dir}")

    state_dict: dict = {}
    for bin_file in bin_files:
        logging.info("Converting legacy weights to safetensors: %s", bin_file)
        sd = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        for k, v in sd.items():
            state_dict[k] = v.contiguous()

    out_path = p / "model.safetensors"
    save_file(state_dict, str(out_path))
    return model_dir


def _build_llama(hf_repo: str, yaml_path: str, max_seq_len: int):
    """Return `(model, build_dict_callable)` for a ttml LlamaCompositeKV built from an HF repo."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from ttml.common.config import TransformerConfig, load_config
    from ttml.models import RunnerType, WeightTyingType
    from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors

    from utils.llama_overrides import LlamaCompositeKV

    raw = load_config(yaml_path)
    tf_config = TransformerConfig(raw["transformer_config"])

    tokenizer = AutoTokenizer.from_pretrained(hf_repo, token=os.environ.get("HF_TOKEN"))
    tf_config.vocab_size = len(tokenizer)
    tf_config.max_sequence_length = int(max_seq_len)

    rope_scaling = LlamaRopeScalingConfig(
        scaling_factor=getattr(tf_config, "scaling_factor", 0.0) or 0.0,
        high_freq_factor=getattr(tf_config, "high_freq_factor", 4.0) or 4.0,
        low_freq_factor=getattr(tf_config, "low_freq_factor", 1.0) or 1.0,
        original_context_length=getattr(tf_config, "original_context_length", 0) or 0,
    )
    runner_type = RunnerType.from_string(str(tf_config.runner_type))
    weight_tying = WeightTyingType.Disabled
    if tf_config.weight_tying:
        weight_tying = WeightTyingType.from_string(str(tf_config.weight_tying))

    llama_cfg = LlamaConfig(
        hidden_size=tf_config.embedding_dim,
        intermediate_size=tf_config.intermediate_dim,
        num_hidden_layers=tf_config.num_blocks,
        num_attention_heads=tf_config.num_heads,
        num_key_value_heads=tf_config.num_groups,
        vocab_size=len(tokenizer),
        max_position_embeddings=tf_config.max_sequence_length,
        rope_theta=tf_config.theta or 10000.0,
        attention_dropout=tf_config.dropout_prob,
        mlp_dropout=tf_config.dropout_prob,
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )

    model = LlamaCompositeKV(llama_cfg)

    model_repo_path = snapshot_download(
        repo_id=hf_repo,
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt"],
        token=os.environ.get("HF_TOKEN"),
    )
    model_repo_path = _ensure_safetensors_dir(model_repo_path)
    load_from_safetensors(model, model_repo_path, llama_cfg)
    gc.collect()

    def build_dict():
        return model.weights_ref_hf_dict()

    return model, build_dict


def _bench_iter(build_dict: Callable[[], dict], mesh_device: Any):
    """One benchmark iteration. Returns `(records, (total_bytes, total_d2h, total_h2d))`.

    Timed regions per tensor:
      * `t_d2h`: `ttnn.to_torch(dict[k])`.
      * `t_h2d`: `ttnn.from_torch(host, bf16/TILE/DRAM/replicate)` into `uploaded[k]`.

    The initial `build_dict()` call is intentionally NOT timed (per user
    request); it runs before the timed region below.
    """
    # Build the source dict OUTSIDE the timed region.
    hf_dict = build_dict()

    records = []  # (key, nbytes, t_d2h, t_h2d)
    total_bytes = 0
    total_d2h = 0.0
    total_h2d = 0.0

    # Output dict of freshly uploaded on-device tensors (NOT the model's
    # parameter store). Values become stale after the immediate deallocate
    # below (DRAM headroom); kept for record-keeping / verifying the schema.
    uploaded: dict = {}

    replicate_mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)

    for k in sorted(hf_dict.keys()):
        t = hf_dict[k]

        # ---- D->H ---------------------------------------------------------
        t0 = time.perf_counter()
        host = ttnn.to_torch(t)
        t_d2h = time.perf_counter() - t0

        nbytes = int(host.numel()) * int(host.element_size())

        # ---- H->D ---------------------------------------------------------
        t0 = time.perf_counter()
        on_device = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_mapper,
        )
        t_h2d = time.perf_counter() - t0

        # Record the (now-alive) handle into the output dict, then release the
        # underlying DRAM buffer so the next iteration doesn't stack up. See
        # the module docstring for why (Qwen3-4B round-trip would OOM the
        # P150's 12 GiB otherwise).
        uploaded[k] = on_device
        ttnn.deallocate(on_device, force=True)

        records.append((k, nbytes, t_d2h, t_h2d))
        total_bytes += nbytes
        total_d2h += t_d2h
        total_h2d += t_h2d

        del host

    del uploaded
    del hf_dict
    gc.collect()

    return records, (total_bytes, total_d2h, total_h2d)


def _fmt_mib(nbytes: int) -> str:
    return f"{nbytes / (1024 * 1024):.1f} MiB"


def _fmt_mibps(nbytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "inf"
    return f"{(nbytes / (1024 * 1024)) / seconds:.1f} MiB/s"


def _run_model(model_meta: Dict[str, Any], args: argparse.Namespace) -> None:
    label = model_meta["label"]
    hf_repo = model_meta["hf_repo"]

    print("", flush=True)
    print("=" * 80, flush=True)
    print(f"[bench] {label}: {hf_repo}", flush=True)
    print("=" * 80, flush=True)

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device([1, 1], [0])
    mesh_device = autograd_ctx.get_device()

    model: Any = None
    build_dict: Any = None
    try:
        t0 = time.perf_counter()
        if model_meta["kind"] == "qwen3":
            model, build_dict = _build_qwen3(hf_repo, int(model_meta["max_seq_len"]))
        elif model_meta["kind"] == "llama":
            model, build_dict = _build_llama(hf_repo, model_meta["yaml"], int(model_meta["max_seq_len"]))
        else:
            raise ValueError(f"unknown model kind: {model_meta['kind']!r}")
        print(
            f"[bench] {label}: model + HF weights loaded in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

        # Warm-up iters (not aggregated).
        for i in range(args.warmup_iters):
            _, (nb, td, th) = _bench_iter(build_dict, mesh_device)
            print(
                f"[bench] {label} warmup {i}: "
                f"d2h={td:.2f}s ({_fmt_mibps(nb, td)})  "
                f"h2d={th:.2f}s ({_fmt_mibps(nb, th)})  total={_fmt_mib(nb)}",
                flush=True,
            )

        # Steady-state iters (aggregated).
        per_d2h: List[float] = []
        per_h2d: List[float] = []
        first_records = None
        first_nb = 0
        for i in range(args.iters):
            records, (nb, td, th) = _bench_iter(build_dict, mesh_device)
            per_d2h.append(td)
            per_h2d.append(th)
            if i == 0:
                first_records = records
                first_nb = nb
            print(
                f"[bench] {label} iter {i}: "
                f"d2h={td:.2f}s ({_fmt_mibps(nb, td)})  "
                f"h2d={th:.2f}s ({_fmt_mibps(nb, th)})  total={_fmt_mib(nb)}",
                flush=True,
            )

        # Per-tensor top-N slowest table for iter 0 (sorted by d2h time).
        if first_records:
            print("", flush=True)
            print(
                f"[bench] {label} top-{args.top_slowest} slowest tensors "
                f"(iter 0, {len(first_records)} tensors, {_fmt_mib(first_nb)} total, "
                f"sorted by d2h):",
                flush=True,
            )
            print(
                f"  {'key':70s}  {'MiB':>8s}  {'d2h (ms)':>10s}  {'d2h (MiB/s)':>13s}  "
                f"{'h2d (ms)':>10s}  {'h2d (MiB/s)':>13s}",
                flush=True,
            )
            for k, nb2, t2, th2 in sorted(first_records, key=lambda r: r[2], reverse=True)[: args.top_slowest]:
                print(
                    f"  {k[:70]:70s}  {nb2 / (1024 * 1024):8.2f}  "
                    f"{t2 * 1000:10.1f}  {_fmt_mibps(nb2, t2):>13s}  "
                    f"{th2 * 1000:10.1f}  {_fmt_mibps(nb2, th2):>13s}",
                    flush=True,
                )

        # Aggregate summary.
        print("", flush=True)
        print(f"[bench] {label} steady-state summary:", flush=True)
        print(f"  iters:              {args.iters}", flush=True)
        print(f"  tensors per iter:   {len(first_records) if first_records else 0}", flush=True)
        print(f"  bytes per iter:     {_fmt_mib(first_nb)}", flush=True)
        if per_d2h:
            print(
                f"  D->H (mean):        {mean(per_d2h):.2f} s ({_fmt_mibps(first_nb, mean(per_d2h))})",
                flush=True,
            )
        if per_h2d:
            print(
                f"  H->D (mean):        {mean(per_h2d):.2f} s ({_fmt_mibps(first_nb, mean(per_h2d))})",
                flush=True,
            )
    finally:
        # Drop references so ttml destructors free the on-device parameter store
        # before the mesh closes.
        model = None
        build_dict = None
        gc.collect()
        try:
            autograd_ctx.close_device()
        except Exception as e:
            print(f"[bench] {label}: close_device raised {type(e).__name__}: {e}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iters", type=int, default=3, help="Number of steady-state iters per model")
    p.add_argument("--warmup-iters", type=int, default=1, help="Number of warm-up iters (not aggregated)")
    p.add_argument("--top-slowest", type=int, default=10, help="Show top-N slowest tensors for iter 0")
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated subset of labels to run. "
            f"Available: {','.join(m['label'] for m in _MODELS)}. "
            "Default: all four in the order above."
        ),
    )
    args = p.parse_args()

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    selected: List[Dict[str, Any]] = _MODELS
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        selected = [m for m in _MODELS if m["label"] in wanted]
        if not selected:
            raise SystemExit(
                f"[bench] no models matched --models={args.models}; " f"available: {[m['label'] for m in _MODELS]}"
            )

    print(f"[bench] running {len(selected)} models: {[m['label'] for m in selected]}", flush=True)

    for m in selected:
        try:
            _run_model(m, args)
        except Exception as e:
            print(
                f"[bench] {m['label']}: FAILED with {type(e).__name__}: {e}",
                flush=True,
            )
            import traceback

            traceback.print_exc()
            # Try to close the device so the next model can open a fresh one.
            try:
                ttml.autograd.AutoContext.get_instance().close_device()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
