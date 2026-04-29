#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop eval of X-VLA against a real LeRobot dataset.

Pulls N samples from a HuggingFace LeRobotDataset, runs the X-VLA
policy (across requested backends), and reports the mean squared /
absolute error of the predicted 30-step action chunk against the GT
actions in the dataset.

Why this — `run_benchmark.py` only exercises synthetic tensors and is
therefore blind to real distributional errors. This script surfaces
them: same pipeline (preprocessor, policy, postprocessor), real images
and proprio, GT actions.

X-VLA's schema expects three image keys named
`observation.images.image{,2,3}` plus `observation.state` of dim 8.
Datasets with different key conventions can be adapted via
`--rename-images` (e.g. `top=image,wrist=image2`).

[FUTURE] This script does OPEN-LOOP eval — one chunk per GT sample,
no closed-loop rollouts. Task-success rates from a simulator (LIBERO /
ALOHA sim / robosuite) are tracked separately; see eval/README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_XV = _HERE.parent
_BENCH = _XV / "benchmark"


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_ttnn_env = _load_file_module("xvla_ttnn_env", _XV / "tt" / "ttnn_env.py")
_ttnn_env.install()
_bootstrap = _load_file_module("xvla_lerobot_bootstrap", _BENCH / "lerobot_bootstrap.py")
_bootstrap.install()


# ------------------------------------------------------------------
# Loaders (reuse the ones from the relative-PCC script)
# ------------------------------------------------------------------

def load_policy(backend: str, weights: Path, steps: int, tokenizer_max_length: int = 32):
    """Load policy. Also caps tokenizer_max_length so the processor pipeline
    doesn't produce a merged seq that blows past SoftPromptedTransformer's
    max_len_seq=512."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    torch.set_grad_enabled(False)
    if backend == "torch_cpu":
        config = PreTrainedConfig.from_pretrained(str(weights))
        config.dtype = "float32"
        config.num_denoising_steps = steps
        config.tokenizer_max_length = tokenizer_max_length
        policy = XVLAPolicy.from_pretrained(str(weights), config=config)
        policy.eval()
        return policy
    if backend == "ttnn":
        tt_policy = _load_file_module("xvla_tt_policy", _XV / "tt" / "policy.py")
        policy = tt_policy.load_policy_ttnn(weights)
        policy.config.num_denoising_steps = steps
        policy.config.tokenizer_max_length = tokenizer_max_length
        return policy
    raise ValueError(backend)


# ------------------------------------------------------------------
# Dataset adaptation: rename image keys to the xvla schema.
# ------------------------------------------------------------------

def _parse_rename(spec: str) -> list[tuple[str, str]]:
    """Parse 'src=dst,src2=dst2'. Entries may be either short suffix form
    (e.g. 'top=image' -> `observation.images.top` -> `observation.images.image`)
    or fully-qualified (dots present) form
    ('observation.image=observation.images.image').

    Returns a list, not a dict, so one source can be duplicated to multiple
    destinations (e.g. 'observation.image=observation.images.image,observation.image=observation.images.image2').
    """
    out: list[tuple[str, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        src, dst = chunk.split("=", 1)
        out.append((src.strip(), dst.strip()))
    return out


def _adapt_sample(sample: dict, renames: list[tuple[str, str]], state_key: str, language_key: str | None):
    """Rewrite a lerobot sample into a batch dict the X-VLA policy understands.

    Rename rules applied IN ORDER:
      - Short form 'top=image' renames `observation.images.top` -> `observation.images.image`.
      - Full form 'observation.image=observation.images.image' does a key-for-key rename.
      - Any rename is COPY-and-keep semantics when both src and dst are fully
        qualified with different names — the source value is placed under the
        dst key, original kept intact. Enables duplicating one camera into
        multiple X-VLA view slots.
    """
    batch = dict(sample)  # start from full sample

    for src, dst in renames:
        is_full_src = "." in src
        is_full_dst = "." in dst
        src_key = src if is_full_src else "observation.images." + src
        dst_key = dst if is_full_dst else "observation.images." + dst
        if src_key in batch:
            batch[dst_key] = batch[src_key]
            if src_key != dst_key and not is_full_src:
                # Short form was a rename; drop the original.
                batch.pop(src_key, None)

    if state_key != "observation.state" and state_key in batch:
        batch["observation.state"] = batch[state_key]

    return batch


def _ensure_batch_dim(t):
    return t.unsqueeze(0) if torch.is_tensor(t) and t.dim() <= 3 else t


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def summarize(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: [N, chunk, action_dim]"""
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    # Per-step MAE along the chunk (useful to see if later steps degrade).
    mae_per_step = np.mean(np.abs(diff), axis=(0, 2))  # [chunk]
    # Per-action-dim MAE normalized by GT std.
    gt_std = gt.std(axis=(0, 1)) + 1e-8
    per_dim_rel = np.mean(np.abs(diff), axis=(0, 1)) / gt_std
    return {
        "mse": mse,
        "mae": mae,
        "mae_first_step": float(mae_per_step[0]),
        "mae_last_step": float(mae_per_step[-1]),
        "per_dim_rel_mean": float(per_dim_rel.mean()),
        "per_dim_rel_max": float(per_dim_rel.max()),
    }


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   help="HF repo id of a LeRobotDataset, e.g. lerobot/libero_10_image")
    p.add_argument("--weights", type=Path,
                   default=_XV / "weights" / "xvla_base")
    p.add_argument("--backends", type=str, default="torch_cpu,ttnn")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--rename-images", type=str, default="",
                   help="Comma list of src=dst pairs to remap image suffixes.")
    p.add_argument("--state-key", type=str, default="observation.state",
                   help="Source key to map to observation.state.")
    p.add_argument("--language-key", type=str, default=None,
                   help="Key carrying the language instruction (optional).")
    p.add_argument("--skip-postprocess", action="store_true",
                   help="Skip the un-normalizer on the predicted actions. Required "
                        "when the dataset's action_dim doesn't match X-VLA's "
                        "max_action_dim (e.g. pusht has 2-D, xvla pads to 20-D).")
    args = p.parse_args()

    rename_images = _parse_rename(args.rename_images)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"Dataset: {args.dataset}  samples: {args.num_samples}  steps: {args.steps}")
    print(f"Backends: {backends}")
    if rename_images:
        print(f"Image rename: {rename_images}")
    print()

    # Load dataset.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    # Request the full action chunk we care about per sample.
    delta = {"action": list(range(args.steps * 3))}  # any finite horizon; sliced to chunk
    ds = LeRobotDataset(args.dataset, delta_timestamps=delta)
    if args.num_samples > len(ds):
        print(f"WARN: dataset has only {len(ds)} samples; reducing to {len(ds)}")
        args.num_samples = len(ds)
    print(f"Dataset loaded: {len(ds)} total samples, first sample keys:")
    example = ds[args.start_index]
    for k in example:
        v = example[k]
        shp = getattr(v, "shape", None)
        print(f"  {k:<36s} {type(v).__name__}{' '+str(tuple(shp)) if shp is not None else ''}")
    print()

    per_backend: dict[str, dict] = {}

    for backend in backends:
        print(f"--- Backend: {backend} ---")
        policy = load_policy(backend, args.weights, args.steps)
        pre, post = make_xvla_pre_post_processors(policy.config, dataset_stats=ds.meta.stats)

        preds, gts = [], []
        for idx in range(args.start_index, args.start_index + args.num_samples):
            sample = ds[idx]
            adapted = _adapt_sample(sample, rename_images, args.state_key, args.language_key)
            try:
                batch = pre(adapted)
            except Exception as e:
                if idx == args.start_index:
                    print(f"Preprocessor failed on sample {idx}: {e}")
                    print("Hint: pass --rename-images to map your image keys to image/image2/image3.")
                    return 1
                else:
                    raise
            with torch.no_grad():
                pred_chunk = policy.predict_action_chunk(batch)
            if args.skip_postprocess:
                pred_chunk = pred_chunk.detach().cpu().float().numpy()
            else:
                try:
                    pred_chunk = post(pred_chunk).detach().cpu().float().numpy()
                except RuntimeError as e:
                    if "must match the size" in str(e) and idx == args.start_index:
                        print("Post-processor action_dim mismatch. "
                              "Re-run with --skip-postprocess for meaningful results.")
                    raise
            gt_action = sample["action"]
            if torch.is_tensor(gt_action):
                gt_action = gt_action.detach().cpu().float().numpy()
            # Align GT to the chunk the model predicted.
            chunk = pred_chunk.shape[1]
            if gt_action.ndim == 2:           # [chunk, A]
                gt_action = gt_action[:chunk]
            elif gt_action.ndim == 3:         # [1, chunk, A]
                gt_action = gt_action[0, :chunk]
            preds.append(pred_chunk[0])
            gts.append(gt_action)
            policy.reset()  # clear the internal action-chunk queue between samples

        preds = np.stack(preds)
        gts = np.stack(gts).astype(np.float32)
        if preds.shape[-1] != gts.shape[-1]:
            # X-VLA pads action_dim; truncate to the dataset's native dim.
            preds = preds[..., : gts.shape[-1]]
        per_backend[backend] = summarize(preds, gts)

    # Report.
    print()
    hdr = f"{'backend':<12s} {'mse':>10s} {'mae':>10s} {'mae[1st]':>10s} {'mae[last]':>10s} {'rel_mean':>10s} {'rel_max':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for b in backends:
        r = per_backend[b]
        print(f"{b:<12s} "
              f"{r['mse']:>10.4e} {r['mae']:>10.4e} "
              f"{r['mae_first_step']:>10.4e} {r['mae_last_step']:>10.4e} "
              f"{r['per_dim_rel_mean']:>10.4f} {r['per_dim_rel_max']:>10.4f}")

    if "torch_cpu" in per_backend and "ttnn" in per_backend:
        delta_mae = per_backend["ttnn"]["mae"] - per_backend["torch_cpu"]["mae"]
        ref_mae = per_backend["torch_cpu"]["mae"]
        print()
        print(f"Implementation overhead: MAE(ttnn) - MAE(torch_cpu_fp32) = {delta_mae:+.4e} "
              f"({100*delta_mae/max(ref_mae,1e-12):+.2f}% of reference error)")

    print()
    print("rel_mean/rel_max are per-action-dim |err| normalized by GT std.")
    print("delta MAE small (<~5%) => port is faithful. Large absolute MAE on a base")
    print("(non-fine-tuned) xvla checkpoint just reflects that the model hasn't seen")
    print("this dataset's task distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
