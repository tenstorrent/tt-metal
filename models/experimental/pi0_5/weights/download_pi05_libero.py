# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Download + prepare a **pi05_libero** checkpoint in the torch (safetensors) layout
this package expects, then verify it loads.

Two variants (pick with `--variant`):

  finetuned  (DEFAULT)  lerobot/pi05_libero_finetuned  — PUBLIC, no HF login.
                        action_horizon=50, MEAN_STD normalization shipped as
                        `policy_preprocessor_step_2_normalizer_processor.safetensors`
                        (state-in-prompt). This is the checkpoint validated on the
                        single-chip path (38/40 = 95% LIBERO). Self-contained: ships
                        its own config.json (chunk_size=50) — no fixup needed.

  upstream              openpi/pi05_libero — GATED (run `huggingface-cli login` first).
                        action_horizon=10, QUANTILE normalization. Distributed
                        canonically as a JAX/Orbax checkpoint; this fetches the torch
                        safetensors mirror and fills in config.json / norm_stats
                        (the `assets/physical-intelligence/libero/norm_stats.json`
                        fetched from the public GCS bucket) if the repo omits them.

Usage:
    # default (finetuned, public):
    python_env/bin/python models/experimental/pi0_5/weights/download_pi05_libero.py \
        --out models/experimental/pi0_5/weights/pi05_libero_finetuned

    # upstream (gated):
    huggingface-cli login
    python_env/bin/python models/experimental/pi0_5/weights/download_pi05_libero.py \
        --variant upstream --out /path/to/pi05_libero_upstream

Then point PI05_CHECKPOINT_DIR at <out> (or pass it as libero_rollout --checkpoint).

If only the upstream JAX/Orbax checkpoint is available (no torch mirror you can pull),
convert it with openpi's exporter, e.g.:
    git clone https://github.com/Physical-Intelligence/openpi && cd openpi
    uv run python scripts/convert_jax_to_pytorch.py --checkpoint gs://openpi-assets/checkpoints/pi05_libero --out <out>
then re-run this script with --variant upstream --skip-download to add config.json/norm_stats + verify.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# openpi upstream config header (this package reads it via
# common/checkpoint_meta.action_horizon_from_checkpoint). action_horizon=10 is the
# upstream training value. The finetuned variant ships its own config.json
# (chunk_size=50), so this is only written for --variant upstream when missing.
_UPSTREAM_CONFIG_JSON = {
    "action_dim": 32,
    "action_horizon": 10,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "precision": "bfloat16",
}
_NORM_STATS_REL = "assets/physical-intelligence/libero/norm_stats.json"
_NORM_STATS_GCS = (
    "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/"
    "assets/physical-intelligence/libero/norm_stats.json"
)

# Per-variant knobs. finetuned is self-contained (ships config.json + a MEAN_STD
# safetensors normalizer), so it needs no config write and no GCS norm_stats fetch —
# injecting the openpi QUANTILE norm_stats.json would make the rollout adapter prefer
# the wrong normalization.
VARIANTS = {
    "finetuned": {
        "repo_id": "lerobot/pi05_libero_finetuned",
        "action_horizon": 50,
        "allow_patterns": ["model.safetensors", "*.json", "*.safetensors"],
        "ensure_config": False,
        "fetch_norm_stats": False,
    },
    "upstream": {
        "repo_id": "openpi/pi05_libero",
        "action_horizon": 10,
        "allow_patterns": ["model.safetensors", "config.json", "assets/**", "*.json"],
        "ensure_config": True,
        "fetch_norm_stats": True,
    },
}


def _download(repo_id: str, out: Path, allow_patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    print(f"[download] snapshot_download({repo_id!r}) → {out}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(out), allow_patterns=allow_patterns)


def _ensure_config(out: Path) -> None:
    cfg = out / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            if "action_horizon" in data or "chunk_size" in data:
                print(f"[config] {cfg} present (horizon key found)", flush=True)
                return
        except json.JSONDecodeError:
            pass
    cfg.write_text(json.dumps(_UPSTREAM_CONFIG_JSON, indent=2))
    print(f"[config] wrote {cfg} (action_horizon=10)", flush=True)


def _ensure_norm_stats(out: Path) -> None:
    dst = out / _NORM_STATS_REL
    if dst.exists():
        print(f"[norm_stats] {dst} present", flush=True)
        return
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[norm_stats] fetching {_NORM_STATS_GCS}", flush=True)
    urllib.request.urlretrieve(_NORM_STATS_GCS, str(dst))
    print(f"[norm_stats] wrote {dst}", flush=True)


def _verify(out: Path, expected_horizon: int) -> bool:
    """Load with this package's own loader — the definitive 'it works' check."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader

    if not (out / "model.safetensors").exists():
        print(f"[verify] FAIL: {out}/model.safetensors missing — is this repo the torch mirror?", flush=True)
        return False
    ah = action_horizon_from_checkpoint(out)
    if ah != expected_horizon:
        print(f"[verify] FAIL: action_horizon={ah} (expected {expected_horizon}) — config.json wrong", flush=True)
        return False
    n = len(Pi0_5WeightLoader(str(out)).categorized_weights)
    print(f"[verify] OK: loader categorized {n} weight groups; action_horizon={ah}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + prepare a pi05_libero checkpoint (torch/safetensors).")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="finetuned",
        help="finetuned (default, public lerobot pi05_libero_finetuned) or upstream (gated openpi pi05_libero).",
    )
    ap.add_argument("--repo-id", default=None, help="override the HF repo id for the chosen variant")
    ap.add_argument("--skip-download", action="store_true", help="only add config.json/norm_stats + verify")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    repo_id = args.repo_id or spec["repo_id"]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"[variant] {args.variant} (repo={repo_id}, action_horizon={spec['action_horizon']})", flush=True)
    if not args.skip_download:
        _download(repo_id, out, spec["allow_patterns"])
    if spec["ensure_config"]:
        _ensure_config(out)
    if spec["fetch_norm_stats"]:
        _ensure_norm_stats(out)
    ok = _verify(out, spec["action_horizon"])
    if ok:
        print(f"\n✅ pi05_libero ({args.variant}) ready at {out}\n   export PI05_CHECKPOINT_DIR={out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
