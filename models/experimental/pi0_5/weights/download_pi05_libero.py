# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Download + prepare the upstream openpi **pi05_libero** checkpoint in the
torch (safetensors) layout this package expects, then verify it loads.

What "the checkpoint that works" means here — a directory with exactly:

    <out>/model.safetensors                                   ~7.2 GB bf16 weights
    <out>/config.json                                         {action_dim, action_horizon=10,
                                                               paligemma_variant, action_expert_variant, precision}
    <out>/assets/physical-intelligence/libero/norm_stats.json state/action mean/std/q01/q99

openpi distributes pi05_libero canonically as a **JAX / Orbax** checkpoint
(`gs://openpi-assets/checkpoints/pi05_libero/`, no config.json). The torch form is
the **safetensors mirror on HuggingFace**, produced by openpi/lerobot's JAX→PyTorch
conversion (which also authors config.json). This script fetches that torch mirror
(the "convert to torch" step is what produced it), fills in config.json / norm_stats
if the repo omits them, and verifies the result with this package's own loader.

Usage:
    python_env/bin/python models/experimental/pi0_5/weights/download_pi05_libero.py \
        --out /home/tt-admin/pi05_cache/pi05_libero_upstream
    # --repo-id defaults to the documented upstream repo; override if you host a mirror.
    # HF auth: the upstream repo is gated — run `huggingface-cli login` first (or set
    # HF_TOKEN). Then point PI05_CHECKPOINT_DIR at <out>.

If only the JAX/Orbax checkpoint is available (no torch mirror you can pull), convert
it with openpi's exporter, e.g.:
    git clone https://github.com/Physical-Intelligence/openpi && cd openpi
    uv run python scripts/convert_jax_to_pytorch.py --checkpoint gs://openpi-assets/checkpoints/pi05_libero --out <out>
then re-run this script with --skip-download to just add config.json/norm_stats + verify.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The 5-key header this package reads via common/checkpoint_meta.action_horizon_from_checkpoint.
# action_horizon=10 is the upstream pi05_libero training value (NOT the bare-config default of 50).
_CONFIG_JSON = {
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


def _download(repo_id: str, out: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"[download] snapshot_download({repo_id!r}) → {out}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out),
        allow_patterns=["model.safetensors", "config.json", "assets/**", "*.json"],
    )


def _ensure_config(out: Path) -> None:
    cfg = out / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            if "action_horizon" in data:
                print(f"[config] {cfg} present (action_horizon={data.get('action_horizon')})", flush=True)
                return
        except json.JSONDecodeError:
            pass
    cfg.write_text(json.dumps(_CONFIG_JSON, indent=2))
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


def _verify(out: Path) -> bool:
    """Load with this package's own loader — the definitive 'it works' check."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader

    if not (out / "model.safetensors").exists():
        print(f"[verify] FAIL: {out}/model.safetensors missing — is this repo the torch mirror?", flush=True)
        return False
    ah = action_horizon_from_checkpoint(out)
    if ah != 10:
        print(f"[verify] FAIL: action_horizon={ah} (expected 10) — config.json wrong", flush=True)
        return False
    n = len(Pi0_5WeightLoader(str(out)).categorized_weights)
    print(f"[verify] OK: loader categorized {n} weight groups; action_horizon={ah}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + prepare upstream pi05_libero (torch/safetensors).")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--repo-id", default="openpi/pi05_libero", help="HF repo with the torch/safetensors mirror")
    ap.add_argument("--skip-download", action="store_true", help="only add config.json/norm_stats + verify")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        _download(args.repo_id, out)
    _ensure_config(out)
    _ensure_norm_stats(out)
    ok = _verify(out)
    if ok:
        print(f"\n✅ pi05_libero ready at {out}\n   export PI05_CHECKPOINT_DIR={out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
