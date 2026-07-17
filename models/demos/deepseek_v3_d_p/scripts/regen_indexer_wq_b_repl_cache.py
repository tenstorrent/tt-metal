# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regenerate ONLY the missing indexer ``wq_b_repl`` tensorbin(s) for a GLM-5.1 ttnn prefill cache.

Why this exists
---------------
PR #49496 (SP×TP seq-sharded DSA indexer) changed the layout of exactly ONE indexer weight:
``indexer.wq_b`` went from TP-column-sharded to **TP-replicated + transposed**, and its cache stem
was deliberately renamed ``wq_b`` -> ``wq_b_repl`` (see TtIndexer._cache_short_name) so a stale
col-sharded tensorbin can never alias the new replicated layout. A ttnn cache built before that PR
therefore has every indexer tensorbin EXCEPT ``wq_b_repl`` — the loader now asks for a file that
isn't there. Every OTHER indexer weight (wk / weights_proj / k_norm / k_norm_bias) is byte-identical,
so only ``wq_b_repl`` needs regenerating.

What this does
--------------
For each GLM-5.1 layer that carries an indexer, load ``indexer.wq_b`` from the HF checkpoint, build
the replicated tensorbin via the model's OWN cache builder (TtIndexer.build_ttnn_cache, device=None)
into a scratch dir, and copy ONLY the ``layer_<i>.mla.indexer_wq_b_repl_*.tensorbin`` file(s) into the
destination cache's ``<glm_5_1_bh_{N}dev>/<sp>x<tp>`` subdir. Nothing else in the destination cache is
touched.

Environment
-----------
  GLM51_HF_MODEL               (required)  GLM-5.1 HF checkpoint dir (safetensors + model.safetensors.index.json).
  GLM_51_LOCAL_PATH            (required)  Writable scratch root; tensorbins are generated here first.
  TT_GLM51_PREFILL_TTNN_CACHE  (required)  Destination cache root; wq_b_repl files are copied into its subdir.
  MESH_ROWS / MESH_COLS        (optional)  Mesh shape (default 8 x 4 -> sp=8, tp=4). Must match the target
                                           cache's <sp>x<tp> subdir, and needs rows*cols devices present so
                                           get_num_devices() yields the same <N>dev token as the target cache.

The scratch and destination subdirs are printed on startup — verify the destination subdir matches the
existing cache (e.g. .../GLM-5_1-Cache/glm_5_1_bh_32dev/8x4) before trusting the copy.

Usage
-----
  GLM51_HF_MODEL=/mnt/models/deepseek-prefill-cache/GLM-5.1-FP8 \
  GLM_51_LOCAL_PATH=/tmp/glm51_cache_scratch \
  TT_GLM51_PREFILL_TTNN_CACHE=/mnt/models/deepseek-prefill-cache/GLM-5_1-Cache \
  python models/demos/deepseek_v3_d_p/scripts/regen_indexer_wq_b_repl_cache.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import load_attention_state_dict
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config, glm_hf_config
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer

# TtIndexer's cache stem for the replicated wq_b (the only weight PR #49496 changed).
WQ_B_REPL_SHORT = "wq_b_repl"
# Adapter identity used to build the cache subdir (mirrors MLAPrefillAdapter.weight_cache_path).
MODEL_NAME = "glm_5_1"

# load_attention_state_dict returns HF-canonical keys; map them to TtIndexer's idx_host keys
# (WEIGHT_NAMES, no ".weight" suffix). The checkpoint's LayerNorm bias (indexer.k_norm.bias) maps to
# the device's separate indexer.k_norm_bias slot.
_STATE_TO_IDX_HOST = {
    "indexer.wq_b.weight": "indexer.wq_b",
    "indexer.wk.weight": "indexer.wk",
    "indexer.k_norm.weight": "indexer.k_norm",
    "indexer.k_norm.bias": "indexer.k_norm_bias",
    "indexer.weights_proj.weight": "indexer.weights_proj",
}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"ERROR: environment variable {name} must be set")
    return value


def _layer_shard_paths(model_dir: str, layer: int) -> list:
    """Local safetensors shard file(s) holding layer `layer`'s self_attn tensors."""
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.exists():
        sys.exit(f"ERROR: {index} not found — GLM51_HF_MODEL must be a local HF checkpoint dir")
    weight_map = json.load(open(index))["weight_map"]
    prefix = f"model.layers.{layer}.self_attn."
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    return [str(Path(model_dir) / s) for s in shards]


def _idx_host_for_layer(model_dir: str, layer: int):
    """Return the 5-key idx_host dict for `layer`, or None if the layer has no indexer weights."""
    shard_paths = _layer_shard_paths(model_dir, layer)
    if not shard_paths:
        return None
    # load_attention_state_dict reads + dequantizes (fp8 -> bf16) the whole self_attn block; we keep
    # only the indexer tensors. Reuse it (rather than a hand-rolled reader) so the fp8 dequant is the
    # exact tested path.
    state = load_attention_state_dict(shard_paths, layer)
    if "indexer.wq_b.weight" not in state:
        return None
    idx_host = {}
    for state_key, host_key in _STATE_TO_IDX_HOST.items():
        if state_key not in state:
            logger.warning(f"layer {layer}: checkpoint missing {state_key}; skipping layer")
            return None
        idx_host[host_key] = state[state_key]
    return idx_host


def main() -> None:
    hf_model = _require_env("GLM51_HF_MODEL")
    scratch_root = Path(_require_env("GLM_51_LOCAL_PATH"))
    dest_root = Path(_require_env("TT_GLM51_PREFILL_TTNN_CACHE"))

    rows = int(os.environ.get("MESH_ROWS", "8"))
    cols = int(os.environ.get("MESH_COLS", "4"))
    sp_axis, tp_axis = 0, 1  # sp = mesh dim 0 (rows), tp = mesh dim 1 (cols)

    logger.info(f"Opening {rows}x{cols} mesh (needs {rows * cols} devices) ...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    try:
        arch = "bh" if is_blackhole() else "wh"
        num_devices = ttnn.get_num_devices()
        # Mirror MLAPrefillAdapter.weight_cache_path: {name}_{arch}_{N}dev / {sp}x{tp}
        subdir = Path(f"{MODEL_NAME}_{arch}_{num_devices}dev") / f"{rows}x{cols}"
        scratch_dir = scratch_root / subdir
        dest_dir = dest_root / subdir
        scratch_dir.mkdir(parents=True, exist_ok=True)
        dest_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"scratch (generate):  {scratch_dir}")
        logger.info(f"dest    (copy into): {dest_dir}")
        logger.info("Verify the dest subdir matches your existing cache before trusting the copy.")

        config = glm_hf_config()
        n_layers = GLM51Config.NUM_LAYERS

        copied = 0
        processed_layers, skipped_layers = [], []
        for layer in range(n_layers):
            idx_host = _idx_host_for_layer(hf_model, layer)
            if idx_host is None:
                logger.info(f"layer {layer}: no indexer weights; skipped")
                skipped_layers.append(layer)
                continue

            # Clear any stale scratch wq_b_repl so as_tensor rebuilds (a cache hit would skip the write).
            for stale in scratch_dir.glob(f"layer_{layer}.mla.indexer_{WQ_B_REPL_SHORT}*.tensorbin"):
                stale.unlink()

            # device=None -> writes tensorbins to disk (no device copy). Builds all 5 indexer tensors;
            # we copy only wq_b_repl. sp/tp axes match the {sp}x{tp} subdir.
            TtIndexer.build_ttnn_cache(
                idx_host, scratch_dir, mesh_device, config, layer, sp_axis=sp_axis, tp_axis=tp_axis
            )

            produced = list(scratch_dir.glob(f"layer_{layer}.mla.indexer_{WQ_B_REPL_SHORT}*.tensorbin"))
            if not produced:
                sys.exit(f"ERROR: layer {layer}: build_ttnn_cache wrote no {WQ_B_REPL_SHORT} tensorbin")
            for f in produced:
                shutil.copy2(f, dest_dir / f.name)
                copied += 1
            processed_layers.append(layer)
            logger.info(f"layer {layer}: copied {[f.name for f in produced]}")

        logger.info(
            f"Done. Copied {copied} {WQ_B_REPL_SHORT} tensorbin(s) into {dest_dir} "
            f"({len(skipped_layers)} layer(s) skipped without indexer weights)."
        )

        _verify(dest_dir, processed_layers, skipped_layers)
    finally:
        ttnn.close_mesh_device(mesh_device)


def _verify(dest_dir: Path, processed_layers: list, skipped_layers: list) -> None:
    """Post-run check: every processed layer must now have a wq_b_repl tensorbin in the destination.
    Exits non-zero if any is missing so a partial/failed copy can't pass silently."""
    logger.info(f"Verifying {WQ_B_REPL_SHORT} tensorbins in {dest_dir} ...")
    missing = [
        layer
        for layer in processed_layers
        if not any(dest_dir.glob(f"layer_{layer}.mla.indexer_{WQ_B_REPL_SHORT}*.tensorbin"))
    ]
    if skipped_layers:
        logger.warning(
            f"{len(skipped_layers)} layer(s) had no indexer weights and were skipped (no {WQ_B_REPL_SHORT} "
            f"expected for them): {skipped_layers}"
        )
    if missing:
        sys.exit(
            f"ERROR: verification FAILED — {len(missing)} processed layer(s) have no {WQ_B_REPL_SHORT} "
            f"tensorbin in {dest_dir}: {missing}"
        )
    logger.info(
        f"Verification PASSED: all {len(processed_layers)} processed layer(s) have a {WQ_B_REPL_SHORT} "
        f"tensorbin in {dest_dir}."
    )


if __name__ == "__main__":
    main()
