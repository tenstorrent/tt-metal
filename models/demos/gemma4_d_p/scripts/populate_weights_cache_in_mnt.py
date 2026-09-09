# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Populate shared Hugging Face and TT weight caches for Gemma4-31B-it."""

import argparse
import fcntl
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=("8x4", "4x8"), default="8x4")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate an already complete TT cache")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")
    os.environ.setdefault("HF_MODEL", "google/gemma-4-31B-it")
    os.environ.setdefault("TT_CACHE_PATH", "/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it")
    model_id = os.environ["HF_MODEL"]
    if model_id != "google/gemma-4-31B-it":
        parser.error("HF_MODEL must be google/gemma-4-31B-it")

    # Set HF environment defaults before importing libraries that read them at import time.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from loguru import logger

    import ttnn
    from conftest import reset_fabric, set_fabric
    from models.common.weight_cache import mark_weight_cache_complete, weight_cache_is_complete
    from models.demos.gemma4_d_p.tt.common import _gemma4_is_host_weight, create_tt_model, weight_cache_identity
    from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
    from models.demos.gemma4_d_p.tt.precision import Gemma4Precision

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
            local_files_only=HF_HUB_OFFLINE,
        )
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", *index["weight_map"].values()}
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        raise RuntimeError(f"Incomplete HF checkpoint at {snapshot}: {missing}. Disable HF_HUB_OFFLINE to download.")
    logger.info("HF checkpoint: {}", snapshot)
    shape = tuple(int(n) for n in args.mesh.split("x"))
    config = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(str(snapshot)))
    precision = Gemma4Precision.load(model_id, shape)
    identity = weight_cache_identity(model_id, config.num_hidden_layers, shape, precision)
    root = Path(os.environ["TT_CACHE_PATH"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"tensor_cache_bf16_mesh{args.mesh}"
    with (root / f".populate_mesh{args.mesh}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not args.rebuild and weight_cache_is_complete(cache, **identity, force_env=None):
            logger.info("TT cache is complete: {}", cache)
            return
        if ttnn.get_arch_name() != "blackhole" or ttnn.get_num_devices() != 32:
            raise RuntimeError("TT cache population requires a single 32-device Blackhole Galaxy")
        with tempfile.TemporaryDirectory(prefix=f".populate_mesh{args.mesh}-", dir=root) as staging:
            os.environ["TT_CACHE_PATH"] = staging
            router = ttnn.FabricRouterConfig()
            router.max_packet_payload_size_bytes = 8192
            set_fabric(ttnn.FabricConfig.FABRIC_1D, fabric_router_config=router)
            try:
                mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(shape))
                try:
                    # Load the resolved snapshot so conversion uses the checked checkpoint.
                    state = Gemma4ModelArgs.load_state_dict(str(snapshot))
                    model_result = create_tt_model(
                        mesh_device=mesh,
                        model_path=model_id,
                        state_dict=state,
                        max_seq_len=8192,
                        prefill_chunk_size=8192,
                    )
                    ttnn.synchronize_device(mesh)
                    del model_result
                finally:
                    ttnn.close_mesh_device(mesh)
            finally:
                reset_fabric(ttnn.FabricConfig.FABRIC_1D)
                os.environ["TT_CACHE_PATH"] = str(root)
            built = Path(staging) / cache.name
            mark_weight_cache_complete(built, state, is_host_weight=_gemma4_is_host_weight, **identity)
            if not weight_cache_is_complete(built, **identity, force_env=None):
                raise RuntimeError(f"TT cache validation failed: {built}")
            backup = root / f".{cache.name}.previous-{uuid.uuid4().hex}"
            if cache.exists():
                cache.rename(backup)
            try:
                built.rename(cache)
            except BaseException:
                if backup.exists():
                    backup.rename(cache)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            logger.info("TT cache populated: {}", cache)


if __name__ == "__main__":
    main()
