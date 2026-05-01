#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Benchmark ``models.tt_transformers.tt.model.Transformer`` initialization.

Times the four phases that come up when you build (or rebuild) the on-device
``Transformer`` for inference, with the same Llama checkpoint your
``pcc_hf_ttml_ttt.py`` script uses:

  * ``model_args.load_state_dict()`` — HF safetensors → Meta-named torch dict.
    This is the ``safetensors load`` cost.
  * ``Transformer(...)`` cold init  — empty cache → run ``pack_as_bfp_tiles``
    (or the bf16 cast) for every weight, write flatbuffer cache files, upload.
  * ``Transformer(...)`` warm init  — populated cache → flatbuffer read +
    upload only.
  * ``Transformer(...)`` fresh-cache init — same as cold but writes to a
    unique throwaway directory, simulating the "transfer ttml→ttt every K
    steps" loop where the cache must be rotated per snapshot since weight
    values change every time. (We can't pass ``weight_cache_path=None``
    because mlp.py / embedding.py crash on it — upstream bug.)

For each dtype we also report ``cold - warm`` so you can see how much of cold
init is the per-element host conversion vs. everything else.

Edit ``MODEL_ID`` / ``DTYPES`` to change the checkpoint or which dtypes get
benchmarked. No CLI args.
"""

from __future__ import annotations

import os

# Silence the noisy ttnn::tilize "Using input shard spec ..." warning and
# other tt-metal log_warning lines so the timing output is readable. Must be
# set before any tt-metal / ttnn import.
os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import gc
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MAX_SEQ_LEN = 256
BATCH_SIZE = 1

# (label, ttnn dtype). bf8 is the script's existing path; bf16 is the
# recommended path for hot-loop ttml→ttt transfer.
DTYPES_LABEL = [("bf8", "bfloat8_b"), ("bf16", "bfloat16")]


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@contextmanager
def stopwatch(label: str, results: Dict[str, float]):
    print(f"[bench] running: {label}")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        results[label] = dt
        print(f"[bench]   -> {label}: {dt:.3f} s")


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def main() -> None:
    os.environ["HF_MODEL"] = MODEL_ID

    import ttnn

    # tt-transformers requires fabric_config set BEFORE the mesh device is
    # opened. FABRIC_2D matches what pcc_hf_ttml_ttt.py uses.
    print(f"[bench] set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    print(f"[bench] open_mesh_device(MeshShape(1, 1))")
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
    print(f"[bench] using {mesh_device.get_num_devices()} device(s)")

    results: Dict[str, float] = {}

    try:
        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

        print(f"[bench] building ModelArgs for {MODEL_ID}")
        model_args = ModelArgs(
            mesh_device,
            instruct=True,
            max_batch_size=BATCH_SIZE,
            optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
            max_seq_len=MAX_SEQ_LEN,
            cache_hf=True,
        )
        print(f"[bench] model_name={model_args.model_name}  n_layers={model_args.n_layers}")

        # ---- Phase 1: HF safetensors → Meta-named torch state_dict ----
        with stopwatch("safetensors load (model_args.load_state_dict)", results):
            state_dict = model_args.load_state_dict()
        n_tensors = len(state_dict)
        print(f"[bench]   state_dict has {n_tensors} tensors")

        paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

        def build_transformer(dtype, weight_cache_path):
            return Transformer(
                args=model_args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                paged_attention_config=paged_attention_config,
            )

        # ---- Phase 2-4: per-dtype timings ----
        for label, dtype_name in DTYPES_LABEL:
            dtype = getattr(ttnn, dtype_name)
            cache_path = model_args.weight_cache_path(dtype)

            print()
            print(f"[bench] === dtype={label} ({dtype_name}) cache_path={cache_path}")

            # Cold: wipe cache, init with cache enabled (will write the cache).
            shutil.rmtree(cache_path, ignore_errors=True)
            with stopwatch(f"{label} cold init  (empty cache, writes cache)", results):
                m = build_transformer(dtype, cache_path)
            del m
            gc.collect()

            # Warm: cache is now populated from the cold call above.
            with stopwatch(f"{label} warm init  (populated cache, reads cache)", results):
                m = build_transformer(dtype, cache_path)
            del m
            gc.collect()

            # Fresh-cache: every init is cold and writes to a throwaway
            # directory that's deleted right after. This simulates the
            # ttml→ttt "rebuild after training step" loop — the cache is
            # rotated per snapshot so it never serves stale weights.
            # weight_cache_path=None would be cleaner but crashes inside
            # mlp.py / embedding.py (their cache_name lambdas don't handle
            # None), so we use a tempdir to dodge that upstream bug.
            fresh_dir = Path(tempfile.mkdtemp(prefix=f"ttt_bench_{label}_"))
            try:
                with stopwatch(f"{label} fresh-cache init (unique throwaway path)", results):
                    m = build_transformer(dtype, fresh_dir)
                del m
                gc.collect()
            finally:
                shutil.rmtree(fresh_dir, ignore_errors=True)

        # ---- Summary ----
        print()
        print("=" * 78)
        print(" Summary  (lower is better)")
        print("=" * 78)
        width = max(len(k) for k in results)
        for k, v in results.items():
            print(f"  {k.ljust(width)}  {v:8.3f} s")

        print()
        print("=" * 78)
        print(" Decomposition: cold = (cold - warm) + warm")
        print(" (cold - warm) is the per-element host conversion + cache-write cost.")
        print(" warm is roughly the cost of flatbuffer-read + DMA upload + scaffolding.")
        print("=" * 78)
        for label, _ in DTYPES_LABEL:
            cold = results.get(f"{label} cold init  (empty cache, writes cache)")
            warm = results.get(f"{label} warm init  (populated cache, reads cache)")
            fresh = results.get(f"{label} fresh-cache init (unique throwaway path)")
            if cold is None or warm is None or fresh is None:
                continue
            print(
                f"  {label:5s}  cold={cold:6.2f}s  warm={warm:6.2f}s  "
                f"diff={cold - warm:6.2f}s  fresh={fresh:6.2f}s  "
                f"(fresh ~ cold confirms per-snapshot path matches first-build cost)"
            )

    finally:
        print()
        print("[bench] closing mesh device")
        try:
            ttnn.close_mesh_device(mesh_device)
        except Exception as exc:
            print(f"[bench] close_mesh_device raised {type(exc).__name__}: {exc}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    main()
