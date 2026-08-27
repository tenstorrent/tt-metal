# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 DP=2 batch-scaling sweep worker (one (local_batch, variant) per run).

Measures dense bidirectional S8192 prefill on one N300 for a given local batch
(per-chip) and implementation variant. Emits a single machine-readable
``RESULT_JSON={...}`` line for the parent driver (dp2_batch_scaling_driver.py).

Variants:
  stock_dram       : stock TTNN SDPA + existing DRAM placements.
  jit_dram         : model-local encoder SDPA (q128/k2048, non-FP32 dest, BF16
                     score), existing DRAM placements.
  jit_l1_<name>    : same JIT math, named handoff moved to sharded L1.

Contract (frozen, see .auto/batch_scaling/environment.md): S=8192, mesh (2,1),
DP=2, no mask, dense non-causal bidirectional, scale 1.0, batch-sharded dim 0,
seed 42 (override with --seed). Each chip gets local_batch full-length seqs.

Run ONE per subprocess with an external timeout:
  TT_VISIBLE_DEVICES=0 python dp2_batch_scaling.py --local-batch 1 --variant jit_dram
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tests.perf.perf import prepare_inputs
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

SEQ_LEN = 8192
MESH = (2, 1)


def _batchsharded(inputs, mesh_device, *, on_device):
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    kwargs = {"mesh_mapper": mapper}
    if on_device:
        kwargs.update(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def convert(t, dtype, layout):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, **kwargs)

    return {
        "input_ids": convert(inputs["input_ids"].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "token_type_ids": convert(inputs["token_type_ids"].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "position_ids": convert(inputs["position_ids"].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    }


def _copy_inputs(host_tensors, device_tensors):
    for key in host_tensors:
        ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key])


def _sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run(local_batch: int, variant: str, seed: int, screen_iters: int, final_iters: int) -> dict:
    global_batch = 2 * local_batch
    use_jit = variant.startswith("jit")
    # jit_l1_* implies the L1 residency path is requested; the specific handoff is
    # selected via BGE_L1_HANDOFF (set by the driver). jit_dram/stock_dram => none.
    l1_handoff = os.environ.get("BGE_L1_HANDOFF", "") if variant.startswith("jit_l1") else ""

    # Strict mode: every jit_* row must PROVE the model-local kernel ran.
    if use_jit:
        os.environ.setdefault("BGE_REQUIRE_ENCODER_SDPA", "1")

    result = {
        "sha": _sha(),
        "local_batch": local_batch,
        "global_batch": global_batch,
        "variant": variant,
        "seed": seed,
        "sdpa_impl": "unknown",
        "l1_tensors": l1_handoff,
        "status": "FAIL",
        "pcc": "",
        "iterations": 0,
        "min_ms": "",
        "p50_ms": "",
        "mean_ms": "",
        "p95_ms": "",
        "std_ms": "",
        "input_copy_ms": "",
        "device_ms": "",
        "global_tokens_per_s": "",
        "per_chip_sequences_per_s": "",
        "profiler_artifact": "",
        "notes": "",
    }

    # Mirror conftest.mesh_device: pop fabric_config, set_fabric() separately,
    # open with remaining params, reset_fabric on close.
    from tests.scripts.common import get_updated_device_params
    from conftest import reset_fabric, set_fabric

    device_params = {
        "trace_region_size": 90_000_000,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    }
    updated = get_updated_device_params(device_params)
    fabric_config = updated.pop("fabric_config", None)
    updated.pop("require_exact_physical_num_devices", False)
    set_fabric(fabric_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH), **updated)
    try:
        assert tuple(mesh_device.shape) == MESH
        assert mesh_device.get_num_devices() == 2

        args, model, _ = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=global_batch,
            max_seq_len=SEQ_LEN,
            dtype=ttnn.bfloat8_b,
            data_parallel=True,
            use_experimental_encoder_sdpa=use_jit,
            encoder_sdpa_q256_vbf4=True,
        )
        assert model._data_parallel, "DP mode not active"

        inputs = prepare_inputs(args.tokenizer, global_batch, SEQ_LEN, args.pad_token_id)
        # Assert per-device shard shape: each chip gets local_batch sequences.
        assert inputs["input_ids"].shape[0] == global_batch, inputs["input_ids"].shape
        host_tensors = _batchsharded(inputs, mesh_device, on_device=False)
        device_tensors = _batchsharded(inputs, mesh_device, on_device=True)
        # Confirm shard shape on device: a batch-sharded tensor reports the
        # PER-DEVICE shard shape, so dim0 == local_batch already.
        shard0 = device_tensors["input_ids"].shape[0]
        result["notes"] += f"shard_dim0={shard0};"
        assert shard0 == local_batch, f"expected per-chip {local_batch}, got {shard0}"

        # Compile (untraced), record program cache before/after.
        out = model.forward(**device_tensors)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)

        # Device-profiler mode: run ONE more untraced, signposted forward and stop
        # (traced replay floods per-core markers; ONE untraced forward gives clean
        # per-op DEVICE KERNEL DURATION for wall-vs-device separation).
        if os.environ.get("BGE_BATCH_DEVICE_PROFILE", "0") == "1":
            try:
                from tracy import signpost

                ttnn.synchronize_device(mesh_device)
                signpost("BGE_BATCH_FWD_START")
            except Exception:
                pass
            out = model.forward(**device_tensors)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)
            result["status"] = "PASS"
            result["notes"] += "device_profile_only;"
            return result

        model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)
        # Warmup >= 5 traced replays.
        for _ in range(5):
            model.execute_trace(blocking=True)
        ttnn.synchronize_device(mesh_device)

        # Program-cache stability check (post-warmup entry count).
        try:
            pc = mesh_device.num_program_cache_entries()
            result["notes"] += f"pcache={pc};"
        except Exception:
            pass

        n_iters = final_iters if variant in ("stock_dram", "jit_dram") else screen_iters
        # Measured traced replay: exclude input-copy from primary latency; time
        # input-copy separately.
        lat_ms = []
        copy_ms = []
        for _ in range(n_iters):
            fresh = prepare_inputs(args.tokenizer, global_batch, SEQ_LEN, args.pad_token_id)
            fresh_host = _batchsharded(fresh, mesh_device, on_device=False)
            tc = time.perf_counter()
            _copy_inputs(fresh_host, device_tensors)
            ttnn.synchronize_device(mesh_device)
            copy_ms.append((time.perf_counter() - tc) * 1000.0)
            t0 = time.perf_counter()
            model.execute_trace(blocking=True)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

        model.release_trace()

        lat_ms.sort()
        result["iterations"] = n_iters
        result["min_ms"] = round(min(lat_ms), 3)
        result["p50_ms"] = round(statistics.median(lat_ms), 3)
        result["mean_ms"] = round(statistics.mean(lat_ms), 3)
        result["p95_ms"] = round(lat_ms[max(0, int(0.95 * len(lat_ms)) - 1)], 3)
        result["std_ms"] = round(statistics.pstdev(lat_ms), 3)
        result["input_copy_ms"] = round(statistics.median(copy_ms), 3)
        p50 = result["p50_ms"]
        result["global_tokens_per_s"] = round(global_batch * SEQ_LEN / (p50 / 1000.0))
        result["per_chip_sequences_per_s"] = round(local_batch / (p50 / 1000.0), 3)
        result["raw_samples"] = lat_ms
        result["sdpa_impl"] = "jit_encoder" if use_jit else "stock_ttnn"
        result["status"] = "PASS"
    finally:
        ttnn.close_mesh_device(mesh_device)
        reset_fabric(fabric_config)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-batch", type=int, required=True, choices=[1, 2, 3, 4, 6])
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-iters", type=int, default=10)
    ap.add_argument("--final-iters", type=int, default=30)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    try:
        res = run(a.local_batch, a.variant, a.seed, a.screen_iters, a.final_iters)
    except Exception as e:  # noqa: BLE001
        import traceback

        res = {
            "local_batch": a.local_batch,
            "global_batch": 2 * a.local_batch,
            "variant": a.variant,
            "seed": a.seed,
            "status": "FAIL",
            "notes": f"{type(e).__name__}: {e}",
        }
        traceback.print_exc()
        print("RESULT_JSON=" + json.dumps(res), flush=True)
        raise SystemExit(2)

    print("RESULT_JSON=" + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
