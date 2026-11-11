# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-profiling harness for PplxDecider.readout_logits (eager or traced prefill).

Run under tracy, e.g.
    PPLX_PROF_LEN=2048 PPLX_PROF_MODE=traced python -m tracy -r -p -v --op-support-count 20000 \\
        -m "pytest models/demos/pplx_decider_v1_27b/tests/test_profile.py"

Env vars:
    PPLX_PROF_LEN     exact prompt token length (default 512)
    PPLX_PROF_LENS    comma list of prompt lengths (overrides PPLX_PROF_LEN), e.g. 512,1024,2048
    PPLX_PROF_MODE    eager | traced | both (default eager); traced calls capture_prefill_trace() first;
                      both = eager sweep over all lengths, then one trace capture, then a traced sweep
    PPLX_PARENT_MESH  unset (default): the model runs on the mesh opened by parametrize_mesh_traced (MESH_DEVICE).
                      "4x8" / "8x4": open that full (Galaxy) mesh with FABRIC_1D_RING and run on a submesh
    PPLX_SUBMESH      submesh shape carved out of PPLX_PARENT_MESH, "1x8" (default) or "1x4"; the first one is used
    PPLX_PROF_CHUNK   optional chunk size for the traced prefill; PplxDecider.capture_prefill_trace takes no
                      argument, so when set the model-level capture_prefill_trace_chunked is called instead
    PPLX_PROF_BUCKET_WARMUP  "0" disables warmup_masked_buckets in capture_prefill_trace_chunked (default "1"; only used
                      when PPLX_PROF_CHUNK is set)
    PPLX_PROF_WARMUP  warmup readout_logits calls before measurement (default 2)
    PPLX_PROF_ITERS   measured readout_logits calls between the start/stop signposts (default 1)
    PPLX_PROF_OUT     if set, a JSON line with the results is appended to this file
    PPLX_SAVE_LOGITS_DIR  if set, the readout logits (what readout_logits returns, fp32 CPU) of the LAST measured
                      iteration of every (mode, len) are written with torch.save to
                      <dir>/<mode>_<len>[_c<chunk>].pt; the top-5 indices are always logged
"""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.pplx_decider_v1_27b.tests.common import model_mesh, parametrize_mesh_traced
from models.demos.pplx_decider_v1_27b.tests.test_perf import _build_input_ids
from models.demos.pplx_decider_v1_27b.tt.model import PplxDecider

try:
    from tracy import signpost
except Exception:  # tracy is only importable when profiling support is installed

    def signpost(header, message=None):
        logger.info(f"signpost (no tracy): {header}")


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _prompt_lengths():
    lens = os.environ.get("PPLX_PROF_LENS")
    if lens:
        return [int(x) for x in lens.split(",") if x.strip()]
    return [_env_int("PPLX_PROF_LEN", 512)]


def _save_readout_logits(save_dir, run_mode, prompt_len, chunk, logits, suffix=""):
    """torch.save the fp32 CPU readout logits to <save_dir>/<mode>_<len>[_c<chunk>]<suffix>.pt; returns the path."""
    os.makedirs(save_dir, exist_ok=True)
    name = f"{run_mode}_{prompt_len}" + (f"_c{chunk}" if chunk is not None else "") + suffix + ".pt"
    path = os.path.join(save_dir, name)
    torch.save(logits.detach().to(torch.float32).cpu().contiguous().clone(), path)
    return path


def _read_device_profiler(mesh_device):
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception as e:  # runs without the profiler enabled too
        logger.warning(f"ttnn.ReadDeviceProfiler skipped: {e!r}")


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(3600)
@parametrize_mesh_traced()
def test_pplx_profile(mesh_device):
    lengths = _prompt_lengths()
    mode = os.environ.get("PPLX_PROF_MODE", "eager")
    chunk = _env_int("PPLX_PROF_CHUNK", None)
    warmup = _env_int("PPLX_PROF_WARMUP", 2)
    iters = _env_int("PPLX_PROF_ITERS", 1)
    save_dir = os.environ.get("PPLX_SAVE_LOGITS_DIR") or None
    assert mode in ("eager", "traced", "both"), f"PPLX_PROF_MODE must be eager|traced|both, got {mode!r}"
    assert iters >= 1 and warmup >= 0 and lengths
    modes = ["eager", "traced"] if mode == "both" else [mode]
    single = len(modes) * len(lengths) == 1  # only then are the plain "start"/"stop" signposts emitted

    parent_shape = tuple(mesh_device.shape)
    sub = model_mesh(mesh_device)
    mesh_shape = tuple(sub.shape)
    try:
        logger.info(f"fabric config: {ttnn.get_fabric_config()} cluster type: {ttnn.cluster.get_cluster_type()}")
    except Exception as e:  # informational only
        logger.warning(f"fabric/cluster query failed: {e!r}")
    logger.info(
        f"profile: parent={parent_shape} mesh={mesh_shape} modes={modes} lens={lengths} chunk={chunk} "
        f"warmup={warmup} iters={iters}"
    )

    # The weight cache path is TT_CACHE_PATH/<device name>/tensor_cache_bf*[_mesh<R>x<C>], already per mesh shape.
    t_build = time.perf_counter()
    decider = PplxDecider.from_pretrained(sub)
    logger.info(f"PplxDecider.from_pretrained (weight load / TT-cache build): {time.perf_counter() - t_build:.1f}s")
    logger.info(
        f"model args: device_name={decider.args.device_name} num_devices={decider.args.num_devices} "
        f"cluster_shape={decider.args.cluster_shape} is_galaxy={decider.args.is_galaxy} "
        f"is_galaxy_cluster={decider.args.is_galaxy_cluster} ccl_topology={decider.args.ccl_topology()} "
        f"cache={decider.args.weight_cache_path()}"
    )
    inputs = {n: _build_input_ids(decider, n) for n in lengths}
    token_ids = list(decider.decision_config.token_ids)
    out_path = os.environ.get("PPLX_PROF_OUT")
    top1_by_run = {}

    for run_mode in modes:
        if run_mode == "traced":
            t_cap = time.perf_counter()
            if chunk is None:
                decider.capture_prefill_trace()
            else:
                bucket_warmup = os.environ.get("PPLX_PROF_BUCKET_WARMUP", "1") != "0"
                decider.model.capture_prefill_trace_chunked(
                    sub, decider._page_table, chunk_size=chunk, warmup_masked_buckets=bucket_warmup
                )
            ttnn.synchronize_device(sub)
            logger.info(f"prefill trace capture: {time.perf_counter() - t_cap:.1f}s")

        for prompt_len in lengths:
            input_ids = inputs[prompt_len]
            for _ in range(warmup):
                decider.readout_logits(input_ids)
            ttnn.synchronize_device(sub)
            _read_device_profiler(sub)  # drop warmup / compile / capture ops from the measured window

            tag = f"{run_mode}_{prompt_len}"
            signpost(f"start_{tag}")
            if single:
                signpost("start")
            wall_ms, top1 = [], []
            for _ in range(iters):
                start = time.perf_counter()
                logits = decider.readout_logits(input_ids)
                ttnn.synchronize_device(sub)
                wall_ms.append((time.perf_counter() - start) * 1000)
                top1.append(int(torch.argmax(logits)))
            if single:
                signpost("stop")
            signpost(f"stop_{tag}")

            ttnn.synchronize_device(sub)
            _read_device_profiler(sub)

            # `logits` is the last measured iteration's readout (fp32 CPU, [len(token_ids)]).
            top5 = torch.topk(logits.reshape(-1), min(5, logits.numel()))
            top5_idx = top5.indices.tolist()
            logger.info(
                f"pplx logits {tag}: top5 idx={top5_idx} token_ids={[token_ids[i] for i in top5_idx]} "
                f"values={[round(v, 4) for v in top5.values.tolist()]}"
            )
            saved_logits = None
            if save_dir:
                saved_logits = _save_readout_logits(save_dir, run_mode, prompt_len, chunk, logits)
                logger.info(f"pplx logits {tag}: saved {tuple(logits.shape)} {logits.dtype} -> {saved_logits}")

            result = {
                "parent_mesh": list(parent_shape),
                "mesh_shape": list(mesh_shape),
                "device_ids": list(sub.get_device_ids()),
                "mode": run_mode,
                "len": prompt_len,
                "chunk": chunk,
                "warmup": warmup,
                "iters": iters,
                "wall_ms": wall_ms,
                "p50_ms": statistics.median(wall_ms),
                "min_ms": min(wall_ms),
                "top1_index": top1,
                "top1_token_id": [token_ids[i] for i in top1],
                "top5_index": top5_idx,
                "saved_logits": saved_logits,
            }
            top1_by_run[(run_mode, prompt_len)] = top1
            logger.info(
                f"pplx profile {tag}: p50={result['p50_ms']:.1f} ms min={result['min_ms']:.1f} ms "
                f"top1={sorted(set(top1))}"
            )
            logger.info(f"pplx profile result: {json.dumps(result)}")
            if out_path:
                with open(out_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

    if mode == "both":
        for prompt_len in lengths:
            e, t = top1_by_run[("eager", prompt_len)], top1_by_run[("traced", prompt_len)]
            consistent = len(set(e)) == 1 and len(set(t)) == 1 and e[0] == t[0]
            logger.info(
                f"top-1 eager vs traced len={prompt_len}: eager={sorted(set(e))} traced={sorted(set(t))} "
                f"consistent={consistent}"
            )
