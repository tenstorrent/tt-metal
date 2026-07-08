# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf (timing) test for the 16-chip decode pipeline (Pi0_5GLX16DecodePipeline).

Layout: TP=8 prefill on row 0 + 8-stage streamed matmul_decode denoise on row 1
(see tt/tt_bh_glx/pipeline_16_decode.py + mesh_setup.open_decode_16_mesh).

What it covers (socket-KV e2e path, PI0_KV_SOCKET=1):
    - SigLIP DP + prefix build + TP=8 prefill, replayed from a captured trace on
      the prefill mesh's CQ0.
    - Device-direct KV-concat sockets prefill(0,c)->denoise(1,c) — no host bounce.
    - 8-stage streamed Euler denoise, replayed from the captured loop trace.
    - NEXT chunk's input H2D on the prefill mesh's CQ1 overlapped with the current
      chunk's compute on CQ0.

Tests:
    test_perf_16_socket_traced_2cq  — 2CQ socket-KV e2e replay; the headline
                                      per-chunk ms.

Run:
    export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
    export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
    python_env/bin/pytest -sq \
      models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_16_e2e_trace_2cq.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# 16-chip decode flags (mirror eval/libero_rollout.py's ttnn_16_decode path).
# Set before apply_production_env_defaults so the production file can't override;
# setdefault semantics mean an explicit shell export still wins.
for _k, _v in {
    "PI0_TP": "8",  # 8-chip tensor parallel for prefill
    "PI0_TP4_ATTN_HEADPAR": "1",  # head-parallel attention split
    "PI0_MLP_BS": "1",  # block-sharded MLP (TP=8 tuned)
    "PI0_MLP_FUSED_RS": "0",  # fused reduce-scatter off (TP=8 uses split RS+AG)
    "PI0_KV_SOCKET": "1",  # device-direct KV handoff (no host bounce) — the 2CQ base
}.items():
    os.environ.setdefault(_k, _v)


def _apply_production_env_defaults():
    """Source _bench_runs/pi05_production.env as DEFAULTS so this test runs the
    full validated production flag set without a manual `source`. setdefault
    semantics — an explicitly-set env var still wins. Must run before any ttnn /
    pi0_5 import so every flag is in place when modules read it."""
    import re as _re

    root = os.environ.get("TT_METAL_HOME") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))
    )
    envf = os.path.join(root, "_bench_runs", "pi05_production.env")
    if not os.path.exists(envf):
        print(f"[16-test] WARN: {envf} not found; production flags NOT applied", flush=True)
        return
    applied = []
    with open(envf) as f:
        for line in f:
            m = _re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            if k not in os.environ:
                os.environ[k] = v
                applied.append(f"{k}={v}")
    print(f"[16-test] production env defaults applied ({len(applied)} flags)", flush=True)


_apply_production_env_defaults()

os.environ.setdefault("PI0_SIGLIP_USE_FOLD", "1")
os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("PI0_NUM_CAMERAS", "3")

import ttnn  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
N_CAMS = int(os.environ["PI0_NUM_CAMERAS"])
LANG_LEN = 256
PERF_ITERS = int(os.environ.get("PERF_ITERS", "20"))
# One warm-up replay by default (not timed) — the pipeline's own socket build +
# warm chunk happens inside the loop call, but a WARMUP_ITERS knob is kept for
# symmetry with the 1×8 test / to absorb any first-call jitter.
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "0"))

_PROD_ENV_KEYS = (
    "PI0_EXPERT_MM_LOFI",
    "PI0_ROPE_TABLES_L1",
    "PI0_MM_SWEEP_V2",
    "PI0_DENOISE_MM_TUNE",
    "PI0_PREFILL_MM_TUNE",
    "PI0_UPSTREAM_MASKS",
    "QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT",
    "QWEN_NLP_CREATE_HEADS_HEAD_SPLIT",
    "PI0_MQA_HEAD_SPLIT",
    "PI0_NUM_CAMERAS",
    "PI0_VLM_CHUNK_SIZE",
    "PI0_VLM_MLP_BF8_OUT",
    "PI0_VLM_MLP_MINIMAL",
    "PI0_VLM_MINIMAL_CFG",
    "PI0_SIGLIP_USE_FOLD",
    "PI0_TP",
    "PI0_TP4_ATTN_HEADPAR",
    "PI0_MLP_BS",
    "PI0_MLP_FUSED_RS",
    "PI0_KV_SOCKET",
    "PI05_NUM_DENOISE_STEPS",
)


def _print_prod_env_status():
    present, missing = [], []
    for k in _PROD_ENV_KEYS:
        v = os.environ.get(k)
        (present.append(f"{k}={v}") if v is not None else missing.append(k))
    print(f"\n[env] {len(present)}/{len(_PROD_ENV_KEYS)} production flags set:")
    for s in present:
        print(f"      {s}")
    if missing:
        print(f"[env] MISSING ({len(missing)}): {', '.join(missing)}")
    print(f"[env] N_CAMS (test) = {N_CAMS}")


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_test_inputs(siglip_cfg):
    torch.manual_seed(SEED)
    H = W = siglip_cfg.image_size
    images = [torch.randn(1, 3, H, W) for _ in range(N_CAMS)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_LEN), dtype=torch.int64)
    return images, lang_tokens


def _make_pipeline(mesh_handles):
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_16_decode import Pi0_5GLX16DecodePipeline

    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5")),
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    pipe = Pi0_5GLX16DecodePipeline(cfg, loader.categorized_weights, mesh_handles)
    return pipe, cfg


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def test_perf_16_socket_traced_2cq():
    """2CQ socket-KV e2e replay: next-chunk H2D on CQ1 overlapped with CQ0 compute.

    Opens the 16-chip mesh with num_command_queues=2, then runs PERF_ITERS
    replays of the socket-KV e2e chunk (prefill trace -> device-direct KV
    sockets -> streamed denoise) with the next chunk's inputs staged on CQ1.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_decode_16_mesh

    _print_prod_env_status()

    with open_decode_16_mesh(
        l1_small_size=24576,
        trace_region_size=256 * 1024 * 1024,
        num_command_queues=2,
    ) as mesh_handles:
        pipe, cfg = _make_pipeline(mesh_handles)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        pipe.set_num_denoising_steps(cfg.num_denoising_steps)

        ah = cfg.action_horizon
        ad = cfg.action_dim

        # Optional extra warm-up replays (each loop already does one internally).
        for _ in range(WARMUP_ITERS):
            _ = pipe.sample_actions_socket_2cq_loop(images, lang_tokens, 1)

        # 1CQ baseline (input H2D serialized on CQ0) then 2CQ (input H2D on CQ1
        # overlapped with CQ0 compute). Same pre-staged host tensors, same socket
        # e2e build — the delta is how much input DMA hides behind compute.
        last_1cq, times_1cq = pipe.sample_actions_socket_1cq_loop(images, lang_tokens, PERF_ITERS)
        last_actions, times = pipe.sample_actions_socket_2cq_loop(images, lang_tokens, PERF_ITERS)
        assert last_actions.shape[-1] == ad, f"action_dim mismatch: {tuple(last_actions.shape)}"
        assert torch.isfinite(last_actions).all(), "non-finite values in actions output"
        assert torch.isfinite(last_1cq).all(), "non-finite values in 1CQ actions output"

        mean_1cq = _mean(times_1cq)
        mean = _mean(times)
        print("\n" + "=" * 72)
        print(f"16-chip pi0.5 socket-KV e2e replay  (N_CAMS={N_CAMS}, steps={cfg.num_denoising_steps})")
        print("=" * 72)
        print(f"  actions shape           : {tuple(last_actions.shape)}  (ah={ah}, ad={ad})")
        print(f"  1CQ mean ({len(times_1cq)} iters)  : {mean_1cq:.2f} ms  (input H2D serial on CQ0)")
        print(f"  2CQ mean ({len(times)} iters)  : {mean:.2f} ms  (input H2D on CQ1 || compute)")
        print(f"  2CQ speedup             : {mean_1cq - mean:+.2f} ms  ({100.0 * (mean_1cq - mean) / mean_1cq:+.1f}%)")
        print("=" * 72)
