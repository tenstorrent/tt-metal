# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf harness for a **single DiT attention block** (self or cross).

Isolates ``TtAceStepAttentionSDPA`` reshape/SDPA/linear path without full denoise loop.
Use to track ``ReshapeViewDeviceOperation`` and ``NLPConcatHeadsDeviceOperation`` after
``ace_step_split_qkv_heads_bhsd`` and manual ``ace_step_nlp_concat_heads`` (permute+reshape).

Run from repository root:

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/experimental/ace_step_v1_5/perf/test_perf_attn_tracy.py::test_perf_ace_step_attn_tracy_profile \\
        -v -s

Analyze device CSV with ``tt-perf-report`` on
``generated/profiler/reports/<timestamp>/cpp_device_perf_report.csv``.

Environment:

- ``ACE_STEP_ATTN_PERF_KIND``: ``self`` (default) or ``cross``
- ``ACE_STEP_ATTN_PERF_ITERS`` (default ``32``): timed forward passes
- ``ACE_STEP_PERF_WARMUP`` (default ``4``)
- ``ACE_STEP_ATTN_PERF_SEQ`` (default ``64``): query sequence length (tile-aligned)
- ``ACE_STEP_ATTN_PERF_ENC_SEQ`` (default ``128``): encoder length for cross-attn
- ``ACE_STEP_ATTN_PERF_LAYERS`` (default ``1``): stack repeated forwards (same weights)
- ``ACE_STEP_ATTN_PERF_GQA=1``: use ``num_key_value_heads=2``, ``num_attention_heads=4``
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import Profiler
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import TorchAceStepDiTCoreRef
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import TtAceStepAttentionSDPA, TtHfRotaryEmbedding
from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
)


def _tracy_signpost(label: str) -> None:
    if os.environ.get("CI", "").lower() in ("true", "1", "yes"):
        return
    try:
        from tracy import signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        signpost(label)
    except Exception:
        pass


def _run_attn_tracy_harness(device: ttnn.Device) -> None:
    kind = os.environ.get("ACE_STEP_ATTN_PERF_KIND", "self").strip().lower()
    if kind not in ("self", "cross"):
        pytest.fail(f"Unknown ACE_STEP_ATTN_PERF_KIND={kind!r}")

    iters = max(1, int(os.environ.get("ACE_STEP_ATTN_PERF_ITERS", "32")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "4")))
    seq_len = max(32, int(os.environ.get("ACE_STEP_ATTN_PERF_SEQ", "64")))
    enc_len = max(32, int(os.environ.get("ACE_STEP_ATTN_PERF_ENC_SEQ", "128")))
    n_layers = max(1, int(os.environ.get("ACE_STEP_ATTN_PERF_LAYERS", "1")))
    use_gqa = os.environ.get("ACE_STEP_ATTN_PERF_GQA", "").lower() in ("1", "true", "yes")

    n_kv = 2 if use_gqa else None
    cfg, sd, d_model, _, _ = tiny_dit_decoder_fixture(
        seq_len=seq_len,
        enc_len=enc_len,
        n_heads=4,
        head_dim=32,
        num_layers=1,
        n_kv_heads=n_kv,
    )
    core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)

    torch.manual_seed(0)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)
    enc = None
    enc_tt = None
    if kind == "cross":
        enc_raw = torch.randn(1, enc_len, 32, dtype=torch.bfloat16)
        enc = core.condition_encoder_hidden_states(enc_raw)
        enc_tt = ttnn.from_torch(enc, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    rotary = None
    if kind == "self":
        rotary = TtHfRotaryEmbedding(
            mesh_device=device,
            head_dim=int(cfg.head_dim),
            max_seq_len=int(cfg.max_position_embeddings),
            rope_theta=float(cfg.rope_theta),
            hidden_size=int(cfg.hidden_size),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            dtype=ttnn.bfloat16,
        )

    attn = TtAceStepAttentionSDPA(
        cfg=cfg,
        state_dict=sd,
        base_address=f"layers.0.{kind}_attn",
        mesh_device=device,
        dtype=ttnn.bfloat16,
        rotary_embedding=rotary,
    )
    x_tt = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    profiler = Profiler()
    profiler.clear()

    def _forward() -> None:
        h = x_tt
        for _ in range(n_layers):
            h = attn(h, encoder_hidden_states=enc_tt, is_causal=False)
        return h

    profiler.start("ace_step_attn_compile", force_enable=True)
    _tracy_signpost("ATTN_COMPILE")
    _forward()
    profiler.end("ace_step_attn_compile", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.start("ace_step_attn_warmup", force_enable=True)
    _tracy_signpost("ATTN_WARMUP")
    for _ in range(warmup):
        _forward()
    profiler.end("ace_step_attn_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.enable()
    profiler.start("ace_step_attn_perf_pass")
    _tracy_signpost("ATTN_PERF_PASS")
    for _ in range(iters):
        _forward()
    ttnn.synchronize_device(device)
    profiler.end("ace_step_attn_perf_pass")
    ace_step_flush_device_profiler(device)

    profiler.print()
    per_iter_ms = profiler.get("ace_step_attn_perf_pass") * 1000.0 / max(1, iters * n_layers)
    logger.info(
        "AceStep attn Tracy (kind={}, seq={}, enc={}, layers={}, gqa={}): "
        "compile={:.3f}s warmup={:.3f}s perf_pass={:.3f}s (~{:.2f}ms/forward)",
        kind,
        seq_len,
        enc_len if kind == "cross" else "n/a",
        n_layers,
        use_gqa,
        profiler.get("ace_step_attn_compile"),
        profiler.get("ace_step_attn_warmup"),
        profiler.get("ace_step_attn_perf_pass"),
        per_iter_ms,
    )
    logger.info(
        "Focus ReshapeViewDeviceOperation vs MatmulDeviceOperation in cpp_device_perf_report.csv "
        "(tt-perf-report on generated/profiler/reports/<timestamp>/)"
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_attn_tracy_profile(device, request):
    """Profile one DiT attention block; submodule Tracy CSV for reshape tuning."""
    ace_step_enable_tracy_profiler_env()

    def _final_flush() -> None:
        for _ in range(2):
            ace_step_flush_device_profiler(device)

    request.addfinalizer(_final_flush)
    ace_step_flush_device_profiler(device)
    _run_attn_tracy_harness(device)
