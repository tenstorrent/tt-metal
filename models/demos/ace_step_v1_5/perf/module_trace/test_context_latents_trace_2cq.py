# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2CQ for ``TtAceStepInstrumentalConditionEncoder.ctx_concat_traced``.

Exercises device ``ttnn.concat([src_latents, chunk_mask], dim=-1)`` with CQ1 input refresh
and CQ0 ``execute_trace``, matching the handler demo path.

Run:

    pytest models/demos/ace_step_v1_5/perf/module_trace/test_context_latents_trace_2cq.py -v -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.ttnn_impl.condition_encoder import TtAceStepInstrumentalConditionEncoder

_CKPT_ROOT_ENV = "ACE_STEP_CHECKPOINT_DIR"
_DEFAULT_CKPT_ROOT = Path("~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints").expanduser()
_DEFAULT_ITERS = int(os.environ.get("ACE_STEP_TRACE_TEST_ITERS", "16"))
_TRACE_VS_COMPILE_PCC = 0.999


def _ckpt_root() -> Path:
    return Path(os.environ.get(_CKPT_ROOT_ENV, str(_DEFAULT_CKPT_ROOT))).expanduser()


def _find_dit_safetensors() -> Path | None:
    root = _ckpt_root()
    for variant in ("acestep-v15-turbo", "acestep-v15-base"):
        p = root / variant / "model.safetensors"
        if p.is_file():
            return p
    for variant_dir in sorted(root.glob("acestep-v15-*")):
        p = variant_dir / "model.safetensors"
        if p.is_file():
            return p
    return None


_SKIP_REASON = (
    "ACE-Step v1.5 DiT checkpoint not found; set ACE_STEP_CHECKPOINT_DIR or download "
    "acestep-v15-base/model.safetensors."
)


@pytest.mark.skipif(_find_dit_safetensors() is None, reason=_SKIP_REASON)
def test_context_latents_concat_trace_2cq(trace_device):
    import ttnn

    device = trace_device
    ckpt = _find_dit_safetensors()
    assert ckpt is not None

    silence = torch.load(str(_ckpt_root() / "acestep-v15-base" / "silence_latent.pt"), map_location="cpu").to(
        torch.float32
    )
    if int(silence.shape[-1]) != 64:
        silence = silence.transpose(1, 2).contiguous()
    frames = 15 * 25
    src_latents = silence[:, :frames, :].contiguous()
    if src_latents.shape[1] < frames:
        rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
        src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()
    chunk_masks = torch.ones((1, frames, 64), dtype=torch.float32)
    src_np = src_latents.numpy()
    chunk_np = chunk_masks.numpy()
    ref = torch.cat([src_latents, chunk_masks], dim=-1).to(torch.bfloat16).float().numpy()

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    enc = TtAceStepInstrumentalConditionEncoder(
        device=device,
        checkpoint_safetensors_path=str(ckpt),
        dtype=ttnn.bfloat16,
    )
    try:
        compile_out = enc.ctx_concat_traced(src_np, chunk_np, use_trace=False)
        compile_np = ttnn.to_torch(compile_out).float().numpy()
        assert_pcc_print("context_latents_compile", torch.from_numpy(ref), torch.from_numpy(compile_np))

        trace_out = enc.ctx_concat_traced(src_np, chunk_np, use_trace=True)
        trace_np = ttnn.to_torch(trace_out).float().numpy()
        assert_pcc_print(
            "context_latents_trace_vs_compile",
            torch.from_numpy(compile_np),
            torch.from_numpy(trace_np),
            pcc_min=_TRACE_VS_COMPILE_PCC,
        )

        t0 = time.perf_counter()
        for _ in range(_DEFAULT_ITERS):
            trace_out = enc.ctx_concat_traced(src_np, chunk_np, use_trace=True)
            ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / _DEFAULT_ITERS
        print(f"context_latents_trace_2cq steady-state: {elapsed_ms:.3f} ms/iter", flush=True)
    finally:
        enc.release_trace()
