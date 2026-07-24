# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: ``TtAceStepAudioCodeDetokenizer`` vs HF ``model.detokenizer`` @ production dims.

5 Hz audio codes expand to 25 Hz hints: ``[1, N_codes × pool_window_size, 64]`` with
``pool_window_size=5`` (e.g. 75 codes → 375 latent frames for 15 s).

Long streams (>200 codes) use chunked TTNN forwards (``ACE_STEP_DETOK_CHUNK_CODES``); the
default demo path must match HF global attention within PCC tolerance.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager

import pytest
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.experimental.ace_step_v1_5.tests._prod_test_helpers import (
    base_model_safetensors,
    ckpt_root,
    ensure_vendored_acestep_on_path,
)

_PCC = float(os.environ.get("ACE_STEP_DETOK_PCC", "0.97"))
_PCC_CHUNKED = float(os.environ.get("ACE_STEP_DETOK_PCC_CHUNKED", "0.965"))


def _pcc_threshold(n_codes: int) -> float:
    """Chunked TTNN detok (no cross-chunk attention) is slightly below HF global at long N."""
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_detok_chunk_n

    if int(n_codes) > ace_step_detok_chunk_n():
        return _PCC_CHUNKED
    return _PCC


def _load_hf_detokenizer():
    from transformers import AutoModel

    ensure_vendored_acestep_on_path()
    model_dir = ckpt_root() / "acestep-v15-base"
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, torch_dtype=torch.float32).eval()
    return model.tokenizer.quantizer, model.detokenizer


def _code_str(n_codes: int, *, seed: int = 0) -> str:
    g = torch.Generator().manual_seed(int(seed))
    ids = torch.randint(100, 60000, (int(n_codes),), generator=g).tolist()
    return "".join(f"<|audio_code_{int(i)}|>" for i in ids)


@contextmanager
def _open_detok_device(mesh_sku: str):
    """Open the 1×1 preprocess device used for detok (P150 or BH_QB Phase-A chip)."""
    from models.experimental.ace_step_v1_5.conftest import _open_kwargs, require_ttnn
    from models.experimental.ace_step_v1_5.utils.tt_device import (
        ace_step_preprocess_num_command_queues,
        close_ace_step_device,
        open_preprocess_device,
    )

    ttnn = require_ttnn()
    saved_mesh = os.environ.pop("MESH_DEVICE", None)
    saved_ace_mesh = os.environ.pop("ACE_STEP_MESH_DEVICE", None)
    dev = None
    try:
        n_cq = ace_step_preprocess_num_command_queues(use_trace=False)
        if mesh_sku == "P150":
            try:
                dev = ttnn.open_device(**_open_kwargs())
            except RuntimeError as exc:
                pytest.skip(f"P150 device unavailable for detok PCC: {exc}")
        else:
            try:
                dev = open_preprocess_device(
                    ttnn,
                    device_id=int(os.environ.get("TT_DEVICE_ID", "0")),
                    num_command_queues=int(n_cq),
                    mesh_sku=mesh_sku,
                )
            except RuntimeError as exc:
                pytest.skip(f"BH_QB preprocess device unavailable for detok PCC: {exc}")
        if dev is not None and hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()
        yield dev
    finally:
        if dev is not None:
            close_ace_step_device(ttnn, dev)
        if saved_mesh is not None:
            os.environ["MESH_DEVICE"] = saved_mesh
        if saved_ace_mesh is not None:
            os.environ["ACE_STEP_MESH_DEVICE"] = saved_ace_mesh


def _chunk_boundary_frames(n_codes: int, chunk_n: int) -> list[int]:
    """25 Hz frame indices at TTNN detok chunk boundaries (each code → 5 frames)."""
    if n_codes <= chunk_n:
        return []
    boundaries: list[int] = []
    for code_idx in range(chunk_n, n_codes, chunk_n):
        boundaries.append(int(code_idx * 5))
    return boundaries


@pytest.mark.parametrize("mesh_sku", ["P150", "BH_QB"])
@pytest.mark.parametrize(
    "n_codes,label",
    [
        (75, "15s_75codes"),
        (150, "30s_150codes"),
        (300, "60s_300codes"),
        (600, "120s_600codes"),
    ],
)
def test_audio_code_detokenizer_pcc_vs_hf(mesh_sku: str, n_codes: int, label: str):
    if base_model_safetensors() is None:
        pytest.skip("ACE-Step v1.5 checkpoints not found; set ACE_STEP_CHECKPOINT_DIR.")

    import ttnn
    from models.experimental.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import TtAceStepAudioCodeDetokenizer
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_detok_chunk_n

    quantizer, detok = _load_hf_detokenizer()

    code_str = _code_str(n_codes, seed=42)
    code_ids = [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str)]
    assert len(code_ids) == int(n_codes)

    indices = torch.tensor(code_ids, dtype=torch.long).reshape(1, n_codes, 1)
    with torch.inference_mode():
        quantized = quantizer.get_output_from_indices(indices)
        ref = detok(quantized).float()

    with _open_detok_device(mesh_sku) as device:
        tt_detok = TtAceStepAudioCodeDetokenizer(
            device=device,
            checkpoint_safetensors_path=str(base_model_safetensors()),
            dtype=getattr(ttnn, "bfloat16", None),
        )
        out_tt = tt_detok.forward(code_str)
        assert out_tt is not None
        got = ttnn.to_torch(out_tt).float()

    t = min(int(ref.shape[1]), int(got.shape[1]))
    c = min(int(ref.shape[2]), int(got.shape[2]))
    ref_s = ref[:, :t, :c]
    got_s = got[:, :t, :c]

    pcc_floor = _pcc_threshold(int(n_codes))
    test_label = f"audio_detokenizer_{mesh_sku}_{label}"
    print(
        f"\n[detok_pcc][{mesh_sku}][{label}] n_codes={n_codes} pcc_floor={pcc_floor} "
        f"ref={tuple(ref.shape)} got={tuple(got.shape)}",
        flush=True,
    )
    score = assert_pcc_print(test_label, ref_s, got_s, pcc=pcc_floor)
    print(
        f"[ace_step_v1_5][PCC] {test_label}_summary: pcc={score:.6f}",
        flush=True,
    )

    chunk_n = ace_step_detok_chunk_n()
    for frame_idx in _chunk_boundary_frames(int(n_codes), int(chunk_n)):
        if frame_idx + 5 > t:
            continue
        window = slice(frame_idx, frame_idx + 5)
        boundary_score = assert_pcc_print(
            f"{test_label}_chunk_boundary_f{frame_idx}",
            ref_s[:, window, :],
            got_s[:, window, :],
            pcc=pcc_floor,
        )
        print(
            f"[ace_step_v1_5][PCC] chunk boundary frame={frame_idx} pcc={boundary_score:.6f}",
            flush=True,
        )
