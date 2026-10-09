# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the pplx-decider PCC tests (real weights, HF goldens)."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.pplx_decider_v1_27b.reference import hf_reference as ref
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import Optimizations, PrecisionPolicy

# Prefill buckets (bge_m3 style, person decision 2026-10-09): the app pads the prompt on the right to
# the next bucket and reads the hidden state of the last real token. 8192 is the app max_length.
SEQ_LENS = [128, 1024, 2048, 4096, 8192]
GOLDEN_SEQ_LEN = 8192  # goldens are the HF stack over one real 8192-token prompt; shorter cases use its prefix
DELTA_LAYERS = [0, 61]
FULL_LAYERS = [3, 63]
ALL_LAYERS = [0, 3, 61, 63]
PCC_THRESHOLD = 0.995
DEVICE_PARAMS = [{"l1_small_size": 24576}]
PCC_LOG = Path(
    os.environ.get("PPLX_DECIDER_PCC_LOG", "/local/ttuser/gtobar/artifacts/pplx_decider/logs/pcc_results.jsonl")
)


def seq_ids(seq_lens=SEQ_LENS):
    return [f"S{s}" for s in seq_lens]


@lru_cache(maxsize=1)
def reader() -> ref.SnapshotReader:
    return ref.SnapshotReader()


@lru_cache(maxsize=1)
def model_args() -> PplxDeciderArgs:
    return PplxDeciderArgs.from_hf_config(reader().text_config)


def build_optimizations(device) -> Optimizations:
    return Optimizations.build(device, policy=PrecisionPolicy.default(), max_seq_len=model_args().max_seq_len)


def build_tt_layer(device, layer_idx: int):
    from models.demos.pplx_decider_v1_27b.tt.decoder import PplxDecoderLayer

    return PplxDecoderLayer.from_state_dict(
        reader().layer_state_dict(layer_idx),
        args=model_args(),
        layer_idx=layer_idx,
        optimizations=build_optimizations(device),
    )


def build_rotary(device):
    from models.demos.pplx_decider_v1_27b.tt.rope import PplxRotary

    a = model_args()
    return PplxRotary(a.rotary_dim, a.rope_theta, a.max_seq_len, device, head_dim=a.head_dim)


def golden_tensor(name: str, seq_len: int) -> torch.Tensor:
    """Prefix ``[:, :seq_len]`` of a streamed HF golden (see reference/hf_reference.py CLI)."""
    path = ref.golden_input_path(ref.DEFAULT_GOLDEN_DIR, name, GOLDEN_SEQ_LEN)
    if not path.exists():
        pytest.fail(
            f"Missing golden {path}. Generate it with: python -m models.demos.pplx_decider_v1_27b.reference.hf_reference "
            f"--seq-len {GOLDEN_SEQ_LEN} --layers {' '.join(map(str, ALL_LAYERS))} --through-final"
        )
    return torch.load(path)[:, :seq_len]


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    """The exact values a BF16 device input holds, back in fp32 for the HF golden."""
    return x.to(torch.bfloat16).to(torch.float32)


@lru_cache(maxsize=1)
def hf_layer_goldens(layer_idx: int, seq_len: int) -> dict[str, torch.Tensor]:
    """HF fp32 outputs of every submodule of one layer on the real-prompt input prefix.

    The layer input is BF16-rounded first so the TT layer and HF see identical values.
    """
    layer = ref.build_decoder_layer(reader(), layer_idx, torch.float32)
    x = bf16_round(golden_tensor(f"L{layer_idx}_input", seq_len))
    outs = ref.layer_submodule_outputs(layer, x, reader().text_config)
    outs["input"] = x
    del layer
    return outs


def to_device(x: torch.Tensor, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(x, device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def to_host(x: ttnn.Tensor, expected_shape) -> torch.Tensor:
    out = ttnn.to_torch(x).to(torch.float32)
    out = out.reshape(out.shape[-len(expected_shape) :]) if out.dim() > len(expected_shape) else out
    assert tuple(out.shape) == tuple(expected_shape), f"Expected {tuple(expected_shape)}, got {tuple(out.shape)}"
    return out


def check_pcc(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    module: str,
    seq_len: int,
    layer: int | None = None,
    threshold: float = PCC_THRESHOLD,
    policy: str | None = None,
) -> float:
    """Assert PCC >= threshold and append the measurement to the PCC log."""
    passing, pcc = comp_pcc(reference, candidate, threshold)
    _, allclose_msg = comp_allclose(reference, candidate)
    pcc = float(pcc)
    # Diagnostic only (not gated): PCC without the first token, which can carry outsized activations.
    pcc_wo_tok0 = None
    if reference.dim() >= 2 and reference.shape[-2] > 1:
        pcc_wo_tok0 = float(comp_pcc(reference[..., 1:, :], candidate[..., 1:, :], threshold)[1])
    policy = policy or PrecisionPolicy.default().name
    kind = model_args().layer_kind(layer) if layer is not None else None
    record = dict(
        module=module,
        layer=layer,
        kind=kind,
        seq_len=seq_len,
        pcc=pcc,
        pcc_wo_tok0=pcc_wo_tok0,
        threshold=threshold,
        passed=bool(passing),
        policy=policy,
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    PCC_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PCC_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(
        f"PCC {module} L{layer} S{seq_len} [{policy}]: {pcc:.6f} (>= {threshold}; w/o tok0 {pcc_wo_tok0}); {allclose_msg}"
    )
    assert passing, f"{module} L{layer} S{seq_len}: PCC {pcc:.6f} < {threshold}; {allclose_msg}"
    return pcc
