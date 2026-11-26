# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A one-row step of a model with several session slots pools its ratio-2 compressor through ``compress_pool_decode``
(:data:`~models.experimental.deepseek_v41_flash.tt.decode.attention.SLOTTED_FUSED_POOL`), pairing the row with its
session's window row of ``pos - 1`` (``prev_slot``), as a one-session model always has: each session then gets the
logits it gets alone.

One persistent TP4 instance on row 4 (every MoE on the TP12 server), ``DEEPSEEK_V4_DECODE_LAYERS`` layers (default 4:
layer 2 is a ratio-2 kv and index source, layer 3 reads its group), run eagerly (``DEEPSEEK_V4_TRACED_DECODE=0``) so
the path switch takes effect between runs. Two fixed token streams are decoded to
``DEEPSEEK_V41_SLOTTED_POOL_POSITIONS`` (default 1040: the ratio-2 select is dense to position 1024 and scored past
it), first by a one-session model (``test_one_session_reference``, which saves its logits in the pytest cache), then
in another device session by a three-slot model whose sessions 1 and 2 (non-zero window slots) decode them interleaved,
one step each per position (``test_slotted_fused_pool_matches_one_session``; a second model build on one open mesh
hangs at its first step, so each test builds one). Run the file, the reference first. The slotted model's fused pool must never take the separate-op pool,
and its logits must equal the one-session model's bit for bit. The separate-op pool
(``DEEPSEEK_V41_SLOTTED_FUSED_POOL=0``: the slot's window copied out with ``fill_cache`` and paired by matmuls) runs
too and its distance is logged: it is not exact (Nov 26: worst logits PCC 0.98, about a tenth of the greedy picks
differ over 1040 steps of this 4-layer model).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_slotted_fused_pool.py
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _L1_SMALL_SIZE,
    _PAGE_BLOCK_SIZE,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _stage_engram_or_skip,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tests.decode.test_multi_session_decode import _TEXTS
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.decode import attention as attention_mod
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_POSITIONS = int(os.environ.get("DEEPSEEK_V41_SLOTTED_POOL_POSITIONS", "1040"))
_STREAMS = 2
# Steps whose full logits are compared; every step's greedy pick is.
_COMPARED = set(range(0, 48)) | set(range(1000, _POSITIONS)) | set(range(0, _POSITIONS, 16))


def _decode(model, sids: list[int], streams: list[list[int]]) -> list[dict]:
    """Session ``sids[k]`` from position 0 over ``streams[k]``, the sessions interleaved (one session: one stream
    after the other); per stream the picks and the compared logits."""
    out = [{"picks": [], "logits": {}} for _ in streams]

    def step(k, sid, pos):
        model.activate_session(sid)
        logits = model.decode_traced(streams[k][pos], pos).reshape(-1).float()
        out[k]["picks"].append(int(torch.argmax(logits)))
        if pos in _COMPARED:
            out[k]["logits"][pos] = logits

    if len(set(sids)) == 1:
        for k in range(len(streams)):
            model.reset_session(sids[0])
            for pos in range(_POSITIONS):
                step(k, sids[0], pos)
    else:
        for sid in sids:
            model.reset_session(sid)
        for pos in range(_POSITIONS):
            for k, sid in enumerate(sids):
                step(k, sid, pos)
    return out


def _distance(ref: dict, other: dict) -> tuple[int, float, int, list[int]]:
    """``(bit-equal compared steps, worst PCC of the rest, its position, positions whose picks differ)``."""
    equal, worst, worst_pos = 0, 1.0, -1
    for pos, r in ref["logits"].items():
        o = other["logits"][pos]
        if torch.equal(r, o):
            equal += 1
            continue
        _, pcc = comp_pcc(r, o, 0.0)
        if float(pcc) < worst:
            worst, worst_pos = float(pcc), pos
    differ = [p for p, (a, b) in enumerate(zip(ref["picks"], other["picks"])) if a != b]
    return equal, worst, worst_pos, differ


_PARAMS = [
    pytest.mark.skipif(
        not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}"
    ),
    pytest.mark.timeout(7200),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
                "num_command_queues": 2,
                "l1_small_size": _L1_SMALL_SIZE,
            }
        ],
        indirect=["device_params"],
        ids=["fabric_1d"],
    ),
    pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_1stage_persistent_moe12"]),
]


def _with_params(fn):
    for mark in reversed(_PARAMS):
        fn = mark(fn)
    return fn


def _reference_path(request):
    return request.config.cache.mkdir("deepseek_v41_slotted_fused_pool") / f"one_session_{_POSITIONS}.pt"


def _run(mesh_device, monkeypatch, num_sessions: int) -> dict:
    """Build the instance with ``num_sessions`` slots and decode the streams: ``{"fused": [...], "fused_calls": ...}``
    and, with slots, the same for ``"separate"``."""
    from transformers import AutoTokenizer

    monkeypatch.setenv("DEEPSEEK_V4_TRACED_DECODE", "0")
    monkeypatch.setenv("DEEPSEEK_V4_DECODE_LAYERS", os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", "4"))
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(int(os.environ["DEEPSEEK_V4_DECODE_LAYERS"]), config.num_hidden_layers)
    assert num_layers > 2 and config.compress_ratios[2] == 2, "layer 2 must be a ratio-2 kv source"
    _stage_engram_or_skip(loader, config)
    text = tokenizer.encode(" ".join(_TEXTS) + " ", add_special_tokens=False)
    streams = [text[7 * k :] for k in range(_STREAMS)]
    streams = [(s * (_POSITIONS // len(s) + 2))[:_POSITIONS] for s in streams]
    max_seq = _traced_max_seq(config, _POSITIONS + 1, num_layers)

    # Count which pool each kv source step takes.
    calls = {"separate": 0, "fused": 0}
    latent, pooled = attention_mod.DeepSeekV41Attention._latent, attention_mod.DeepSeekV41Attention._pooled_latent

    def counted_latent(self, *args, **kwargs):
        calls["separate"] += 1
        return latent(self, *args, **kwargs)

    def counted_pooled(self, *args, **kwargs):
        calls["fused"] += 1
        return pooled(self, *args, **kwargs)

    monkeypatch.setattr(attention_mod.DeepSeekV41Attention, "_latent", counted_latent)
    monkeypatch.setattr(attention_mod.DeepSeekV41Attention, "_pooled_latent", counted_pooled)

    results = {}
    with contextlib.ExitStack() as stack:
        model, lm_head, _, _ = _construct_model(
            mesh_device, stack, loader=loader, config=config, num_stages=1, persistent_moe=True
        )
        model.prepare_static_decode(
            _build_rope(config, loader.snapshot_dir, max_seq),
            max_seq,
            lm_head=lm_head,
            num_sessions=num_sessions,
            total_tokens=num_sessions * max_seq,
            block_size=_PAGE_BLOCK_SIZE,
            device_sampling=False,
        )
        assert model._slotted == (num_sessions > 1)
        opened = [model.open_session() for _ in range(num_sessions)]
        # One session decodes the streams in turn; with slots, sessions 1 and 2 decode them interleaved.
        sids = [opened[0]] * _STREAMS if num_sessions == 1 else opened[1 : 1 + _STREAMS]
        for sid in set(sids):
            model.ensure_session_capacity(max_seq - 1, sid)
        for name, fused in (("fused", True), ("separate", False)):
            if num_sessions == 1 and not fused:
                continue  # one session has no window offset: always the fused pool
            monkeypatch.setattr(attention_mod, "SLOTTED_FUSED_POOL", fused)
            calls.update(separate=0, fused=0)
            results[name] = _decode(model, sids, streams)
            results[name + "_calls"] = dict(calls)
        model.shutdown()
    return results


@_with_params
@torch.no_grad()
def test_one_session_reference(mesh_device, reset_seeds, monkeypatch, request) -> None:
    alone = _run(mesh_device, monkeypatch, 1)
    assert alone["fused_calls"]["separate"] == 0 and alone["fused_calls"]["fused"] > 0, alone["fused_calls"]
    torch.save(alone["fused"], _reference_path(request))
    logger.info(f"one-session reference of {_STREAMS} streams x {_POSITIONS} steps saved to {_reference_path(request)}")


@_with_params
@torch.no_grad()
def test_slotted_fused_pool_matches_one_session(mesh_device, reset_seeds, monkeypatch, request) -> None:
    path = _reference_path(request)
    if not path.exists():
        pytest.fail(f"no one-session reference at {path}: run test_one_session_reference first")
    alone = torch.load(path)
    slotted = _run(mesh_device, monkeypatch, _STREAMS + 1)
    logger.info(f"pool calls: slotted fused {slotted['fused_calls']}, slotted separate-op {slotted['separate_calls']}")
    assert slotted["fused_calls"]["separate"] == 0 and slotted["fused_calls"]["fused"] > 0, slotted["fused_calls"]
    assert slotted["separate_calls"]["separate"] > 0, slotted["separate_calls"]

    failures = []
    for k in range(_STREAMS):
        compared = len(alone[k]["logits"])
        for name in ("fused", "separate"):
            equal, worst, worst_pos, differ = _distance(alone[k], slotted[name][k])
            logger.info(
                f"stream {k} (session {k + 1}), slotted {name} vs one session: {equal}/{compared} compared steps "
                f"bit-equal, worst PCC {worst:.6f} at pos {worst_pos}, picks differ at {len(differ)} of "
                f"{_POSITIONS}: {differ[:10]}"
            )
            if name == "fused" and (equal != compared or differ):
                failures.append(
                    f"stream {k}: slotted fused pool differs from one session (worst PCC {worst:.6f} at pos "
                    f"{worst_pos}, picks differ at {differ[:10]})"
                )
    assert not failures, "; ".join(failures)
