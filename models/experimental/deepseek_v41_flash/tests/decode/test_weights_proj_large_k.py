# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The indexer's ``weights_proj`` as ``matmul_decode_large_k`` (:data:`WEIGHTS_PROJ_LARGE_K_CORES` cores, the weight
in L1, each core's K-slice read out of the tokens replica) against the one-core ``matmul_decode`` it replaces, on the
tokens a real decode feeds it.

One persistent TP4 instance on row 4 (every MoE on the TP12 server), ``DEEPSEEK_V4_DECODE_LAYERS`` layers (default 4:
layer 2 runs a scoring select), one session, eager (``DEEPSEEK_V4_TRACED_DECODE=0``), decoded to
``DEEPSEEK_V41_WEIGHTS_PROJ_POSITIONS`` (default 1100: the ratio-2 select scores from position 1024). Every scoring
select's head weights are computed both ways; the large-K path must be taken on every call and match within PCC
0.999.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_weights_proj_large_k.py
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

_POSITIONS = int(os.environ.get("DEEPSEEK_V41_WEIGHTS_PROJ_POSITIONS", "1100"))
_PCC = 0.999


def _first(t: ttnn.Tensor) -> torch.Tensor:
    """One rank's copy (the projection is replicated over the TP ranks)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(-1)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@torch.no_grad()
@pytest.mark.parametrize(
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
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_1stage_persistent_moe12"])
def test_weights_proj_large_k(mesh_device, reset_seeds, monkeypatch) -> None:
    from transformers import AutoTokenizer

    assert attention_mod.WEIGHTS_PROJ_LARGE_K_CORES > 0, "DEEPSEEK_V41_WEIGHTS_PROJ_LARGE_K_CORES=0 turns it off"
    monkeypatch.setenv("DEEPSEEK_V4_TRACED_DECODE", "0")
    monkeypatch.setenv("DEEPSEEK_V4_DECODE_LAYERS", os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", "4"))
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(int(os.environ["DEEPSEEK_V4_DECODE_LAYERS"]), config.num_hidden_layers)
    _stage_engram_or_skip(loader, config)
    text = tokenizer.encode(" ".join(_TEXTS) + " ", add_special_tokens=False)
    tokens = (text * (_POSITIONS // len(text) + 2))[:_POSITIONS]
    max_seq = _traced_max_seq(config, _POSITIONS + 1, num_layers)

    calls = {"large_k": 0, "fallback": 0}
    worst = {"pcc": 1.0, "max_abs": 0.0}
    head_weights = attention_mod.DeepSeekV41Indexer._head_weights

    def compared(self, tokens_tt):
        new = head_weights(self, tokens_tt)
        if self.weights_proj_l1 is None or not self._large_k_reads(tokens_tt, self.weights_proj_l1):
            if not calls["fallback"]:
                w = self.weights_proj_l1
                logger.warning(
                    f"weights_proj fell back: tokens {tokens_tt.shape} {tokens_tt.layout} {tokens_tt.dtype} "
                    f"tile {tokens_tt.get_tile().tile_shape} {tokens_tt.memory_config()}; "
                    f"large-K weight {None if w is None else w.memory_config()}"
                )
            calls["fallback"] += 1
            return new
        calls["large_k"] += 1
        old = self.weights_proj(attention_mod._decode_activation(self.weights_proj, tokens_tt))
        a, b = _first(old), _first(new)
        _, pcc = comp_pcc(a, b, 0.0)
        worst["pcc"] = min(worst["pcc"], float(pcc))
        worst["max_abs"] = max(worst["max_abs"], float((a - b).abs().max()))
        ttnn.deallocate(old)
        return new

    monkeypatch.setattr(attention_mod.DeepSeekV41Indexer, "_head_weights", compared)

    with contextlib.ExitStack() as stack:
        model, lm_head, _, _ = _construct_model(
            mesh_device, stack, loader=loader, config=config, num_stages=1, persistent_moe=True
        )
        model.prepare_static_decode(
            _build_rope(config, loader.snapshot_dir, max_seq),
            max_seq,
            lm_head=lm_head,
            num_sessions=1,
            total_tokens=max_seq,
            block_size=_PAGE_BLOCK_SIZE,
            device_sampling=False,
        )
        sid = model.open_session()
        model.ensure_session_capacity(max_seq - 1, sid)
        model.activate_session(sid)
        for pos in range(_POSITIONS):
            model.decode_traced(tokens[pos], pos)
        model.shutdown()

    logger.info(
        f"weights_proj over {_POSITIONS} steps: {calls}; large-K vs matmul_decode worst PCC {worst['pcc']:.6f}, "
        f"max abs diff {worst['max_abs']:.3e}"
    )
    assert calls["large_k"] > 0 and calls["fallback"] == 0, calls
    assert worst["pcc"] >= _PCC, worst
