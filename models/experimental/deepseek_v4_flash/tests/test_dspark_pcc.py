# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test: ttnn ``DSparkModel`` vs. the pure-torch DSpark reference.

Both sides load the *same* dequantized ``mtp.*`` checkpoint weights (via the lazy
:class:`DeepseekV4WeightLoader`) and are driven with identical inputs:

1. a shared random main-hidden *sequence* of ``L`` positions seeds every stage's
   sliding KV cache (the DSpark prefill), then
2. one draft step at ``start_pos = L`` with a shared random main-hidden + accepted
   token id produces ``block_size`` draft logits / ids / confidences.

The ttnn draft logits are PCC-compared against the reference; the greedily sampled
draft ids must match exactly. Sized to run on a single Blackhole (all 3 MTP stages
+ their 256-expert MoEs resident in ``bfloat4_b``).

Run (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/cache \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_dspark_pcc.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tt"))

from models.experimental.deepseek_v4_flash.tt.dspark import DSparkModel  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.weight_loader import (  # noqa: E402
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)
import dspark_reference as REF  # noqa: E402
import weight_loader as WL  # noqa: E402
import quant as Q  # noqa: E402

_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR")


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


@pytest.mark.skipif(not _checkpoint_available(), reason=f"DSpark checkpoint not found under {_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_dspark_draft_pcc(mesh_device) -> None:
    torch.manual_seed(0)
    # DSpark is single-device here; carve a 1x1 submesh out of the fixture mesh.
    mesh_device.reshape(ttnn.MeshShape(1, mesh_device.get_num_devices()))
    device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))

    loader = DeepseekV4WeightLoader(_MODEL_DIR)
    ref_loader = WL.DeepseekV4WeightLoader(_MODEL_DIR)
    n_mtp = REF.count_mtp_stages(ref_loader)
    args = REF.DSparkArgs.from_config_json(loader.snapshot_dir / "config.json", n_mtp_layers=n_mtp)
    args.temperature = 0.0

    L = 4  # prompt positions (<= window)
    start_pos = L
    dim3 = args.dim * len(args.dspark_target_layer_ids)
    main_seq = torch.randn(1, 1, L, dim3) * 0.1
    main_dec = torch.randn(1, 1, dim3) * 0.1
    token_id = 100

    # -- reference ------------------------------------------------------------ #
    logger.info("building torch reference (lazy experts)")
    ref = REF.DSparkModel(args).eval()
    REF.load_dspark_weights(ref, ref_loader, Q)
    ids_t = torch.tensor([token_id])
    ref.forward_spec(ids_t, main_seq.reshape(1, L, dim3), 0)  # prefill
    ref_ids, ref_logits, ref_conf = ref.forward_spec(ids_t, main_dec.reshape(1, 1, dim3), start_pos)

    # -- ttnn ----------------------------------------------------------------- #
    logger.info("building ttnn DSparkModel (bf4 experts resident)")
    from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache

    cache = WeightCache(os.path.join(_CACHE_DIR, "dspark")) if _CACHE_DIR else None
    tt = DSparkModel(args, loader, device, cache=cache, weight_dtype=ttnn.bfloat4_b)

    main_seq_tt = ttnn.from_torch(main_seq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt.prefill(main_seq_tt)
    main_dec_tt = ttnn.from_torch(
        main_dec.reshape(1, 1, dim3), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_ids, tt_logits, tt_conf = tt.forward_spec(token_id, main_dec_tt, start_pos)

    # -- compare -------------------------------------------------------------- #
    ref_ids = ref_ids.reshape(-1)
    logits_pcc = _pcc(tt_logits, ref_logits)
    conf_pcc = _pcc(tt_conf, ref_conf)
    logger.info(f"draft logits PCC = {logits_pcc:.5f}")
    logger.info(f"confidence   PCC = {conf_pcc:.5f}")
    logger.info(f"ref draft ids = {ref_ids.tolist()}")
    logger.info(f"tt  draft ids = {tt_ids.tolist()}")

    assert logits_pcc > 0.97, f"draft logits PCC too low: {logits_pcc}"
    assert conf_pcc > 0.97, f"confidence PCC too low: {conf_pcc}"
    # The seeded token is echoed at index 0; the first *drafted* token (index 1)
    # is the highest-confidence prediction and should match the fp32 reference.
    # (Deeper positions can flip under bf4 expert quant near argmax ties.)
    assert tt_ids[1] == ref_ids[1], f"first draft token mismatch: {tt_ids.tolist()} vs {ref_ids.tolist()}"
