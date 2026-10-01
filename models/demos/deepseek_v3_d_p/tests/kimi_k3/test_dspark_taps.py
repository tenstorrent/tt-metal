# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's DSpark taps against the GPU taps the drafter golden was built from.

DSpark taps target layer L at layer L+1's AttnRes read before attention (`attn_input_tap`), not at the
running sum `layer_tap` gives. This runs K3's first layers once and scores every tap they reach against
`tap_layer_{L}.safetensors` in `$PREFILL_DFLASH_GOLDEN_TAPS_DIR`. The tokens come from the 1M trace, which
is the same prompt: its first 1024 positions match the taps to bf16 rounding, the rest to PCC 0.9998
because the two GPU runs chunked the prefill differently.
"""

import os
from pathlib import Path

import pytest
from loguru import logger
from safetensors import safe_open

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_1M, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import (
    DEEP_LAYER_PCC,
    PLACEMENTS,
    SEQ_LEN,
    SHALLOW_LAYER_PCC,
    SP_AXIS,
    TP_AXIS,
    _compose,
    _model_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache

# The same taps test_dflash_golden_pcc feeds the drafter.
GOLDEN_TAPS_ENV = "PREFILL_DFLASH_GOLDEN_TAPS_DIR"


@pytest.mark.timeout(4800)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", [9, 25], ids=["L9", "L25"])
def test_dspark_taps_match_golden(mesh_device, device_params, num_layers):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_1M)
    taps_dir = os.environ.get(GOLDEN_TAPS_ENV)
    if checkpoint is None or trace is None or not taps_dir:
        pytest.skip(f"needs KIMI_K3_HF_MODEL, the 1M golden trace and {GOLDEN_TAPS_ENV}")
    taps_dir = Path(taps_dir)
    # Target L is read inside layer L + 1, so only the targets below num_layers - 1 are reached.
    targets = sorted(int(p.stem.removeprefix("tap_layer_")) for p in taps_dir.glob("tap_layer_*.safetensors"))
    targets = [t for t in targets if t + 1 < num_layers]
    assert targets, f"no tap in {taps_dir} is reached by {num_layers} layers"

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)
    state_dict = _model_state_dict(checkpoint, num_layers, root, cache)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        tt_ccl=get_tt_ccl(mesh_device),
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden, block_residual=None):
        walk = TtAttnResWalk(
            attn_res,
            hidden,
            list(attn_res.weights.pre),
            list(attn_res.weights.post),
            attn_res.weights.output,
            num_layers,
        )
        return TtAttnResResidual(walk)

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        state_dict,
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=cache,
    )
    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=SEQ_LEN,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
        )
    tokens_tt = prepare_prefill_input_tensor(
        trace.token_ids(SEQ_LEN)[0].tolist(),
        mesh_device,
        tuple(mesh_device.shape)[SP_AXIS],
        False,
        tuple(mesh_device.shape),
        SP_AXIS,
    )

    got = {}

    def tap(local_idx, hidden):
        if local_idx - 1 in targets:
            logger.info(f"tap {local_idx - 1}: per-device {tuple(hidden.shape)} {hidden.dtype} {hidden.layout}")
            got[local_idx - 1] = _compose(mesh_device, hidden)

    try:
        model.forward(tokens_tt, kvpe_cache=kvpe, attn_input_tap=tap)
    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    for t in targets:
        with safe_open(taps_dir / f"tap_layer_{t}.safetensors", framework="pt") as f:
            want = f.get_slice(f"tap_layer_{t}")[:SEQ_LEN].float()
        bar = DEEP_LAYER_PCC if t + 1 > KimiK3Config.ATTN_RES_BLOCK_SIZE else SHALLOW_LAYER_PCC
        ok, pcc = comp_pcc(want, got[t].float(), bar)
        logger.info(f"L{num_layers} tap {t} (layer {t + 1} attention input) vs tap_layer_{t}: {pcc} (bar {bar})")
        assert ok, f"tap {t}: PCC {pcc} < {bar}"
