# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-parallel prefill (DecoderLayer._prefill_forward_sp) vs the replicated residual stream.

One-layer GPT-OSS model (random weights at the model's init scale), 1x4 Blackhole mesh, S = 1024 (the shortest
prefill that runs sequence-parallel). The same model runs the prefill with GPT_OSS_PREFILL_SP=0 (replicated residual
stream) and GPT_OSS_PREFILL_SP=1 (each device keeps S / 4 rows), and compares the final hidden states (all rows,
the all-gather end of the SP path) and the last-token logits tile (the tile-gather end):
  * SP with the MoE expert input gathered in bf16 (tt/mlp.py PREFILL_SP_EXPERT_INPUT_DTYPE): every op sees the same
    values as on the replicated path (residual adds and norms on local rows, the router on local rows), so the
    outputs must be bit-identical;
  * SP as committed (expert input gathered in bf8): that typecast rounds differently from the bf8 pack inside the
    expert op, so the outputs differ by rounding; PCC vs replicated >= COMMITTED_MIN_PCC.

Run: pytest models/demos/gpt_oss/tests/unit/test_prefill_sequence_parallel.py -k 1x4
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt import mlp as mlp_module
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.model import PREFILL_SP_ENV, PREFILL_SP_MIN_SEQ, Model
from models.demos.gpt_oss.utils.general_utils import get_default_num_links
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ...tt.experts.prefill_routed import ROUTED_PREFILL_ENV, ROUTED_PREFILL_NUM_EXPERTS, routed_prefill_mesh_supported
from ..test_factory import TestFactory, parametrize_mesh_with_fabric, skip_1x4_on_galaxy

SEQ_LEN = 1024
COMMITTED_EXPERT_INPUT_DTYPE = mlp_module.PREFILL_SP_EXPERT_INPUT_DTYPE
COMMITTED_MIN_PCC = 0.999


def _prefill(model, mesh_device, tokens, get_last_token, skip_lm_head):
    tt_tokens = ttnn.from_torch(
        tokens.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x = ttnn.embedding(tt_tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_tokens.deallocate(True)
    if len(x.shape) == 3:
        x = ttnn.unsqueeze_to_4D(x)
    out = model.ttnn_prefill_forward(
        x=x, user_id=0, page_table=None, kv_cache=None, get_last_token=get_last_token, skip_lm_head=skip_lm_head
    )
    # Hidden states are replicated over the mesh (take device 0); the LM head output is split over the vocab.
    dims = (0, -1) if not skip_lm_head else (0, 1)
    out_torch = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )
    out.deallocate(True)
    if skip_lm_head:
        out_torch = out_torch[:, :1]
    return out_torch


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_sequence_parallel_bit_identical(mesh_device, device_params, monkeypatch, reset_seeds):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    if not routed_prefill_mesh_supported(mesh_device):
        pytest.skip("sequence-parallel prefill needs the routed prefill experts (Blackhole, one mesh row)")
    assert SEQ_LEN >= PREFILL_SP_MIN_SEQ
    monkeypatch.setenv(ROUTED_PREFILL_ENV, "1")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    if config.num_local_experts != ROUTED_PREFILL_NUM_EXPERTS:
        pytest.skip(f"the routed prefill path is built for {ROUTED_PREFILL_NUM_EXPERTS} experts (gpt-oss-20b)")
    config.num_hidden_layers = 1
    config._attn_implementation = "eager"
    reference = GptOssForCausalLM(config)
    state_dict = convert_hf_qkv_to_meta_format(reference.state_dict(), config.head_dim)
    del reference

    mesh_shape = tuple(mesh_device.shape)
    model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=get_default_num_links(mesh_device)),
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0])),
        create_kv_cache=True,
        max_local_batch_size=1,
        use_throughput_experts=False,
    )

    # Count the sequence-parallel layer calls, so the SP=1 run is known to take that path.
    layer = model.layers[0]
    sp_calls = []
    sp_forward = layer._prefill_forward_sp

    def counted(*args, **kwargs):
        sp_calls.append(1)
        return sp_forward(*args, **kwargs)

    monkeypatch.setattr(layer, "_prefill_forward_sp", counted)

    tokens = torch.randint(0, config.vocab_size, (1, SEQ_LEN))
    last_tile = SEQ_LEN - 32

    def run(sp, expert_input_dtype):
        monkeypatch.setenv(PREFILL_SP_ENV, sp)
        monkeypatch.setattr(mlp_module, "PREFILL_SP_EXPERT_INPUT_DTYPE", expert_input_dtype)
        n = len(sp_calls)
        hidden = _prefill(model, mesh_device, tokens, get_last_token=-1, skip_lm_head=True)
        logits = _prefill(model, mesh_device, tokens, get_last_token=last_tile, skip_lm_head=False)
        assert len(sp_calls) - n == (2 if sp == "1" else 0), f"{PREFILL_SP_ENV}={sp}: SP layer calls {sp_calls}"
        return {"hidden": hidden, "logits": logits}

    replicated = run("0", COMMITTED_EXPERT_INPUT_DTYPE)
    # Expert input gathered in bf16: every op of the SP path sees the same values as the replicated path.
    sp_bf16 = run("1", ttnn.bfloat16)
    # As committed: the expert input is gathered in bf8 (MLP.prefill_sequence_parallel), a rounding change.
    sp_committed = run("1", COMMITTED_EXPERT_INPUT_DTYPE)

    for name, rep in replicated.items():
        assert torch.isfinite(rep.float()).all(), f"{name}: non-finite values on the replicated path"
        for label, sp, exact in (("bf16 expert input", sp_bf16[name], True), ("committed", sp_committed[name], False)):
            assert rep.shape == sp.shape, f"{name} {label}: {rep.shape} vs {sp.shape}"
            max_abs = (rep.float() - sp.float()).abs().max().item()
            _, pcc = comp_pcc(rep.float(), sp.float(), COMMITTED_MIN_PCC)
            logger.info(f"{name} {tuple(rep.shape)} SP ({label}) vs replicated: max abs diff {max_abs}, PCC {pcc}")
            if exact:
                assert torch.equal(rep, sp), f"{name}: SP ({label}) vs replicated not bit-identical (max {max_abs})"
            else:
                assert torch.isfinite(sp.float()).all(), f"{name}: non-finite values on the SP path ({label})"
                assert pcc >= COMMITTED_MIN_PCC, f"{name}: SP ({label}) vs replicated PCC {pcc} < {COMMITTED_MIN_PCC}"
