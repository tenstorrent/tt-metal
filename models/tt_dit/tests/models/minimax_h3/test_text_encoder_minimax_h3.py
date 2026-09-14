# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# MiniMax-H3 text conditioner, t2va (text-only) scope: the *unnormalized* hidden_states[50]
# of the 64-layer Qwen3-VL decoder (no LM head, no final norm), vs the HF reference on the
# released weights. Large-host test: ~62 GiB of shards and RAM; skips when unavailable.

import dataclasses

import pytest
import torch
import transformers
from loguru import logger

import ttnn

from ....encoders.qwen3vl.loader_minimax_h3 import MINIMAX_H3_TEXT_ENCODER_LAYER as TAP
from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlEncoder
from ....parallel.config import EncoderParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from .common import CONDITIONER_SUBFOLDER, conditioner_checkpoint_dir, load_reference_conditioner

# scoped: only these text_encoder files must be present for the test to run
_PATTERNS = [
    f"{CONDITIONER_SUBFOLDER}/config.json",
    f"{CONDITIONER_SUBFOLDER}/*.safetensors",
    f"{CONDITIONER_SUBFOLDER}/model.safetensors.index.json",
]


def _reference_lm(path: str):
    hf = load_reference_conditioner(path)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    return lm.eval()


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp"),
    [
        pytest.param((4, 8), (4, 8), (8, 1), id="tp8_axis1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize("seq_len", [128])
def test_minimax_h3_text_conditioner(
    *, mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], tp: tuple[int, int], seq_len: int
) -> None:
    """The layer-50 tap MiniMax-H3 conditions on, under TP, against the HF golden."""
    torch.manual_seed(0)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    path = conditioner_checkpoint_dir(_PATTERNS)
    text_config = transformers.AutoConfig.from_pretrained(path).text_config
    assert text_config.num_hidden_layers > TAP, (
        f"MiniMax-H3 conditions on hidden_states[{TAP}], which needs more than {TAP} decoder layers, "
        f"but the conditioner config declares {text_config.num_hidden_layers}."
    )

    lm = _reference_lm(path)
    cfg = lm.config
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    with torch.no_grad():
        out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True)
    golden = out.hidden_states[TAP].float()
    assert golden.shape == (1, seq_len, cfg.hidden_size)

    # The reference's language model carries no head.
    config = dataclasses.replace(Qwen3VlEncoder.config_from_hf(cfg), final_linear=False)
    encoder = Qwen3VlEncoder(
        config,
        device=submesh,
        parallel_config=EncoderParallelConfig.from_tuples(tp=tp, sp=None),
        ccl_manager=CCLManager(submesh, num_links=2, topology=ttnn.Topology.Linear),
    )
    encoder.load_torch_state_dict(
        Qwen3VlEncoder.convert_state({f"model.language_model.{k}": v for k, v in lm.state_dict().items()})
    )

    tt_ids = tensor.from_torch(ids, device=submesh, dtype=ttnn.uint32)
    hidden_states = encoder.forward(tt_ids, skip_final_linear=True, output_hidden_states=True)
    actual = tensor.to_torch(hidden_states[TAP])

    logger.info(f"minimax-h3 conditioner TP={tp[0]} axis={tp[1]} layer {TAP} of {cfg.num_hidden_layers}:")
    assert actual.shape[-2:] == (seq_len, cfg.hidden_size)
    assert_quality(golden, actual, pcc=0.99)

    assert not torch.allclose(golden, out.last_hidden_state.float(), atol=1e-2), (
        f"layer {TAP} output is indistinguishable from the normalized final hidden state; the tap is "
        "not exercising the mid-stack read MiniMax-H3 depends on."
    )
