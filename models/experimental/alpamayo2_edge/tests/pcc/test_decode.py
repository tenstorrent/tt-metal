# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced TT CoT decode against the torch reference on one Blackhole P100."""

import pytest
import torch
from loguru import logger

from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.forced_cot import driving_text_cot
from models.experimental.alpamayo2_edge.reference.inputs import prepare_sample_inputs, synthetic_sample
from models.experimental.alpamayo2_edge.reference.model import ReferenceModel
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tests.conftest import P100_MESH
from models.experimental.alpamayo2_edge.tt.generate import generate_cot
from models.experimental.alpamayo2_edge.tt.text import AlpamayoCache, TtLanguageModel
from models.tt_dit.utils import tensor

PCC = 0.99


def _pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    stacked = torch.stack([golden.flatten().double(), actual.flatten().double()])
    return torch.corrcoef(stacked)[0, 1].item()


def _reference_forced_decode(cfg, inputs, forced: torch.Tensor) -> dict:
    """The vision embeddings the LM consumed and the teacher-forced CoT with its per-step logits."""
    model = ReferenceModel.from_pretrained(cfg)
    captured = {}
    model.vision.register_forward_hook(lambda _m, _a, out: captured.update(vision=out))
    steps = []

    def progress(_m, _a, _out):
        steps.append(None)
        print(f"reference LM forward {len(steps)}", flush=True)

    model.lm.text.register_forward_hook(progress)
    _, cache = model.prefill(inputs)
    cot = model.decode(inputs, cache, forced=forced)
    return {"vision": captured["vision"], "tokens": cot.tokens, "offsets": cot.offsets, "logits": cot.logits}


@P100_MESH
@pytest.mark.timeout(1500)
def test_decode_teacher_forced(mesh_device, cfg):
    inputs = prepare_sample_inputs(cfg, synthetic_sample(0))
    forced = driving_text_cot(cfg)[None]
    golden = golden_cache.cached(
        "lm_forced_decode",
        {
            "input_ids": inputs.input_ids,
            "position_ids": inputs.position_ids,
            "patches": inputs.patches,
            "grid": inputs.image_grid_thw,
            "forced": forced,
        },
        lambda: _reference_forced_decode(cfg, inputs, forced),
        cfg=cfg,
    )
    seq = inputs.input_ids.shape[1]
    steps = forced.shape[1]
    assert torch.equal(golden["tokens"], forced)
    logger.info(f"S={seq}, {steps} forced CoT tokens, {cfg.text.layers} layers")

    model = TtLanguageModel(TtLanguageModel.config_from(cfg.text), device=mesh_device)
    model.load_torch_state_dict(
        TtLanguageModel.convert_state(
            load_weights(cfg.model_dir, "vlm.model.language_model."), load_weights(cfg.model_dir, "vlm.lm_head.")
        )
    )
    cache = AlpamayoCache(device=mesh_device, size=AlpamayoCache.size_for(seq), batch_size=1)

    cot = generate_cot(
        model,
        inputs,
        tensor.from_torch(golden["vision"], device=mesh_device),
        tokens=cfg.tokens,
        cache=cache,
        forced=forced,
        return_logits=True,
    )

    assert torch.equal(cot.tokens, forced)
    assert cot.logits.shape == golden["logits"].shape == (1, steps, cfg.text.vocab)
    pccs = torch.tensor([_pcc(golden["logits"][:, t], cot.logits[:, t]) for t in range(steps)])
    argmax_equal = (golden["logits"].argmax(-1) == cot.logits.argmax(-1)).float().mean().item()
    logger.info(
        f"per-step logits PCC min {pccs.min() * 100:.4f} % (step {int(pccs.argmin())}), "
        f"mean {pccs.mean() * 100:.4f} %, step 0 {pccs[0] * 100:.4f} %, last {pccs[-1] * 100:.4f} %; "
        f"argmax agreement {argmax_equal * 100:.1f} %"
    )

    assert cot.offsets.tolist() == golden["offsets"].tolist() == [seq + steps]
    assert cot.prefix_length == seq + steps
    assert cot.cache is cache
    assert int(tensor.to_torch(cache.position)[0]) == cot.prefix_length
    failing = (pccs < PCC).nonzero().flatten().tolist()
    assert not failing, f"steps {failing} have logits PCC below {PCC}: {pccs[failing].tolist()}"
