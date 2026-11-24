# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""DSpark drafter on card vs the torch reference, and greedy speculative decode == plain greedy.

Needs the weight cache + ``<cache dir>/mtp`` (python -m models.demos.inkling2_mini.convert.mtp), the checkpoint's
mtp.safetensors (``$INKLING_CKPT``) and the golden prompts (``$INKLING_GOLDEN``); skips without them.
"""

import os

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.demo.golden_check import GOLDEN
from models.demos.inkling2_mini.tests.mtp_reference import Reference
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.spec_generator import SpecGenerator, build

pytestmark = pytest.mark.skipif(
    GOLDEN is None or not (GOLDEN / "chat").exists() or not os.environ.get("INKLING_CKPT"),
    reason="golden prompts or checkpoint missing (set INKLING_GOLDEN and INKLING_CKPT)",
)


@pytest.fixture(scope="module")
def setup():
    dev = ttnn.open_device(device_id=0, trace_region_size=640 << 20)
    model, drafter = build(dev, 2048, log=lambda s: None)
    yield model, drafter
    ttnn.close_device(dev)


def test_drafter_vs_reference(setup):
    """Eager drafts on the TT model's own features agree with the torch reference (first token >= 80%)."""
    model, drafter = setup
    gen = SpecGenerator(model, drafter)
    ref = Reference()
    ids = torch.load(GOLDEN / "chat" / "input_ids.pt")[0].tolist()
    model.reset()
    seq, feats, out, first = list(ids), [], None, []
    for pos in range(len(ids) + 12):
        if pos >= len(ids):
            seq.append(gen.read_token(out))
            if pos > len(ids):
                first.append(gen._draft_eager(pos)[0] == ref.draft(seq, torch.stack(feats), pos)[0])
        model.set_inputs(seq[pos], pos)
        out = model.step(collect=True)
        feats.append(ttnn.to_torch(out["feats"]).float()[0, 0, :4].clone())
    assert sum(first) / len(first) >= 0.8, first


def test_spec_equals_greedy(setup):
    model, drafter = setup
    ids = torch.load(GOLDEN / "math" / "input_ids.pt")[0].tolist()
    spec = SpecGenerator(model, drafter)
    spec.capture()
    try:
        greedy, _ = Generator.generate(spec, ids, 48)
        new, st = spec.generate_spec(ids, 48, k=4)
    finally:
        spec.release()
    assert new == greedy
    assert st["passes"] < len(new)  # some drafts accepted
