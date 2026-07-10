# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched TP generation (``generate_tp_batched``) functional check (Step 1).

Runs the full 27B TP model with B = max_batch_size = 2 sequences decoded in
lockstep. Each prompt is prefilled into its own batch lane (``prefill_seed_tp``),
the GDN recurrent/conv state is assembled with ``finalize_seed``, then all lanes
decode together (one batched forward per step). Asserts:

  * lane 0 (the France prompt) still yields the oracle first token 'Paris' —
    i.e. batching did not corrupt lane 0's KV / GDN state, and cross-lane
    contamination did not occur;
  * every lane is non-degenerate;
  * the two lanes produce different continuations (independent state).

This anchors the batched oracle path against test_generate_tp.py (which proves the
B=1 France prompt yields 'Paris'). On-device only; needs a 4-chip P150x4 mesh.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_batched_decode_tp.py -v -s
"""
import os

from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_generate_tp_batched(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    B = 2
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=256)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts_text = [
        "The capital of France is",
        "Water is made of hydrogen and",
    ]
    prompts = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts_text]

    out = model.generate_tp_batched(prompts, max_new_tokens=8)
    assert len(out) == B

    for b, new_ids in enumerate(out):
        text = tok.decode(prompts[b] + new_ids)
        logger.info(f"[lane {b}] GENERATED: {text!r}")
        assert len(set(new_ids)) > 1, f"lane {b} degenerate: {new_ids}"

    # Lane 0 must reproduce the B=1 oracle answer (see test_generate_tp.py).
    first0 = tok.decode([out[0][0]]).strip()
    logger.info(f"[lane 0] first generated token: {first0!r}")
    assert first0 == "Paris", f"lane 0 expected 'Paris', got {first0!r} (batched seeding/decode corrupted lane 0)"

    # The two lanes decode independently → different continuations.
    assert out[0] != out[1], "lanes produced identical output — batch lanes are not independent"
    logger.info("PASSED: batched generate_tp_batched — lane 0 correct, lanes independent")
