# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Continuous-batching generation (generate_tp_batched_continuous) check (Step 4, Part A).

Runs N=3 prompts through B=2 decode slots: the two slots fill with the first two prompts,
and when they finish the freed slot is reused in place for the third prompt (reseed_slot_tp).
Asserts:

  * the first-batch sequences (slots 0/1) match a plain batched run — the continuous-batching
    machinery does not perturb sequences already in flight;
  * the third sequence, generated in a REUSED slot, follows its prompt correctly (oracle
    first token 'Paris').

On-device only; needs a 4-chip P150x4 mesh.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_batched_continuous_tp.py -v -s
"""
import os

from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_generate_tp_batched_continuous(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    B = 2
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=256)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    ids = lambda s: tok(s, return_tensors="pt").input_ids[0].tolist()
    P0 = ids("The largest ocean on Earth is")
    P1 = ids("Photosynthesis converts sunlight into")
    P2 = ids("The capital of France is")  # reused slot → oracle 'Paris'
    STEPS = 6

    # Reference: first two prompts as a plain batched run.
    ref = model.generate_tp_batched([P0, P1], max_new_tokens=STEPS, use_trace=False)

    # Continuous: three prompts through two slots (slot reused for P2).
    cont = model.generate_tp_batched_continuous([P0, P1, P2], max_new_tokens=STEPS)
    assert len(cont) == 3

    for i, seq in enumerate(cont):
        logger.info(f"[seq {i}] {tok.decode(seq)!r}")
        assert len(set(seq)) > 1, f"seq {i} degenerate: {seq}"

    # First-batch sequences unperturbed by the continuous machinery.
    assert cont[0] == ref[0], f"seq 0 differs from plain batched\n  ref={ref[0]}\n  cont={cont[0]}"
    assert cont[1] == ref[1], f"seq 1 differs from plain batched\n  ref={ref[1]}\n  cont={cont[1]}"

    # Reused slot ran P2 correctly.
    first2 = tok.decode([cont[2][0]]).strip()
    logger.info(f"[seq 2] first token (reused slot): {first2!r}")
    assert first2 == "Paris", f"reused-slot seq 2 expected 'Paris', got {first2!r}"
    logger.info("PASSED: continuous batching — first-batch unperturbed, reused slot correct")
