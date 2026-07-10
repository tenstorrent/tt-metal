# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched TP decode TRACE (generate_tp_batched use_trace=True) check (Step 2).

Captures the B-lane batched decode step ONCE as a ttnn trace and replays it with
DMA-updated inputs. The traced replay is bit-faithful to the eager batched decode (execute_trace re-issues
the identical program, and the traced path uses the SAME rot_mats_decode rope as eager),
so this asserts:

  * traced output == eager output, token-for-token, for every lane;
  * lane 0 reproduces the oracle first token 'Paris';
  * lanes are non-degenerate and independent.

Prompts have different lengths → the traced path carries per-user decode positions in a
fixed buffer (updated in place each step). On-device only; needs a 4-chip P150x4 mesh.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_batched_decode_trace_tp.py -v -s
"""
import os

from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_generate_tp_batched_traced(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    B = 2
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=256)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts_text = [
        "The capital of France is",
        "The largest ocean on Earth is",
    ]
    prompts = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts_text]
    logger.info(f"prompt lengths: {[len(p) for p in prompts]}")

    eager = model.generate_tp_batched(prompts, max_new_tokens=8, use_trace=False)
    traced = model.generate_tp_batched(prompts, max_new_tokens=8, use_trace=True)

    for b in range(B):
        logger.info(f"[lane {b}] eager : {tok.decode(prompts[b] + eager[b])!r}")
        logger.info(f"[lane {b}] traced: {tok.decode(prompts[b] + traced[b])!r}")
        assert len(set(traced[b])) > 1, f"lane {b} traced degenerate: {traced[b]}"

    # Core Step-2 assertion: the captured trace replay is bit-faithful to eager decode.
    assert traced == eager, f"traced != eager\n  eager={eager}\n  traced={traced}"

    # Lane 0 oracle + independence (same anchors as the eager Step-1 test).
    first0 = tok.decode([traced[0][0]]).strip()
    logger.info(f"[lane 0] first generated token: {first0!r}")
    assert first0 == "Paris", f"lane 0 expected 'Paris', got {first0!r}"
    assert traced[0] != traced[1], "lanes produced identical output — batch lanes not independent"
    logger.info("PASSED: batched decode trace — traced==eager, lane 0 correct, lanes independent")
