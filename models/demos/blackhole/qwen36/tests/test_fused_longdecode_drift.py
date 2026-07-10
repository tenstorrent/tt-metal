# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-decode drift check for the fused GDN decode (bf16 recurrent state).

The fused kernel carries the gated-delta recurrent state in bf16 (the python path keeps fp32 for
decode-drift mitigation). Over a long decode, coarse bf16 quantization of decay=exp(g)~1.0 can
accumulate and cause late-generation repetition collapse. This generates N tokens (traced) and
reports a repetition metric over the tail so we can see whether the fused bf16-state path holds up
for serving-length decodes. Set QWEN_GDN_FUSED_DECODE=1 for fused; unset for the fp32 python path.

Run: QWEN_GDN_FUSED_DECODE=1 QWEN_LONGDECODE_N=256 QWEN_BENCH_B=8 MESH_DEVICE=P150x4 \
       HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_fused_longdecode_drift.py -v -s
"""
import os

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


def _tail_repetition(ids, window=64):
    """Fraction of REPEATED tokens in the last `window` (1 - unique/len). ~0=diverse, ~1=collapsed."""
    tail = ids[-window:]
    if not tail:
        return 0.0
    return 1.0 - len(set(tail)) / len(tail)


def _max_run(ids):
    """Longest run of an identical token (collapse indicator)."""
    best = run = 1
    for i in range(1, len(ids)):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, run)
    return best


@parametrize_mesh_tp()
def test_fused_longdecode_drift(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    B = int(os.environ.get("QWEN_BENCH_B", "8"))
    N = int(os.environ.get("QWEN_LONGDECODE_N", "256"))
    fused = os.environ.get("QWEN_GDN_FUSED_DECODE") == "1"
    tag = "FUSED_bf16" if fused else "PYTHON_fp32"

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=2048)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = tok("Write a short story about a robot learning to paint.", return_tensors="pt").input_ids[0].tolist()
    prompts = [prompt] * B

    outs = model.generate_tp_batched(prompts, max_new_tokens=N, use_trace=True)
    ids0 = outs[0]
    rep = _tail_repetition(ids0, 64)
    mx = _max_run(ids0)
    text = tok.decode(ids0)
    logger.info(f"LONGDECODE[{tag}] N={len(ids0)} tail_rep(64)={rep:.3f} max_run={mx}")
    logger.info(f"LONGDECODE[{tag}] head: {tok.decode(ids0[:60])!r}")
    logger.info(f"LONGDECODE[{tag}] tail: {tok.decode(ids0[-60:])!r}")
    # Collapse guard: a healthy decode has diverse tail (rep < ~0.6) and no long identical run.
    logger.info(f"LONGDECODE_SUMMARY[{tag}] tail_rep={rep:.3f} max_run={mx} verdict={'COLLAPSE' if (rep>0.7 or mx>=12) else 'OK'}")
