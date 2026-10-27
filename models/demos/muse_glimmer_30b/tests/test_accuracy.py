# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: teacher-forcing top-1/top-5 accuracy of the full 52-layer model against a golden reference."""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import DEFAULT_CKPT_DIR, MuseGlimmerModelArgs

CKPT = os.environ.get("HF_MODEL") or DEFAULT_CKPT_DIR
GOLDEN = os.path.join(os.path.dirname(__file__), "reference_outputs", "tf_ref_photosynthesis.refpt")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")) or not os.path.exists(GOLDEN),
    reason="checkpoint or golden file not available",
)

MIN_TOP1 = 0.90
MIN_TOP5_CONTAINS = 0.98


def _top5(logits: torch.Tensor):
    return torch.topk(logits.float(), 5).indices.tolist()


@mesh_test_params(trace_region_size=200_000_000)
@pytest.mark.parametrize("use_trace", [False, True], ids=["eager", "trace"])
def test_teacher_forcing_accuracy(mesh_device, tp, use_trace):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel
    from models.demos.muse_glimmer_30b.tt.tokenizer_utils import decode, load_tokenizer

    gold = torch.load(GOLDEN)
    tokens = gold["tokens"].reshape(-1).to(torch.int64)
    prompt_len = int(gold["prompt_len"])
    g_ids = gold["top5_ids"]  # row t = predictions for token at position t+1 given tokens[:t+1]
    g_vals = gold["top5_vals"]
    n_tok = tokens.shape[0]

    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=sub, max_batch_size=1, max_seq_len=1024, max_prefill_len=1024
    )
    assert args.n_layers == 52
    model = MuseGlimmerModel(args, sub)

    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = ttnn.from_torch(
        pt,
        device=sub,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # rows[t] = logits predicting token t+1
    tt_top5 = {}
    tt_top1_text_ids = {}

    def record(t, logits):
        assert logits.shape[-1] == args.vocab_size
        ids = _top5(logits)
        tt_top5[t] = ids
        tt_top1_text_ids[t] = ids[0]

    # ---- prefill
    logits = model.prefill_forward(tokens[:prompt_len], 0, pt_tt)
    record(prompt_len - 1, logits)

    # ---- teacher-forced decode
    step_times = []
    dev_in = model.allocate_decode_inputs(1, nb)
    trace_id, trace_out = None, None
    for t in range(prompt_len, n_tok - 1):
        t_step = time.time()
        host_in = model.prepare_decode_inputs_host(tokens[t : t + 1], torch.tensor([t]), pt)
        model.copy_decode_inputs(host_in, dev_in)
        if use_trace and trace_id is not None:
            ttnn.execute_trace(sub, trace_id, cq_id=0, blocking=True)
            out_tt = trace_out
        else:
            out_tt = model.decode_forward(*dev_in)
            if use_trace:
                # eager step done (compiled); next steps replay a captured trace
                ttnn.synchronize_device(sub)
                ttnn.deallocate(out_tt)
                trace_id = ttnn.begin_trace_capture(sub, cq_id=0)
                trace_out = model.decode_forward(*dev_in)
                ttnn.end_trace_capture(sub, trace_id, cq_id=0)
                # the capture did not execute anything: run this step via the trace
                ttnn.execute_trace(sub, trace_id, cq_id=0, blocking=True)
                out_tt = trace_out
        logits_t = model.logits_to_torch(out_tt, 1)[0]
        step_times.append(time.time() - t_step)
        record(t, logits_t)
        if not use_trace:
            ttnn.deallocate(out_tt)

    if trace_id is not None:
        ttnn.release_trace(sub, trace_id)

    if len(step_times) > 3:
        steady = step_times[2:]
        logger.info(
            f"TIMING [{'trace' if use_trace else 'eager'}] decode step (host input copy + run + logits readback): "
            f"mean {sum(steady) / len(steady) * 1000:.1f} ms over {len(steady)} steps (first two excluded)"
        )

    # ---- metrics
    positions = sorted(tt_top5)
    assert positions == list(range(prompt_len - 1, n_tok - 1))
    top1_hits, g1_in_tt5, tt1_in_g5 = [], [], []
    mismatches = []
    for t in positions:
        g = g_ids[t].tolist()
        tt = tt_top5[t]
        top1_hits.append(tt[0] == g[0])
        g1_in_tt5.append(g[0] in tt)
        tt1_in_g5.append(tt[0] in g)
        if tt[0] != g[0]:
            mismatches.append((t, g, tt, float(g_vals[t, 0] - g_vals[t, 1])))
    n = len(positions)
    top1 = sum(top1_hits) / n
    top5_contains = sum(g1_in_tt5) / n
    tt1_in_top5 = sum(tt1_in_g5) / n

    logger.info("pos | golden top5 | tt top5 | golden top1-top2 gap")
    for t, g, tt, gap in mismatches:
        logger.info(f"{t:4d} | {g} | {tt} | {gap:.4f}")
    logger.info(
        f"[{'trace' if use_trace else 'eager'}] positions={n} top1={top1:.4f} "
        f"golden_top1_in_tt_top5={top5_contains:.4f} tt_top1_in_golden_top5={tt1_in_top5:.4f}"
    )

    try:
        tok = load_tokenizer(args.ckpt_dir)
        tt_text = decode(tok, [tt_top5[t][0] for t in positions])
        gold_text = decode(tok, [g_ids[t, 0].item() for t in positions])
        logger.info(f"TT greedy (teacher-forced) text:\n{tt_text}")
        logger.info(f"Golden greedy text:\n{gold_text}")
    except Exception as e:  # decoding is for eyeballing only
        logger.warning(f"tokenizer decode failed: {e}")

    assert top5_contains >= MIN_TOP5_CONTAINS, f"golden top-1 in TT top-5 {top5_contains:.4f} < {MIN_TOP5_CONTAINS}"
    assert top1 >= MIN_TOP1, f"top-1 accuracy {top1:.4f} < {MIN_TOP1}"
