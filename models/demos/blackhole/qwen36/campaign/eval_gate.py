# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for the Qwen3.8-27B campaign — the precision ladder's go/no-go.

Two independent, self-contained stages (no external eval service), both running
the model's own TP serving path (chunked per-user prefill + traced decode with
the per-shard greedy readback — the same machinery bench_decode measures):

1. test_gpqa_diamond_10 — GPQA-diamond 10-doc subset, free generation, letter
   extraction, accuracy over 10. Includes the NON-EMPTY-RESPONSE assert: the
   known failure mode of a collapsed serving path is empty/degenerate replies
   masquerading as a low score (fmf autoport lesson), so an empty reply FAILS
   the run rather than scoring 0.
2. test_top1_agreement — the cheap per-rung gate: N teacher-forced decode steps
   over a fixed corpus, recording the greedy top-1 at every step. Two configs
   are compared by agreement % (and top-1 sha256 for eyeballing across logs):
   run the reference config with QWEN38_EVAL_DUMP_REF=/path once, then each
   precision rung with QWEN38_EVAL_REF=/path. Teacher forcing keeps the token
   prefix identical across configs, so disagreement measures pure numerics.

Both stages emit one greppable EVAL_JSON line (same record shape as BENCH_JSON;
run_eval.sbatch appends them to eval_results.jsonl).

GPQA data (offline-friendly): the gate reads a 10-doc JSON from
QWEN38_GPQA_PATH (default /data/ayerofieiev/qwen38/eval_data/gpqa_diamond_10.json)
and SKIPS with instructions when absent. GPQA is HF-gated and cluster egress is
slow, so fetch the tiny subset once on a workstation and rsync it:

    pip install datasets  # and `huggingface-cli login` + accept Idavidrein/gpqa terms
    python - <<'EOF'
    import json, random
    from datasets import load_dataset
    rows = sorted(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"),
                  key=lambda r: r["Record ID"])[:10]
    docs = []
    for r in rows:
        rng = random.Random(r["Record ID"])          # deterministic per-doc shuffle
        ch = [r["Correct Answer"], r["Incorrect Answer 1"],
              r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4)); rng.shuffle(order)
        docs.append({"id": r["Record ID"], "question": r["Question"].strip(),
                     "choices": [ch[i].strip() for i in order],
                     "answer": "ABCD"[order.index(0)]})
    json.dump(docs, open("gpqa_diamond_10.json", "w"), indent=1)
    EOF
    rsync gpqa_diamond_10.json <exabox>:/data/ayerofieiev/qwen38/eval_data/

Run (P150x8):
    MESH_DEVICE=P150x8 HF_MODEL=/path/to/qwen38-27b-weights \\
        pytest models/demos/blackhole/qwen36/campaign/eval_gate.py -v -s

Knobs (env):
    QWEN38_GPQA_PATH       10-doc JSON path (see fetch recipe above)
    QWEN38_EVAL_MAX_NEW    generation cap per GPQA doc (default 768)
    QWEN38_EVAL_TF_ISL     teacher-forced prefix length (default 512)
    QWEN38_EVAL_TF_STEPS   teacher-forced steps (default 200)
    QWEN38_EVAL_DUMP_REF   write this run's top-1 list to a JSON file (reference config)
    QWEN38_EVAL_REF        compare this run's top-1 list against a dumped reference
"""

import hashlib
import json
import os
import re
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.campaign.bench_common import (
    bench_prompt,
    git_ref,
    restore_gdn_tp_staged,
    snapshot_gdn_tp,
    stage_gdn_tp,
)
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, DEVICE_PARAMS, _MESH_SHAPE, _MULTI
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device

_GPQA_PATH = os.environ.get("QWEN38_GPQA_PATH", "/data/ayerofieiev/qwen38/eval_data/gpqa_diamond_10.json")
_MAX_NEW = int(os.environ.get("QWEN38_EVAL_MAX_NEW", "768"))
_TF_ISL = int(os.environ.get("QWEN38_EVAL_TF_ISL", "512"))
_TF_STEPS = int(os.environ.get("QWEN38_EVAL_TF_STEPS", "200"))
_B = 8


def emit_eval_json(kind, config, metrics):
    """One greppable EVAL_JSON line, mirroring bench_common.emit_bench_json."""
    record = {
        "kind": kind,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ref": git_ref(),
        "mesh": os.environ.get("MESH_DEVICE", "P150x4"),
        "hf_model": os.environ.get("HF_MODEL", ""),
        "config": config,
        "metrics": metrics,
    }
    print("EVAL_JSON " + json.dumps(record), flush=True)
    return record


class _ServedDecoder:
    """The serving-path generation loop the gates share.

    Owns the model + paged KV, prefills a batch of prompts through the per-user
    chunked path, then steps the SAME traced decode + per-shard greedy readback
    bench_decode uses (host staging every step, so teacher forcing works).
    """

    def __init__(self, mesh_device, bpu):
        self.bpu = bpu
        self.max_seq_len = bpu * BLOCK_SIZE
        self.model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=_B, max_seq_len=self.max_seq_len)
        assert self.model.sampling is not None, "eval gate needs the on-device sampler topology (1x4/1x8)"
        m = self.model
        self.mesh = m.mesh_device
        self.vocab = m.args.vocab_size
        self.nd = m.num_devices
        self.per_shard = self.vocab // self.nd
        kv_shape = [_B * bpu, m.args.n_local_kv_heads, BLOCK_SIZE, m.args.head_dim]
        m.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=_B)
        self.page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(_B)])
        self._comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        c = 32
        self._mv_r = (((self.per_shard + c - 1) // c) + 31) // 32 * 32
        self._mv_c = c

    def prefill(self, token_rows, valid_lens):
        """token_rows: list of B [1, T_u] int tensors. Returns first greedy token per user."""
        logits = self.model.prefill_chunked_peruser(token_rows, self.page_table, valid_lens=valid_lens)
        ttnn.synchronize_device(self.mesh)
        return [
            int(ttnn.to_torch(logits[u], mesh_composer=self._comp).reshape(-1, self.vocab)[0].float().argmax())
            for u in range(_B)
        ]

    def _argmax_dev(self, sharded_logits):
        Bn = sharded_logits.shape[2]
        logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        padded = ttnn.pad(
            sharded_logits, [(0, 0), (0, 0), (0, 0), (0, self._mv_r * self._mv_c - self.per_shard)], value=-1e30
        )
        grid = ttnn.reshape(padded, (1, Bn, self._mv_r, self._mv_c))
        part = ttnn.max(grid, dim=-1)
        part_row = ttnn.reshape(part, (1, 1, Bn, self._mv_r))
        val = ttnn.max(part_row, dim=-1)
        for t in (padded, grid, part, part_row):
            ttnn.deallocate(t)
        return idx, val

    def _pick(self, idx_t, val_t):
        Bn = idx_t.shape[-1]
        idxs = ttnn.to_torch(idx_t, mesh_composer=self._comp).reshape(self.nd, Bn)[:, :_B].to(torch.int64)
        vals = ttnn.to_torch(val_t, mesh_composer=self._comp).reshape(self.nd, Bn)[:, :_B]
        d = torch.argmax(vals, dim=0)
        return (d * self.per_shard + idxs[d, torch.arange(_B)]).tolist()

    def decode_loop(self, first_tokens, start_pos, num_steps, forced=None, stop_ids=None):
        """Traced greedy decode for num_steps (or until every row stops).

        forced: optional list of token ids — teacher forcing: step i's INPUT for
        every row is forced[i] (forced[0] replaces first_tokens), and the return
        value is the model's greedy top-1 at each step, NOT the fed sequence.
        Returns per-row generated/top-1 token lists (num_steps long unless all
        rows hit stop_ids earlier).
        """
        m = self.model
        dev = m.prepare_inputs_decode(
            torch.tensor(first_tokens, dtype=torch.int32).reshape(_B, 1),
            torch.tensor(start_pos, dtype=torch.int32),
            page_table=self.page_table,
        )

        def _fwd():
            return m.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)

        def _update(tokens_row, positions):
            host = m.prepare_decode_inputs_host(
                torch.tensor(tokens_row, dtype=torch.int32).reshape(_B, 1),
                torch.tensor(positions, dtype=torch.int32),
                page_table=None,
            )
            copy_host_to_device(host[:3], device_tensors=dev[:3])

        # Trace capture runs the forward twice; GDN state is non-idempotent, so
        # snapshot/stage/restore around it (the bench_decode discipline).
        snap = snapshot_gdn_tp(m)
        staging = stage_gdn_tp(m, snap)
        warm = _fwd()
        wi, wv = self._argmax_dev(warm)
        ttnn.deallocate(wi)
        ttnn.deallocate(wv)
        trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        tt_logits = _fwd()
        tt_idx, tt_val = self._argmax_dev(tt_logits)
        ttnn.end_trace_capture(self.mesh, trace_id, cq_id=0)
        restore_gdn_tp_staged(m, staging)

        out = [[] for _ in range(_B)]
        done = [False] * _B
        cur = list(first_tokens) if forced is None else [forced[0]] * _B
        pos = list(start_pos)
        for i in range(num_steps):
            _update(cur, pos)
            ttnn.execute_trace(self.mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            toks = self._pick(tt_idx, tt_val)
            for u in range(_B):
                if not done[u]:
                    out[u].append(toks[u])
                    if stop_ids and toks[u] in stop_ids:
                        done[u] = True
            pos = [p + 1 for p in pos]
            if forced is not None:
                cur = [forced[i + 1]] * _B if i + 1 < num_steps else cur
            else:
                cur = toks
                if all(done):
                    break
        ttnn.release_trace(self.mesh, trace_id)
        del staging
        return out


def _extract_letter(text):
    """Answer letter from a reply; reasoning (<think>) is skipped first."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    m = re.findall(r"[Aa]nswer[^A-D]{0,10}([ABCD])\b", text)
    if m:
        return m[-1]
    m = re.findall(r"\(([ABCD])\)|\b([ABCD])\b", text)
    if m:
        return next(x for x in m[-1] if x)
    return None


def _doc_prompt(doc, tokenizer):
    lines = [doc["question"], ""]
    for letter, choice in zip("ABCD", doc["choices"]):
        lines.append(f"({letter}) {choice}")
    lines.append("")
    lines.append('Choose the correct answer. End your reply with "Answer: <letter>".')
    msgs = [{"role": "user", "content": "\n".join(lines)}]
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return torch.tensor(ids, dtype=torch.int32).reshape(1, -1)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gpqa_diamond_10(mesh_device):
    if not _MULTI:
        pytest.skip("eval gate runs the TP serving path; set MESH_DEVICE=P150x4 or P150x8")
    if not os.path.isfile(_GPQA_PATH):
        pytest.skip(
            f"GPQA subset not found at {_GPQA_PATH} — fetch it once on a workstation and rsync "
            "(see the module docstring for the exact recipe); the top1-agreement gate still runs."
        )
    docs = json.load(open(_GPQA_PATH))
    assert len(docs) >= 1 and all({"question", "choices", "answer"} <= set(d) for d in docs)

    from transformers import AutoTokenizer

    # Rounds of B slots; short rounds are padded by cycling docs (pads are not scored).
    n_docs = len(docs)
    padded = [docs[i % n_docs] for i in range(-(-n_docs // _B) * _B)]

    # Block budget needs the worst prompt length BEFORE the model exists, so the
    # tokenizer is loaded from HF_MODEL directly (the campaign env contract).
    probe_tok = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"], trust_remote_code=True)
    max_prompt = max(_doc_prompt(d, probe_tok).shape[1] for d in padded)
    bpu = ((max(8, -(-(max_prompt + _MAX_NEW + 16) // BLOCK_SIZE)) + 7) // 8) * 8

    srv = _ServedDecoder(mesh_device, bpu)
    tokenizer = AutoTokenizer.from_pretrained(srv.model.args.CKPT_DIR, trust_remote_code=True)
    stop_ids = {tokenizer.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0:
            stop_ids.add(tid)

    responses = []
    for r0 in range(0, len(padded), _B):
        batch = padded[r0 : r0 + _B]
        rows = [_doc_prompt(d, tokenizer) for d in batch]
        lens = [int(t.shape[1]) for t in rows]
        first = srv.prefill(rows, lens)
        gen = srv.decode_loop(first, lens, _MAX_NEW - 1, stop_ids=stop_ids)
        for u in range(_B):
            ids = [first[u]] + gen[u]
            if ids and ids[-1] in stop_ids:
                ids = ids[:-1]
            responses.append(tokenizer.decode(ids))
    responses = responses[:n_docs]

    # Serving-collapse guard BEFORE scoring: empty replies must fail, not score 0.
    empty = [i for i, t in enumerate(responses) if not t.strip()]
    n_correct = n_parsed = 0
    for doc, text in zip(docs, responses):
        letter = _extract_letter(text)
        n_parsed += letter is not None
        n_correct += letter == doc["answer"]
        logger.info(f"[gpqa] doc={doc.get('id')} gold={doc['answer']} got={letter} chars={len(text)}")

    metrics = {
        "accuracy": round(n_correct / n_docs, 4),
        "n_docs": n_docs,
        "n_correct": n_correct,
        "n_parsed": n_parsed,
        "empty_responses": len(empty),
        "mean_response_chars": round(sum(len(t) for t in responses) / n_docs, 1),
    }
    emit_eval_json("gpqa_diamond_10", {"max_new": _MAX_NEW, "batch": _B, "gpqa_path": _GPQA_PATH}, metrics)
    assert not empty, f"EMPTY RESPONSES for docs {empty} — serving collapse, score is not trustworthy"
    assert n_parsed >= max(1, n_docs // 2), f"only {n_parsed}/{n_docs} replies contained an answer letter"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_top1_agreement(mesh_device):
    if not _MULTI:
        pytest.skip("eval gate runs the TP serving path; set MESH_DEVICE=P150x4 or P150x8")
    bpu = ((max(8, -(-(_TF_ISL + _TF_STEPS + 16) // BLOCK_SIZE)) + 7) // 8) * 8
    srv = _ServedDecoder(mesh_device, bpu)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(srv.model.args.CKPT_DIR, trust_remote_code=True)
    corpus = bench_prompt(_TF_ISL + _TF_STEPS + 1, tokenizer)[0].tolist()

    prompt = torch.tensor(corpus[:_TF_ISL], dtype=torch.int32).reshape(1, -1)
    srv.prefill([prompt for _ in range(_B)], [_TF_ISL] * _B)
    forced = corpus[_TF_ISL : _TF_ISL + _TF_STEPS]
    tf_out = srv.decode_loop([forced[0]] * _B, [_TF_ISL] * _B, _TF_STEPS, forced=forced)

    top1 = tf_out[0]
    rows_identical = all(row == top1 for row in tf_out)
    digest = hashlib.sha256(json.dumps(top1).encode()).hexdigest()[:16]

    dump = os.environ.get("QWEN38_EVAL_DUMP_REF")
    ref_path = os.environ.get("QWEN38_EVAL_REF")
    agreement = first_div = None
    if dump:
        os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
        json.dump({"ref": git_ref(), "isl": _TF_ISL, "steps": _TF_STEPS, "top1": top1}, open(dump, "w"))
        logger.info(f"[agreement] reference top-1 stream dumped to {dump}")
    if ref_path:
        ref = json.load(open(ref_path))
        assert ref["isl"] == _TF_ISL and ref["steps"] == _TF_STEPS, "reference was dumped with different TF config"
        matches = [a == b for a, b in zip(top1, ref["top1"])]
        agreement = round(100.0 * sum(matches) / len(matches), 2)
        first_div = matches.index(False) if False in matches else None

    metrics = {
        "n_steps": _TF_STEPS,
        "isl": _TF_ISL,
        "top1_sha256": digest,
        "rows_identical": rows_identical,
        "agreement_pct": agreement,
        "first_divergence": first_div,
        "ref_file": ref_path,
        "dumped_ref": dump,
    }
    emit_eval_json("top1_agreement", {"batch": _B, "teacher_forced": True}, metrics)
    assert rows_identical, "identical forced rows produced different top-1 streams — nondeterminism in the step"
    if agreement is not None:
        logger.info(f"[agreement] {agreement}% over {_TF_STEPS} steps (first divergence: {first_div})")
