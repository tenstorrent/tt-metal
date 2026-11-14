# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.45 (owner 2026-11-11, MoE expert replication study): MOE_COUNTS=on records every MoE layer's expert counts per
prefill chunk (tt/moe_block_main_adapter.py _record_counts / flush_counts, called from TtDS41Model.sync).

The ordinary device prefill of a 56,320-token prompt in 11 chunks of the spec's chunk (5,120): every chunk from the
device's own state (no golden prefix), one model sync a chunk, which writes <MOE_COUNTS_DIR>/<MOE_COUNTS_TAG>_chunk<NN>
.json. WL selects the prompt:
  novel = the ladder's prompt (the s56320 rung's ids: $ART/input/prompt.json, A Tale of Two Cities as a model turn
          after 'Recite the Project Gutenberg text of A Tale of Two Cities by Charles Dickens, from the start.');
  code  = the first 56,320 tokens of CODE_TEXT (C++ sources of ttnn operations) as a model turn after 'Print the C++
          source of tt-metal TTNN operations, file by file.' (reference/prompt.py build_ids, the spec's wrap).
Checks: one file a chunk, written by this run; every layer that ran is in it; every full chunk's counts per layer are
non-negative integers summing to chunk x topk (30,720); the device concat's rows match one layer's buffer read alone.
Metric counts_chunks_<WL>: the chunk files this run wrote (11). Run bare (no DS41_MOE_COUNTS, as P.52's gate unit),
the test switches MOE_COUNTS on itself and writes the files into pytest's tmp_path.

    PYTHONPATH=$PWD DS41_MOE_COUNTS=on DS41_MOE_COUNTS_DIR=/scratch/dnijemcevic/replication/counts \\
        DS41_MOE_COUNTS_TAG=novel WL=novel DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all \\
        --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_moe_counts.py
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference import prompt as P
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec

S = spec()
SEQ, N_CHUNKS = 56320, 11
CODE_TEXT = "/scratch/dnijemcevic/moeplace/code.txt"
CODE_REQUEST = "Print the C++ source of tt-metal TTNN operations, file by file."
NOVEL_REQUEST = "Recite the Project Gutenberg text of A Tale of Two Cities by Charles Dickens, from the start."


def _ids(wl: str) -> list[int]:
    if wl == "novel":  # (the ladder's ids: its goldens are made from this file; the .bz2 source is not in the tree)
        rec = P.load(S)
        assert rec is not None, f"no prompt file {P.prompt_path(S)}"
        assert rec.get("request") == NOVEL_REQUEST and rec.get("wrap") == "model_turn", (
            rec.get("wrap"),
            rec["request"],
        )
        ids = rec["token_ids"]
    else:
        from models.demos.common.bringup.reference.golden import tokenizer

        sc = copy.copy(S)
        sc.data = copy.deepcopy(S.data)
        sc.data["text"].update(source=CODE_TEXT, request=CODE_REQUEST, wrap="model_turn")
        ids, info = P.build_ids(sc, tokenizer(S), SEQ)
        logger.info(f"code prompt: {info['source']} request {info['request']!r}, {info['prefix_tokens']} prefix tokens")
    assert len(ids) >= SEQ, (wl, len(ids))
    return list(ids[:SEQ])


@mesh_parametrize(timeout_s=2 * 3600)
def test_moe_counts(mesh_device, monkeypatch, tmp_path):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    wl = os.environ.get("WL", "novel")
    assert wl in ("novel", "code"), f"WL={wl}: novel or code"
    if not os.environ.get("DS41_MOE_COUNTS"):  # (a gate unit runs it bare (P.52): switch counting on, files in tmp)
        monkeypatch.setenv("DS41_MOE_COUNTS", "on")
        if not os.environ.get("DS41_MOE_COUNTS_DIR"):
            monkeypatch.setenv("DS41_MOE_COUNTS_DIR", str(tmp_path))
        monkeypatch.setenv("DS41_MOE_COUNTS_TAG", os.environ.get("DS41_MOE_COUNTS_TAG", wl))
        settings.thaw()
    assert settings.get("MOE_COUNTS") == "on", "set DS41_MOE_COUNTS=on"
    out_dir, tag = settings.get("MOE_COUNTS_DIR"), settings.get("MOE_COUNTS_TAG")
    assert out_dir and Path(out_dir).is_dir(), f"DS41_MOE_COUNTS_DIR={out_dir!r}: an existing directory"
    chunk = int(S.get("target.chunk"))
    assert chunk * N_CHUNKS == SEQ, (chunk, SEQ)
    tokens = torch.tensor(_ids(wl))
    layers = S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    state = model.new_state(SEQ)
    topk = int(model.cfg.topk_experts)
    t_start = time.time()
    for c in range(N_CHUNKS):
        t0, st = time.time(), c * chunk
        t = tokens[st : st + chunk]
        begin_chunk(model, t, st)
        h = model.embed(t)
        for i in layers:
            h2 = model.layer(i, h, st, state)
            model.free(h)
            h = h2
        if layers[-1] == S.num_layers - 1:
            h2 = model.final_norm(h)
            model.free(h)
            h = h2
        model.free(h)
        model.sync()  # (MOE_COUNTS: the chunk's counts file)
        logger.info(f"WL={wl} chunk {c} ({st}..{st + chunk}) {time.time() - t0:.1f} s")

    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    fails, written, last = [], 0, None
    for c in range(N_CHUNKS):
        path = Path(out_dir) / f"{tag}_chunk{c:02d}.json"
        if not path.exists() or path.stat().st_mtime < t_start:
            fails.append(f"chunk {c}: no file {path} from this run")
            continue
        written += 1
        rec = json.loads(path.read_text())
        last = rec
        if rec["chunk"] != c or rec["start"] != c * chunk:
            fails.append(f"chunk {c}: file says chunk {rec['chunk']} start {rec['start']}")
        if not rec["layers"]:
            fails.append(f"chunk {c}: no layer")
        spc = int(rec["slots_per_chip"])
        for L, v in rec["layers"].items():
            a = v["chips"]
            ok_vals = all(isinstance(x, int) and x >= 0 for row in a for x in row)
            n = sum(sum(row) for row in a)
            if not ok_vals:
                fails.append(f"chunk {c} layer {L}: a count is not a non-negative integer")
            if n != chunk * topk:
                fails.append(f"chunk {c} layer {L}: counts sum {n} != {chunk} x {topk}")
            off = sum(x for k, row in enumerate(a) for j, x in enumerate(row) if j // spc != k)
            if off:
                logger.warning(f"chunk {c} layer {L}: {off} counts outside the chips' own slots")
            if len(v["perm"]) != int(model.cfg.n_experts):
                fails.append(f"chunk {c} layer {L}: perm of {len(v['perm'])}")
        if c == 0:
            logger.info(f"chunk 0 layers {sorted(int(x) for x in rec['layers'])}")
    # (the device concat's order: one layer's buffer read alone == its rows in the last chunk's file)
    lay = next((l for l in _moe_layers(model) if l.counts_buf is not None), None)
    if lay is not None and last is not None and str(lay.layer) in last["layers"]:
        alone = ttnn.to_torch(lay.counts_buf, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        alone = alone.reshape(mesh_device.get_num_devices(), -1).to(torch.int64).tolist()
        if alone != last["layers"][str(lay.layer)]["chips"]:
            fails.append(f"layer {lay.layer}: the file's rows differ from its buffer read alone")
    else:
        fails.append("no MoE layer with a counts buffer (MOE_BLOCK_SRC=main?)")
    assert A.counts_on()
    metrics.record(f"counts_chunks_{wl}", written)
    for f in fails:
        logger.error(f"MOE_COUNTS: {f}")
    assert not fails, f"{len(fails)} failed checks: {fails[:5]}"


def _moe_layers(model):
    """The model's MoE layers (TtDS41Block.moe_block), in layer order."""
    return [b.moe_block for _, b in sorted(model.blocks.items()) if getattr(b, "moe_block", None) is not None]
