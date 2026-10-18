# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: BABILong needle-in-a-haystack (RMT-team/babilong, qa1 / qa2: "Where is X?" -> a room) on the device model,
prefill only - no decode. Each sample's prompt (BABILong's instruction, the haystack with the facts, the question, an
answer prefix "X is in the") is prefilled from position 0 in chunks; the next-token logits at the last real row are
scored: the target room's rank among the 6 rooms (each one token; chance 1/6) and whether it is the overall top-1.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec_babilong.yaml \\
    scripts/run_safe_pytest.sh --mesh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_babilong.py -s

GLM_BL_DIR (default /localdev/$USER/bringup/babilong: data/<task>/<length>.json), GLM_BL_TASKS (default qa1),
GLM_BL_LENS (default 0k,4k,16k,32k), GLM_BL_N (samples per bucket, default 20). The spec's target.seq must cover the
longest prompt (spec_babilong.yaml: 256000)."""

import json
import os
import re
import time

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.smoke import tokenizer

S = spec()
pytestmark = device_timeout(S)
ROOMS = ["bathroom", "hallway", "kitchen", "office", "garden", "bedroom"]
INSTRUCTION = (
    "I will give you context with the facts about positions of different persons hidden in some random text and a "
    "question. You need to answer the question based only on the information from the facts. If a person was in "
    "different locations, use the latest location to answer the question.\n\n"
)


def _prompt(row):
    q = row["question"].strip()
    m = re.match(r"Where is (.+)\?$", q)
    assert m, q
    subj = m.group(1)
    subj = subj[0].upper() + subj[1:]
    return f"{INSTRUCTION}<context>\n{row['input'].strip()}\n</context>\n\nQuestion: {q}\nAnswer: {subj} is in the"


@mesh_parametrize
def test_babilong(mesh_device):
    d = os.environ.get("GLM_BL_DIR", f"/localdev/{os.environ['USER']}/bringup/babilong")
    tasks = os.environ.get("GLM_BL_TASKS", "qa1").split(",")
    lens = os.environ.get("GLM_BL_LENS", "0k,4k,16k,32k").split(",")
    n = int(os.environ.get("GLM_BL_N", "20"))
    chunk = int(S.get("target.chunk"))
    max_seq = int(S.get("target.seq"))
    tok = tokenizer(S)
    room_ids = []
    for r in ROOMS:
        ids = tok(" " + r, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (r, ids)
        room_ids.append(ids[0])
    t0 = time.time()
    model = S.hooks().device_model(mesh_device, S, S.layers(), lm_head=True)
    print(f"[bl] model loaded in {time.time() - t0:.0f} s (max_seq {max_seq}, chunk {chunk})", flush=True)
    summary = []
    for task in tasks:
        for length in lens:
            rows = json.load(open(f"{d}/data/{task}/{length}.json"))[:n]
            hit_room = hit_top1 = done = 0
            ranks, toks, t_bucket = [], [], time.time()
            for i, row in enumerate(rows):
                ids = tok(_prompt(row), add_special_tokens=False)["input_ids"]
                L = len(ids)
                if L > max_seq:
                    print(f"[bl] {task} {length} #{i}: {L} tokens > max_seq {max_seq}, skipped", flush=True)
                    continue
                padded = torch.tensor(ids + [0] * (-L % chunk), dtype=torch.long)
                hid = None
                for start in range(0, len(padded), chunk):
                    h = model.embed(padded[start : start + chunk])
                    for li in S.layers():
                        h2 = model.layer(li, h, start, None)
                        model.free(h)
                        h = h2
                    if start + chunk >= L:  # (the chunk with the last real row)
                        hid = model.final_norm(h)
                    model.free(h)
                lg = model.logits(hid, [(L - 1) % chunk]).float().reshape(-1)
                model.free(hid)
                target = ROOMS.index(row["target"].strip())
                sc = lg[room_ids]
                rank = int((sc > sc[target]).sum())
                top = int(lg.argmax())
                hit_room += rank == 0
                hit_top1 += top == room_ids[target]
                done += 1
                ranks.append(rank)
                toks.append(L)
                print(
                    f"[bl] {task} {length} #{i}: {L} tokens, target {row['target'].strip()}, room pick "
                    f"{ROOMS[int(sc.argmax())]} (target rank {rank}), top-1 '{tok.decode([top])}'",
                    flush=True,
                )
            if done:
                line = (
                    f"[bl] SUMMARY {task} {length}: room acc {hit_room / done:.2f} ({hit_room}/{done}), strict top-1 "
                    f"{hit_top1 / done:.2f}, mean target rank {sum(ranks) / done:.2f}, ~{sum(toks) // done} tokens, "
                    f"{(time.time() - t_bucket) / done:.1f} s a sample"
                )
                print(line, flush=True)
                summary.append(line)
    print("\n".join(summary), flush=True)
