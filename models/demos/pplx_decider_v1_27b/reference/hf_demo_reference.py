# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF bf16 reference answers for the demo rows (the snapshot ``inference.py`` text example).

Streams the 64 HF decoder layers one at a time on CPU (``hf_decision_golden.stream_layers``) for the
demo's two decisions and writes ``Decider.predict``-shaped answers, so the TT demo output can be
compared against HF without loading the 54 GB model::

    python -m models.demos.pplx_decider_v1_27b.reference.hf_demo_reference --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.demos.pplx_decider_v1_27b.demo.decider import DEMO_STATE, demo_questions
from models.demos.pplx_decider_v1_27b.reference.decision_prompts import DEFAULT_SNAPSHOT, AppTokenizer
from models.demos.pplx_decider_v1_27b.reference.hf_decision_golden import (
    decision_head,
    embed_prompts,
    position_embeddings,
    stream_layers,
)
from models.demos.pplx_decider_v1_27b.reference.hf_reference import (
    SnapshotReader,
    build_decoder_layer,
    build_final_norm,
    build_readout,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.time()
    tok = AppTokenizer(DEFAULT_SNAPSHOT)
    reader = SnapshotReader(DEFAULT_SNAPSHOT)
    questions = demo_questions()
    rows = {name: {"state": DEMO_STATE, "question": q} for name, q in questions.items()}
    ids = {name: tok.input_ids(row) for name, row in rows.items()}
    dtype = torch.bfloat16
    hs = embed_prompts(reader, list(ids.values()), dtype)
    pes = [position_embeddings(reader.text_config, len(v), dtype) for v in ids.values()]
    stream_layers(hs, pes, lambda i: build_decoder_layer(reader, i, dtype), range(reader.text_config.num_hidden_layers))
    norm, readout = build_final_norm(reader, dtype), build_readout(reader, dtype)
    out = {"dtype": "bf16", "state": DEMO_STATE, "runtime_s": None, "results": {}}
    for (name, row), h in zip(rows.items(), hs):
        count = tok.count(row)
        logits, probs = decision_head(norm(h)[:, -1], readout, count, reader.temperature)
        out["results"][name] = {
            "seq_len": len(ids[name]),
            "probabilities": probs.tolist(),
            "answer": tok.autojev.answer(row["question"], probs.tolist()),
        }
    out["runtime_s"] = round(time.time() - start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
