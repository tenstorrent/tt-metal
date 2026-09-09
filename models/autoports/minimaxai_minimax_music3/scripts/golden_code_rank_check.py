# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only control for the "teacher-forced argmax != golden sampled code" observation.

For the golden frames, run the fp32 torch depth decoder teacher-forced and report, per head, the
rank of the golden sampled code in the conditional logits, the argmax probability and the top-50
probability mass. Writes doc/depth_decoder/pcc/golden_code_ranks.json. No device needed:

    source ~/mm3-bringup/common.sh && cd $MM3_WT && $MM3_PY $MM3_MODEL_DIR/scripts/golden_code_rank_check.py
"""
import json
from pathlib import Path

import torch

from models.autoports.minimaxai_minimax_music3.reference import depth_decoder_ref as REF
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CODE_OFFSET

OUT = Path(__file__).resolve().parents[1] / "doc" / "depth_decoder" / "pcc" / "golden_code_ranks.json"
FRAMES = [0, 1, 2, 3]

if __name__ == "__main__":
    torch.set_num_threads(8)
    root = R.reference_dir()
    frame_hiddens = torch.load(root / "frame_hiddens.pt")
    codes = torch.load(root / "sampled_codes.pt")
    embed_weight = R.load_embed_weight()
    model = REF.load_reference(R.weights_dir(), torch.float32)
    result = {}
    for f in FRAMES:
        gh = frame_hiddens[0, f, :4096].reshape(1, -1).repeat(2, 1).float()
        se = embed_weight[int(codes[f, 0]) + AUDIO_CODE_OFFSET].reshape(1, -1).repeat(2, 1).float()
        rc = codes[f, 1:].reshape(1, -1).repeat(2, 1)
        _, logits = REF.teacher_forced_depth_loop(model, gh, se, rc)
        rows = []
        for k, l in enumerate(logits):
            cond = l[0]
            probs = torch.softmax(cond, -1)
            order = torch.argsort(cond, descending=True)
            gold = int(rc[0, k])
            rank = int((order == gold).nonzero()[0, 0])
            rows.append(
                {
                    "head": k + 1,
                    "golden_code": gold,
                    "rank_in_conditional_logits": rank,
                    "in_top50": rank < 50,
                    "argmax_prob": round(float(probs.max()), 4),
                    "golden_code_prob": round(float(probs[gold]), 4),
                    "top50_mass": round(float(probs.topk(50).values.sum()), 4),
                }
            )
        result[f"frame_{f + 1}"] = rows
        print(
            f"frame {f + 1}: ranks {[r['rank_in_conditional_logits'] for r in rows]}, argmax probs {[r['argmax_prob'] for r in rows]}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", OUT)
