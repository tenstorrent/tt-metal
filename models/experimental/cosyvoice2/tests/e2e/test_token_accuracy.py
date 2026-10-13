# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced token accuracy against the PyTorch reference: #54104 Stage 1's "Token-level accuracy > 95%".

Inputs (skipped without them):
- `COSYVOICE2_INPUTS`: `scripts/prepare_inputs.py`'s directory.
- `COSYVOICE2_TOKEN_REF`: `scripts/token_accuracy_reference.py`'s directory, the reference's top-5 along its own
  generated sequences.

Method: CosyVoice1's teacher-forced measure. For each corpus case, the reference's generated speech tokens are forced
through TT's own decode loop: the prefill, then the traced decode of the reported configuration, one step per token
(`TtQwen2LM.teacher_forced_topk`). At each of the N + 1 positions, TT's top-1 is compared with the reference's.

Every case with a reference file counts: the primary set, the parity sentence and, when its files are there, the
token-accuracy extension (scripts/corpus.py). A subtotal is printed per corpus set.

Gated: top-1 agreement over all positions of all cases. Also reported:
- how often TT's top-1 is inside the reference's top-5;
- the reference's own top-1 / top-2 log-prob margin where the two disagree. A near-tie there is an ordering
  flip, not a different prediction.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

from models.experimental.cosyvoice2.tests.perf import gates

INPUTS_DIR = os.environ.get("COSYVOICE2_INPUTS", "")
TOKEN_REF_DIR = os.environ.get("COSYVOICE2_TOKEN_REF", "")


@pytest.mark.skipif(not (INPUTS_DIR and TOKEN_REF_DIR), reason="set COSYVOICE2_INPUTS and COSYVOICE2_TOKEN_REF")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 50_000_000}], indirect=True)
@pytest.mark.timeout(0)  # a device job is never killed mid-op (pytest.ini sets 300 s)
def test_device_teacher_forced_token_accuracy(device):
    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN
    from models.experimental.cosyvoice2.tt.prompt import PromptContext

    refs = sorted(glob.glob(os.path.join(TOKEN_REF_DIR, "*.npz")))
    assert refs, TOKEN_REF_DIR
    pipe = CosyVoice2TTNN(device)
    rows, agree_all, top5_all, margins, by_set = [], 0, 0, [], {}
    try:
        for path in refs:
            case_id = os.path.basename(path)[: -len(".npz")]
            ref = np.load(path)
            ctx = PromptContext.from_npz(os.path.join(INPUTS_DIR, f"{case_id}.npz"))
            case_set = ctx.meta["case"]["set"]
            text_ids = torch.cat(
                [ctx.prompt_text_ids.long(), torch.tensor([ctx.meta["segment_text_ids"][0]], dtype=torch.long)], dim=1
            )
            forced = ref["tokens"].tolist()
            idx, _ = pipe.llm.teacher_forced_topk(text_ids, ctx.llm_prompt_speech_tokens.long(), forced, k=5)
            tt_top1 = idx[:, 0].numpy()
            ref_top5 = ref["top5"]
            agree = tt_top1 == ref_top5[:, 0]
            in_top5 = np.array([t in r for t, r in zip(tt_top1, ref_top5)])
            gap = ref["logprob_top5"][:, 0] - ref["logprob_top5"][:, 1]
            margins.extend(gap[~agree].tolist())
            agree_all, top5_all = agree_all + int(agree.sum()), top5_all + int(in_top5.sum())
            rows.append((case_id, len(agree), agree.mean(), in_top5.mean()))
            n_set, a_set, c_set = by_set.get(case_set, (0, 0, 0))
            by_set[case_set] = (n_set + len(agree), a_set + int(agree.sum()), c_set + 1)
    finally:
        pipe.release()

    n = sum(r[1] for r in rows)
    print("\n  | case | positions | top-1 agreement | TT top-1 in reference top-5 |\n  |---|---|---|---|")
    for case_id, positions, a, t5 in rows:
        print(f"  | {case_id} | {positions} | {100 * a:.2f} % | {100 * t5:.2f} % |")
    accuracy = 100.0 * agree_all / n
    print(f"  | **all** | {n} | **{accuracy:.2f} %** | {100.0 * top5_all / n:.2f} % |")
    for case_set, (n_set, a_set, c_set) in by_set.items():
        print(f"  set {case_set}: {c_set} cases, {n_set} positions, top-1 agreement {100.0 * a_set / n_set:.2f} %")
    if margins:
        m = np.array(margins)
        print(
            f"  reference top-1 / top-2 log-prob margin at the {len(m)} disagreements: median {np.median(m):.3f}, "
            f"90th percentile {np.quantile(m, 0.9):.3f}, max {m.max():.3f}"
        )
    line = gates.enforce("token_accuracy", accuracy, device, extra=f"{n} positions, {len(rows)} cases")
    gates.report([line], "Stage 1, token accuracy (teacher-forced)")
