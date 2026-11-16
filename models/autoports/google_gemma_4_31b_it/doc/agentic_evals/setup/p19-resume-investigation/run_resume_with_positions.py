"""Preserve per-position TT comparisons without changing readiness thresholds."""
import sys

import torch

from readiness_check import run_prefill_resume_check as runner

original_compare = runner.compare_logits


def compare_with_positions(reference, candidate):
    result = original_compare(reference, candidate)
    ref_values, ref_ids = torch.topk(reference[0].float(), 2, dim=-1)
    cand_values, cand_ids = torch.topk(candidate[0].float(), 2, dim=-1)
    result["per_position"] = [
        {
            "window_index": i,
            "pcc": runner._pcc(reference[0, i], candidate[0, i]),
            "reference_top1": int(ref_ids[i, 0]),
            "candidate_top1": int(cand_ids[i, 0]),
            "reference_top1_top2_gap": float(ref_values[i, 0] - ref_values[i, 1]),
            "candidate_top1_top2_gap": float(cand_values[i, 0] - cand_values[i, 1]),
        }
        for i in range(reference.shape[1])
    ]
    return result


runner.compare_logits = compare_with_positions
sys.exit(runner._main())
