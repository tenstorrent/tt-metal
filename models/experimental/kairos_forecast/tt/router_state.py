# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-owned router state: the Kairos gate's 5-element bias, which the reference mutates on every forward.

Update (moe.py Gate, no training guard):
    bias += rate * (target_dist * total - bincount(indices, weights)) / total
Settled against tsfm/model/kairos/moe.py and modeling_kairos.py at 16eb4b98 (2026-10-28, D2 fix 2):
  (a) the bias is added before the softmax, so the top-k weights come from the biased scores.
  (b) softmax in fp32 over all 5 experts (3 real + 2 null), then the 3 picked weights are renormalized to sum 1.
  (c) the gate input is the instance-normed context (NaN where masked), patched to [P, 128], nan_to_num(0).
  (d) nothing is masked: padded / all-NaN patches route as zeros and count in the bincount and the total.
      total is the sum of the weights (= P patches at route_scale 1), not P * 3.
The reference updates the bias on every forward (no training guard); the flip runs x then -x sequentially
(_autoregressive_generate twice), so the -x half sees the bias after the x half.
"""
import torch

TARGET_DIST = (0.05, 0.10, 0.55, 0.15, 0.15)
UPDATE_RATE = 0.01
N_EXPERTS = 5
TOP_K = 3


class RouterState:
    def __init__(self, bias):
        self.bias0 = bias.detach().to(torch.float32).clone()
        self.bias = self.bias0.clone()

    def restore(self):
        """Suite and bench call this before every forward call (the e2e stream excepted)."""
        self.bias = self.bias0.clone()

    def route(self, patches, w_gate):
        """patches [P, 128] fp32 (NaN allowed), w_gate [5, 128]. Returns (weights [P, 3], indices [P, 3], margin)."""
        x = torch.nan_to_num(patches.to(torch.float32), nan=0.0)
        scores = torch.softmax(x @ w_gate.T.to(torch.float32) + self.bias, dim=-1)
        top = torch.topk(scores, TOP_K + 1, dim=-1)
        indices = top.indices[:, :TOP_K]
        w = top.values[:, :TOP_K]
        weights = w / w.sum(dim=-1, keepdim=True)
        margin = float((top.values[:, TOP_K - 1] - top.values[:, TOP_K]).min())
        return weights, indices, margin

    def update(self, weights, indices):
        """Apply after each forward's routing; for the flip, before routing the -x half."""
        counts = torch.bincount(indices.flatten(), weights=weights.flatten().to(torch.float32), minlength=N_EXPERTS)
        total = counts.sum()
        target = torch.tensor(TARGET_DIST, dtype=torch.float32)
        self.bias = self.bias + UPDATE_RATE * ((target * total - counts) / total)  # moe.py op order


def top3_margin(ref_scores):
    """Smallest gap between the 3rd and 4th of the reference's selection scores (whatever it passes to topk)."""
    top = torch.topk(ref_scores.to(torch.float32), TOP_K + 1, dim=-1).values
    return float((top[..., TOP_K - 1] - top[..., TOP_K]).min())


def routing_record(ref_indices, dev_indices, ref_bias, dev_bias, ref_scores):
    """The D0 amendment-1 keys, in every suite and bench record (D2 fix 3).

    ref_* are the reference's captured values from the goldens, never a host RouterState replay, so a shared
    mistake in (a)-(d) above cannot pass both sides. Indices are [rows, patches, 3]."""
    ref_i, dev_i = ref_indices.sort(-1).values, dev_indices.sort(-1).values
    bad_patch = (ref_i != dev_i).any(-1)  # [rows, patches]
    return {
        "route_mismatch_rows": int(bad_patch.flatten(1).any(-1).sum()),  # series rows with any mismatched patch
        "route_mismatch_patches": int(bad_patch.sum()),  # report-only
        "bias_max_abs_diff": float((ref_bias - dev_bias).abs().max()),
        "min_top3_margin": top3_margin(ref_scores),
    }
