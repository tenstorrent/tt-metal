"""Correctness of topk_large_indices at the EXACT production GLM-5.2 configuration.

Coverage gap this closes (as of 2026-08-14): the in-tree tests assert correctness at n=51200 only
for k=1536, assert k=2048 only when n==k (a tiny pool), and NEVER pass `valid_length` to a
correctness test — it appears only in a perf check. Production calls

    topk_large_indices(logits, k=2048, valid_length=<written KV depth>)

over a 56320-wide row. The prefill accuracy collapse scales with key-pool size while every layer is
individually correct, so a top-k that is subtly wrong only at large n / large k / with valid_length
would explain it exactly.

Random values (not the synthetic ramp the in-tree tests use) because ties and near-ties at the
top-k boundary are the failure mode of interest: the ramp gives every element a distinct, widely
separated value, which is the easiest possible input for a selection op.
"""

import numpy as np
import pytest
import torch

import ttnn

CHUNK = 5120
SEQ_CACHE = 56320


def _rand_scores(num_rows: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn((num_rows, n), generator=g, dtype=torch.float32).to(torch.bfloat16)


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _selected_set_agreement(torch_input: torch.Tensor, tt_indices: ttnn.Tensor, k: int, valid_length=None):
    """Fraction of rows' selected SETS that match torch, plus the mean per-row overlap.

    Set comparison, not positional: ties may be ordered differently and that is legitimate. What is
    NOT legitimate is selecting a different set of keys, which is what changes attention output.
    """
    ref_src = torch_input.float()
    if valid_length is not None:
        ref_src = ref_src[:, :valid_length]
    _, ref = torch.topk(ref_src, k, dim=-1, largest=True, sorted=True)
    got = ttnn.to_torch(tt_indices).to(torch.int64)
    rows = ref.shape[0]
    exact_rows, overlaps = 0, []
    for r in range(rows):
        a, b = set(ref[r].tolist()), set(got[r].tolist())
        overlaps.append(len(a & b) / k)
        exact_rows += a == b
    return exact_rows / rows, float(np.mean(overlaps))


def _concentrated_scores(num_rows: int, n: int, k: int) -> torch.Tensor:
    """The in-tree test's input shape: every top-k value in the LAST k columns.

    With llk chunking (k=2048 -> 2048-wide chunks) this puts all winners in the final chunk, so a
    merge that mishandled candidates spread across chunks would still pass. Used here purely as a
    CONTROL against the random (spread) input.
    """
    values = torch.zeros((num_rows, n), dtype=torch.bfloat16)
    hi16 = (0x3F80 + np.arange(k, dtype=np.uint32)).astype(np.uint32)
    values[:, -k:] = torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)
    return values


def test_topk_concentrated_control(device):
    """Control: production shape but winners concentrated in the last chunk (in-tree style)."""
    num_rows, n, k = 640, SEQ_CACHE, 2048
    torch_input = _concentrated_scores(num_rows, n, k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    exact_frac, mean_overlap = _selected_set_agreement(torch_input, tt_indices, k)
    print(f"\n  CONTROL concentrated n={n} k={k}: rows exact {exact_frac:.4f}, overlap {mean_overlap:.6f}")
    assert mean_overlap > 0.999, f"even the concentrated control failed: {mean_overlap:.6f}"


@pytest.mark.parametrize(
    "num_rows,n,k,valid_length",
    [
        (640, SEQ_CACHE, 2048, None),  # production shape, no bound
        (640, SEQ_CACHE, 2048, SEQ_CACHE),  # bound == full width
        (640, SEQ_CACHE, 2048, 51200),  # warm_cache: bound just under full
        (640, SEQ_CACHE, 2048, CHUNK),  # cold chunk 0: heavy truncation
        (640, 51200, 1536, None),  # the config the in-tree test already covers (control)
    ],
    ids=["prod_2048_nobound", "prod_2048_fullbound", "prod_2048_51200", "prod_2048_chunk0", "known_good_1536"],
)
def test_topk_production_config_matches_torch(device, num_rows, n, k, valid_length):
    torch_input = _rand_scores(num_rows, n)
    kwargs = {"k": k}
    if valid_length is not None:
        kwargs["valid_length"] = valid_length
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), **kwargs)

    exact_frac, mean_overlap = _selected_set_agreement(torch_input, tt_indices, k, valid_length)

    # Are the differing picks TIES (equally-scored, benign) or STRICTLY WORSE keys (a real defect)?
    # Attending to a lower-scored key is a genuine selection error; picking a different key with the
    # same score is not. This is the difference between "the op is buggy" and "tie-breaking differs".
    src = torch_input.float()[:, :valid_length] if valid_length else torch_input.float()
    ref_vals, ref_idx = torch.topk(src, k, dim=-1, largest=True, sorted=True)
    got_idx = ttnn.to_torch(tt_indices).to(torch.int64)
    worse, tied, checked = 0, 0, 0
    for r in range(min(64, src.shape[0])):
        a, b = set(ref_idx[r].tolist()), set(got_idx[r].tolist())
        missed, extra = sorted(a - b), sorted(b - a)
        thresh = ref_vals[r, -1].item()  # score of torch's k-th (worst kept) element
        for e in extra:
            checked += 1
            v = src[r, e].item()
            if v < thresh:
                worse += 1
            else:
                tied += 1
    print(
        f"  differing picks over 64 rows: {checked} total -> {worse} STRICTLY WORSE than torch's k-th, "
        f"{tied} tied/at-threshold"
    )
    print(
        f"\n  n={n} k={k} valid_length={valid_length}: rows exactly matching torch {exact_frac:.4f}, "
        f"mean per-row overlap {mean_overlap:.6f}"
    )
    # Any selected key that torch would not have selected changes the attended set. Allow a hair of
    # slack for genuine bf16 ties at the boundary, but a real bug shows up far below this.
    assert mean_overlap > 0.999, f"top-k selected a different key set: mean overlap {mean_overlap:.6f}"
