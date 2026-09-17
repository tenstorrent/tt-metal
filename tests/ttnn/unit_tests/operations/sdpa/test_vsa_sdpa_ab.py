# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""A/B helper: run vsa_sdpa (streaming) once on a mid-size production-like shape under the current TT_VSA_* environment
and save the output to VSA_AB_OUT (a .pt). Compare two runs with different environments in a third process."""

import os

import torch
import ttnn
from models.common.utility_functions import skip_for_wormhole_b0

from .test_vsa_sdpa_perf import make_inputs, bstride_order


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
def test_vsa_sdpa_ab(device):
    spec = dict(
        s_local=int(os.environ.get("VSA_AB_S", "4800")),
        n_blocks=int(os.environ.get("VSA_AB_BLOCKS", "688")),
        row_blocks=int(os.environ.get("VSA_AB_K", "80")),
        dense_rows=int(os.environ.get("VSA_AB_DENSE", "2")),
        order="model",
    )
    q, k, v, idx, counts, _ = make_inputs(device, **spec)
    kw = dict(streaming=True, dense_row_hint=list(range(spec["dense_rows"])))
    order = os.environ.get("VSA_ORDER")
    if order and order != "identity":
        perm = bstride_order(counts.shape[-1], order).reshape(1, 1, 1, -1)
        kw["stream_order"] = ttnn.from_torch(perm, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    out = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts, **kw)
    out2 = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts, **kw)  # cache hit + determinism
    t1, t2 = ttnn.to_torch(out), ttnn.to_torch(out2)
    d = (t1.float() - t2.float()).abs()
    rows_bad = torch.nonzero(d.amax(dim=-1).reshape(t1.shape[1], -1).amax(dim=0) > 0).reshape(-1)
    print(
        f"\nVSA_AB launch-to-launch: equal={torch.equal(t1, t2)} max|diff|={d.max().item():.4g} "
        f"bad_tokens={int((d.amax(dim=-1) > 0).sum())} bad_qrows(64-tok)={sorted(set((rows_bad // 64).tolist()))[:20]}"
    )
    path = os.environ.get("VSA_AB_OUT")
    if path:
        torch.save(t1, path)
        print(f"\nVSA_AB saved {path} shape {tuple(t1.shape)}")
