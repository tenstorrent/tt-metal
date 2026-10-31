# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fused head planner's golden answers (head_plan.fit of the reference tree, mstaletovic/moe-trim) for
test_head_plan_parity.py. Host only, no device. Run in the reference tree's environment:

    P=/data/mstaletovic/tt-metal-trimb; R=/tmp/$USER/pyroot-trimb; mkdir -p $R/ttnn && ln -sf $P/ttnn/ttnn/* $R/ttnn/ \
      && ln -sf $P/build_Release/lib/_ttnn*.so $R/ttnn/
    cd $P && TT_METAL_HOME=$P LD_LIBRARY_PATH=$P/build_Release/lib PYTHONPATH=$R:$P python \
      <this tree>/tests/ttnn/unit_tests/operations/experimental/moe_block/fused_head/dump_head_plan_golden.py [out.json.gz]

Cases: the five models of the block matrix (MoEBlock's FusedHead.shape on head_plan's host defaults: grid 12 x 10, the
model's fabric payload, L1_FREE - L1_MARGIN) at every rows-a-mesh-row of the matrix on the Galaxy (8 x 4: line and
torus) and the LoudBox meshes (2 x 4 line, 1 x 4 ring), plus fit()'s keyword variants (router placement, K step, no
fallback, split)."""

import dataclasses
import gzip
import json
import math
import sys
from pathlib import Path

from models.demos.moe_block.moe_block import DSV41_FLASH, GLM_53, GLM_53_FLASH, KIMI_K3, KIMI_K27, _payload_of
from models.demos.moe_block.tt import head_plan as hp

MODELS = {"glm53": GLM_53, "k3": KIMI_K3, "k27": KIMI_K27, "glm53_flash": GLM_53_FLASH, "dsv41_flash": DSV41_FLASH}
GALAXY_EPC = {"k3": 28, "k27": 12}  # (test_moe_block_matrix.mesh_model)
EPC_MAX = 64
ROWS = (128, 256, 512, 640, 768, 1024, 1280, 2048)
# (R, G, tp ring, sp ring)
FULL = 3
MESHES = [(8, 4, False, False), (8, 4, True, True), (2, 4, False, False), (1, 4, True, False)]


def fold_of(name, E, chips):
    epc = GALAXY_EPC.get(name, E // chips)
    F = 1
    while E // F // chips > EPC_MAX or (name in GALAXY_EPC and E // F // chips > epc):
        F *= 2
    return F


def shape_of(name, rows, R, G, ring, sp_ring, A=16, links=2):
    m = MODELS[name]
    F = fold_of(name, m.num_experts, R * G)
    return hp.HeadShape(
        rows=rows,
        H=m.emb_dim,
        I=m.shared_expert_hidden_dim // G,
        G=G,
        E=m.num_experts,
        K=m.top_k,
        latent=m.latent_dim,
        fold=F,
        links=links,
        ring=ring,
        grid=hp.GRID,
        R=R,
        sp_ring=sp_ring,
        payload=_payload_of(m),
        A=A,
        w_dtype="bf8",
    )


def _num(v):
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return v


def planned(p):
    if p is None:
        return None
    return dict(
        lay=dataclasses.asdict(p.lay),
        arena=p.arena,
        sizes=dict(p.sizes),
        t=_num(p.t),
        parts={k: _num(v) for k, v in p.parts.items()},
        note=p.note,
    )


def run(shape, kw):
    f = hp.fit(shape, **kw)
    # (every planned layout with its prediction; the first FULL in full - arena, region sizes, the cost's parts)
    return dict(
        ok=f.ok,
        why=f.why,
        note=f.note,
        planned=[planned(p) for p in f.planned[:FULL]],
        ranked=[[*dataclasses.astuple(p.lay), _num(p.t), p.arena] for p in f.planned],
        free=planned(f.free),
    )


def cases():
    out = []
    for name in MODELS:
        for R, G, ring, sp in MESHES:
            for rows in ROWS:
                out.append((name, rows, R, G, ring, sp, 16, {}))
    for name in MODELS:  # (fit's keyword variants and other accumulator counts at the 8 x 4, chunk 5120)
        for kw in (
            dict(on_engine=True),
            dict(on_engine=False),
            dict(fallback=False),
            dict(split=True),
            dict(placement="model"),
            dict(KB=4),
            dict(largest=False, explain=False),
        ):
            out.append((name, 640, 8, 4, True, True, 16, kw))
        for A in (8, 12):
            out.append((name, 640, 8, 4, True, True, A, {}))
        out.append((name, 640, 8, 4, True, True, 16, dict(l1_free=900 * 1024)))
        out.append((name, 1024, 8, 4, True, True, 16, dict(l1_free=700 * 1024)))
        out.append((name, 640, 4, 2, False, False, 16, {}))  # (TP 2)
        out.append((name, 640, 4, 8, True, True, 16, {}))  # (TP 8)
        out.append((name, 640, 8, 4, True, True, 16, dict(links=1)))
    return out


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("head_plan_golden.json.gz"))
    from models.demos.moe_block.tt.se_engine3 import arena_cap_of

    res = []
    for name, rows, R, G, ring, sp, A, kw in cases():
        kw = dict(kw)
        links = kw.pop("links", 2)
        l1_free = kw.pop("l1_free", hp.L1_FREE - hp.L1_MARGIN)
        sh = shape_of(name, rows, R, G, ring, sp, A, links)
        call = dict(kw)
        call["arena_max"] = lambda kb, sh=sh, l1=l1_free: arena_cap_of(l1, kb, payload=sh.payload)
        got = run(sh, call)
        shape = dataclasses.asdict(sh)
        shape["grid"] = list(sh.grid)
        res.append(dict(model=name, shape=shape, l1_free=l1_free, kwargs=kw, fit=got))
    text = json.dumps(
        dict(reference="mstaletovic/moe-trim models/demos/moe_block/tt/head_plan.py", cases=res), separators=(",", ":")
    )
    with gzip.open(path, "wt") as f:
        f.write(text)
    print(f"wrote {len(res)} cases to {path}")


if __name__ == "__main__":
    main()
