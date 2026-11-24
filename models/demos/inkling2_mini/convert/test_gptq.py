# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the GPTQ tools (gptq_calib.py, gptq_codes.py): no card, no weights.

    pytest models/demos/inkling2_mini/convert/test_gptq.py

``test_corpus`` needs the model's tokenizer (``$INKLING_CKPT``) and skips without it.
"""

import json
import os

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.convert import gptq_calib as G
from models.demos.inkling2_mini.convert import gptq_codes as C
from models.demos.inkling2_mini.convert.moe_pcc import bfp_roundtrip


@pytest.mark.parametrize("mbits, dtype, name", [(3, ttnn.bfloat4_b, "BFP4"), (7, ttnn.bfloat8_b, "BFP8")])
def test_gptq_on_the_bfp_grid(mbits, dtype, name):
    """The solved values are stored by ttnn bit for bit (W^T tile orientation: one exponent per 16 outputs at one
    input) and give a lower output error on the calibration inputs than plain rounding."""
    torch.manual_seed(0)
    B, R, K, N = 3, 64, 96, 600
    X = torch.randn(B, N, K) @ (torch.randn(K, K) / K**0.5) + 0.1 * torch.randn(B, N, K)  # correlated inputs
    W = torch.randn(B, R, K) * torch.rand(B, R, 1) * 0.05
    q, e = G.gptq(W, X.transpose(1, 2) @ X, damp=0.01, block=32, mbits=mbits)
    assert int(q.abs().max()) <= 2**mbits - 1
    v = G.decode(q, e, mbits)
    assert torch.equal(C.decode_codes(q, e, mbits), v)
    vt = v.transpose(1, 2).contiguous()  # W^T [B, in, out]
    assert torch.equal(bfp_roundtrip(vt, name), vt)
    stored = ttnn.to_torch(ttnn.from_torch(vt.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT))
    assert torch.equal(stored.float(), vt)
    ref = X @ W.transpose(1, 2)
    err = lambda w: float(((X @ w.transpose(1, 2) - ref) ** 2).sum())  # noqa: E731
    assert err(v) < err(G.decode(*G.rtn(W, mbits), mbits))


def test_codes_layer_file(tmp_path):
    """A ``codes:`` layer file (4-bit packed gate_up, 8-bit down) comes back as those values with their dtypes."""
    torch.manual_seed(1)
    q4 = torch.randint(-7, 8, (2, 32, 16), dtype=torch.int8)
    q8 = torch.randint(-127, 128, (2, 16, 32), dtype=torch.int8)
    e4 = torch.randint(110, 130, (2, 2, 16), dtype=torch.uint8)
    e8 = torch.randint(110, 130, (2, 1, 32), dtype=torch.uint8)
    assert torch.equal(C.unpack4(G.pack4(q4)), q4)
    torch.save({"gate_up": {"q4": G.pack4(q4), "e": e4}, "down": {"q8": q8, "e": e8}}, tmp_path / "L05.pt")
    source = f"codes:{tmp_path}"
    assert C.layer_dtypes(source, 5) == {"experts.w_gate_up": "BFP4", "experts.w_down": "BFP8"}
    store, exact = C.layer_store(source, 5)
    assert exact
    p = "model.language_model.layers.5.mlp.experts."
    assert torch.equal(store.experts(p + "gate_up_proj", 0, 2).float(), G.decode(q4, e4, 3))
    assert torch.equal(store.experts(p + "down_proj", 1, 2).float(), G.decode(q8, e8, 7)[1:2])


def test_layer_inputs(tmp_path):
    """A loop layer's inputs are those of its three applications; --weights picks and weights the sequences."""
    for app, seq, n in ((6, 0, 3), (18, 0, 3), (30, 2, 5), (7, 0, 4), (6, 1, 2)):
        z = {"x": torch.full((n, G.H), float(app), dtype=torch.bfloat16), "ids": torch.zeros(n, 8, dtype=torch.int16)}
        torch.save(z | {"s": 0}, tmp_path / f"a{app:02d}_s{seq:02d}_p00000.pt")
    x, ids, w = G.layer_inputs(tmp_path, 6)
    assert x.shape == (13, G.H) and ids.shape == (13, 8) and w.tolist() == [1.0] * 13
    assert sorted(set(x[:, 0].tolist())) == [6.0, 18.0, 30.0]
    x, ids, w = G.layer_inputs(tmp_path, 6, {0: 2.0, 2: 0.5})
    assert sorted(w.tolist()) == [0.5] * 5 + [2.0] * 6


def test_weights(tmp_path):
    """Each kind gets its share of the rows; the mean weight over all tokens is 1."""
    torch.save({"ids": torch.zeros(3, 2048, dtype=torch.int32), "kinds": ["chat", "chat", "code"]}, tmp_path / "a.pt")
    torch.save({"ids": torch.zeros(1, 8192, dtype=torch.int32), "kinds": ["long"], "first": 3}, tmp_path / "b.pt")
    out = tmp_path / "w.json"
    G.main(["weights", "--ids", f"{tmp_path}/a.pt", f"{tmp_path}/b.pt", "--share", "chat=.5,code=.25,long=.25"]
           + ["--out", str(out)])  # fmt: skip
    assert json.loads(out.read_text()) == {"0": 1.75, "1": 1.75, "2": 1.75, "3": 0.4375}


@pytest.mark.skipif(not os.environ.get("INKLING_CKPT"), reason="tokenizer missing (set INKLING_CKPT)")
def test_corpus(tmp_path):
    """Sequences come out in command-line order, one kind per source, numbered from --first."""
    chat = tmp_path / "chat.jsonl"
    talk = lambda i: [
        {"role": "user", "content": f"Count to {i}."},
        {"role": "assistant", "content": str(list(range(i)))},
    ]  # noqa: E731
    chat.write_text("".join(json.dumps(talk(i)) + "\n" for i in range(20, 60)))
    docs = tmp_path / "build_docs"  # --ask skips by the path below its root, not by the root's own name
    (docs / "generated").mkdir(parents=True)
    (docs / "generated" / "a.md").write_text("Left out: a generated file. " * 200)
    (docs / "z.md").write_text("The card has seven DRAM banks. " * 200)
    book = tmp_path / "book.txt"
    book.write_text("It was a dark and stormy night. " * 400)
    out = tmp_path / "ids.pt"
    ask = f"docs:1:{docs}:.md:Summarize this document."
    G.main(["corpus", "--out", str(out), "--seq", "64", "--first", "5", "--chat", f"chat:2:{chat}", "--ask", ask]
           + ["--text", f"book:{book}@0.5"])  # fmt: skip
    d = torch.load(out)
    assert tuple(d["ids"].shape) == (4, 64) and d["ids"].dtype == torch.int32
    assert d["kinds"] == ["chat", "chat", "docs", "book"] and d["first"] == 5
    assert d["src"][2] == "z.md"
