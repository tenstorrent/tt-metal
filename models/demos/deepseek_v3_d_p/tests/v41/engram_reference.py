# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Cached CPU expected results of one DeepSeek-V4.1 Engram layer (bead F5), from the vendored reference.

For a schedule whose first layer is an Engram layer (e.g. ``real_spec((1,), S)``), the Engram input is the
expanded embedding; the reference hash (``NgramHashState``, optionally with an image mask), row lookup
(``ParallelEngramEmbedding`` on the synthetic rows the prompt hashes to) and ``Engram.forward`` run once and are
cached on disk (``oracle.CACHE_DIR``). No ttnn import: run it in a plain torch process before a device run::

    python -m models.demos.deepseek_v3_d_p.tests.v41.engram_reference

Result: tokens [S], mask [S] bool or None, hash_ids [S, n_hash_cols] (global rows), table (weight fp8
[R, head_dim], scale E8M0 [R, head_dim/32], row_ids [R] sorted: the rows the prompt needs), rows [S, n_hash_cols
* head_dim] bf16 (reference lookup), x_in [S, hc, D] bf16, out_fp32 [S, hc, D] (``Engram.forward`` on the fp32
view of x_in: the update without the final bf16 rounding), out [S, hc, D] bf16 (the reference output).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.v41.small_config import small_spec

VERSION = 2
SMALL_SEQ, REAL_SEQ = 512, 5120
# name -> (spec, prompt length, image span or None); layer 1 is the first Engram layer (SWA-only, no sources)
CASES = {
    "small": (lambda: small_spec((1,), SMALL_SEQ), SMALL_SEQ, None),
    "small_padded": (lambda: small_spec((1,), SMALL_SEQ), SMALL_SEQ - 12, None),
    "small_image": (lambda: small_spec((1,), SMALL_SEQ), SMALL_SEQ, (250, 270)),  # spans the 256 chunk boundary
    "real": (lambda: orc.real_spec((1,), REAL_SEQ), REAL_SEQ, None),
}


def image_mask(seq: int, start: int, end: int) -> torch.Tensor:
    """[S] bool token mask: False (image token) on [start, end)."""
    mask = torch.ones(seq, dtype=torch.bool)
    mask[start:end] = False
    return mask


def case_inputs(name: str) -> tuple[orc.OracleSpec, torch.Tensor, torch.Tensor | None]:
    """(spec, tokens [S], token mask [S] or None) of a named case."""
    make_spec, length, span = CASES[name]
    spec = make_spec()
    tokens = orc.random_tokens(spec, length)[0]
    return spec, tokens, None if span is None else image_mask(length, *span)


def _path(spec: orc.OracleSpec, tokens: torch.Tensor, mask: torch.Tensor | None):
    sha = lambda t: None if t is None else hashlib.sha256(t.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
    key = orc._digest(
        asdict(spec.args),
        asdict(spec.model_args),
        list(spec.layer_ids),
        spec.seed,
        sha(tokens),
        sha(mask),
        orc._reference_digest(),
        torch.__version__,
        VERSION,
    )
    return orc.CACHE_DIR / f"engram-{key}.pt"


def _load_rows(emb, seed: int, layer_id: int, ids: torch.Tensor):
    """Materialize the synthetic rows ``ids`` hash to in the reference table (as oracle.load_engram_rows)."""
    rows = torch.unique(ids.flatten())
    weight, scale = orc.synthetic_engram_rows(seed, layer_id, rows, emb.dim)
    emb.weight = torch.nn.Parameter(weight, requires_grad=False)
    emb.scale = torch.nn.Parameter(scale, requires_grad=False)
    emb.vocab_end_idx = len(rows)
    emb.oracle_rows = rows
    return {"weight": weight, "scale": scale, "row_ids": rows}


@torch.no_grad()
def engram_case(
    spec: orc.OracleSpec, tokens: torch.Tensor, mask: torch.Tensor | None = None, model: v41.Transformer | None = None
) -> dict:
    """Expected Engram results of the spec's first layer (an Engram layer) for ``tokens`` [S] (cached)."""
    path = _path(spec, tokens, mask)
    if path.is_file():
        return torch.load(path)
    assert 0 in spec.args.engram_layer_ids, "the schedule's first layer must be an Engram layer"
    model = model if model is not None else orc.build_reference(spec)
    engram = model.layers[0].engram
    started = time.time()
    with torch.inference_mode():
        ids = model.engram_hash(tokens[None], 0, None if mask is None else mask[None])[0, :, engram.layer_hash_index]
    table = _load_rows(engram.embed, spec.seed, spec.layer_ids[0], ids)
    with v41.set_dtype(torch.bfloat16), torch.inference_mode():
        rows = engram.embed(ids).flatten(-2)
        x_in = model.embed(tokens[None]).unsqueeze(2).repeat(1, 1, spec.args.hc_mult, 1)
        token_mask = None if mask is None else mask[None]
        out = engram(x_in, ids[None], token_mask)[0]
        out_fp32 = engram(x_in.float(), ids[None], token_mask)[0]
    result = {
        "tokens": tokens.clone(),
        "mask": None if mask is None else mask.clone(),
        "hash_ids": ids.clone(),
        "table": table,
        "rows": rows.clone(),
        "x_in": x_in[0].clone(),
        "out_fp32": out_fp32.clone(),
        "out": out.clone(),
        "seconds": time.time() - started,
    }
    orc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(result, tmp)
    tmp.replace(path)
    return result


if __name__ == "__main__":
    # precompute the cases of tests/v41/test_engram_v41.py
    import sys

    torch.set_num_threads(16)
    for name in sys.argv[1:] or CASES:
        started = time.time()
        engram_case(*case_inputs(name))
        print(f"{name}: {time.time() - started:.1f} s")
