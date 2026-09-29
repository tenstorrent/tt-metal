# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the routed-expert dtype study (dev-spec D-B, ``test_v41_expert_dtype``); torch only, no ttnn.

The MoE input of real layer ``layer`` is the §6 block oracle's ``ffn_in`` (the same 2048-token real-weight oracle
``test_block_v41`` uses: schedule ``stack`` for layer 2, ``swa`` for layer 0), so MoE and block numbers of this
study share one input. The observable MoE parts (``moe_reference.reference_parts``) are computed on those 2048
rows and checked bit for bit against the oracle's ``ffn_out``; the device runs a production 5120-token chunk made
by tiling the rows (every MoE part is row-local, so the expected parts tile the same way).

Warm the cache outside the device lock:
``python -m models.demos.deepseek_v3_d_p.tests.v41.expert_dtype_reference 2 0``.
"""

import hashlib
import sys
import time

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as O
from models.demos.deepseek_v3_d_p.tests.v41.moe_reference import reference_parts

ORACLE_SEQ = 2048
CHUNK = 5120  # production prefill chunk the device MoE runs
SCHEDULES = {2: (2, 3, 20, 21, 24), 20: (2, 3, 20, 21, 24), 0: (0,)}  # test_block_v41 SCHEDULES: stack, swa
REFERENCE_VERSION = 1


def block_spec(layer: int, layers: tuple[int, ...] | None = None) -> O.OracleSpec:
    """``test_block_v41``'s real-weight spec of the schedule holding ``layer`` (or of ``layers``)."""
    return O.real_spec(layers or SCHEDULES[layer], ORACLE_SEQ, candidate_topk_blocks=96, checkpoint=O.HF_SNAPSHOT)


def tile_rows(t: torch.Tensor, rows: int = CHUNK) -> torch.Tensor:
    reps = -(-rows // t.shape[0])
    return t.repeat(reps, *([1] * (t.dim() - 1)))[:rows]


def moe_parts(layer: int) -> dict:
    """Reference MoE parts of real ``layer`` on its block-oracle input (cached), tiled to ``CHUNK`` rows."""
    spec = block_spec(layer)
    tokens = O.random_tokens(spec)
    name = O.cache_path(spec, tokens).name
    key = hashlib.sha256(f"{name}:{layer}:{REFERENCE_VERSION}".encode()).hexdigest()[:20]
    path = O.CACHE_DIR / f"moe-dtype-parts-{key}.pt"
    if path.is_file():
        start = time.perf_counter()
        parts = torch.load(path)
        logger.info(f"MoE parts layer {layer}: cached {path.name} {time.perf_counter() - start:.1f}s")
    else:
        start = time.perf_counter()
        result = O.oracle(spec, tokens)  # cached by the block tests
        x = result["blocks"][layer]["ffn_in"]
        logger.info(f"block oracle {name} loaded {time.perf_counter() - start:.1f}s")
        start = time.perf_counter()
        model = O.build_reference(block_spec(layer, (layer,)))  # one real layer: only its MoE is used
        logger.info(f"reference layer {layer} built {time.perf_counter() - start:.1f}s")
        start = time.perf_counter()
        with v41.set_dtype(torch.bfloat16):
            parts = reference_parts(model.layers[0].ffn, x)
        assert torch.equal(parts["final"], result["blocks"][layer]["ffn_out"]), "MoE split differs from MoE.forward"
        parts["x"] = x
        logger.info(f"MoE parts layer {layer} computed {time.perf_counter() - start:.1f}s")
        tmp = path.with_suffix(".tmp")
        torch.save(parts, tmp)
        tmp.replace(path)
    return {k: tile_rows(v) for k, v in parts.items()}


if __name__ == "__main__":
    torch.set_num_threads(8)
    for arg in sys.argv[1:]:
        parts = moe_parts(int(arg))
        print(arg, {k: tuple(v.shape) for k, v in parts.items()}, flush=True)
