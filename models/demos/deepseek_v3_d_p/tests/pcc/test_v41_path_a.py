# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DS41F-0067: the V4.1 prefill's CSA attention on the sparse path A vs the dense path B, chunk by chunk, REAL weights.

Two copies of layer 2 (index / KV source) and layer 3 (its Reuse consumer) are built from the same checkpoint weights: one
with ``PREFILL_V41_PATH_A`` off (path B everywhere, the validated reference: PCC 0.996 vs model.py at 1024 tokens), one with
it on (path A from the first chunk whose context holds a window + top-k worth of entries; chunk 0 stays on path B). Both get
the SAME random-normal attention inputs (distinct keys, so top-k near-ties are rare) for ``V41_PA_CHUNKS`` chunks of
``V41_PA_CHUNK`` tokens; every chunk's output of each layer is compared. Chunk 0 must be identical (same path); later chunks
compare the two paths (same math: softmax over [window | the indexer's top-k]; different kernels -> rounding).

    PYTHONPATH=<this worktree>:<cpu_stub> pytest models/demos/deepseek_v3_d_p/tests/pcc/test_v41_path_a.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tests.pcc.test_v41_attention import _MESH_CONFIGS, _to_device_hidden, _to_host_hidden
from models.demos.deepseek_v3_d_p.tt.v41 import attention as v41a
from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

CHUNK = int(os.environ.get("V41_PA_CHUNK", "4096"))
N_CHUNKS = int(os.environ.get("V41_PA_CHUNKS", "3"))
# path A vs path B: two approximations of model.py, each PCC ~0.9967 / 0.9974 vs model.py on the real 1536-token golden in
# 512-token chunks, path A == path B to 6 digits there (test_v41_attention_chunked.py, 2026-10-15 06:25). On these random inputs
# (many top-k near-ties) the two land at ~0.995 against EACH OTHER, so 0.999 is no gate; 0.99 catches a broken path.
PCC = float(os.environ.get("V41_PA_PCC", "0.99"))
RMS = float(os.environ.get("V41_PA_RMS", "1.0"))  # the attention input is post-RMSNorm: rms ~ the norm weight's scale


def _build(mesh_device, cfg, ck, path_a: bool, max_seq: int):
    v41a.PATH_A = path_a  # read by V41CSA.from_weights
    rot = DeepseekV4RotaryEmbedding(v41a.v41_hf_config(cfg, max_seq=max_seq))
    src = v41a.V41CSA.from_weights(mesh_device, cfg, 2, ck.layer(2), rot)
    con = v41a.V41CSAConsumer.from_weights(mesh_device, cfg, 3, ck.layer(3), rot, source=src)
    states = (src.alloc_state(max_seq, chunk_tokens=CHUNK), con.alloc_state(max_seq, chunk_tokens=CHUNK))
    return src, con, states


@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v41_path_a_vs_path_b(mesh_device, device_params):
    cfg = V41Config.load()
    ck = checkpoint()
    max_seq = CHUNK * N_CHUNKS
    sets = {"B": _build(mesh_device, cfg, ck, False, max_seq), "A": _build(mesh_device, cfg, ck, True, max_seq)}
    v41a.PATH_A = False
    gen = torch.Generator().manual_seed(1234)
    worst = {}
    for c in range(N_CHUNKS):
        h = torch.randn(CHUNK, cfg.dim, generator=gen) * RMS
        outs = {}
        for name, (src, con, (s_src, s_con)) in sets.items():
            ys = src(_to_device_hidden(mesh_device, h), seq_len_actual=CHUNK, state=s_src)
            yc = con(_to_device_hidden(mesh_device, h), seq_len_actual=CHUNK, state=s_con)
            outs[name] = (_to_host_hidden(mesh_device, ys), _to_host_hidden(mesh_device, yc), src.last_path)
        for k, lid in ((0, 2), (1, 3)):
            a, b = outs["A"][k], outs["B"][k]
            _, pcc = comp_pcc(b, a)
            rel = float((a - b).norm() / b.norm())
            same = bool(torch.equal(a, b))
            logger.info(
                f"[v41 path A] chunk {c} layer {lid}: path {outs['A'][2]} vs {outs['B'][2]}: PCC {pcc:.6f}, rel {rel:.5f}, "
                f"identical {same}, rms {float(b.square().mean().sqrt()):.4f}"
            )
            worst[(c, lid)] = pcc
            if c == 0:
                assert (
                    outs["A"][2] == "B" and same
                ), f"chunk 0 must take path B in both sets and match exactly (layer {lid})"
    later = [p for (c, _), p in worst.items() if c > 0]
    assert outs["A"][2] == "A", "path A never engaged (the last chunk ran path B)"
    assert min(later) >= PCC, f"path A vs B: min PCC {min(later):.6f} < {PCC} ({worst})"
