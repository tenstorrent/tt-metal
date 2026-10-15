# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DS41F-0067: V4.1 prefill CSA attention CHUNKED vs the checkpoint's ``inference/model.py`` on REAL inputs -- the check that
decides whether the sparse path A is as accurate as the dense path B.

Layers 2 (index / KV source) and 3 (its Reuse consumer) run over a prefill golden (``V41_PREFILL_TRACE``, default the
1536-token needle-4096 trace) in ``V41_CHUNK``-token chunks (default 512). With 512-token chunks, chunk 2 (tokens 1024..1535)
is the first whose context holds a window + top-k worth of entries (top-k 512 x rate 2), so ``PREFILL_V41_PATH_A=1`` runs
it on path A and chunks 0 / 1 on path B; ``=0`` runs every chunk on path B. Every chunk's output rows are compared with
model.py's rows (computed on the whole sequence at once, i.e. chunk-independent). Run once per setting and compare chunk 2.

    PYTHONPATH=<this worktree>:<cpu_stub> PREFILL_V41_PATH_A=1 pytest models/demos/deepseek_v3_d_p/tests/pcc/test_v41_attention_chunked.py
"""

from __future__ import annotations

import os

import pytest
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tests.pcc import test_v41_attention as base
from models.demos.deepseek_v3_d_p.tt.v41.attention import V41CSA, V41CSAConsumer, v41_hf_config
from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

CHUNK = int(os.environ.get("V41_CHUNK", "512"))
PCC = float(os.environ.get("V41_TEST_PCC", "0.99"))


@pytest.mark.parametrize("mesh_device, device_params", base._MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v41_attention_chunked_vs_model_py(mesh_device, device_params):
    seq = base.SEQ
    assert seq % CHUNK == 0, (seq, CHUNK)
    cfg = V41Config.load()
    ref = base._reference_attention([2, 3])
    ck = checkpoint()
    rot = DeepseekV4RotaryEmbedding(v41_hf_config(cfg, max_seq=seq))
    src = V41CSA.from_weights(mesh_device, cfg, 2, ck.layer(2), rot)
    con = V41CSAConsumer.from_weights(mesh_device, cfg, 3, ck.layer(3), rot, source=src)
    states = {2: src.alloc_state(seq, chunk_tokens=CHUNK), 3: con.alloc_state(seq, chunk_tokens=CHUNK)}
    mods = {2: src, 3: con}
    path_a = os.environ.get("PREFILL_V41_PATH_A", "0") == "1"
    worst = 1.0
    for c in range(seq // CHUNK):
        r0, r1 = c * CHUNK, (c + 1) * CHUNK
        for lid in (2, 3):
            h = ref[lid][0][r0:r1]
            y = mods[lid](base._to_device_hidden(mesh_device, h), seq_len_actual=CHUNK, state=states[lid])
            got = base._to_host_hidden(mesh_device, y)[:CHUNK]
            want = ref[lid][1][r0:r1]
            _, pcc = comp_pcc(want, got)
            rel = float((got - want).norm() / want.norm())
            logger.info(
                f"[v41 chunked] PATH_A={int(path_a)} chunk {c} [{r0}, {r1}) layer {lid}: path {getattr(src, 'last_path', '?')}, "
                f"PCC vs model.py {pcc:.6f}, rel {rel:.4f}"
            )
            worst = min(worst, pcc)
    assert worst >= PCC, f"chunked attention vs model.py: min PCC {worst:.6f} < {PCC}"
