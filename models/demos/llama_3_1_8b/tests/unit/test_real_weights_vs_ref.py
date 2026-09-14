# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device vs the package's own reference, REAL weights, the FULL 10240-token trace length.

The acceptance test measures the device against the golden trace, and that comparison is bounded by
the trace's own rope convention (see ``tests/torch_ref/test_golden_trace_rope_precision.py``). This
test removes the trace from the loop: same real checkpoint, same token ids, same full length, but the
oracle is ``reference/model.py``, which is independently pinned to HuggingFace at PCC 0.99999.

It is **depth-reduced to layer 0 and full-length in the sequence** — the opposite reduction from
``test_model_sp_vs_ref.py``, and chosen because layer 0's K and V need no attention on the host side
(embedding -> norm -> k/v projection -> rope), so a full 10240-token oracle costs seconds. What it
isolates is exactly the thing the trace comparison cannot separate: whether the device's error grows
with position (a phase problem) or stays flat (arithmetic).

Both prefill modes run, so it also states P2's property against a non-trace oracle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference import model as ref
from models.demos.llama_3_1_8b.reference.config import LlamaConfig
from models.demos.llama_3_1_8b.tests.common import assert_pcc, galaxy_mesh, pcc
from models.demos.llama_3_1_8b.tt.attention.kv_cache import read_slot_kv
from models.demos.llama_3_1_8b.tt.model_config import load_state_dict
from models.demos.llama_3_1_8b.tt.runners.prefill_kv_validation import naturalize
from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import PrefillRuntimeConfig, TtPrefillRuntime
from models.demos.llama_3_1_8b.utils.rope_layout import hf_to_meta_perm

CHECKPOINT = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"
TRACE = os.getenv("PREFILL_TRACE_DIR") or f"{CHECKPOINT}/golden/synthetic_10240"
CHUNK = 5120

pytestmark = pytest.mark.skipif(
    not (Path(CHECKPOINT, "model.safetensors.index.json").exists() and Path(TRACE, "metadata.json").exists()),
    reason=f"needs the real checkpoint at {CHECKPOINT} and the golden trace at {TRACE} (for its token ids)",
)


def _reference_layer0_kv(cfg, sd, ids):
    """Layer 0's post-RoPE K and raw V from the torch reference — no attention needed."""
    from models.demos.llama_3_1_8b.reference.model import REF_DTYPE, Attention, RMSNorm

    n = ids.shape[-1]
    emb = torch.nn.functional.embedding(ids, sd["model.embed_tokens.weight"].to(REF_DTYPE))
    norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    norm.weight.data = sd["model.layers.0.input_layernorm.weight"].to(REF_DTYPE)
    attn = Attention(cfg)
    attn.load_state_dict(
        {
            k[len("model.layers.0.self_attn.") :]: v.to(REF_DTYPE)
            for k, v in sd.items()
            if k.startswith("model.layers.0.self_attn.")
        },
        strict=True,
    )
    _, k, v = attn.project(norm(emb))
    cos, sin = ref.rope_cos_sin(cfg, n, dtype=torch.float32)
    return ref.apply_rope(k, cos, sin).float(), v.float()


@galaxy_mesh()
@pytest.mark.parametrize("chunked", [False, True], ids=["one_shot", "chunked"])
def test_layer0_real_weights_vs_ref_full_length(mesh_device, device_params, chunked, topology_name):
    from models.demos.llama_3_1_8b.conftest import CCL_TOPOLOGY

    cfg = LlamaConfig.from_json()
    cfg.num_hidden_layers = 1
    ids_list = json.load(open(Path(TRACE, "metadata.json")))["token_ids"]
    n = len(ids_list)
    padded = ((n + CHUNK - 1) // CHUNK) * CHUNK

    sd = load_state_dict(CHECKPOINT, num_layers=1)
    runtime = TtPrefillRuntime(
        mesh_device,
        cfg,
        sd,
        PrefillRuntimeConfig(
            num_layers=1,
            max_seq_len=padded,
            chunk_size=CHUNK if chunked else padded,
            mesh_shape=tuple(mesh_device.shape),
            topology=CCL_TOPOLOGY,
        ),
    )
    kv_cache = runtime.allocate_kv_cache()
    runtime.prefill_sequence(ids_list, kv_cache)

    k_blk, v_blk = read_slot_kv(mesh_device, kv_cache, 0, 1)
    c = runtime.config
    dev_k = naturalize(k_blk[0], n, c.sp, c.chunk_size, c.max_seq_len).unsqueeze(0)
    dev_v = naturalize(v_blk[0], n, c.sp, c.chunk_size, c.max_seq_len).unsqueeze(0)

    ids = torch.tensor(ids_list).unsqueeze(0)
    ref_k, ref_v = _reference_layer0_kv(cfg, sd, ids)
    perm = hf_to_meta_perm(cfg.head_dim)
    ref_k = ref_k[..., perm]  # reference is HF-ordered; the device holds Meta order

    mode = "chunked" if chunked else "one_shot"
    p_k, p_v = pcc(ref_k, dev_k), pcc(ref_v, dev_v)
    assert_pcc(f"real_weights_L0_K[{mode},{topology_name}]", p_k)
    assert_pcc(f"real_weights_L0_V[{mode},{topology_name}]", p_v)

    # The point of the test: the device's error must be FLAT across position. A phase error (which
    # is what the golden trace carries) would show as monotonic decay here too.
    blocks = [
        (s, pcc(ref_k[:, :, s : s + 2048], dev_k[:, :, s : s + 2048])) for s in range(0, n, 2048)
    ]
    logger.info(
        f"[{mode}] device vs reference, layer 0, real weights, {n} tokens: K={p_k:.6f} V={p_v:.6f}; "
        + "per-2048-token K: " + ", ".join(f"{p:.6f}" for _, p in blocks)
    )
    spread = max(p for _, p in blocks) - min(p for _, p in blocks)
    assert spread < 5e-3, f"device K error varies with position by {spread:.4f} — that is a phase error, not arithmetic"
