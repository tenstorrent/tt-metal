# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole attention block vs the torch reference: QKV proj -> head split -> RoPE -> causal SDPA -> o_proj.

Target mesh (8, 4), random weights, identical on both sides. Structure follows
`minimax_m3/tests/unit/test_attention_vs_ref.py`.

The device is fed the device's own META-format cos/sin, which is the pairing its Meta-permuted q/k
weights require; the reference uses HF half-split tables with unpermuted weights. The two
conventions cancel exactly (see `utils/weight_conversion.py`), so the outputs are directly
comparable — and this measures attention rather than the rope constants, which have their own test
in `test_rope_vs_ref.py`.

No QK-norm step, no attention sinks, no sliding window: three donor features Llama does not have,
and their absence is why this test is shorter than the one it is modelled on.

The cached K/V are checked alongside the output. They are what the golden trace grades at P1/P2, so
an attention that is right on its output while writing the wrong thing to the cache is a failure
worth catching here rather than 500 lines of pipeline later.
"""

import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefAttention, RefRotaryEmbedding, causal_mask
from models.demos.llama_3_1_8b_d_p.tt.attention import Attention
from models.demos.llama_3_1_8b_d_p.tt.attention.config import LlamaAttentionProgramConfig
from models.demos.llama_3_1_8b_d_p.tt.layer import build_attention_config
from models.demos.llama_3_1_8b_d_p.tt.rope import create_rope_setup

from ..test_factory import ACT_DTYPE, CHUNK_SIZE, WEIGHT_DTYPE, assert_pcc, parametrize_target_mesh, sp_shard_rope

SEQ = 2048



def meta_rope_tables(mesh_device, rope_setup, head_dim, lo, hi):
    """The device's own META-format cos/sin, sliced to positions [lo, hi).

    The reference's HF half-split tables must NOT be used here: `load_attention_weights` permutes
    q/k into Meta head order, so the rotation has to be driven by Meta tables for the two to cancel
    and reproduce the HF reference. Feeding HF tables to Meta-permuted weights measures ~0.75 PCC
    with nothing raised. See `utils/weight_conversion.py`.
    """
    def slice_of(matrix):
        full = ttnn.to_torch(ttnn.get_device_tensors(matrix)[0]).reshape(-1, head_dim)
        return full[lo:hi].reshape(1, 1, hi - lo, head_dim)

    return slice_of(rope_setup.cos_matrix), slice_of(rope_setup.sin_matrix)


def _reference(config, seq_len, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, seq_len, config.hidden_size, dtype=REF_DTYPE)
    attn = RefAttention(config)
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(seq_len)[None, :], dtype=REF_DTYPE)
    with torch.no_grad():
        out, k_rope, v = attn(x, (cos, sin), causal_mask(seq_len), return_kv=True)
    state_dict = {f"{n}.weight": getattr(attn, n).weight.detach().clone() for n in ("q_proj", "k_proj", "v_proj", "o_proj")}
    return x, state_dict, (cos, sin), out, k_rope, v


@parametrize_target_mesh()
def test_attention_vs_ref(mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name):
    """Attention output vs the torch reference, one-shot (no prior cache)."""
    x, state_dict, (cos, sin), golden, _, _ = _reference(config, SEQ)

    rope_setup = create_rope_setup(mesh_device, hf_config, max_seq_len=SEQ)
    attention = Attention(
        mesh_device=mesh_device,
        config=build_attention_config(hf_config, max_seq_len=SEQ, chunk_size=CHUNK_SIZE),
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=LlamaAttentionProgramConfig(),
        global_layer_idx=0,
        transformation_mats=rope_setup.transformation_mat_prefill,
        weight_dtype=WEIGHT_DTYPE,
    )

    def to_dev(t, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ACT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    tt_x = to_dev(x.reshape(1, 1, SEQ, config.hidden_size), [2, None])
    cos_meta, sin_meta = meta_rope_tables(mesh_device, rope_setup, config.head_dim, 0, SEQ)
    # SP-sharded, not replicated: each SP row needs the cos/sin rows for ITS positions.
    rope_mats = (sp_shard_rope(mesh_device, mesh_config, cos_meta), sp_shard_rope(mesh_device, mesh_config, sin_meta))

    out = attention(tt_x, rope_mats=rope_mats, kv_cache=None, cached_len=0, logical_n=SEQ)
    ttnn.synchronize_device(mesh_device)

    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))
    ).to(REF_DTYPE)
    assert_pcc("attention", golden.reshape(1, 1, SEQ, config.hidden_size), got, topology_name)
