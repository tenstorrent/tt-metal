# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RoPE on the target mesh vs the torch reference.

Not a row of the recipe's decoder table, but written anyway (gpt_oss_d_p has its own
`test_indexed_rope_vs_ref.py` for the same reason): rope carries two failure modes that produce
plausible output rather than an error, and both would otherwise surface as an unexplained PCC drop
inside the attention test.

1. **The rotation convention.** HF Llama is half-split (`emb = cat(freqs, freqs)` + `rotate_half`);
   the original Meta checkpoints are interleaved. Same weights, different pairing of head columns.
2. **llama3 frequency scaling.** It only bites past the original 8192-token context, so a table
   built without it is indistinguishable from a correct one at short ISL and destroys accuracy at
   the 131072 the spec asks for. Covered here at positions on both sides of that boundary.

Full-width rotation: `rotary_dim == head_dim == 128`, so `apply_rope` takes its plain branch and the
donor's partial-rotary slice/concat is never entered.

**Everything on the device side is in META (interleaved) order.** `rotary_embedding_llama` and
`rotary_embedding_indexed` both consume Meta-format tables, which is what `RotarySetup` builds, so
this package permutes the q/k projection weights from HF to Meta at load — see
`utils/weight_conversion.py` for why that is free (a dot product is invariant under a permutation
applied to both operands, and v/o are never rotated). These tests therefore compare the device
against `hf_to_meta_head_dim(<reference>)` rather than against the reference directly. Comparing
against the raw HF-order reference measures PCC 0.75 — plausible output, silently wrong.
"""

from dataclasses import replace

import pytest
import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefRotaryEmbedding, apply_rotary_pos_emb
from models.demos.llama_3_1_8b_d_p.tt.attention.operations import apply_rope
from models.demos.llama_3_1_8b_d_p.tt.rope import create_rope_setup
from models.demos.llama_3_1_8b_d_p.utils.weight_conversion import hf_to_meta_head_dim

from ..test_factory import ACT_DTYPE, assert_pcc, comp_pcc, parametrize_target_mesh

SEQ = 1024


@parametrize_target_mesh()
def test_rope_tables_vs_ref(mesh_device, device_params, config, hf_config, topology_name):
    """The device cos/sin tables against the reference's, over the first SEQ positions."""
    rope_setup = create_rope_setup(mesh_device, hf_config, max_seq_len=SEQ)
    cos = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.cos_matrix)[0]).to(REF_DTYPE)
    sin = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.sin_matrix)[0]).to(REF_DTYPE)

    ref = RefRotaryEmbedding(config)
    cos_ref, sin_ref = ref(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)

    cos = cos.reshape(-1, config.head_dim)[:SEQ]
    sin = sin.reshape(-1, config.head_dim)[:SEQ]
    # Device tables are Meta-interleaved; the reference is HF half-split. Compare like with like.
    assert_pcc("rope.cos", hf_to_meta_head_dim(cos_ref[0]), cos, topology_name)
    assert_pcc("rope.sin", hf_to_meta_head_dim(sin_ref[0]), sin, topology_name)
    # And confirm the two conventions really are distinguishable here, so the test is not vacuous.
    assert comp_pcc(cos_ref[0], cos) < 0.999, "Meta and HF cos tables are indistinguishable; check the op convention"


@parametrize_target_mesh()
def test_rope_scaling_is_applied(mesh_device, device_params, config, hf_config, topology_name):
    """The device tables must carry llama3 scaling, checked where unscaled rope would differ.

    Positions past the original 8192 context are where the two diverge; below it the low-frequency
    band is already scaled, so the check is meaningful at both ends.
    """
    long_seq = 16384
    rope_setup = create_rope_setup(mesh_device, hf_config, max_seq_len=long_seq)
    cos = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.cos_matrix)[0]).to(torch.float32)
    cos = cos.reshape(-1, config.head_dim)[:long_seq]

    scaled = RefRotaryEmbedding(config)
    cos_scaled, _ = scaled(torch.arange(long_seq)[None, :], dtype=torch.float32)

    # dataclasses.replace, not `type(config)(**config.__dict__)`: the config carries cached_property
    # values (head_dim) that are not constructor fields.
    unscaled = RefRotaryEmbedding(replace(config, rope_scaling=dict(config.rope_scaling, factor=1.0)))
    cos_unscaled, _ = unscaled(torch.arange(long_seq)[None, :], dtype=torch.float32)

    assert_pcc("rope.cos[16k, llama3-scaled]", hf_to_meta_head_dim(cos_scaled[0]), cos, topology_name)
    pcc_unscaled = comp_pcc(hf_to_meta_head_dim(cos_unscaled[0]), cos)
    assert pcc_unscaled < 0.999, (
        f"device tables match UNSCALED rope at 16k (PCC {pcc_unscaled:.6f}) - llama3 scaling was lost"
    )


@parametrize_target_mesh()
def test_apply_rope_vs_ref(mesh_device, device_params, config, hf_config, mesh_config, topology_name):
    """Rotated Q and K on device vs the reference, fed the SAME cos/sin tables.

    Sharing the tables is deliberate: it makes this measure the rotation op rather than the constants,
    which `test_rope_tables_vs_ref` already covers on its own.
    """
    rope_setup = create_rope_setup(mesh_device, hf_config, max_seq_len=SEQ)
    ref = RefRotaryEmbedding(config)
    cos_ref, sin_ref = ref(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)

    n_q_local = config.q_heads_per_chip(mesh_config.tp)  # 8
    n_kv_local = config.kv_heads_per_chip(mesh_config.tp)  # 2
    torch.manual_seed(0)
    q = torch.randn(1, n_q_local, SEQ, config.head_dim, dtype=REF_DTYPE)
    k = torch.randn(1, n_kv_local, SEQ, config.head_dim, dtype=REF_DTYPE)
    # Rotate in HF space, then permute the RESULT to Meta order: that is what the device — fed
    # Meta-permuted inputs and Meta tables — must reproduce.
    q_ref, k_ref = apply_rotary_pos_emb(q, k, cos_ref, sin_ref)
    q_ref, k_ref = hf_to_meta_head_dim(q_ref), hf_to_meta_head_dim(k_ref)

    def to_dev(t, dtype=ACT_DTYPE):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # The DEVICE's own Meta-format tables, sliced to this sequence; and Meta-permuted q/k inputs.
    cos_dev = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.cos_matrix)[0]).reshape(-1, config.head_dim)[:SEQ]
    sin_dev = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.sin_matrix)[0]).reshape(-1, config.head_dim)[:SEQ]
    rope_mats = (
        to_dev(cos_dev.reshape(1, 1, SEQ, config.head_dim)),
        to_dev(sin_dev.reshape(1, 1, SEQ, config.head_dim)),
    )
    trans = rope_setup.transformation_mat_prefill

    q_dev = apply_rope(to_dev(hf_to_meta_head_dim(q)), rope_mats, trans, is_decode_mode=False)
    k_dev = apply_rope(to_dev(hf_to_meta_head_dim(k)), rope_mats, trans, is_decode_mode=False)
    ttnn.synchronize_device(mesh_device)

    assert_pcc("rope.q", q_ref, ttnn.to_torch(ttnn.get_device_tensors(q_dev)[0]), topology_name)
    assert_pcc("rope.k", k_ref, ttnn.to_torch(ttnn.get_device_tensors(k_dev)[0]), topology_name)


@parametrize_target_mesh()
def test_apply_rope_rejects_partial_tables(mesh_device, device_params, config):
    """A rope table narrower than head_dim means the donor's partial-rotary path leaked in."""
    half = config.head_dim // 2
    narrow = torch.zeros(1, 1, 32, half, dtype=REF_DTYPE)
    tensor = torch.zeros(1, 1, 32, config.head_dim, dtype=REF_DTYPE)

    def to_dev(t):
        return ttnn.from_torch(
            t, device=mesh_device, dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    with pytest.raises(AssertionError, match="full head"):
        apply_rope(to_dev(tensor), (to_dev(narrow), to_dev(narrow)), None, is_decode_mode=False)
