# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RoPE setup for Llama-3.1-8B prefill.

Borrowed from `minimax_m3/tt/model.py:create_rope_setup`, which builds `tt_transformers`'
`RotarySetup`. That path already carries llama3 rope scaling through
`rope_scaling_model_factory` -> `RopeScalingLlama3`, so the model this bring-up is for is the one
that plumbing was written for — the donor's own partial-rotary branch is what does not apply here.

Two things to hold onto:

* **Full rotation.** `rotary_dim == head_dim == 128`, so the tables are built at the full head width
  and `apply_rope` takes its plain full-rotation branch. The donor builds its tables at
  `rotary_dim` 64 and slices each head into rotate / pass-through halves; carrying that over would
  silently leave half of every head unrotated.
* **The tables must be the half-split convention.** `reference/model.py` is explicit that HF Llama
  uses `emb = cat(freqs, freqs)` + `rotate_half`, not the Meta interleaved pairing.
  `tests/unit/test_rope_vs_ref.py` measures the device tables against the reference's, which is
  where a convention mismatch surfaces — it produces plausible-looking output either way.

`tt_transformers` is not a `common/prefill` package and never ran on this engine, so it is a source
for the rope MATH only. Nothing else is taken from it.
"""

import ttnn
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup


def create_rope_setup(
    mesh_device,
    hf_config,
    *,
    max_seq_len=None,
    max_local_batch_size=1,
    users_row_sharded=False,
    datatype=ttnn.bfloat16,
    shard_batch_to_mesh_dim=0,
):
    """Build the `RotarySetup` carrying this model's cos/sin and transformation matrices.

    Args:
        max_seq_len: table length. Defaults to the config's `max_position_embeddings` (131072).
            A chunked run wants the WHOLE-cache tables (rounded up to a chunk boundary), because
            indexed rope derives each chunk's start row on-device from them.
    """
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    rope_scaling_params = getattr(hf_config, "rope_scaling", None)
    assert rope_scaling_params, "Llama-3.1 has llama3 rope scaling; an empty rope_scaling means it was lost"
    assert rope_scaling_params.get("rope_type") == "llama3", (
        f"expected llama3 rope scaling, got {rope_scaling_params.get('rope_type')!r}"
    )
    rope_scaling = rope_scaling_model_factory(rope_scaling_params)

    # rope_theta lives in different places across the transformers 4.x/5.x boundary; take whichever
    # the installed version exposes rather than defaulting silently to the wrong base.
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(hf_config, "rope_parameters", {}).get("rope_theta")
    assert rope_theta, "rope_theta is missing from the config; see reference/config.py:to_hf_config"

    batch_size = max_local_batch_size * mesh_device.shape[0] if users_row_sharded else max_local_batch_size
    return RotarySetup(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=head_dim,  # FULL head: no partial-rotary narrowing
        max_seq_len=max_seq_len or hf_config.max_position_embeddings,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        datatype=datatype,
        shard_batch_to_mesh_dim=shard_batch_to_mesh_dim,
    )
