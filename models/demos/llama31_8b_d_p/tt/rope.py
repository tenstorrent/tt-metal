# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3 scaled RoPE tables. **The only place theta and the scaling parameters are read.**

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding` (`rope_type`
`"llama3"`). **Template:** the scaling math is *imported*, not rewritten (`DEC-015`) —
`models/tt_transformers/tt/common.py:489` `precompute_freqs` -> `:437` `apply_scaling` -> `:405`
`compute_llama3_parameters`; table assembly `:534` `get_prefill_rot_mat`; transformation matrix
`:562` `get_rot_transformation_mat`. Structural template for the indexed builder:
`models/demos/gpt_oss_d_p/tt/rope.py:115`.

**Why "the only place" is a load-bearing claim.** On `transformers` 5.12.1 a live config object has
no `rope_theta` attribute, so `getattr(cfg, "rope_theta", 10000.0)` — the pattern at
`models/demos/gpt_oss_d_p/tt/model_config.py:76` — **succeeds** and returns 10000.0 against Llama's
500000.0, giving a RoPE that is wrong at every position with no exception anywhere. This is the
highest-severity silent-wrongness trap in the bring-up (recipe P1 trap 1, `07_RISKS.md` R-005).
Theta and scaling are therefore read exactly once, here, through
`models/tt_transformers/tt/common.py:165` `get_rope_theta` and `:183` `get_rope_scaling`, on the
**raw `config.json` dict** — not `cfg.to_dict()`, which has neither key — and asserted non-`None`.

**Convention: Meta / interleaved** (`ttnn.experimental.rotary_embedding_llama` plus a
transformation matrix), which is what both prefill templates use
(`models/demos/gpt_oss_d_p/tt/attention/operations.py:87`,
`models/demos/minimax_m3/tt/attention/operations.py:93`), so the surrounding prefill scaffolding
already assumes it. The cost is that Q/K **projection weights** must be `reverse_permute`d at load
(`models/tt_transformers/tt/load_checkpoints.py:891`); that happens inside
`tt/attention/weights.py` so a weight can never reach the device un-swizzled by a path that forgot.
The alternative is `ttnn.experimental.rotary_embedding_hf`, which removes the permute and is the
likely direction of travel (`models/tt_transformers/tt/model_config.py:623` defaults
`use_hf_rope=False` today, issue #37605) — `DEC-011`, `DEC-033`.

**Two builders, deliberately separate.** `build_prefill_rope` is contiguous and asserts
`start_pos <= seq_len`, because `models/tt_transformers/tt/common.py:525` `gather_cos_sin` indexes
a table of `seq_len * 2` rows and a chunked call would read out of bounds — the
`RuntimeError: index N is out of bounds` landmine, whose message names neither RoPE nor chunking.
Chunked prefill (P7) must reach for `build_indexed_rope` instead.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.tt_transformers.tt.common import (
    get_prefill_rot_mat,
    get_rope_scaling,
    get_rope_theta,
    get_rot_transformation_mat,
    precompute_freqs,
)

from .config import derive_head_dim

# `models/tt_transformers/tt/common.py:407-408` hard-codes these as LOCAL CONSTANTS inside
# `compute_llama3_parameters`; they are NOT read from `config.json`. Benign for Llama-3.x, silently
# wrong for any model that changes them, so `assert_llama3_factors` checks the config against them
# rather than trusting them (`07_RISKS.md` R-010).
_HELPER_LOW_FREQ_FACTOR = 1
_HELPER_HIGH_FREQ_FACTOR = 4
_EXPECTED_ROPE_TYPE = "llama3"


def assert_llama3_factors(hf) -> None:
    """Check the config's llama3 scaling factors against the helper's hard-coded literals.

    Closes `07_RISKS.md` R-010: `compute_llama3_parameters` takes only
    `(freqs, scale_factor, orig_context_len)`, so `low_freq_factor` / `high_freq_factor` from the
    config are never consulted. If a checkpoint ever ships different ones, the imported math is
    silently wrong and nothing raises.
    """
    scaling = get_rope_scaling(hf)
    assert scaling is not None, "rope_scaling is missing; get_rope_scaling returned None"
    assert scaling.get("rope_type") == _EXPECTED_ROPE_TYPE, f"expected llama3 rope, got {scaling.get('rope_type')!r}"
    assert float(scaling["low_freq_factor"]) == float(_HELPER_LOW_FREQ_FACTOR), (
        f"config low_freq_factor {scaling['low_freq_factor']} != the {_HELPER_LOW_FREQ_FACTOR} "
        f"hard-coded at models/tt_transformers/tt/common.py:407 — the imported scaling math would be wrong"
    )
    assert float(scaling["high_freq_factor"]) == float(_HELPER_HIGH_FREQ_FACTOR), (
        f"config high_freq_factor {scaling['high_freq_factor']} != the {_HELPER_HIGH_FREQ_FACTOR} "
        f"hard-coded at models/tt_transformers/tt/common.py:408 — the imported scaling math would be wrong"
    )


def rope_params(hf) -> tuple[float, float, int]:
    """`(theta, scale_factor, original_max_position_embeddings)`, read once and asserted non-`None`.

    Every other function in this module goes through here, and nothing outside this module reads
    those three values (recipe P1 trap 1).
    """
    assert_llama3_factors(hf)
    theta = get_rope_theta(hf)
    scaling = get_rope_scaling(hf)
    assert theta is not None, "rope_theta is None — get_rope_theta found neither key (recipe P1 trap 1)"
    scale_factor = scaling.get("factor")
    orig_context_len = scaling.get("original_max_position_embeddings")
    assert scale_factor is not None, "rope_scaling.factor is None"
    assert orig_context_len is not None, "rope_scaling.original_max_position_embeddings is None"
    return float(theta), float(scale_factor), int(orig_context_len)


def llama3_freqs(hf, seq_len: int, *, scaled: bool = True):
    """Half-dimension `(cos, sin)` tables, `[seq_len, head_dim/2]`, from the imported helper.

    `scaled=False` disables the llama3 piecewise scaling and exists for one purpose: `G-ROPE` has
    to prove the scaling actually took effect, and a test that passes with scaling silently
    disabled is worthless (`BRINGUP_RECIPE.md:1340-1349`).
    """
    theta, scale_factor, orig_context_len = rope_params(hf)
    return precompute_freqs(
        derive_head_dim(hf),
        seq_len,
        theta=theta,
        scale_factor=scale_factor if scaled else None,
        orig_context_len=orig_context_len,
        rope_type=_EXPECTED_ROPE_TYPE,
    )


def _interleave_meta(half: torch.Tensor) -> torch.Tensor:
    """`[S, head_dim/2]` -> `[1, 1, S, head_dim]` in the Meta convention `[c0, c0, c1, c1, ...]`.

    The same stacking `models/tt_transformers/tt/common.py:525` `gather_cos_sin` does. Getting this
    wrong — concatenating the halves instead, which is the HF convention — is *the* classic RoPE
    bug, and the negative control in `G-ROPE` is exactly that mistake.
    """
    return torch.stack([half, half], dim=-1).flatten(-2)[None, None]


def build_prefill_rope(mesh_device, hf, seq_len: int, start_pos: int = 0):
    """Contiguous prefill cos/sin, `[1, 1, seq_len, head_dim]` bf16 TILE, **replicated**.

    Positions `start_pos .. start_pos + seq_len - 1`. Delegates to
    `models/tt_transformers/tt/common.py:534` `get_prefill_rot_mat`, which builds a table of
    `seq_len * 2` rows and gathers from it. The helper builds **bf16** tensors and takes no dtype
    argument, so neither does this.

    Returns `[cos, sin]` as `ttnn` tensors.
    """
    assert start_pos <= seq_len, (
        f"start_pos {start_pos} > seq_len {seq_len}: get_prefill_rot_mat builds a table of "
        f"{2 * seq_len} rows and gather_cos_sin would read out of bounds "
        f"(models/tt_transformers/tt/common.py:525). Chunked prefill must use build_indexed_rope."
    )
    theta, scale_factor, orig_context_len = rope_params(hf)
    return get_prefill_rot_mat(
        derive_head_dim(hf),
        mesh_device,
        seq_len,
        theta=theta,
        scale_factor=scale_factor,
        orig_context_len=orig_context_len,
        start_pos=start_pos,
    )


def build_indexed_rope(mesh_device, hf, *, max_seq_len: int, chunk_size: int, sp_axis: int = 0, dtype=ttnn.bfloat16):
    """Whole-cache, block-cyclic, SP-sharded cos/sin for the **indexed** on-device RoPE, built ONCE.

    Covers every cache position up to `max_seq_len`, block-cyclic-reordered by the per-chip chunk
    (`chunk_size // sp`) and then SP-sharded on `sp_axis`, so device `c`'s contiguous shard holds —
    in local-cache-row order — the RoPE for every global position it will carry.
    `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` then picks this chunk's rows
    on-device from a single `kv_actual_global` runtime argument, with no per-chunk host reshard.

    Persistent: reused across all chunks, so do **not** deallocate it per chunk. Consumed in P7;
    written here because P5.3 owns "expose a chunk-offset table builder separately from the
    contiguous one" (`BRINGUP_RECIPE.md:1332-1334`).

    Returns `[cos, sin]`.
    """
    sp = mesh_device.shape[sp_axis]
    assert (
        chunk_size % (ttnn.TILE_SIZE * sp) == 0
    ), f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
    assert max_seq_len % chunk_size == 0, f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})"
    chunk_local = chunk_size // sp

    cos_half, sin_half = llama3_freqs(hf, max_seq_len)
    cos = block_cyclic_reorder(_interleave_meta(cos_half), chunk_local, sp, seq_dim=2)
    sin = block_cyclic_reorder(_interleave_meta(sin_half), chunk_local, sp, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2  # SP-shard the seq dim; replicate across TP
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(shard_dims))

    def _to_dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    return [_to_dev(cos), _to_dev(sin)]


def build_transformation_mat(mesh_device, dtype=ttnn.bfloat16):
    """The `[1, 1, 32, 32]` replicated RoPE transformation matrix.

    `models/tt_transformers/tt/common.py:562` `get_rot_transformation_mat` is called with **no
    arguments**: `:564` reassigns `dhead = 32` regardless of what was passed, so a `dhead=128` call
    would silently be ignored (recipe P1 trap 4). The op works on a single tile.
    """
    return ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
