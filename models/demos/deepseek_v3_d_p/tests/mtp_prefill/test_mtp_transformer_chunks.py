# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Three 5120-token chunks of GLM-5.2 MTP4 through the real ``TtPrefillTransformer`` (#53533).

The MTP token contract: per chunk the socket delivers ``C + n_mtp`` ids (5120 + 32), cut into each
chip's trunk row and the lookahead row that abuts it; level ``k`` then reads the local slice
``[k+1 : k+1+L)`` of the on-device union those two are gathered into, and that slice -- embedded,
not a hidden state -- is level ``k``'s input. Three things in it can be wrong, none of them a shape
error:

1. **Stream / socket formation.** Chip ``c``'s two rows must join into the contiguous
   ``stream[c*L : c*L + L + n_mtp]``, which is what makes level ``k``'s window the SAME local slice
   on every chip. The last chunk's lookahead is pad, because the ids it wants do not exist yet: the
   DEVICE generates them, one per level, out of each level's own LM head.
2. **Slicing / upload.** ``t_{p+k+1}`` must land on the row whose hidden sits at ``p``. This runs
   the production path for it: the same ``MTPUnionEmbedding.from_ids`` the runtime calls, the same
   per-level ``union.window(k+1)``, and on the last chunk the same ``MTPDeviceGeneration`` -- so
   ``mtp_prefill/device_windows.py`` is under test here rather than bypassed. Nothing in this file
   slices a window; it only builds the two id tensors the H2D socket would have delivered.
3. **Numerics, chunked.** ``test_mtp.py::test_mtp_predictor_pcc`` gates ``TtMTPPredictor``
   single-shot; nothing gated it *chunked*, where each level's KV cache is written three times and
   every chunk after the first attends over keys the earlier ones left behind. The reference is
   **teacher-forced** -- level ``k`` is fed the DEVICE's own ``H^{k-1}`` -- so each level's PCC is a
   statement about that level alone. Its embedding is a HOST gather of the ids the level *should*
   have asked for: bit-identical to the device's when the window is right, unrelated to it when it
   is not, so a wrong window and wrong math fail the same assertion.

Two axes on top of that. ``num_layers`` picks the trunk depth -- 78 (GLM-5.2's own) or 1, a shallow
variant for iterating on the MTP path without paying for the trunk. ``skip_pcc`` picks the numerics
-- ``pcc`` is the real test, ``nopcc`` keeps the device path and claim 1 and drops the rest. Neither
is a cheaper way to VALIDATE: the shallow trunk still runs the full CPU reference, and ``nopcc``
runs none of it. ``layers1-nopcc`` is the fast iteration leg and the one that proves least.

What the test keeps for itself is the EXPECTATION, never the input. :func:`_expected_window` derives
the ids each level should have read from the definition of MTP, and the reference is driven by a HOST
gather of those. Feed the reference the device's own windows instead and a level that read the wrong
rows would agree with it perfectly -- which is why there is no recorder anywhere in here.

The prompt id at absolute position ``p`` is ``p + 1`` (0 stays reserved for pad), so a failure reads
as "row j is carrying position q".

What it deliberately does NOT claim
-----------------------------------
* **A legible diagnosis of a wrong window.** Claim 2 detects one decisively, but reports it as a
  collapsed PCC rather than as "level 2 asked for id X, expected Y":
  ``glm_mtp_predictor_reference`` takes *embeddings* and never sees a token id, and the device's
  windows never become ids at all -- they are row slices of an embedding. Recording what the device
  asked for is not an option either: it has no counterpart in production, and feeding the reference
  the device's own rows makes the numerics blind to the mapping outright. Assertion order recovers
  most of the localisation: fused projection first, at 0.999.
* **The D2D half of the union.** ``MTPUnionEmbedding.from_ids`` and every window slice are covered;
  ``from_embedding``, ``.parts`` and ``_mtp_pack_activation`` are not, because they exist for a
  downstream rank and this test is single-galaxy. The branch's own argument is that the two
  constructions agree by construction -- ``slice(embed(ids)) == embed(slice(ids))`` row for row --
  which is sound for the rows and says nothing about a mis-sized socket.
* **The KVPE cache contents.** Level outputs are compared; the slots they wrote are not
  (``test_mtp.py::test_mtp_predictor_pcc`` does that single-shot). A slot *collision* still shows up
  from chunk 1 on -- two levels sharing a slot read each other's keys -- but a consistent off-by-N
  into slots nothing else uses does not.
* **Mid-slab chunk starts.** All three chunks start at a multiple of the global chunk size, so
  ``rotated_chip_positions``' KV-pad-aware rotation is degenerate. A non-aligned boundary is a
  placement question, and nothing covers it today.
* **Multi-rank.** ``TtPrefillTransformer`` asserts MTP needs an embedding table on the rank that
  runs the tail, which today means single-galaxy (``is_first_rank == is_last_rank``).
* **A checkpoint-free run.** Every weight is a real GLM-5.2 one: the trunk, the embedding table and
  the LM head come out of the TTNN weight cache, layer 78 out of the checkpoint. There is no random
  leg left, so the test skips on a box with neither rather than degrading into one. This holds at
  the shallow depth too -- ``layers1`` shortens the trunk, it does not synthesize it.
* **Real text.** The prompt is still ``p + 1`` at absolute position ``p``, the property claims 1, 3
  and 4 read their failures through. Real token ids would make ``h^0`` a realistic hidden state;
  they would also make every window assertion illegible.
"""

from __future__ import annotations

import copy
import gc
import os
import time
from pathlib import Path
from typing import Sequence

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.runner_utils import num_mtp_tokens
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import glm_mtp_predictor_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import CHAIN_FROM_NORM, TtMTPPredictor
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import MTP_CACHE_ENV, MTP_CACHE_PREFIX, enable_mtp_indexer_slot
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor, prepare_prefill_mtp_tokens
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from tests.ttnn.utils_for_testing import assert_with_pcc

SP_AXIS, TP_AXIS = 0, 1


# Restated from tests/mtp_prefill/test_mtp.py, not imported: importing a test module from a test
# module is how one file ends up collected twice under two names. Same claims -- two RMSNorms and a
# matmul earn the rmsnorm/ffn threshold, and the MTP layer is an ordinary GLM MoE block, so it earns
# test_prefill_block.py's block threshold. No per-level drift allowance: the reference is
# teacher-forced, so nothing is inherited.
FUSED_MTP_PCC = 0.999

# The block-output allowance. MEASURED, but in a regime this test no longer runs: against a RANDOM
# trunk and random layer-78 weights, L0's block output was 0.99319 / 0.99286 / 0.99268 by chunk;
# swapping in the four real MTP tensors while everything else stayed random moved it to
# 0.98259 / 0.97435 / 0.97085, sliding by -0.0082 then -0.0035. Teacher forcing pins each level's
# hidden INPUT but not the KV caches -- each side accumulates K/V from its own outputs -- so chunk
# 0's divergence seeds chunk 1's and compounds; the deltas halve per chunk, so it saturates rather
# than diverging. 0.96 covered that plus 1.6e-4 to 2.9e-4 of run-to-run noise.
#
# At full depth on real weights this is PROVISIONAL and could move either way. The mechanism behind
# the drop above is a TRAINED gate scoring a SYNTHETIC input, which ``test_glm_prefill_block``
# measures at ~0.1 in isolation against ~0.995 in context -- and a real 78-layer trunk is the "in
# context" end of that, so the number may well come back up. Against it: bfloat4_b experts, and 78
# layers of accumulated KV on both sides. Re-measure before treating it as a floor.
MTP_MODULE_OUTPUT_PCC = 0.96

# Real routing is not uniform -- a trained gate concentrates tokens on some experts -- so the MoE
# dispatch buffer needs headroom the random leg never did. 8 is what
# test_prefill_transformer_chunked.py runs the pretrained 78-layer model with; the default is 2.
DISPATCH_BUFFER_CAPACITY_FACTOR = 8

# Where layer 78's cache goes: a SIBLING of the trunk cache, derived from it rather than hardcoded,
# so it follows wherever TT_GLM52_PREFILL_TTNN_CACHE points. Layer 78 is not part of the transformer
# the trunk cache was built for, and that directory is read-only to us anyway -- but its parent is
# writable, which is what makes a sibling the right home:
#
#   /mnt/models/deepseek-prefill-cache/glm52_ttnn_cache/glm_5_2_bh_32dev/8x4      trunk, 401 GB
#   /mnt/models/deepseek-prefill-cache/glm52_mtp_ttnn_cache/glm_5_2_bh_32dev/8x4  layer 78, 5.5 GB
#
# It must be on SHARED storage. ``~/.cache`` is node-local ext4 on these galaxy nodes -- a cache
# written there is invisible from the login node and is rebuilt from scratch by the next job that
# lands on a different node. /mnt/models is NFS and is visible everywhere.
#
# 48 files, written on the first run and read on every run after. ONE layer, not four:
# TtMTPPredictor builds a single TtMTPModule and replays it K times, so all four levels share it.
#
# MTP_CACHE_ENV / MTP_CACHE_PREFIX are imported from tt/mtp_prefill/utils.py, not redeclared: the
# prefill runner reads the cache THIS test writes, and a divergence would surface as nothing worse
# than "cache incomplete" at serving time.

CHUNK = 5 * 1024  # 5120 -- the "5k chunk" of #53533, and TtPrefillTransformer.seq_len
NUM_CHUNKS = 3
TOTAL = CHUNK * NUM_CHUNKS  # 15360 prompt tokens

PAD_TOKEN = 0
"""Id written into pad positions. 0 is reserved for it -- the prompt's id at position p is p + 1."""

MTP_LEVEL_AXIS = (4, 7)
"""The level counts this test runs: MTP4 and MTP7.

Both ship, so both are gated. K = 1 is deliberately NOT here -- it is the debugging configuration,
and ``test_mtp.py::test_mtp_predictor_pcc`` keeps it as its regression leg, single-shot and cheap.

MTP7 needs no transport change at all, which is the point worth knowing: ``num_mtp_tokens`` rounds K
up to a whole tile, and 4 and 7 both round to **32**, so the socket row, the H2D page and the union's
height are byte-identical between them. What actually differs is the KVPE cache depth
(``num_layers + K``), the number of levels replayed, and the CPU reference's cost -- 7 levels is
~1.75x the host torch of 4."""

# The two depths turn out to COMPLEMENT each other on claim (4), which is worth knowing before
# dropping either. Claim (4) can only catch a patch SWAP when the generated ids differ, and which ids
# come out depends on h^0, i.e. on the trunk: at depth 78 the draft chain collapses onto repeats
# ([198, 659, 659, 154842] -- 659 twice, adjacently), at depth 1 it does not ([89467, 55969, 635,
# 429]). So the shallow leg is the one that actually gates the permutation, and the deep leg says so
# in a warning rather than pretending otherwise (see seam_is_decisive in the body).

# What a level count has to satisfy: the union must be tall enough for the deepest window
# (``window(K)`` ends at row ``K + L``) and for the K generated positions past ``actual_end``.
# ``num_mtp_tokens`` is what guarantees both, so assert it rather than trust the axis.
for _k in MTP_LEVEL_AXIS:
    assert num_mtp_tokens(_k) >= _k, f"num_mtp_tokens({_k}) = {num_mtp_tokens(_k)} cannot cover {_k} levels"


def _shard_dims():
    dims = [None, None]
    dims[TP_AXIS] = -1
    dims[SP_AXIS] = -2
    return dims


def _from_device(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """``[1, 1, C/sp, H/tp]`` per chip -> ``[1, 1, C, H]`` in POSITION order.

    Position order because this test runs ``is_balanced=False``: ``prepare_prefill_input_tensor``
    then shards by a plain ``reshape(sp, 1, C // sp)``, so concatenating the chips back along ``-2``
    is the exact inverse. Under ``is_balanced=True`` it is not, and the CPU reference -- which reads
    row ``j`` as absolute position ``actual_start + j`` -- would need
    ``reverse_reorder_tensor_chunks`` first.
    """
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=_shard_dims(), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# The MTP input: the production device path
# ---------------------------------------------------------------------------
# The windows are built ON DEVICE, by the same code serving uses: one ``MTPUnionEmbedding`` per
# chunk, from which every level slices its own rows, and on the last chunk an ``MTPDeviceGeneration``
# that fills the positions the prompt does not reach out of each level's own lm_head. This file
# builds the two id tensors the H2D socket would have delivered and hands them to the same
# ``from_ids`` the runtime calls (``tt_prefill_runtime._mtp_prepare_input``); everything after that
# is production code.
#
# What the test keeps for itself is the EXPECTATION, not the input: :func:`_expected_window` derives
# the ids each level should have read straight from the definition of MTP, and the CPU reference is
# driven by a HOST gather of those. That is what keeps the numerics sensitive to the mapping -- feed
# the reference the device's own windows and a level that read the wrong rows would agree with it
# perfectly.


def mtp_chunk_stream(
    all_tokens: Sequence[int],
    chunk_idx: int,
    chunk_size: int,
    n_mtp: int,
    *,
    pad_token: int = 0,
) -> tuple[list[int], int]:
    """A whole prompt + a chunk index -> that chunk's ``(stream, real_len)``.

    ``stream`` is ``chunk_size + n_mtp`` ids: this chunk's own ``C`` positions followed by the
    ``n_mtp`` that come after them, right-padded once the prompt runs out. One slice covers both
    kinds of chunk -- an interior one finds its lookahead in the prompt, the last one finds pad --
    because the last chunk's missing ids are not this function's job: the DEVICE generates them
    (``MTPDeviceGeneration``), and the only thing that has to know which chunk is last is the
    ``is_last_chunk`` flag on ``forward``.

    ``n_mtp`` is ``num_mtp_tokens(K)`` -- a whole tile, so 32 at MTP4, not 4. It is the socket row's
    width, and :func:`~models.demos.deepseek_v3_d_p.tt.runners.input_prep.prepare_prefill_mtp_tokens`
    slices chip ``c``'s row out of exactly this list.

    Args:
        all_tokens: the request's REAL prompt ids, unpadded. ``len(all_tokens)`` is ``P``.
        chunk_idx: which ``chunk_size``-sized chunk of them to build.
        chunk_size: ``C``, the padded chunk length.
        n_mtp: lookahead ids the socket row carries past each chip's shard.
        pad_token: id written into pad positions. Any in-vocab id works -- pad rows sit past
            ``actual_end``, where the trunk's own hidden and KV are already garbage, and on the last
            chunk the generation keep-mask zeroes the rows it is about to write.

    Returns:
        ``(stream, real_len)`` -- ``C + n_mtp`` ids indexed by chunk-local position, and this chunk's
        real-token count, which is its ``actual_isl``.
    """
    total = len(all_tokens)
    c, n = int(chunk_size), int(n_mtp)
    s = int(chunk_idx) * c
    assert 0 <= s < total, f"chunk {chunk_idx} starts at {s}, past the {total} real tokens"
    assert n > 0, f"n_mtp must be positive, got {n}"

    real = min(c, total - s)
    stream = list(all_tokens[s : s + c + n])
    stream += [pad_token] * (c + n - len(stream))
    assert len(stream) == c + n, f"stream is {len(stream)}, expected {c} + {n}"
    return stream, real


def assert_socket_rows(stream: Sequence[int], sp_factor: int, chunk_size: int, n_mtp: int) -> None:
    """Claim (1): each chip's row is the CONTIGUOUS slice of the stream at its own offset.

    ``prepare_prefill_input_tensor`` gives chip ``c`` ``stream[c*L : (c+1)*L]`` and
    ``prepare_prefill_mtp_tokens`` gives it ``stream[(c+1)*L : (c+1)*L + n_mtp]``; joined, that is
    ``stream[c*L : c*L + L + n_mtp]``, and it is the reason level ``k`` reads the SAME local slice
    ``[k, k+L)`` on every chip with no SP ring-shift. Restated here over the host lists, because on
    device the two tensors are separate blocks and nothing else says they abut.
    """
    isl = chunk_size // sp_factor
    for c in range(sp_factor):
        trunk = list(stream[c * isl : (c + 1) * isl])
        mtp = list(stream[(c + 1) * isl : (c + 1) * isl + n_mtp])
        assert len(trunk) == isl and len(mtp) == n_mtp, (
            f"chip {c}: trunk {len(trunk)} + mtp {len(mtp)} ids, expected {isl} + {n_mtp}; the stream "
            f"is {len(stream)} long"
        )
        assert trunk + mtp == list(stream[c * isl : c * isl + isl + n_mtp]), (
            f"chip {c}'s trunk row and MTP row do not abut: joined they must be the contiguous "
            f"stream[{c * isl}:{c * isl + isl + n_mtp}]"
        )


def _mtp_union(transformer: TtPrefillTransformer, stream: Sequence[int], n_mtp: int, num_levels: int):
    """Build this chunk's :class:`MTPUnionEmbedding` the way the runtime does.

    Mirrors ``tt_prefill_runtime._mtp_prepare_input``'s first-rank branch: upload the two id tensors
    the H2D row is cut into, gather each with the model's OWN embedding
    (``transformer.mtp_embed_ids``), and hand both blocks to ``from_ids``. The union's leading block
    is then this chunk's model input (``union.trunk``), which is why ``forward`` is called with
    ``input_is_embedded=True`` -- gathering the trunk twice is the one cost this arrangement exists
    to avoid.

    The id tensors are consumed here, as in the runtime. The union owns ``trunk`` and frees it in
    :meth:`~...MTPUnionEmbedding.deallocate`, so the caller must not free the model input.
    """
    chunk_ids = prepare_prefill_input_tensor(
        list(stream[: transformer.seq_len]),
        transformer.mesh_device,
        transformer.sp_factor,
        transformer.is_balanced,
        transformer.mesh_shape,
        transformer.sp_axis,
    )
    mtp_ids = prepare_prefill_mtp_tokens(
        list(stream),
        transformer.mesh_device,
        transformer.sp_factor,
        transformer.mesh_shape,
        transformer.sp_axis,
        num_mtp_tokens=n_mtp,
    )
    union = MTPUnionEmbedding.from_ids(chunk_ids, mtp_ids, transformer.mtp_embed_ids, num_levels=num_levels)
    ttnn.deallocate(chunk_ids)
    ttnn.deallocate(mtp_ids)
    return union


def _mtp_cache_dir(preferred: Path, fallback_root: Path) -> Path:
    """``preferred`` if this run can use it, else the same leaf under ``fallback_root``.

    The shared MTP cache tree belongs to whoever built it -- mode 775 with no group everyone shares
    -- so it is READABLE by other users and writable by none of them. That is fine for the depth the
    tree already holds: a complete cache is only ever read. It is not fine for a depth nobody has
    built, whose directory does not exist yet, and ``mkdir`` there raises ``PermissionError`` before
    a single weight is touched.

    So: use ``preferred`` when it can be created, or when it already holds weights (read-only reuse
    needs no write permission). Otherwise relocate under ``fallback_root`` and say so loudly -- the
    relocated leg pays a full cache build, which is the fp8 dequant plus ~5.5 GiB written.

    ``TT_GLM52_MTP_TTNN_CACHE`` overrides the root outright and skips all of this.
    """
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        if os.access(preferred, os.W_OK) or any(preferred.glob("*.tensorbin")):
            return preferred
        reason = "exists, holds no weights, and is not writable by this user"
    except PermissionError as exc:
        reason = f"cannot be created ({exc.strerror})"

    fallback = fallback_root / preferred.parent.name / preferred.name
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning(
        f"[mtp chunks] MTP weight cache {preferred} {reason}; falling back to {fallback}. "
        f"That directory starts empty, so this run BUILDS the MTP layer's cache (~5.5 GiB) instead "
        f"of loading it. Set {MTP_CACHE_ENV} to a writable root to choose the location yourself."
    )
    return fallback


def _next_token_fn(transformer: TtPrefillTransformer, actual_isl: int):
    """``H^k -> int``: the greedy token at the last real row, through the trunk's own LM head.

    This does NOT feed the device -- ``mtp_generate_embedding`` runs the identical chain on device
    and the id never comes back. It is how the TEST learns which id the device must have generated,
    so :func:`_expected_window` can place it and the reference can embed it. Costs one 32-row LM head
    call per level: ``TtLMHead.forward`` narrows to the single tile holding the target row before the
    vocab matmul.

    Greedy (argmax), not the transformer's sampler, because that is what the device does: sampling
    would make level ``k+1``'s *input* depend on the trunk's temperature.

    It is not circular. Both sides argmax the same ``H^k``, so a device that generated the WRONG id
    -- or wrote the right one into the wrong union row -- still disagrees with a window built from
    this one, on exactly the seam rows claim (4) checks. What it does not re-derive is the LM head
    itself, which is the trunk's and is covered by the trunk's own tests.
    """

    def next_token(h_normed):
        _, logits = transformer._lm_head_and_extract(h_normed, actual_isl)
        flat = logits.reshape(-1)
        assert (
            flat.numel() == transformer.lm_head.vocab_size
        ), f"expected full-vocab logits, got {flat.numel()} of {transformer.lm_head.vocab_size}"
        return int(torch.argmax(flat).item())

    return next_token


# ---------------------------------------------------------------------------
# The token-level reference
# ---------------------------------------------------------------------------


def _expected_window(full_seq: list[int], chunk_idx: int, level: int) -> list[int]:
    """The ONE statement this whole test is checking, written independently of production code.

    Level ``k`` of chunk ``c`` sees ``full_seq[c*C + k + 1 : c*C + k + 1 + C]``. ``full_seq`` is the
    prompt followed by the ``K`` tokens the last chunk generated, so one expression covers interior
    chunks (lookahead borrowed from the next chunk) and the last one (tail = generated) with no
    special case. Derived from the definition of MTP, not from ``mtp_chunk_stream`` -- the two live
    in the same file now, but this one indexes the full sequence directly while that one composes
    per-chunk lookahead, so they stay independent derivations. Written off ``mtp_chunk_stream`` it
    would agree with it however wrong it was.
    """
    start = chunk_idx * CHUNK + level + 1
    return list(full_seq[start : start + CHUNK])


def _host_window_embedding(embed_table: torch.Tensor, window_ids: list[int]) -> torch.Tensor:
    """The embedding the device MUST have produced for ``window_ids`` [C, H], derived here.

    Bit-identical to the device's, not merely close, and that is what lets it drive the CPU
    reference. Driving the reference from the EXPECTED ids rather than the ones
    the device actually asked for is what makes the numerics sensitive to the token->window mapping.

    Bit-identical because the device path from ids to window is arithmetic-free: an id upload,
    ``ttnn.embedding``'s row gather (``transformer.mtp_embed_ids``, the model's own), a row slice out
    of the union, and an optional multiply by a 0/1 mask. ``TtParallelEmbedding`` stores the table as
    ``ttnn.bfloat16``, which is what this gathers from. Slicing after the gather rather than before
    changes nothing -- ``slice(embed(ids)) == embed(slice(ids))`` row for row.

    The position-0 mask is deliberately NOT applied here: ``fused_mtp_reference`` zeroes row 0 itself
    from the ``positions`` this test passes, and leaving it out is what lets the row-0 check below
    tell a device that masked from one that did not.
    """
    return embed_table[torch.tensor(window_ids, dtype=torch.long)]


_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=GLM52Config.FABRIC_PAYLOAD_SIZE,
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
# Trunk depth. 78 is GLM-5.2's real one; 1 is a shallow variant for iterating on the MTP path
# without paying for the trunk.
#
# Full depth fits on ONE mesh because the trunk is loaded from the TTNN weight cache --
# ``state_dict={}``, every module reads its own .tensorbin straight to device -- so there is no
# host-side peak at all, and the routed experts are stored bfloat4_b: 401 GB of cache over 32 chips
# is 12.5 GiB/chip against 32 GB. Only the MTP layer costs host memory (~19 GiB of fp8 dequant),
# because it is the one layer the cache has no entry for -- and that cost is the SAME at both
# depths, as is the CPU reference. What a shallow trunk saves is device weight load, the trunk
# forward, and the KVPE/index caches (depth ``num_layers + K``, so 5 instead of 82).
#
# Two things made 78 the only workable depth before, and the config block in the test body is where
# each is now handled explicitly rather than by the coincidence that full depth satisfies both:
#
#   * ``TtPrefillBlock`` derives ``is_moe = layer_idx >= NUM_DENSE_LAYERS`` (3), and every block
#     cache key is ``f"layer_{layer_idx}"``. Deriving the MTP block's index as ``num_layers`` would
#     therefore build layer 78's MoE weights into a DENSE block at depths 1 and 2. The index is
#     ``max(num_layers, NUM_DENSE_LAYERS)`` instead -- past the trunk's range, still on the MoE
#     path -- and the shallow cache gets its own directory so its ``layer_3`` files never mix with
#     full depth's ``layer_78``.
#   * The index-K stride the trunk derives (``full_indexer_rank(first_layer_idx + layer_num)``) must
#     equal the one TtMTPModule derives (``num_full_indexer_layers``); they share one cache. At 78
#     both are 22 against the model's own 78-entry map. A shallow trunk gets a map TRUNCATED to its
#     own depth, which brings them back into agreement (both 4 at depth 1) -- and the assert below
#     is what checks that rather than assuming it.
@pytest.mark.parametrize("num_layers", [1, 78], ids=["layers1", "layers78"])
# Numerics axis. ``pcc`` is the real test; ``nopcc`` runs every device op and claim (1) but skips the
# CPU reference and every PCC comparison -- claim (2) and the two slice claims (3) and (4) all need
# ``glm_mtp_predictor_reference``, which is 45-140s of host torch per chunk and this test's dominant
# cost. It also drops the three things only the reference needs: the host embedding table (~1.8 GiB
# off the checkpoint shards), the four persistent ``SparseMLAReference`` instances, and every device
# readback.
#
# What survives is worth having on its own -- the full device path, all K levels through the real
# predictor and the real LM head, the C+K stream, the per-level windows ``_expected_window`` derives
# independently, the interior/last split, and that the last chunk generated exactly K tokens. So it
# catches a crash, a hang, a shape, a cache path or a stream-formation bug.
#
# What it CANNOT tell you is whether any number is right, so it is not a way to sign off a change:
# the run logs a warning per chunk saying the chunk is not validated. An axis rather than a CLI flag
# so the two legs are separate node ids and ``nopcc`` can never be mistaken for a green ``pcc``.
@pytest.mark.parametrize("skip_pcc", [False, True], ids=["pcc", "nopcc"])
# Prediction levels: both shipping configurations. See :data:`MTP_LEVEL_AXIS` for why MTP7 costs
# nothing on the wire and why K = 1 lives in test_mtp.py instead.
@pytest.mark.parametrize("mtp_levels", MTP_LEVEL_AXIS, ids=[f"mtp{k}" for k in MTP_LEVEL_AXIS])
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
# Weight axis: pretrained only. Everything is real -- the trunk, the embedding table and the LM head
# out of the TTNN cache, layer 78's MLA + indexer + 256-expert MoE and the four MTP tensors out of
# the checkpoint. There is no random leg because there is no random 78-layer trunk: the cache is the
# only reason full depth fits, and it holds one particular model's weights. The shallow depth reads
# the SAME cache -- it just stops after layer 0 -- so it has no random leg either.
# ``test_mtp.py`` keeps the checkpoint-free coverage of the same modules at depth 1.
#
# What the four MTP tensors look like, measured off the checkpoint at layer 78:
#
#     eh_proj embed half   std 0.0149   max|w| 0.227   max/mean|w|    19.2
#     eh_proj hidden half  std 0.0238   max|w| 2.359   max/mean|w|   211.6
#     enorm gain           std 0.0033   max|w| 0.053   max/mean|w|     1.3
#     hnorm gain           std 0.0110   max|w| 0.459   max/mean|w|     6.2
#
# The half-to-half asymmetry and the 211:1 tail are what the concat, the bf16 matmul and the
# reduce_scatter carry at four levels and across chunk seams; ``random_mtp_state_dict`` draws both
# halves alike from ``randn * (2H)**-0.5`` and has neither.
@pytest.mark.parametrize("use_pretrained", [True], ids=["pretrained"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_transformer_chunks(
    variant,
    config_only,
    model_path,
    weight_cache_path,
    mesh_device,
    device_params,
    num_links,
    num_layers,
    mtp_levels,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
    mtp_layer_state_dict,
    skip_pcc,
    monkeypatch,
):
    """3 x 5120 tokens of GLM-5.2 MTP4 end to end: every window, every level, exact ids.

    Assertions, most-local first so a failure localises itself:

    ===  ======================================================================================
     #   claim
    ===  ======================================================================================
     1   ``mtp_chunk_stream`` hands the transformer ``C + K`` ids: for an interior chunk exactly
         ``prompt[cC : cC + C + K]``, for the last chunk the prompt tail plus ``K`` sentinels.
     2   Every level's fused projection, block output and post-``shared_head.norm`` match
         ``glm_mtp_predictor_reference`` on every chunk, teacher-forced from the device's own
         ``H^{k-1}`` and driven by the embedding of :func:`_expected_window`'s ids -- so this is
         also the next-token-formation claim: adjacent token embeddings are unrelated, so a level
         that embedded a shifted window decorrelates against it outright.
     3   Absolute position 0 -- and only it, and only on chunk 0 -- is masked, vLLM's
         ``torch.where(positions == 0, 0, inputs_embeds)``. Asserted on row 0 alone: one row in
         ``C`` moves the whole-tensor PCC by less than its run-to-run noise.
     4   The last chunk's ``K`` sentinels became the ``K`` LM-head tokens, level by level: level
         ``k``'s window ends with ``generated[:k+1]``. Asserted on those ``k+1`` rows, same reason.
    ===  ======================================================================================

    Claim 2 is where the runtime goes: 12 CPU evaluations of a 256-expert GLM MoE decoder layer over
    a KV cache growing to 15360 positions -- 45.7 / 48.9 / 77.8s per chunk, rising with the chunk's
    start because that is how much cache each level attends over. It is not in any CI yaml.
    """
    torch.manual_seed(42)
    # Bound here rather than threaded through every reference below: K and the socket row's width are
    # fixed for the whole test once the axis picks them, and the body reads them ~20 times.
    NUM_LEVELS = mtp_levels
    N_MTP = num_mtp_tokens(NUM_LEVELS)
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")

    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape[SP_AXIS], mesh_shape[TP_AXIS]

    # copy.copy, not the shared object: config_only is lru_cached, so the mutations below would
    # otherwise follow every other GLM-5.2 test. They rebind attributes, so shallow is enough.
    config = copy.copy(config_only)
    config.max_seq_len = TOTAL
    full_depth = int(config_only.num_hidden_layers)
    assert 1 <= num_layers <= full_depth, f"trunk depth {num_layers} outside [1, {full_depth}]"
    shallow = num_layers < full_depth

    # A shallow trunk gets an indexer map truncated to its OWN depth, so that the stride both sides
    # derive from it agrees again (see the axis comment). ``copy.copy`` is shallow, so rebind the
    # attribute rather than slicing in place -- ``config_only`` is lru_cached and every other
    # GLM-5.2 test reads the same list object.
    if shallow:
        config.indexer_types = list(config_only.indexer_types)[:num_layers]

    # One index-K cache is shared by the trunk and the MTP block, and its per-user slot stride is
    # derived TWICE: the trunk from ``full_indexer_rank(first_layer_idx + layer_num)``, the MTP block
    # from ``num_full_indexer_layers`` (TtMTPModule passes no first_layer_idx, so it reads the
    # whole-model count). update_padded_kv_cache TT_FATALs if the cache's batch dim is not a multiple
    # of it, before any assertion below is reached -- so they are asserted equal here.
    #
    # At full depth they agree against the model's OWN 78-entry map: appending one "full" slot at 78
    # makes both 22. A shallow trunk agrees against the TRUNCATED map instead -- at depth 1 the map
    # is [full] extended to the MTP index, so both are 4 -- which is why the truncation above and
    # the MTP index below are chosen together, not independently.

    # The index the MTP BLOCK is built at -- which is not necessarily the index its WEIGHTS come
    # from. ``mtp_cfg.mtp_layer_idx`` (78) always sources the weights; this is where the block sits
    # for ``is_moe``, its cache key and its indexer slot. At full depth the two coincide.
    mtp_block_idx = max(num_layers, variant.model_config.NUM_DENSE_LAYERS)
    layer_idx = enable_mtp_indexer_slot(config, mtp_block_idx)
    assert layer_idx == mtp_block_idx >= num_layers, (
        f"the MTP block must sit past the trunk's layers [0, {num_layers}) so it cannot claim one of "
        f"their cache keys; got {layer_idx}"
    )
    assert layer_idx >= variant.model_config.NUM_DENSE_LAYERS, (
        f"TtPrefillBlock derives is_moe = layer_idx >= {variant.model_config.NUM_DENSE_LAYERS}, so an "
        f"MTP block at {layer_idx} would be built DENSE while its weights (checkpoint layer "
        f"{mtp_cfg.mtp_layer_idx}) are a 256-expert MoE"
    )
    if not shallow:
        assert (
            layer_idx == mtp_cfg.mtp_layer_idx == num_layers
        ), f"at full depth the MTP block sits on the checkpoint's own layer {mtp_cfg.mtp_layer_idx}, got {layer_idx}"
    trunk_stride = full_indexer_rank(config, num_layers + NUM_LEVELS)
    mtp_stride = num_full_indexer_layers(config)
    assert trunk_stride == mtp_stride, (
        f"index-K cache stride disagrees: trunk {trunk_stride} vs MTP {mtp_stride}. They share one "
        "cache, so update_padded_kv_cache raises TT_FATAL 'cache batch dim must be a multiple of "
        "num_layers' on the first indexer write -- before any assertion below is reached."
    )
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size

    logger.info(
        f"[mtp chunks] mesh={mesh_shape} chunks={NUM_CHUNKS}x{CHUNK} total={TOTAL} K={NUM_LEVELS} "
        f"trunk_layers={num_layers} mtp_layer={layer_idx} vocab={config.vocab_size} "
        f"cache={weight_cache_path}"
    )

    # --- Prompt: id at absolute position p is p + 1, so a decoded id names its own position -------
    prompt = list(range(1, TOTAL + 1))
    logger.info(f"Prompt size is: {len(prompt)}")
    logger.info(f"Prompt is: {prompt}")
    assert max(prompt) < config.vocab_size

    # --- Weights ---------------------------------------------------------------------------------
    # The trunk never becomes a host tensor. ``state_dict={}`` puts TtPrefillTransformer in its
    # load-from-cache mode, where every module hands ttnn.as_tensor a cache_file_name and the tensor
    # goes .tensorbin -> device with no torch intermediate. That is the only reason 78 layers fit,
    # and it is checked here rather than discovered as a missing-file error 40 layers in.
    effective_cache_path = weight_cache_path / f"{sp_factor}x{tp_factor}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp_factor * tp_factor)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    # Layer 78 is the one layer the trunk cache has no entry for -- it is not part of the transformer
    # that cache was built for -- so it gets its own directory, and it is NOT asserted: on a first run
    # the constructor writes it (as_tensor prefers an existing cache file and creates it otherwise),
    # on every run after it is loaded. Logged either way so a slow first run explains itself.
    # weight_cache_path is <TT_GLM52_PREFILL_TTNN_CACHE>/<variant>_<arch>_<n>dev, so .parent.parent
    # is the directory holding the trunk cache and the sibling lands beside it. See MTP_CACHE_ENV.
    mtp_cache_root = Path(os.getenv(MTP_CACHE_ENV) or weight_cache_path.parent.parent / "glm52_mtp_ttnn_cache")
    mtp_cache_path = mtp_cache_root / f"{variant.name}_{'bh' if is_blackhole() else 'wh'}_{ttnn.get_num_devices()}dev"
    # A shallow run builds the MTP block at a DIFFERENT index (layer_3, not layer_78), so its
    # .tensorbin names differ and it needs its own directory -- otherwise one depth's partial set
    # sits beside the other's and only ``check_cache_complete`` can tell them apart. Full depth
    # keeps the unsuffixed path it has always used, so its prebuilt cache still hits.
    mtp_cache_path = mtp_cache_path / (f"{sp_factor}x{tp_factor}" + (f"_L{num_layers}" if shallow else ""))
    # Not a bare mkdir: the shared tree is another user's, so a depth it has no directory for cannot
    # be created there. See :func:`_mtp_cache_dir`.
    mtp_cache_path = _mtp_cache_dir(mtp_cache_path, Path(ttnn.CONFIG.cache_path) / "glm52_mtp_ttnn_cache")

    # check_cache_complete resolves every pattern against the PROCESS-GLOBAL checker directory that
    # init_checker last set -- tt_distributed_rms_norm.check_cache_complete takes a cache_path and
    # ignores it -- so the checker has to be aimed at this directory before it is asked about it,
    # as tests/test_prefill_block.py:879 does. Without this the call above (TtPrefillTransformer's,
    # which inits the checker to the TRUNK dir) leaves it pointed there, and a complete MTP cache is
    # reported ABSENT. Only the message was ever wrong -- as_tensor does its own per-file check and
    # loaded all 48 correctly -- but a status line that cannot say "present" is not worth printing.
    init_checker(mtp_cache_path)
    mtp_cached = TtMTPPredictor.check_cache_complete(
        mtp_cache_path,
        layer_idx,
        cache_name_prefix=MTP_CACHE_PREFIX,
        experts_per_chip=experts_per_chip,
        model_cfg=variant.model_config,
    )
    init_checker(effective_cache_path)  # restore: the trunk owns the checker for its own build
    logger.info(
        f"[mtp chunks] MTP layer {layer_idx} cache at {mtp_cache_path}: "
        f"{'present -- loading it' if mtp_cached else 'ABSENT -- building it (first run)'}"
    )

    # The embedding table, host side, for the MTP window embeddings claim 2 is teacher-forced from.
    # bf16 is the dtype TtParallelEmbedding stores it in, so the gather matches the device row for
    # row. Pulled from the checkpoint rather than the cache because the cache holds it sharded.
    # Reference-only, so the ``nopcc`` leg does not pay for it: 154880 x 6144 bf16 is ~1.8 GiB off the
    # checkpoint's shards, and the DEVICE reads its own copy out of the TTNN cache either way.
    embed_table = None
    if not skip_pcc:
        embed_table = load_hf_state_dict_filtered(str(model_path), ["model.embed_tokens."])[
            "model.embed_tokens.weight"
        ].to(torch.bfloat16)
        assert list(embed_table.shape) == [
            config.vocab_size,
            config.hidden_size,
        ], f"embedding table {list(embed_table.shape)} != [{config.vocab_size}, {config.hidden_size}]"

    # Layer 78's own decoder weights, kept: claim 2's CPU reference needs the SAME tensors the device
    # got, so the dict must outlive construction. ~19 GiB resident (256 x 3 x [2048, 6144] bf16 plus
    # the shared expert), the one large host allocation in this test.
    mtp_layer_sd = mtp_layer_state_dict
    mla_weights = mtp_layer_sd["mla_weights"]
    ref_moe_weights = {k: mtp_layer_sd[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")}

    # first_cache_slot / layer_num are the two the transformer asserts on: levels write KV slots
    # [num_layers, num_layers + K) and every block must stride users by the cache's true depth.
    # is_chunked/max_seq_len/slot_num reach TtPrefillBlock through TtMTPModule's **block_kwargs, so
    # the MTP layer is chunked exactly like the trunk's.
    predictor = TtMTPPredictor(
        mesh_device,
        config,
        variant.model_config,
        {"mtp": mtp_state_dict, "layer": mtp_layer_sd},
        mtp_cfg,
        seq_len=CHUNK,
        num_levels=NUM_LEVELS,
        layer_idx=layer_idx,
        first_cache_slot=num_layers,
        tp_axis=TP_AXIS,
        sp_axis=SP_AXIS,
        num_links=num_links,
        topology=topology,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        dispatch_buffer_capacity_factor=DISPATCH_BUFFER_CAPACITY_FACTOR,
        weight_cache_path=mtp_cache_path,
        cache_name_prefix=MTP_CACHE_PREFIX,
        is_chunked=True,
        max_seq_len=TOTAL,
        slot_num=1,
        layer_num=num_layers + NUM_LEVELS,
    )

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        weight_cache_path=effective_cache_path,
        num_layers=num_layers,
        seq_len=CHUNK,
        max_seq_len=TOTAL,
        dispatch_buffer_capacity_factor=DISPATCH_BUFFER_CAPACITY_FACTOR,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        padding_side="right",
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        lm_head_is_column_parallel=True,
        is_chunked=True,
        slot_num=1,
        mtp_predictor=predictor,
    )
    gc.collect()
    ttnn.synchronize_device(mesh_device)

    assert transformer.num_kvpe_cache_layers == num_layers + NUM_LEVELS
    assert transformer.num_mtp_levels == NUM_LEVELS

    # Teacher forcing hands level k the device's ``out_head_normed[k-1]``, which is H^{k-1} only
    # under the "norm" chaining convention. Asserted, not assumed: chain_from is a constructor
    # argument, and flipping it makes every level>0 comparison below meaningless, with no shape
    # error to say so.
    assert predictor.chain_from == CHAIN_FROM_NORM, (
        f"the reference below is teacher-forced with out_head_normed, so the device must chain from "
        f"it too; predictor.chain_from is {predictor.chain_from!r}"
    )

    # One persistent SparseMLAReference PER LEVEL, carried across all three chunks. Per level because
    # each level owns its own KV cache -- one shared instance would let level k attend to level k-1's
    # keys. Persistent because the caches and the fill watermark live on the instance, so a fresh one
    # per chunk would make every chunk attend over itself alone: exactly the bug this test exists to
    # catch, baked into its reference instead.
    ref_mla = None if skip_pcc else [SparseMLAReference(config, mla_weights, seq_len=TOTAL) for _ in range(NUM_LEVELS)]

    # --- Caches ------------------------------------------------------------------------------------
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=transformer.num_kvpe_cache_layers,
        num_users=1,
    )
    # Sized over the compacted full-indexer space of the map built above, which both sides agree on
    # (asserted there). TtIndexer derives each block's slot from its static layer_idx: the trunk
    # takes slots [0, num_layers), MTP the appended one.
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_full_indexer_layers(config),
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    mesh_device.enable_program_cache()

    # --- Drive the three chunks --------------------------------------------------------------------
    # h^0 -- the post-``model.norm`` trunk output the predictor is seeded with -- is the one thing
    # this test needs that no production path hands back: not in forward()'s return value, not on
    # ``MTPPredictorOutput``. So it is taken where it is passed; the call site is
    # ``self.run_mtp(...)``, so an instance attribute wins at call time.
    h0_host: dict = {}
    real_run_mtp = transformer.run_mtp
    # Per-chunk switch for the hook below. ``t0`` -- the id the device's level 1 must have generated
    # -- can only be taken here: it comes off h^0, which forward does not hand back and may free.
    derive_gen: dict = {"on": False, "isl": 0}

    def _capture_h0(h_normed, *args, **kwargs):
        # None on the ``nopcc`` leg: the key still records that run_mtp fired (asserted below), but the
        # readback exists only to seed the reference's level 0.
        h0_host["h"] = None if skip_pcc else _from_device(h_normed, mesh_device)
        if derive_gen["on"]:
            h0_host["t0"] = _next_token_fn(transformer, derive_gen["isl"])(h_normed)
        return real_run_mtp(h_normed, *args, **kwargs)

    monkeypatch.setattr(transformer, "run_mtp", _capture_h0)

    captured: dict = {}

    def _on_mtp_complete(mtp_out, mtp_generated):
        captured["out"] = mtp_out
        captured["generated"] = list(mtp_generated)

    # Both exist only to build the reference's input, so both stay empty on the ``nopcc`` leg --
    # which is why the skip below sits BEFORE they are filled, not after.
    windows: list[list[list[int]]] = []  # [chunk][level] -> ids in POSITION order
    generated: list[int] = []

    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK
        is_last = chunk_idx == NUM_CHUNKS - 1

        stream, real_len = mtp_chunk_stream(prompt, chunk_idx, CHUNK, N_MTP)
        logger.info(f"Processing chunk {chunk_idx}")
        logger.info(f"Len(stream) is {len(stream)}")
        # Head and tail, not the whole thing: the stream is C + n_mtp = 5152 ids at the production
        # shape, and the tail is the interesting half -- it is where an interior chunk's borrowed
        # lookahead ends and the last chunk's pad (which the device generates over) begins.
        logger.info(f"Stream is : {stream[:8]} ... {stream[CHUNK - 4:CHUNK]} | mtp tail {stream[CHUNK:CHUNK + 8]} ...")

        # (1) what the socket delivers: C + n_mtp ids, cut into a per-chip trunk row and a per-chip
        # lookahead row that abut. One slice covers both kinds of chunk -- an interior one finds its
        # lookahead in the prompt, the last one finds pad and the DEVICE generates over it -- so the
        # interior/last split lives only in the `is_last_chunk` flag below, as it does in production.
        assert len(stream) == CHUNK + N_MTP, f"chunk {chunk_idx} stream is {len(stream)}, expected {CHUNK + N_MTP}"
        assert real_len == CHUNK, f"chunk {chunk_idx} real_len {real_len}; all three chunks are full"
        assert stream[:CHUNK] == prompt[start : start + CHUNK]
        expected_tail = (prompt[start + CHUNK : start + CHUNK + N_MTP] + [PAD_TOKEN] * N_MTP)[:N_MTP]
        assert stream[CHUNK:] == expected_tail, (
            f"chunk {chunk_idx}'s lookahead must be the {N_MTP} stream positions after it, padded "
            f"where the prompt has ended; got {stream[CHUNK:][:8]}... expected {expected_tail[:8]}..."
        )
        assert_socket_rows(stream, sp_factor, CHUNK, N_MTP)

        # The production input: two id tensors -> one on-device union, whose leading block is the
        # model input. Exactly what tt_prefill_runtime._mtp_prepare_input builds on a first rank.
        union = _mtp_union(transformer, stream, N_MTP, NUM_LEVELS)
        h0_host.clear()
        captured.clear()
        derive_gen.update(on=(is_last and not skip_pcc), isl=real_len)

        logger.info(f"[mtp chunks] chunk {chunk_idx}: start={start} real_len={real_len} last={is_last}")
        transformer.forward(
            union.trunk,
            kvpe_cache,
            actual_isl=real_len,
            actual_start=start,
            actual_end=start + real_len,
            cache_user_id=0,
            index_kv_cache=index_kv_cache,
            mtp_union=union,
            is_last_chunk=is_last,
            # union.trunk IS this chunk's embedding, so the first rank must not gather it again.
            input_is_embedded=True,
            on_mtp_complete=_on_mtp_complete,
        )
        ttnn.synchronize_device(mesh_device)
        # Nothing below reads the union -- the level outputs are their own tensors -- so free it here,
        # as the runtime's last rank does. It owns the model input, which is why that is not freed.
        union.deallocate()

        assert "generated" in captured, "on_mtp_complete never fired -- the MTP branch did not run"
        res = captured["out"]
        assert len(res.x) == NUM_LEVELS
        assert captured["generated"] == [], (
            f"chunk {chunk_idx}: the device path argmaxes, embeds and consumes its generated ids on "
            f"device, so generated_tokens must come back empty on every chunk; got {captured['generated']}"
        )

        assert "h" in h0_host, "run_mtp never ran, so h^0 was never captured"

        if skip_pcc:
            logger.warning(
                f"[mtp chunks] chunk {chunk_idx}: PCC comparison SKIPPED (skip_pcc) -- "
                f"the device ran all {NUM_LEVELS} levels and claim (1) passed, but claims (2)/(3)/(4) "
                "did NOT run. This is not a validated chunk."
            )
            del res
            continue

        # The K ids the device MUST have generated, derived rather than observed: level k's window
        # needs t_{P+k} = argmax lm_head(H^{k-1}) at the last real row, and H^{k-1} is h^0 for k=1
        # (taken in the hook, where it is still live) and the device's own level k-2 output after.
        # See _next_token_fn for why running the same head here is not circular.
        seam_is_decisive = True
        if derive_gen["on"]:
            next_token = _next_token_fn(transformer, real_len)
            generated = [h0_host["t0"]] + [next_token(res.out_head_normed[k]) for k in range(NUM_LEVELS - 1)]
            assert len(generated) == NUM_LEVELS
            logger.info(f"[mtp chunks] chunk {chunk_idx}: device must have generated {generated}")

            # Claim (4) can only localise a mis-placed patch when the ids it places are telling
            # apart. Two levels that generated the SAME id sit in adjacent union rows, so swapping
            # their patches is invisible to a PCC over those rows -- the reference would embed the
            # same vector either way. On this prompt that happens: the ids are synthetic (p+1), the
            # draft chain has no signal, and the LM head collapses onto a handful of tokens.
            #
            # So say so rather than let the seam look stronger than it is. The claim still runs --
            # a patch in the wrong PLACE entirely, or a wrong id, still fails it -- but the
            # permutation-within-duplicates case is not covered, and the geometry evidence for that
            # is the host-side simulation with distinct ids, not this assertion.
            dupes = len(generated) - len(set(generated))
            if dupes:
                seam_is_decisive = False
                logger.warning(
                    f"[mtp chunks] chunk {chunk_idx}: generated ids {generated} contain {dupes} "
                    "duplicate(s), so the generation seam below CANNOT distinguish a patch swap "
                    "between the levels that share an id. It still catches a wrong id and a patch "
                    "written outside the generated rows."
                )

        # Complete only once the last chunk has generated; interior chunks never index past TOTAL.
        full_seq = prompt + generated
        windows.append([])

        # The window each level MUST embed, derived from the definition of MTP rather than observed.
        # The reference below is driven by these ids -- claim (2) -- and adjacent token embeddings
        # are unrelated, so a level that embedded a shifted window fails its fused PCC outright.
        windows[chunk_idx].extend(_expected_window(full_seq, chunk_idx, level) for level in range(NUM_LEVELS))
        assert all(len(w) == CHUNK for w in windows[chunk_idx]), (
            f"chunk {chunk_idx}: a window came out short -- full_seq is {len(full_seq)} ids, which "
            f"does not cover this chunk's shift-{NUM_LEVELS} lookahead"
        )
        # --- (2) numerics, teacher-forced ---------------------------------------------------------
        host_embeds = [_host_window_embedding(embed_table, w) for w in windows[chunk_idx]]

        # H^{k-1} for the reference is the DEVICE's, never the reference's own previous output:
        # level 0 gets h^0 off the trunk, level k>0 gets level k-1's device output.
        #
        # EVERY device readback happens here, before the reference runs. The reference is 45-140s of
        # pure host torch, and pulling a device tensor across that gap killed the first run: SIGBUS
        # inside ttnn.to_torch at chunk 4, on the first device touch after the longest gap (137.6s).
        # Stale mapping vs. the host failing to back a page at peak RSS was never established (no
        # dmesg here), but neither can happen if the device is untouched until the next chunk. It
        # costs ~60 MiB per extra host copy.
        dev_x = [_from_device(res.x[k], mesh_device) for k in range(NUM_LEVELS)]
        dev_out = [_from_device(res.out[k], mesh_device) for k in range(NUM_LEVELS)]
        dev_normed = [_from_device(res.out_head_normed[k], mesh_device) for k in range(NUM_LEVELS)]
        del res
        ref_hiddens = [h0_host["h"]] + dev_normed[:-1]

        t0 = time.monotonic()
        ref_xs, ref_outs, ref_normeds, _ = glm_mtp_predictor_reference(
            config,
            mla_weights,
            mtp_state_dict,
            mtp_layer_sd["attn_norm_weight"],
            mtp_layer_sd["ffn_norm_weight"],
            [e.unsqueeze(0) for e in host_embeds],
            ref_hiddens[0].squeeze(0),
            TOTAL,
            moe_weights=ref_moe_weights,
            num_levels=NUM_LEVELS,
            index_share=predictor.index_share,
            hiddens=[h.squeeze(0) for h in ref_hiddens],
            mla_refs=ref_mla,
            # Row j of this chunk is absolute position start + j. Without this, fused_mtp_reference
            # defaults to arange(C) and every chunk zeroes its own row 0, which the device does only
            # on chunk 0.
            positions=torch.arange(start, start + CHUNK),
            actual_start=start,
            actual_end=start + real_len,
        )
        logger.info(f"[mtp chunks] chunk {chunk_idx}: CPU reference took {time.monotonic() - t0:.1f}s")

        for level in range(NUM_LEVELS):
            # Most-local first: the fused projection is two norms and a matmul, the block output
            # adds a whole DSA-MLA + 256-expert MoE, the third is that plus one more norm.
            _, msg = assert_with_pcc(ref_xs[level].unsqueeze(0), dev_x[level], FUSED_MTP_PCC)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: fused projection PCC {msg}")
            _, msg = assert_with_pcc(ref_outs[level].unsqueeze(0), dev_out[level], MTP_MODULE_OUTPUT_PCC)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: block output PCC {msg}")
            _, msg = assert_with_pcc(ref_normeds[level].unsqueeze(0), dev_normed[level], MTP_MODULE_OUTPUT_PCC)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: shared_head.norm PCC {msg}")

            # Two claims the whole-tensor PCC cannot resolve: each turns on at most K rows out of C,
            # ~1e-4 of the elements, under the gate AND under the 1.6e-4 to 2.9e-4 run-to-run noise.
            # On their own rows a wrong token is a total mismatch.
            if chunk_idx == 0:
                # (3) the position-0 mask: the reference zeroes row 0 from the positions passed
                # above and _host_window_embedding does not, so this row agrees only if the DEVICE
                # masked it too.
                _, msg = assert_with_pcc(ref_xs[level].unsqueeze(0)[:, :, :1], dev_x[level][:, :, :1], FUSED_MTP_PCC)
                logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: position-0 mask PCC {msg}")
            if is_last:
                # (4) the generation seam: the last level+1 rows carry ids only the LM head can
                # produce, and the reference got them from _expected_window, not from the device.
                # See seam_is_decisive above for what a duplicate id costs this claim.
                seam = CHUNK - level - 1
                _, msg = assert_with_pcc(
                    ref_xs[level].unsqueeze(0)[:, :, seam:], dev_x[level][:, :, seam:], FUSED_MTP_PCC
                )
                logger.info(
                    f"[mtp chunks] chunk {chunk_idx} L{level}: generation seam PCC {msg}"
                    f"{'' if seam_is_decisive else ' (duplicate ids -- swap-blind, see warning above)'}"
                )

        del dev_x, dev_out, dev_normed, ref_hiddens, ref_xs, ref_outs, ref_normeds

    logger.info(f"[mtp chunks] generated tokens for the final K slots: {generated}")
