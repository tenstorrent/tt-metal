# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for GLM-5.3-Flash on the resident mesh model: the MTP module (``tt/mtp.py``, checkpoint
layer 45) drafts ``K`` tokens, the main model verifies them in ONE batched decode step, the longest matching
prefix is accepted (lossless greedy: every committed token is the main model's own greedy id).

Row layout of a verify step (per mesh row, ``B`` user slots, ``K`` drafts): ``R = B (K + 1)`` candidate-major
rows, row ``j B + u`` = user ``u``, candidate ``j`` (``j = 0`` the anchor token, ``j >= 1`` draft ``d_j``) at
position ``p_u + j``. Candidate-major keeps every candidate a contiguous ``[B]`` slab (the per-candidate latent
appends, the KDA launches and the index commit address the real user slots with ``[B]`` tensors). Policy:
``R <= 32`` keeps the rows inside one tile row (free head-major views, one-row-per-core programs, the
weight-bandwidth-bound regime): ``K = min(K_ctx, 32 // B - 1)`` (``spec_draft_len``), 0 = plain decode.

One iteration (host loop ``GreedySpecLoop``; device graphs ``SpecDecoder``):

1. **V (verify)**: embedding of the ``R`` tokens -> every layer on ``R`` rows (mHC / norms / router / experts /
   head row-wise; DSA: latent appended per candidate, dense flash-MLA over the user's pages through a page table
   whose user rows are replicated per candidate with ``cur_pos = p_u + j`` (causal among the candidates for
   free); KDA: ``K + 1`` sequential ``kda_decode`` launches on the candidate slabs, state backed up first) ->
   greedy ids of all rows -> MTP level 1 on ALL rows (teacher-forced with the row's greedy id, MTP position
   ``p_u + j``) -> its draft ids and normed hidden. Outputs copied into persistent buffers.
2. host accept: ``m_u`` = matched prefix; committed = ``d_1 .. d_m`` + the bonus ``g_m``; next anchor ``g_m`` at
   ``p_u + m_u + 1``; the next level-1 draft is row ``m_u``'s MTP output.
3. **C (commit)**: KDA: replay the candidate rows on the backup from the projections cached in V, per-user
   masks hand each user's live state the replay after its own ``m_u`` (``KdaSpecState.commit``); DSA: the
   pooled-key index maintenance of the committed rows in order (``DsaSpecState.commit``).
4. **D (draft chain)**: MTP levels ``2 .. K`` from the selected row (one-hot matmul gather of the level-1 hidden,
   exact), each level a decode row at ``p_u + m_u + k - 1``.

State discipline (plan section 2c, refined):

* DSA latent cache: position-indexed writes, attention bounded by ``cur_pos``: tentative rows (rejected
  candidates, chain rows) are overwritten by the next committed tokens, no rollback.
* pooled-key index: a tentative row must NEVER reach the ring (``[slots, 1, 32, D]``, rows ``0..3`` = the last 4
  positions): a rejected row at position 8 overwrites the slot still holding committed position 4 before pool 1
  (positions 4..7) completes, and pool 1 would be built from ``{8, 5, 6, 7}`` for ever. So V and D do not touch
  the rings / pools at all (the dense-range attention never reads them); the raw k / gate rows of the candidates
  are cached in V and C writes the committed rows ``0 .. m_u`` in order (ring row, pool rewrite) with the
  rejected rows routed to the ring's dead rows ``4 .. 31`` (unused tile rows of the ring block) and their pool
  rewrite landing on a not-yet-complete pool (rewritten at completion, never selectable before).
* KDA recurrent + conv: the only non-idempotent state. Two commit modes (``GLM53_SPEC_COMMIT``, ``spec_commit_mode``):
  **blend** (the default, today's path): V backs up ``S_0`` (``ttnn.copy`` of the recurrent ``[B, H, K, V]`` fp32
  and the tap-major conv history ``[3, B, W]`` bf16) before its ``K + 1`` launches leave the live state at
  ``S_{K+1}``; C replays launches ``0 .. K`` ON the backup from the cached ``qkv`` / gate projections (the same
  programs on the same inputs) and after launch ``j`` blends ``live = live * keep_j + backup * take_j``
  (``take_j[u] = 1`` iff ``m_u == j``; 1.0 / 0.0 products and adds are exact; binary_ng computes fp32 tensors in
  fp32 DST), so every user's live state ends as ``S_{m_u + 1}``. **gated** (SP1): V copies the live state into a
  SCRATCH (the same ``bk`` tensors) and runs its ``K + 1`` launches there, so the live state is untouched by V; C
  replays launches ``0 .. K`` on the LIVE state through ``kda_decode``'s per-user write gate (``KdaDecode(gated=True)``:
  a ``[1, 32]`` int32 device page per launch, ``gate_j[u] = 1`` iff ``m_u >= j``, refreshed by the host before C):
  launch ``j`` reads every user's state but writes only the gated users' (the writer skips the state tiles, the
  reader merges the history shift row-wise), so user ``u`` takes exactly launches ``0 .. m_u`` -- the same programs
  on the same inputs as the plain path -- and keeps the bytes of ``S_{m_u + 1}`` afterwards. No blends, no restore:
   C is ``K + 1`` launches instead of ``K + 1`` launches + ``6 (K + 1)`` full-state elementwise ops. **fused** (SP2):
  the fused ``T = K + 1`` ``kda_decode`` variant (``KdaDecode(fused_steps=T)``) takes the whole ``[1, 1, R, W]``
  candidate-major slabs and applies, per user, its candidates in order with the state kept in L1: V is ONE dry launch
  per layer on the live state (read once, never written: no backup, no slices, no concat; the ``[R, H V]`` output
  feeds ``o_proj`` directly) and C is ONE commit launch per layer on the cached ``qkv_all`` / ``g2_all`` with the
  per-user step page ``n_u = m_u + 1`` (a ``[1, 32]`` int32 device page per chip refreshed by the host before C):
  user ``u`` applies candidates ``0 .. m_u`` on its live state and is written once. No blends, no gates, no scratch
  state. One commit per verify in every mode.

Trace safety: every per-iteration input is a persistent tensor refreshed with ``copy_host_to_device_tensor``
(ids, positions, ring slots, pool ids, masks / write gates, the select matrix); every buffer the graphs write for the host or
for another graph predates the first capture (allocated in ``SpecDecoder.__init__``: backups, projection
caches, output copies); the compile pass (``warmup``) is the identical op set; nothing allocated after a capture
is persistent; reshaped id rows (views) are never deallocated.

Scope of v1 (device validation pending): dense range only (``max position + K <= 2050``), non-partitioned
caches, block-mode KDA (``GLM53_KDA_FUSED=1``, the default), greedy. The sparse-range verify needs the index
scorer with ``K + 1`` rows per user (plan item 4) and is refused.

Served form (``SpecDecoder(plain_draft=True)``, the vLLM adapter ``tt/generator_vllm.py`` + ``tt/vllm_spec.py``): the
same V / C / D graphs plus **P** (the plain decode step on the model's persistent inputs + MTP level 1 on every user
slot, its draft ids / normed hidden kept in ``plain_d1`` / ``plain_hidden``: the ordinary decode of a speculating
launch, so the drafter's cache follows every position) and **D0** (levels ``2 .. K`` from P's output, no gather).
The plugin owns the accept walk: V runs in ``decode_forward(spec_mode="argmax_ids")``, C + D (or D0) in
``propose_draft_tokens`` once the runner's accepted counts are known.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.kda_decode.op import GATE_WORDS
from models.demos.glm53_flash.tt.dsa import TtDSA, TtLatentCache, _meta_mapper, _meta_upload
from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.mtp import TtMTPBlock
from models.demos.glm53_flash.tt.ttkda.kda import KdaState

if TYPE_CHECKING:  # no import cycle at runtime (mesh_model does not import this module)
    from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

TILE = 32
MAX_VERIFY_ROWS = TILE  # B (K + 1) <= 32: one tile row of candidates per mesh row
SPEC_COMMIT_MODES = ("blend", "gated", "fused")


def spec_commit_mode(env=None) -> str:
    """``GLM53_SPEC_COMMIT``: ``blend`` (the default: V's launches on the live state after a backup, C replays them on
    the backup and blends ``live = live * keep_j + backup * take_j`` after every launch) or ``gated`` (SP1: V's launches
    on a scratch copy, C replays them on the LIVE state through ``kda_decode``'s per-user write gate, launch ``j``
    writing user ``u`` iff ``m_u >= j``; no blends, no restore) or ``fused`` (SP2: V = one dry fused T = K + 1 launch per
    layer on the live state, C = one fused launch per layer applying ``m_u + 1`` candidates per user in place)."""
    env = os.environ if env is None else env
    mode = env.get("GLM53_SPEC_COMMIT", "blend").strip().lower()
    if mode not in SPEC_COMMIT_MODES:
        raise ValueError(f"GLM53_SPEC_COMMIT={mode!r}: expected one of {SPEC_COMMIT_MODES}")
    return mode


# ============================================================================================ host-side policy
def spec_draft_len(users_per_row: int, k_ctx: int, max_rows: int = MAX_VERIFY_ROWS) -> int:
    """The draft length for ``users_per_row`` user slots per mesh row: ``min(k_ctx, max_rows // B - 1)``; 0 means
    plain decode (at ``B (K + 1) > 32`` the verify leaves the weight-bandwidth-bound regime and every K lost)."""
    if users_per_row <= 0:
        raise ValueError(f"users_per_row must be positive, got {users_per_row}")
    return max(0, min(int(k_ctx), max_rows // users_per_row - 1))


def candidate_rows(users_per_row: int, draft_len: int) -> int:
    """``R = B (K + 1)`` verify rows per mesh row (refused above ``MAX_VERIFY_ROWS``)."""
    R = users_per_row * (draft_len + 1)
    if R > MAX_VERIFY_ROWS or draft_len < 0:
        raise ValueError(f"B={users_per_row} K={draft_len}: B (K + 1) = {R} exceeds {MAX_VERIFY_ROWS} rows")
    return R


def dead_ring_row(j: int, kpool: int) -> int:
    """A ring-block row (``kpool .. 31``) no reader looks at: where a rejected candidate's ring write goes."""
    return kpool + (j % (TILE - kpool))


def accept_greedy(
    drafts: torch.Tensor, verify: torch.Tensor, valid: int | None = None
) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
    """Greedy acceptance. ``drafts`` ``[N, K]`` the drafted ids, ``verify`` ``[N, K + 1]`` the main model's greedy id
    of every candidate row (row ``j`` = the prediction after ``[anchor, d_1 .. d_j]``), ``valid`` = how many
    leading drafts are real (0 on the first iteration: the rows carry placeholders). Returns ``(m [N], committed,
    bonus [N])``: ``m_u`` = the longest prefix with ``d_j == verify[u, j - 1]``, ``committed[u] = d_1 .. d_m +
    [verify[u, m]]`` (the bonus token is the main model's own prediction after the accepted prefix, so the result
    is the plain greedy continuation), ``bonus = verify[u, m]`` = the next anchor."""
    drafts, verify = drafts.long(), verify.long()
    N, K = drafts.shape
    if tuple(verify.shape) != (N, K + 1):
        raise ValueError(f"verify {tuple(verify.shape)} does not match drafts {tuple(drafts.shape)}")
    valid = K if valid is None else int(valid)
    if valid:
        miss = drafts[:, :valid] != verify[:, :valid]
        m = torch.where(miss.any(1), miss.long().argmax(1), torch.full((N,), valid, dtype=torch.long))
    else:
        m = torch.zeros(N, dtype=torch.long)
    bonus = verify[torch.arange(N), m]
    committed = [drafts[u, : int(m[u])].tolist() + [int(bonus[u])] for u in range(N)]
    return m, committed, bonus


def verify_row_positions(positions: list[int], users_per_row: int, draft_len: int, rows: int = 1) -> list[int]:
    """The ``cur_pos`` of every verify row, mesh-row-major then candidate-major: for mesh row ``r``, candidate
    ``j``, user ``u``: ``positions[r B + u] + j`` (``positions`` lists the ``rows * B`` anchor positions)."""
    B, K = users_per_row, draft_len
    if len(positions) != rows * B:
        raise ValueError(f"{len(positions)} positions for {rows} x {B} user slots")
    out = []
    for r in range(rows):
        for j in range(K + 1):
            out += [positions[r * B + u] + j for u in range(B)]
    return out


def verify_tokens_row_major(tokens: torch.Tensor, users_per_row: int, rows: int = 1) -> torch.Tensor:
    """``tokens`` ``[rows B, K + 1]`` (user-major: row ``u`` = ``[anchor_u, d_1 .. d_K]``) -> the flat candidate-major
    id list ``[rows B (K + 1)]`` the verify embeds (mesh row r's block first, inside it candidate j's ``B`` users)."""
    B = users_per_row
    N, P = tokens.shape
    if N != rows * B:
        raise ValueError(f"{N} token rows for {rows} x {B} user slots")
    return tokens.reshape(rows, B, P).permute(0, 2, 1).reshape(-1).long()


def verify_ids_user_major(flat: torch.Tensor, users_per_row: int, draft_len: int, rows: int = 1) -> torch.Tensor:
    """The inverse of ``verify_tokens_row_major`` for the verify's outputs: flat ``[rows R]`` -> ``[rows B, K + 1]``."""
    B, P = users_per_row, draft_len + 1
    return flat.reshape(rows, P, B).permute(0, 2, 1).reshape(rows * B, P).long()


def select_rows(m: torch.Tensor, users_per_row: int) -> list[int]:
    """The verify row (inside the user's mesh row block) holding user ``u``'s accept point: ``m_u B + (u mod B)``."""
    return [int(m[u]) * users_per_row + (u % users_per_row) for u in range(len(m))]


def select_onehot(m: torch.Tensor, users_per_row: int, draft_len: int, rows: int = 1) -> torch.Tensor:
    """``[rows, 1, B, R]`` one-hot rows (mesh row ``r``'s block ``[1, B, R]``): user ``u``'s row picks verify row
    ``m_u B + u``; the matmul ``onehot @ hidden`` is an exact row gather (one 1.0 product, fp32 accumulation)."""
    B, R = users_per_row, users_per_row * (draft_len + 1)
    sel = torch.zeros(rows, 1, B, R)
    for n, r_idx in enumerate(select_rows(m, B)):
        sel[n // B, 0, n % B, r_idx] = 1.0
    return sel


def kda_commit_masks(m: torch.Tensor, draft_len: int) -> torch.Tensor:
    """``[K + 1, N]`` fp32: ``take[j, u] = 1.0`` iff ``m_u == j`` (the launch after which user ``u``'s state is the
    committed one, ``S_{m_u + 1}``)."""
    N = len(m)
    take = torch.zeros(draft_len + 1, N)
    for u in range(N):
        take[int(m[u]), u] = 1.0
    return take


def kda_commit_gates(m: torch.Tensor, draft_len: int) -> torch.Tensor:
    """``[K + 1, N]`` int32: ``gate[j, u] = 1`` iff ``m_u >= j`` -- launch ``j`` of the gated commit writes user ``u``'s
    state, so user ``u`` takes launches ``0 .. m_u`` and ends at ``S_{m_u + 1}``. Monotone in ``j`` (``gate[0]`` all ones);
    ``take_j = gate_j - gate_{j + 1}`` recovers ``kda_commit_masks``."""
    m = torch.as_tensor(m).long().reshape(1, -1)
    j = torch.arange(draft_len + 1).reshape(-1, 1)
    return (m >= j).to(torch.int32)


def commit_gate_rows(gate_j: torch.Tensor, users_per_row: int, rows: int, words: int = GATE_WORDS) -> torch.Tensor:
    """One launch's ``[N]`` gates -> the ``[rows, words]`` int32 host tensor whose row ``r`` is mesh row ``r``'s gate page
    (its ``B`` user slots in the first ``B`` words, the padding words 0): the kernel reads page ``tb`` = ``[32]`` words.
    """
    B = users_per_row
    g = torch.as_tensor(gate_j).to(torch.int32).reshape(rows, B)
    out = torch.zeros(rows, words, dtype=torch.int32)
    out[:, :B] = g
    return out


def kda_commit_steps(m: torch.Tensor) -> torch.Tensor:
    """``[N]`` int32: ``n_u = m_u + 1`` -- the candidates (the anchor + the accepted drafts) the fused commit launch
    applies to user ``u`` (``sum_j gate[j, u]`` of ``kda_commit_gates``)."""
    return (torch.as_tensor(m).long().reshape(-1) + 1).to(torch.int32)


def commit_steps_page(m: torch.Tensor, users_per_row: int, rows: int, words: int = GATE_WORDS) -> torch.Tensor:
    """The fused commit's ``[rows, words]`` int32 step page (row ``r`` = mesh row ``r``'s ``B`` user slots in the first
    ``B`` words, padding 0): the kernels read page ``tb`` = ``[32]`` words, word ``u`` = ``m_u + 1``."""
    return commit_gate_rows(kda_commit_steps(m), users_per_row, rows, words)


def ring_slots_masked(positions: list[int], m: torch.Tensor, j: int, kpool: int) -> list[int]:
    """The ring row candidate ``j`` of every user writes in the commit: ``(p_u + j) mod P`` when the candidate is
    committed (``j <= m_u``), else a dead row of the ring block."""
    return [(p + j) % kpool if j <= int(m[u]) else dead_ring_row(j, kpool) for u, p in enumerate(positions)]


# ================================================================================== host loop (device-free)
@dataclass
class SpecStats:
    iterations: int = 0
    tokens: int = 0  # committed tokens over all users (excluding the seed token)
    accepted: list[int] = field(default_factory=list)  # m per (iteration, user)
    seconds: float = 0.0

    @property
    def tokens_per_iteration(self) -> float:
        return self.tokens / self.iterations if self.iterations else 0.0

    @property
    def mean_accepted(self) -> float:
        return sum(self.accepted) / len(self.accepted) if self.accepted else 0.0

    def summary(self) -> str:
        return (
            f"{self.iterations} iterations, {self.tokens} tokens ({self.tokens_per_iteration:.2f} per iteration "
            f"over all users), mean accepted drafts {self.mean_accepted:.2f}, {self.seconds:.2f} s"
        )


class GreedySpecLoop:
    """The lossless greedy speculative loop over three callables (the device graphs, or fakes in the host test):

    * ``verify(tokens [N, K + 1] long, positions [N]) -> (ids [N, K + 1], d1 [N, K + 1])``: the main model's greedy
      id of every candidate row and the MTP level-1 draft of every row (teacher-forced with that id), the row's
      context = the committed context + ``tokens[u, : j + 1]``;
    * ``commit(m [N] long, positions [N])``: make the committed rows ``0 .. m_u`` permanent (states / index);
    * ``draft_chain(m [N] long, d1 [N] long, positions [N]) -> [N, K - 1]`` (``K > 1`` only): MTP levels
      ``2 .. K`` chained from the level-1 output of verify row ``m_u`` (its draft ``d1[u]``), ``positions`` the
      anchor positions of that verify (level ``k`` sits at ``p_u + m_u + k - 1``).

    Every iteration verifies ``[anchor, d_1 .. d_K]`` and commits ``m_u + 1`` tokens per user (the accepted drafts
    plus the main model's bonus token, which is the next anchor). The first iteration carries placeholder drafts
    (``valid = 0``): it is a plain step that also seeds the drafts. ``eos``: a user stops after producing it (its
    later rows keep running as padding and are dropped)."""

    def __init__(
        self,
        users_per_row: int,
        draft_len: int,
        *,
        verify: Callable,
        commit: Callable,
        draft_chain: Callable | None = None,
        rows: int = 1,
    ):
        self.B, self.K, self.rows = users_per_row, draft_len, rows
        self.N = users_per_row * rows
        candidate_rows(users_per_row, draft_len)
        if draft_len > 1 and draft_chain is None:
            raise ValueError("K > 1 needs the draft chain (MTP levels 2 .. K)")
        self.verify, self.commit, self.draft_chain = verify, commit, draft_chain

    def run(
        self, anchors: list[int], positions: list[int], max_new_tokens: int, eos: int | None = None
    ) -> tuple[list[list[int]], SpecStats]:
        """``anchors``: the committed token every user decodes next (the prompt's first generated token),
        ``positions``: its position (= the prompt length). Returns the committed continuation of every user
        (``max_new_tokens`` ids after the anchor, or up to and including ``eos``) and the statistics."""
        N, K = self.N, self.K
        if len(anchors) != N or len(positions) != N:
            raise ValueError(f"{len(anchors)} anchors / {len(positions)} positions for {N} users")
        anchors_t = torch.tensor(anchors, dtype=torch.long)
        pos = list(positions)
        drafts = anchors_t[:, None].repeat(1, K) if K else torch.zeros(N, 0, dtype=torch.long)
        valid = 0
        out: list[list[int]] = [[] for _ in range(N)]
        done = [False] * N
        stats = SpecStats()
        t0 = time.perf_counter()
        while not all(done):
            tokens = torch.cat([anchors_t[:, None], drafts], dim=1)  # [N, K + 1]
            ids, d1 = self.verify(tokens, pos)
            m, committed, bonus = accept_greedy(drafts, ids, valid)
            for u in range(N):
                if done[u]:
                    continue
                for t in committed[u]:
                    out[u].append(t)
                    stats.tokens += 1
                    if len(out[u]) >= max_new_tokens or (eos is not None and t == eos):
                        done[u] = True
                        break
                stats.accepted.append(int(m[u]))
            stats.iterations += 1
            self.commit(m, pos)
            if all(done):
                break
            # the next iteration: anchor = the bonus token at p + m + 1, drafts = the MTP chain from row m
            next_pos = [pos[u] + int(m[u]) + 1 for u in range(N)]
            if K:
                d1_sel = d1[torch.arange(N), m]
                rest = self.draft_chain(m, d1_sel, pos) if K > 1 else torch.zeros(N, 0, dtype=torch.long)
                drafts = torch.cat([d1_sel[:, None], rest.long().reshape(N, K - 1)], dim=1)
                valid = K
            anchors_t = bonus
            pos = next_pos
        stats.seconds = time.perf_counter() - t0
        return [o[:max_new_tokens] for o in out], stats


# ==================================================================================== device helpers
def _mapper(mesh: GLMMesh, rows_dim: int | None):
    if mesh.is_single:
        return None
    if rows_dim is not None and mesh.rows > 1:
        return mesh.shard_rows(rows_dim)
    return mesh.replicate()


def _upload(mesh: GLMMesh, t: torch.Tensor, dtype, layout, rows_dim: int | None = None) -> ttnn.Tensor:
    """Persistent device tensor from host values; ``rows_dim`` = the host dim split over the mesh rows."""
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, rows_dim),
    )


def _refresh(mesh: GLMMesh, t: torch.Tensor, dst: ttnn.Tensor, rows_dim: int | None = None):
    """Host values -> the persistent device tensor ``dst`` (same shape / dtype / layout / placement)."""
    host = ttnn.from_torch(t.contiguous(), dtype=dst.dtype, layout=dst.layout, mesh_mapper=_mapper(mesh, rows_dim))
    ttnn.copy_host_to_device_tensor(host, dst)


def _up_i32(mesh: GLMMesh, vals: list[int]) -> ttnn.Tensor:
    """``[n]`` int32 ROW_MAJOR per chip from a ``rows * n`` host list (row-sharded like the decode positions)."""
    return _meta_upload(mesh.device, torch.tensor(vals, dtype=torch.int32), ttnn.int32, mesh, False)


def _set_i32(mesh: GLMMesh, vals: list[int], dst: ttnn.Tensor):
    host = ttnn.from_torch(
        torch.tensor(vals, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=_meta_mapper(mesh, False),
    )
    ttnn.copy_host_to_device_tensor(host, dst)


def _zeros_like(t: ttnn.Tensor, device) -> ttnn.Tensor:
    return ttnn.zeros(
        tuple(t.shape), dtype=t.dtype, layout=t.layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _zeros(shape, dtype, device, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.zeros(tuple(shape), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rows(t: ttnn.Tensor, j: int, B: int) -> ttnn.Tensor:
    """Candidate ``j``'s ``[1, 1, B, W]`` slab of a ``[1, 1, R, W]`` tensor (a fresh tensor)."""
    W = int(t.shape[-1])
    return ttnn.slice(t, [0, 0, j * B, 0], [1, 1, (j + 1) * B, W])


def _hifi4_fp32(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


# ================================================================================== per-iteration inputs
class VerifyInputs:
    """The persistent inputs of the verify (V) and commit (C) graphs for one mesh: the candidate-major ids, the
    per-row ``cur_pos``, per-candidate ``[B]`` positions (latent appends), and the commit's per-candidate ring
    rows (masked) and pool ids. Refreshed on the host before every replay."""

    def __init__(self, mesh: GLMMesh, cfg, users_per_row: int, draft_len: int, max_dense_pos: int):
        self.mesh, self.cfg = mesh, cfg
        self.B, self.K = users_per_row, draft_len
        self.R = candidate_rows(users_per_row, draft_len)
        self.rows = mesh.rows
        self.N = self.B * self.rows
        self.P = cfg.index_kpool
        self.max_dense_pos = max_dense_pos
        zeros_pos = [0] * self.N
        self.ids = _upload(
            mesh, torch.zeros(1, 1, 1, self.rows * self.R, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 3
        )
        self.rows_pos = _up_i32(mesh, verify_row_positions(zeros_pos, self.B, self.K, self.rows))
        self.pos = [_up_i32(mesh, zeros_pos) for _ in range(self.K + 1)]
        self.ring = [_up_i32(mesh, zeros_pos) for _ in range(self.K + 1)]
        self.pool = [_up_i32(mesh, zeros_pos) for _ in range(self.K + 1)]
        self.host_positions: list[int] = zeros_pos

    def check_positions(self, positions: list[int]):
        if len(positions) != self.N:
            raise ValueError(f"{len(positions)} positions for {self.N} user slots")
        top = max(positions) + self.K
        if top > self.max_dense_pos:
            raise NotImplementedError(
                f"verify rows reach position {top} > {self.max_dense_pos} (the dense range of the DSA decode and the "
                "caches' max_seq_len): the sparse-range verify (index scorer with K + 1 rows per user) is not "
                "wired; use plain decode beyond"
            )

    def set_verify(self, tokens: torch.Tensor, positions: list[int]):
        """``tokens`` ``[N, K + 1]`` (user-major), ``positions`` the ``N`` anchor positions."""
        self.check_positions(positions)
        flat = verify_tokens_row_major(tokens, self.B, self.rows).to(torch.int32)
        if int(flat.min()) < 0 or int(flat.max()) >= self.cfg.vocab_size:
            raise ValueError("token id out of range")
        _refresh(self.mesh, flat.reshape(1, 1, 1, -1), self.ids, 3)
        _set_i32(self.mesh, verify_row_positions(positions, self.B, self.K, self.rows), self.rows_pos)
        for j in range(self.K + 1):
            _set_i32(self.mesh, [p + j for p in positions], self.pos[j])
        self.host_positions = list(positions)

    def set_commit(self, m: torch.Tensor, positions: list[int]):
        """Per-candidate ring rows (rejected candidates to dead rows) and pool ids of the committed rows."""
        for j in range(self.K + 1):
            _set_i32(self.mesh, ring_slots_masked(positions, m, j, self.P), self.ring[j])
            _set_i32(self.mesh, [(p + j) // self.P for p in positions], self.pool[j])


# ============================================================================= DSA layers (and the MTP block)
class DsaSpecState:
    """One DSA attention (``TtDSA`` + its ``TtLatentCache``: a main layer or the MTP block) under speculation.

    * ``verify_attention(h, vin)``: the composite decode stems on the ``R`` rows, the latent appended per candidate
      (``[B]`` positions, the real page table), the raw indexer k / gate rows of all candidates copied into the
      persistent ``k_all`` / ``g_all`` (for the commit), then the dense flash-MLA decode over the user's pages
      through ``table`` (the page table with every user row replicated ``K + 1`` times, rows candidate-major)
      with ``cur_pos = p_u + j`` per row. No ring / pool write.
    * ``chain_attention(h, pos)``: a ``[B]``-row tentative step (MTP levels 2 .. K): latent append + dense
      attention over the real table at ``pos``; no index write, nothing cached.
    * ``commit(vin)``: the pooled-key maintenance of candidates ``0 .. K`` in order from ``k_all`` / ``g_all``
      (``TtDSAIndexer.maintain_decode``: k LayerNorm, ring row write at ``vin.ring[j]``, pool rewrite at
      ``vin.pool[j]``); the rejected candidates' ring rows land on dead rows, their pool rewrite on an
      incomplete pool.

    Tensors read / written: ``cache.cache`` (latent pages: written per candidate), ``cache.page_table`` (read),
    ``cache.index.ring_k`` / ``ring_g`` / ``pool_cache`` (commit only, through the indexer's own methods), the
    attention weights of ``attn`` (read)."""

    def __init__(self, attn: TtDSA, cache: TtLatentCache, mesh: GLMMesh, users_per_row: int, draft_len: int):
        if cache.partition:
            raise NotImplementedError("the speculative verify is not wired for the user-partitioned cache")
        self.attn, self.cache, self.mesh = attn, cache, mesh
        self.B, self.K = users_per_row, draft_len
        self.R = candidate_rows(users_per_row, draft_len)
        if cache.users_per_chip != users_per_row:
            raise ValueError(f"cache has {cache.users_per_chip} user slots, the verify {users_per_row}")
        D = attn.indexer.D
        self.k_all = _zeros((1, 1, self.R, D), ttnn.bfloat16, mesh.device)
        self.g_all = _zeros((1, 1, self.R, D), ttnn.bfloat16, mesh.device)
        self.table = self._replicated_table()

    def _replicated_table(self) -> ttnn.Tensor:
        """``[R, blocks_per_seq]`` int32 (replicated on every chip): row ``j B + u`` = the page-table row of user ``u``."""
        pt = self.cache.page_table_host[: self.B]
        rep = pt.repeat(self.K + 1, 1).contiguous()  # candidate-major: [pt; pt; ...]
        return _upload(self.mesh, rep, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def _stems(self, h: ttnn.Tensor):
        attn = self.attn
        T = int(h.shape[-2])
        stem = attn.lin_stem(h, dtype=ttnn.bfloat16, compute_kernel_config=attn.hifi2)
        parts = {k: ttnn.slice(stem, [0, 0, 0, a], [1, 1, T, b]) for k, (a, b) in attn.stem_bounds.items()}
        ttnn.deallocate(stem)
        q_resid = ttnn.rms_norm(
            parts["qa"], epsilon=attn.cfg.rms_norm_eps, weight=attn.qa_norm, compute_kernel_config=attn.hifi4
        )
        c = ttnn.rms_norm(
            parts["kva"], epsilon=attn.cfg.rms_norm_eps, weight=attn.kva_norm, compute_kernel_config=attn.hifi4
        )
        for k in ("qa", "kva", "w"):
            ttnn.deallocate(parts[k])
        return q_resid, c, parts["k"], parts["g"]

    def _dense_attention(self, q_resid: ttnn.Tensor, table: ttnn.Tensor, cur_pos: ttnn.Tensor) -> ttnn.Tensor:
        attn = self.attn
        T = int(q_resid.shape[-2])
        q = attn._q(q_resid)  # [1,1,T,H*Dq]
        q_lat = attn._absorb_q(q)  # [1,H,T,L]
        ttnn.deallocate(q)
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,T,H,L]
        ttnn.deallocate(q_lat)
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            self.cache.cache,
            head_dim_v=attn.L,
            page_table_tensor=table,
            cur_pos_tensor=cur_pos,
            scale=attn.scale,
            program_config=attn._decode_sdpa_config(T),
            compute_kernel_config=attn.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,T,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,T,L]
        ttnn.deallocate(o_lat)
        y = attn._o_proj(attn._o_lat_to_heads(oh))
        ttnn.deallocate(oh)
        return y

    def verify_attention(self, h: ttnn.Tensor, vin: VerifyInputs) -> ttnn.Tensor:
        """``h`` ``[1, 1, R, 4096]`` bf16 (input_layernorm applied) -> ``[1, 1, R, 4096]`` bf16."""
        B, K, L = self.B, self.K, self.attn.L
        if int(h.shape[-2]) != self.R:
            raise ValueError(f"verify rows {int(h.shape[-2])} != R {self.R}")
        q_resid, c, k_raw, g_raw = self._stems(h)
        ttnn.copy(k_raw, self.k_all)
        ttnn.copy(g_raw, self.g_all)
        ttnn.deallocate(k_raw)
        ttnn.deallocate(g_raw)
        for j in range(K + 1):  # one paged write per candidate: a user's rows share a block (multi-row writes race)
            c_j = _rows(c, j, B)
            self.cache.update(ttnn.reshape(c_j, [1, B, 1, L]), vin.pos[j])
            ttnn.deallocate(c_j)
        ttnn.deallocate(c)
        y = self._dense_attention(q_resid, self.table, vin.rows_pos)
        ttnn.deallocate(q_resid)
        return y

    def chain_attention(self, h: ttnn.Tensor, pos: ttnn.Tensor) -> ttnn.Tensor:
        """``h`` ``[1, 1, B, 4096]`` bf16 at the ``[B]`` positions ``pos`` (a tentative row per user)."""
        B, L = self.B, self.attn.L
        if int(h.shape[-2]) != B:
            raise ValueError(f"chain rows {int(h.shape[-2])} != B {B}")
        q_resid, c, k_raw, g_raw = self._stems(h)
        ttnn.deallocate(k_raw)
        ttnn.deallocate(g_raw)
        self.cache.update(ttnn.reshape(c, [1, B, 1, L]), pos)
        ttnn.deallocate(c)
        y = self._dense_attention(q_resid, self.cache.page_table, pos)
        ttnn.deallocate(q_resid)
        return y

    def commit(self, vin: VerifyInputs):
        """Ring rows + pool rewrites of the committed candidates in order (the sequential decode's index state)."""
        for j in range(self.K + 1):
            k_j = _rows(self.k_all, j, self.B)
            g_j = _rows(self.g_all, j, self.B)
            # ``x`` is unused when ``kg`` is given; the indexer deallocates k_j / g_j itself
            self.attn.indexer.maintain_decode(k_j, self.cache.index, vin.pool[j], vin.ring[j], kg=(k_j, g_j))


# ============================================================================================ KDA layers
class KdaSpecState:
    """One KDA layer (``FunctionalDecoder`` with its persistent ``kda_batched`` state) under speculation.

    ``verify_attention(h)``: the block decode's projections on all ``R`` rows (``lin_qkv``, ``lin_g1`` ->
    ``lin_g2``: the weights read once), copied into the persistent ``qkv_all`` ``[1, 1, R, 3 H K]`` bf16 and
    ``g2_all`` ``[1, 1, R, H K + H V + Hb]`` fp32; the live state backed up (``bk``); then ``K + 1`` sequential
    ``kda_decode`` launches on the candidate slabs (launch ``j`` reads the state after candidates ``< j``: the
    causal recurrence of every row) leaving the live state at ``S_{K+1}``; the per-launch outputs concatenated
    (``[1, 1, R, H V]``) feed one ``o_proj`` + TP all-reduce.

    ``commit()`` (after ``set_masks(m)``), mode ``blend`` (today's path): for ``j = 0 .. K``: launch ``j`` on ``bk``
    (``S_0`` advancing to ``S_{j+1}``) from the cached slab, then ``live = live * keep_j + bk * take_j`` (recurrent
    with ``[B, H, K, 1]`` fp32 masks, conv with ``[3, B, 1]`` bf16 masks: exact products / adds); afterwards live =
    ``S_{m_u + 1}`` per user and ``bk`` is spent (one commit per verify). Tensors read: ``qkv_all``, ``g2_all``, the
    masks, the layer's ``taps_blk`` / ``chan_consts`` / ``o_norm_w32`` / ``selectors``; read and written in place:
    ``bk.recurrent`` / ``bk.convolution`` (the replay) and ``kda_batched.recurrent`` / ``.convolution`` (the blend).

    Mode ``gated`` (SP1, ``GLM53_SPEC_COMMIT=gated``): ``verify_attention`` copies the live state into ``bk`` (now a
    scratch) and runs its ``K + 1`` launches THERE (the live state is untouched by V); ``commit()`` runs launches
    ``0 .. K`` on the live state through the gated program (``kda_gated = attn.kda_decode.gated_twin()``) with the
    per-launch gate pages ``gate[j]`` (``[1, 32]`` int32 per chip, ``gate[j][u] = 1`` iff ``m_u >= j``): a gated user's
    launch is the plain program's update (same inputs: the live state after its launches ``< j``, the cached slab),
    an ungated user's launch leaves its recurrent tiles and history rows byte for byte. No masks, no blends. Tensors
    read: ``qkv_all``, ``g2_all``, ``gate[j]``, the layer constants; written in place: ``bk`` (V) and
    ``kda_batched.recurrent`` / ``.convolution`` (C, gated per user).

    Mode ``fused`` (SP2, ``GLM53_SPEC_COMMIT=fused``): ``verify_attention`` runs ONE dry fused launch
    (``kda_fused_v = attn.kda_decode.fused_twin(K + 1, dry=True)``) on the ``[1, 1, R, W]`` slabs against the live
    state (read only: no backup, no scratch, no slices, no concat: its ``[1, 1, R, H V]`` output is ``o_proj``'s input);
    ``commit()`` runs ONE fused commit launch (``kda_fused_c``, ``dry=False``) on ``qkv_all`` / ``g2_all`` with the step
    page ``steps`` (``[1, 32]`` int32 per chip, word ``u`` = ``m_u + 1``, refreshed by ``set_masks``): user ``u`` applies
    candidates ``0 .. m_u`` in order on its live state in L1 and is written once (the plain state after ``m_u + 1``
    steps, bit for bit). Tensors read: ``qkv_all``, ``g2_all``, ``steps``, the layer constants; written in place:
    ``kda_batched.recurrent`` / ``.convolution`` (C only). No ``bk``, no masks, no gates."""

    def __init__(
        self, dec: FunctionalDecoder, mesh: GLMMesh, users_per_row: int, draft_len: int, commit: str = "blend"
    ):
        if not dec.is_kda or dec.kda_batched is None:
            raise ValueError(f"layer {dec.layer_idx} is not a KDA layer")
        attn = dec._require_weights().attn
        if attn.fused_decode != "block":
            raise NotImplementedError("the speculative verify needs the block KDA decode (GLM53_KDA_FUSED=1)")
        self.dec, self.attn, self.mesh = dec, attn, mesh
        self.state: KdaState = dec.kda_batched
        self.B, self.K = users_per_row, draft_len
        self.R = candidate_rows(users_per_row, draft_len)
        if int(self.state.recurrent.shape[0]) != users_per_row:
            raise ValueError(f"KDA state holds {int(self.state.recurrent.shape[0])} slots, the verify {users_per_row}")
        device = mesh.device
        H, Kd, V = attn.H, attn.K, attn.V
        Hb = -(-H // TILE) * TILE
        self.taps = int(self.state.convolution.shape[0])
        self.bk = (
            None
            if commit == "fused"
            else KdaState(
                recurrent=_zeros_like(self.state.recurrent, device),
                convolution=_zeros_like(self.state.convolution, device),
            )
        )  # blend: the backup; gated: the verify's scratch; fused: none
        self.qkv_all = _zeros((1, 1, self.R, 3 * H * Kd), ttnn.bfloat16, device)
        self.g2_all = _zeros((1, 1, self.R, H * Kd + H * V + Hb), ttnn.float32, device)
        N = users_per_row * mesh.rows
        if commit not in SPEC_COMMIT_MODES:
            raise ValueError(f"commit mode {commit!r}: expected one of {SPEC_COMMIT_MODES}")
        self.commit_mode = commit
        self.rows = mesh.rows
        self.kda_gated = self.kda_fused_v = self.kda_fused_c = None
        self.gate: list = []
        self.steps = None
        self.take_rec = self.keep_rec = self.take_conv = self.keep_conv = []
        if commit == "fused":
            # the fused T = K + 1 programs (the verify's dry one, the commit's) and the per-user step page: one
            # [1, 32] int32 ROW_MAJOR page per chip (mesh row r's B user slots in words 0 .. B-1), read by the commit
            # kernels at run time; the verify's dry launch takes the same tensor and never reads it
            self.kda_fused_v = attn.kda_decode.fused_twin(self.K + 1, dry=True)
            self.kda_fused_c = attn.kda_decode.fused_twin(self.K + 1, dry=False)
            zero_page = torch.zeros(mesh.rows, GATE_WORDS, dtype=torch.int32)
            self.steps = _upload(mesh, zero_page, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, 0)
            self.bk = None  # no backup / scratch state in the fused mode
        elif commit == "gated":
            # the per-launch write gates: one [1, 32] int32 ROW_MAJOR page per chip (mesh row r's B user slots in
            # words 0 .. B-1), read by the gated kernels at run time; the gated twin of the layer's program
            self.kda_gated = attn.kda_decode.gated_twin()
            zero_page = torch.zeros(mesh.rows, GATE_WORDS, dtype=torch.int32)
            self.gate = [_upload(mesh, zero_page, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, 0) for _ in range(self.K + 1)]
        else:
            rec_mask = torch.zeros(N, H, Kd, 1)
            conv_mask = torch.zeros(self.taps, N, 1)
            self.take_rec = [_upload(mesh, rec_mask, ttnn.float32, ttnn.TILE_LAYOUT, 0) for _ in range(self.K + 1)]
            self.keep_rec = [_upload(mesh, rec_mask, ttnn.float32, ttnn.TILE_LAYOUT, 0) for _ in range(self.K + 1)]
            self.take_conv = [_upload(mesh, conv_mask, ttnn.bfloat16, ttnn.TILE_LAYOUT, 1) for _ in range(self.K + 1)]
            self.keep_conv = [_upload(mesh, conv_mask, ttnn.bfloat16, ttnn.TILE_LAYOUT, 1) for _ in range(self.K + 1)]

    @property
    def gated(self) -> bool:
        return self.commit_mode == "gated"

    @property
    def fused(self) -> bool:
        return self.commit_mode == "fused"

    def set_masks(self, m: torch.Tensor):
        """``m`` ``[N]``: user ``u``'s state is taken after launch ``m_u`` (blend: the take / keep masks; gated: the
        write gates ``gate_j[u] = m_u >= j``; fused: the step page ``n_u = m_u + 1``)."""
        if self.fused:
            _refresh(self.mesh, commit_steps_page(m, self.B, self.rows), self.steps, 0)
            return
        if self.gated:
            gates = kda_commit_gates(m, self.K)  # [K+1, N]
            for j in range(self.K + 1):
                _refresh(self.mesh, commit_gate_rows(gates[j], self.B, self.rows), self.gate[j], 0)
            return
        take = kda_commit_masks(m, self.K)  # [K+1, N]
        H, Kd = self.attn.H, self.attn.K
        for j in range(self.K + 1):
            t = take[j]
            rec = t.reshape(-1, 1, 1, 1).expand(-1, H, Kd, 1)
            conv = t.reshape(1, -1, 1).expand(self.taps, -1, 1)
            _refresh(self.mesh, rec, self.take_rec[j], 0)
            _refresh(self.mesh, 1.0 - rec, self.keep_rec[j], 0)
            _refresh(self.mesh, conv, self.take_conv[j], 1)
            _refresh(self.mesh, 1.0 - conv, self.keep_conv[j], 1)

    def _launch(
        self, qkv_j: ttnn.Tensor, g2_j: ttnn.Tensor, st: KdaState, gate: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        """One ``kda_decode`` program on a candidate slab against the state ``st`` (updated in place); with ``gate`` the
        gated twin (users whose gate word is 0 keep their state bytes)."""
        attn = self.attn
        o = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.B, attn.H * attn.V]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        args = (
            qkv_j,
            g2_j,
            st.convolution,
            st.recurrent,
            attn.taps_blk,
            attn.chan_consts,
            attn.o_norm_w32,
            attn.selectors,
            o,
        )
        if gate is None:
            attn.kda_decode(*args)
        else:
            self.kda_gated(*args, gate=gate)
        return o

    def _launch_fused(self, qkv_all: ttnn.Tensor, g2_all: ttnn.Tensor, st: KdaState, dry: bool) -> ttnn.Tensor:
        """One fused ``T = K + 1`` program on the ``[1, 1, R, W]`` candidate-major slabs against the state ``st``:
        ``dry`` = the verify (``K + 1`` steps per user, the state read only), else the commit (``steps`` page: ``m_u + 1``
        steps per user, the state updated in place). Returns the ``[1, 1, R, H V]`` output (candidate-major)."""
        attn = self.attn
        o = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.R, attn.H * attn.V]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        op = self.kda_fused_v if dry else self.kda_fused_c
        op(
            qkv_all,
            g2_all,
            st.convolution,
            st.recurrent,
            attn.taps_blk,
            attn.chan_consts,
            attn.o_norm_w32,
            attn.selectors,
            o,
            steps=self.steps,
        )
        return o

    def backup(self):
        """``bk`` := the live state (``S_0`` of the iteration; a whole-tensor copy, fp32 / bf16 preserved). Blend: the
        backup the commit replays on; gated: the scratch the verify's launches run on."""
        ttnn.copy(self.state.recurrent, self.bk.recurrent)
        ttnn.copy(self.state.convolution, self.bk.convolution)

    def verify_attention(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """``h`` ``[1, 1, R, 4096]`` bf16 (input_layernorm applied) -> ``[1, 1, R, 4096]`` bf16."""
        attn, B, K = self.attn, self.B, self.K
        if int(h.shape[-2]) != self.R:
            raise ValueError(f"verify rows {int(h.shape[-2])} != R {self.R}")
        qkv = attn.lin_qkv(h, dtype=ttnn.bfloat16, compute_kernel_config=attn.hifi2)  # [1,1,R,3HK]
        g1 = attn.lin_g1(h, dtype=ttnn.bfloat16, compute_kernel_config=attn.hifi2)
        g2 = attn.lin_g2(g1, dtype=ttnn.float32, compute_kernel_config=attn.fp32)  # [1,1,R,HK+HV+Hb]
        ttnn.deallocate(g1)
        ttnn.copy(qkv, self.qkv_all)
        ttnn.copy(g2, self.g2_all)
        if self.fused:
            # one dry launch on the whole slab against the live state (read only): no backup, no slices, no concat
            o_all = self._launch_fused(qkv, g2, self.state, dry=True)  # [1,1,R,HV] candidate-major
            ttnn.deallocate(qkv)
            ttnn.deallocate(g2)
        else:
            self.backup()  # S_0 of this iteration (blend: the commit's replay base; gated: the verify's scratch)
            target = self.bk if self.gated else self.state  # gated: the live state is untouched by the verify
            outs = []
            for j in range(K + 1):
                qkv_j, g2_j = _rows(qkv, j, B), _rows(g2, j, B)
                outs.append(self._launch(qkv_j, g2_j, target))
                ttnn.deallocate(qkv_j)
                ttnn.deallocate(g2_j)
            ttnn.deallocate(qkv)
            ttnn.deallocate(g2)
            if K == 0:
                o_all = outs[0]
            else:
                o_all = ttnn.concat(outs, dim=-2)  # [1,1,R,HV] candidate-major
                for o in outs:
                    ttnn.deallocate(o)
        y = attn.lin_o(o_all, dtype=ttnn.bfloat16, compute_kernel_config=attn.hifi2)
        ttnn.deallocate(o_all)
        return attn.mesh.all_reduce_cols(y)

    def commit(self):
        """live := ``S_{m_u + 1}`` per user (``set_masks`` first). Blend: the backup ``bk`` (``S_0``) is advanced in
        place by the launches ``0 .. K`` on the cached projections (the same programs on the same inputs as the
        verify's) and after launch ``j`` the live state takes ``bk`` for the users with ``m_u == j`` (``live = live *
        keep_j + bk * take_j``: exact products / adds). Every user is taken exactly once, so the live state's
        post-verify content never survives. The backup is consumed: one commit per verify (the next verify backs up
        again). Gated: the launches ``0 .. K`` run on the live state (``S_0``: the verify ran on the scratch) through
        the gated program, launch ``j`` writing the users with ``m_u >= j`` only; no blend, nothing consumed. Fused: ONE
        launch on the cached slabs applies ``m_u + 1`` candidates per user on the live state (the step page) and writes
        every user once; nothing consumed."""
        if self.fused:
            o = self._launch_fused(self.qkv_all, self.g2_all, self.state, dry=False)
            ttnn.deallocate(o)
            return
        if self.gated:
            for j in range(self.K + 1):
                qkv_j, g2_j = _rows(self.qkv_all, j, self.B), _rows(self.g2_all, j, self.B)
                o = self._launch(qkv_j, g2_j, self.state, gate=self.gate[j])
                ttnn.deallocate(o)
                ttnn.deallocate(qkv_j)
                ttnn.deallocate(g2_j)
            return
        st, bk = self.state, self.bk
        for j in range(self.K + 1):
            qkv_j, g2_j = _rows(self.qkv_all, j, self.B), _rows(self.g2_all, j, self.B)
            o = self._launch(qkv_j, g2_j, bk)
            ttnn.deallocate(o)
            ttnn.deallocate(qkv_j)
            ttnn.deallocate(g2_j)
            ttnn.multiply(st.recurrent, self.keep_rec[j], output_tensor=st.recurrent)
            t = ttnn.multiply(bk.recurrent, self.take_rec[j])
            ttnn.add(st.recurrent, t, output_tensor=st.recurrent)
            ttnn.deallocate(t)
            ttnn.multiply(st.convolution, self.keep_conv[j], output_tensor=st.convolution)
            t = ttnn.multiply(bk.convolution, self.take_conv[j])
            ttnn.add(st.convolution, t, output_tensor=st.convolution)
            ttnn.deallocate(t)


# ====================================================================================== the decoder
def prefill_with_mtp(
    model: "GLM53MeshModel", mtp: TtMTPBlock, prompt_ids: torch.Tensor, user: int = 0
) -> tuple[int, int]:
    """The prompt through the main model's eager prefill (every layer, the head on the last token) and the MTP
    cache / index of ``user`` from the prompt rows (row i: the main model's normed hidden at i and token
    ``t_{i+1}``; the last row takes the prompt's first generated token). On a multi-row mesh the MTP sees the
    same slabs the layers left their streams in. Returns ``(first token, its position = S)``."""
    mesh = model.mesh
    S = int(prompt_ids.numel())
    streams, logits, owner = model.prefill(prompt_ids, user=user)
    tok = model.first_token(logits, owner)
    T_pad = int(streams.shape[-2]) * mesh.rows
    next_ids = torch.cat([prompt_ids.reshape(-1).long()[1:], torch.tensor([tok])])
    host = model.embedding.tokens_to_host(next_ids, pad_to=T_pad)  # [1,1,1,T_pad]
    ids_dev = _upload(mesh, host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 3 if mesh.rows > 1 else None)
    h = model.head.collapse(streams)  # this row's slab [1,1,T_loc,H] fp32
    hidden = model.head.norm(h)  # bf16: the LM-head input of every prompt row
    if h is not streams:
        ttnn.deallocate(h)
    out = mtp.prefill_cache(ids_dev, hidden, user=user, seq_len=S)
    for t in (out, hidden, ids_dev, streams, logits):
        ttnn.deallocate(t)
    return tok, S


def verify_layer(
    dec: FunctionalDecoder,
    streams: ttnn.Tensor,
    state: DsaSpecState | KdaSpecState,
    vin: VerifyInputs,
    *,
    collapse: bool,
) -> ttnn.Tensor:
    """One decoder layer on the ``R`` candidate rows: ``TtDecoderLayer.decode_forward`` with the speculative
    attention (the mHC wraps, norms and the FFN are row-wise and unchanged)."""
    layer = dec._require_weights()
    if dec.is_kda:
        attention = lambda h: state.verify_attention(h)
    else:
        attention = lambda h: state.verify_attention(h, vin)
    streams = layer._hc_call(layer.attn_hc, streams, attention, layer.input_norm_w)
    return layer._hc_call(layer.ffn_hc, streams, layer.mlp, layer.post_norm_w, collapse=collapse)


class SpecDecoder:
    """The device side of speculative decoding on a ``GLM53MeshModel`` with an attached ``TtMTPBlock``.

    Construction allocates every persistent tensor (states, inputs, outputs) -- before any trace capture of the
    model. ``warmup`` compiles the three graphs eagerly (consuming dummy tokens: re-prefill afterwards, like
    ``GLM53MeshModel.capture``), ``capture`` traces them, ``prefill`` fills the main and MTP caches of a user,
    ``generate`` runs the ``GreedySpecLoop`` over the traced (or eager) graphs."""

    def __init__(
        self,
        model: "GLM53MeshModel",
        mtp: TtMTPBlock,
        draft_len: int,
        *,
        keep_logits: bool = True,
        commit_mode: str | None = None,
        plain_draft: bool = False,
    ):
        """``plain_draft``: also own the served ordinary step (graph P: the plain decode on the model's persistent
        inputs + MTP level 1 on every user slot, ``plain_d1`` / ``plain_hidden`` kept for the proposal) and the
        plain-sourced draft chain (graph D0: levels ``2 .. K`` from that hidden, no gather) -- the vLLM adapter's
        speculating launch, where a step without drafts is the plain decode and the next proposal still needs
        the MTP (``tt/generator_vllm.py``)."""
        if model.head is None:
            raise ValueError("the speculative decoder needs the model's head")
        self.model, self.mtp, self.K = model, mtp, int(draft_len)
        self.commit_mode = spec_commit_mode() if commit_mode is None else str(commit_mode).strip().lower()
        if self.commit_mode not in SPEC_COMMIT_MODES:
            raise ValueError(f"commit_mode {commit_mode!r}: expected one of {SPEC_COMMIT_MODES}")
        self.mesh: GLMMesh = model.mesh
        self.cfg = model.cfg
        self.B = model.max_batch
        self.rows = self.mesh.rows
        self.N = model.total_users
        self.R = candidate_rows(self.B, self.K)
        if mtp.max_batch != self.B or mtp.mesh.rows != self.rows:
            raise ValueError(f"MTP block ({mtp.max_batch} users/row) does not match the model ({self.B})")
        if mtp.max_seq_len < model.max_seq_len:
            raise ValueError("the MTP cache must hold at least the model's max_seq_len positions")
        dsa_layers = [d for d in model.layers if not d.is_kda]
        dense_max = min([d.layer.attn.dense_exact_max_pos for d in dsa_layers] + [mtp.attn.dense_exact_max_pos])
        self.vin = VerifyInputs(self.mesh, self.cfg, self.B, self.K, min(dense_max, model.max_seq_len - 1))
        self.states: dict[int, DsaSpecState | KdaSpecState] = {}
        for dec in model.layers:
            if dec.is_kda:
                self.states[dec.layer_idx] = KdaSpecState(dec, self.mesh, self.B, self.K, commit=self.commit_mode)
            else:
                self.states[dec.layer_idx] = DsaSpecState(dec.layer.attn, dec.cache, self.mesh, self.B, self.K)
        self.mtp_state = DsaSpecState(mtp.attn, mtp.cache, self.mesh, self.B, self.K)
        H, V = self.cfg.hidden_size, self.cfg.vocab_size
        device = self.mesh.device
        rm, u32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32
        # V outputs (persistent copies): the main greedy ids, the MTP level-1 ids and normed hidden of every row
        self.out_ids = _zeros((1, 1, self.R, 1), u32, device, rm)
        self.out_d1 = _zeros((1, 1, self.R, 1), u32, device, rm)
        self.out_hidden = _zeros((1, 1, self.R, H), ttnn.bfloat16, device)
        self.out_logits = _zeros((1, 1, self.R, V), ttnn.bfloat16, device) if keep_logits else None
        # D inputs / outputs: the select matrix, the selected level-1 draft ids, the per-level MTP positions
        self.sel = _upload(self.mesh, torch.zeros(self.rows, 1, self.B, self.R), ttnn.bfloat16, ttnn.TILE_LAYOUT, 0)
        self.d1_ids = _upload(self.mesh, torch.zeros(1, 1, 1, self.N, dtype=torch.int32), u32, rm, 3)
        self.level_pos = {k: _up_i32(self.mesh, [0] * self.N) for k in range(2, self.K + 1)}
        self.out_draft = {k: _zeros((1, 1, self.B, 1), u32, device, rm) for k in range(2, self.K + 1)}
        # P outputs (``plain_draft``): the ordinary step's MTP level-1 ids and normed hidden of every user slot, the
        # D0 chain's source (persistent: allocated here, before any capture)
        self.plain_draft = bool(plain_draft)
        self.plain_d1 = _zeros((1, 1, self.B, 1), u32, device, rm) if plain_draft else None
        self.plain_hidden = _zeros((1, 1, self.B, H), ttnn.bfloat16, device) if plain_draft else None
        self.plain_outputs = None  # the captured P step's (logits, ids), read after every replay
        self.hifi4 = _hifi4_fp32(device)
        self.trace_v = self.trace_c = self.trace_d = self.trace_d0 = self.trace_p = None
        self.compiled = False

    # ------------------------------------------------------------------------------------------ graphs
    def _verify_graph(self):
        """V: the main model on the R rows + MTP level 1 on every row; outputs copied into the persistent buffers."""
        m, vin = self.model, self.vin
        streams = m.embedding(vin.ids)  # [1,1,R,16384] fp32
        last = len(m.layers) - 1
        for k, dec in enumerate(m.layers):
            collapse = k == last and dec.supports_collapse
            out = verify_layer(dec, streams, self.states[dec.layer_idx], vin, collapse=collapse)
            ttnn.deallocate(streams)
            streams = out
        hn, logits, ids = m.head_forward_with_hidden(streams)
        ttnn.deallocate(streams)
        ttnn.copy(ids, self.out_ids)
        if self.out_logits is not None:
            ttnn.copy(logits, self.out_logits)
        ttnn.deallocate(logits)
        # MTP level 1 on all rows, teacher-forced with the row's greedy id (the id row is a view of ``ids``)
        x = self.mtp.project(self.mtp.ids_row(ids), hn)
        ttnn.deallocate(hn)
        ttnn.deallocate(ids)
        out = self.mtp.block_rows(x, lambda h: self.mtp_state.verify_attention(h, vin))
        normed, mlogits, mids = self.mtp.head_forward(out)
        ttnn.copy(mids, self.out_d1)
        ttnn.copy(normed, self.out_hidden)
        for t in (out, normed, mlogits, mids):
            ttnn.deallocate(t)

    def _commit_graph(self):
        """C: the KDA state commit of every KDA layer, the index commit of every DSA layer and of the MTP."""
        for dec in self.model.layers:
            st = self.states[dec.layer_idx]
            if dec.is_kda:
                st.commit()
            else:
                st.commit(self.vin)
        self.mtp_state.commit(self.vin)

    def _draft_graph(self, source: str = "verify"):
        """D: MTP levels 2 .. K from the selected level-1 rows (exact one-hot gather of the level-1 hidden);
        ``source="plain"`` (D0, ``plain_draft``): the same chain from the ordinary step's ``plain_d1`` /
        ``plain_hidden`` of every user slot (no gather; the persistent source is never deallocated)."""
        if self.K < 2:
            return
        if source == "verify":
            hid = ttnn.matmul(
                self.sel, self.out_hidden, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4
            )  # [1,1,B,H]
            ids = self.d1_ids  # [1,1,1,B] persistent
            owned = True  # ``hid`` is this graph's transient
        elif source == "plain":
            if not self.plain_draft:
                raise RuntimeError("the plain-sourced draft chain needs SpecDecoder(plain_draft=True)")
            hid = self.plain_hidden
            ids = self.mtp.ids_row(self.plain_d1)  # a view of the persistent ids: never deallocated
            owned = False
        else:
            raise ValueError(f"draft source {source!r}: expected 'verify' or 'plain'")
        garbage: list[ttnn.Tensor] = []
        for k in range(2, self.K + 1):
            x = self.mtp.project(ids, hid)
            pos = self.level_pos[k]
            out = self.mtp.block_rows(x, lambda h, pos=pos: self.mtp_state.chain_attention(h, pos))
            normed, logits, dids = self.mtp.head_forward(out)
            ttnn.copy(dids, self.out_draft[k])
            garbage += [out, logits]
            if owned:
                garbage.append(hid)
            hid, owned = normed, True
            ids = self.mtp.ids_row(dids)  # a view of dids: dids is freed after the next level consumed it
            garbage.append(dids)
        if owned:
            garbage.append(hid)
        for t in garbage:
            ttnn.deallocate(t)

    def _plain_graph(self):
        """P (``plain_draft``): the plain decode step on the model's persistent inputs (``ids_dev`` / ``positions``,
        the packed unpack included) + MTP level 1 on every user slot at ``model.mtp_positions`` (the same values):
        ``GLM53MeshModel.decode_step_with_draft`` with the drafter's ids and normed hidden copied into the persistent
        ``plain_d1`` / ``plain_hidden`` (the D0 chain's source). Returns ``(logits [1,1,B,V], ids [1,1,B,1])`` like
        ``decode_step``: the served ordinary step's outputs. Device ops only (trace-capturable)."""
        if not self.plain_draft:
            raise RuntimeError("the plain-with-draft step needs SpecDecoder(plain_draft=True)")
        m = self.model
        if m.mtp_positions is None:
            raise RuntimeError("set the MTP positions first (GLM53MeshModel.set_mtp_positions)")
        m._unpack_inputs(m.ids_dev, m.positions)
        streams = m.embedding(m.ids_dev)
        last = len(m.layers) - 1
        for k, dec in enumerate(m.layers):
            collapse = k == last and dec.supports_collapse
            out = dec.decode_forward(streams, positions=m.positions, collapse=collapse)
            ttnn.deallocate(streams)
            streams = out
        hn, logits, ids = m.head_forward_with_hidden(streams)
        ttnn.deallocate(streams)
        # the id row is a view of ``ids`` (never deallocated); ``project`` reads it and ``hn`` without consuming
        out, normed, mlogits, draft = self.mtp.decode_forward(self.mtp.ids_row(ids), hn, m.mtp_positions)
        ttnn.copy(draft, self.plain_d1)
        ttnn.copy(normed, self.plain_hidden)
        for t in (out, normed, mlogits, draft, hn):
            ttnn.deallocate(t)
        return logits, ids

    # ------------------------------------------------------------------------------------ compile / trace
    def warmup(self, positions: list[int] | None = None, cq_id: int = 0):
        """Run V, C and D once eagerly on placeholder inputs (program compile, lazily built constants). Advances
        the KDA states by one anchor row and writes placeholder rows into the caches: re-prefill afterwards."""
        positions = [0] * self.N if positions is None else list(positions)
        tokens = (torch.arange(self.N * (self.K + 1)) % 1000 + 1000).reshape(self.N, self.K + 1)
        self.vin.set_verify(tokens, positions)
        m = torch.zeros(self.N, dtype=torch.long)
        self._set_commit(m, positions)
        self._set_draft(m, torch.zeros(self.N, dtype=torch.long), positions)
        if self.plain_draft:
            self._set_draft_plain(positions)
        t0 = time.perf_counter()
        self._verify_graph()
        self._commit_graph()
        self._draft_graph()
        if self.plain_draft and self.K >= 2:
            self._draft_graph("plain")
        ttnn.synchronize_device(self.mesh.device)
        self.compiled = True
        logger.info(
            f"spec decode: graphs compiled in {time.perf_counter() - t0:.1f} s (B {self.B} K {self.K} R {self.R}, "
            f"KDA commit {self.commit_mode}{', plain-sourced chain' if self.plain_draft else ''})"
        )

    def warmup_plain(self, ids: torch.Tensor, host_positions: list[int]):
        """``GLM53MeshModel.warmup_decode`` for the P step (``plain_draft``): allocate the model's persistent decode
        inputs (the packed segments with the knob) and the MTP positions, run P once eagerly (compiles the plain
        programs and the MTP's decode row). Consumes the token (the states advance): reset / re-prefill after."""
        m = self.model
        m._stage_decode_inputs(ids, host_positions)
        m.set_mtp_positions(host_positions)
        for t in self._plain_graph():
            ttnn.deallocate(t)
        ttnn.synchronize_device(m.device)

    def capture_plain(self, ids: torch.Tensor, host_positions: list[int], cq_id: int = 0):
        """Warm-up + capture of the P step on the model's persistent inputs (``GLM53MeshModel.capture`` with the
        drafter): returns ``(trace id, (logits, ids))``, transient outputs read after every ``replay``. Both passes
        consume the token."""
        self.warmup_plain(ids, host_positions)
        dev = self.mesh.device
        t0 = time.perf_counter()
        tid = ttnn.begin_trace_capture(dev, cq_id=cq_id)
        outs = self._plain_graph()
        ttnn.end_trace_capture(dev, tid, cq_id=cq_id)
        self.mesh.allocations_frozen[0] = True  # no lazily allocated constants from here on (tt/kda.py)
        ttnn.synchronize_device(dev)
        self.trace_p, self.plain_outputs = tid, outs
        logger.info(f"spec decode: trace P (plain step + MTP level 1) captured in {time.perf_counter() - t0:.1f} s")
        return tid, outs

    def capture(self, cq_id: int = 0):
        """Capture V, C and D (and D0 with ``plain_draft``; after ``warmup``: every persistent tensor already
        exists). Call ``warmup`` before ANY trace capture of the model (the plain decode trace included): a
        collective of a new shape compiled after a capture places its semaphores in the L1 the captured trace
        overwrites, and the replay hangs."""
        if not self.compiled:
            self.warmup(cq_id=cq_id)
        dev = self.mesh.device
        graphs = [("V", self._verify_graph), ("C", self._commit_graph), ("D", self._draft_graph)]
        if self.plain_draft:
            graphs.append(("D0", lambda: self._draft_graph("plain")))
        for name, fn in graphs:
            if name in ("D", "D0") and self.K < 2:
                continue
            t0 = time.perf_counter()
            tid = ttnn.begin_trace_capture(dev, cq_id=cq_id)
            fn()
            ttnn.end_trace_capture(dev, tid, cq_id=cq_id)
            ttnn.synchronize_device(dev)
            setattr(self, {"V": "trace_v", "C": "trace_c", "D": "trace_d", "D0": "trace_d0"}[name], tid)
            logger.info(f"spec decode: trace {name} captured in {time.perf_counter() - t0:.1f} s")

    def release(self):
        for attr in ("trace_v", "trace_c", "trace_d", "trace_d0", "trace_p"):
            tid = getattr(self, attr)
            if tid is not None:
                ttnn.release_trace(self.mesh.device, tid)
                setattr(self, attr, None)
        self.plain_outputs = None

    def _run(self, trace_id, fn, cq_id: int = 0):
        if trace_id is not None:
            ttnn.execute_trace(self.mesh.device, trace_id, cq_id=cq_id, blocking=True)
        else:
            fn()

    # ------------------------------------------------------------------------------------- the three steps
    def _ids_host(self, t: ttnn.Tensor) -> torch.Tensor:
        return self.model.ids_to_host(t)

    def verify_forward(self, tokens: torch.Tensor, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens`` ``[N, K + 1]`` long (user-major ``[anchor, d_1 .. d_K]``), ``positions`` the ``N`` anchor
        positions -> ``(ids [N, K + 1], d1 [N, K + 1])``: the main model's greedy id and the MTP level-1 draft of
        every candidate row. Traced after ``capture``, eager before."""
        self.vin.set_verify(tokens, positions)
        self._run(self.trace_v, self._verify_graph)
        ids = verify_ids_user_major(self._ids_host(self.out_ids), self.B, self.K, self.rows)
        d1 = verify_ids_user_major(self._ids_host(self.out_d1), self.B, self.K, self.rows)
        return ids, d1

    def verify_logits(self) -> torch.Tensor | None:
        """The last verify's main logits ``[N, K + 1, V]`` fp32 (``keep_logits``): acceptance diagnostics."""
        if self.out_logits is None:
            return None
        t = self.mesh.to_host(self.out_logits, rows=2 if self.rows > 1 else None).float()[0, 0]  # [rows R, V]
        V = t.shape[-1]
        return t.reshape(self.rows, self.K + 1, self.B, V).permute(0, 2, 1, 3).reshape(self.N, self.K + 1, V)

    def _set_commit(self, m: torch.Tensor, positions: list[int]):
        self.vin.set_commit(m, positions)
        for st in self.states.values():
            if isinstance(st, KdaSpecState):
                st.set_masks(m)

    def commit(self, m: torch.Tensor, positions: list[int]):
        """Make candidates ``0 .. m_u`` of every user permanent (KDA states, pooled-key index)."""
        self._set_commit(m, positions)
        self._run(self.trace_c, self._commit_graph)

    def _set_draft(self, m: torch.Tensor, d1: torch.Tensor, positions: list[int]):
        _refresh(self.mesh, select_onehot(m, self.B, self.K, self.rows), self.sel, 0)
        _refresh(self.mesh, d1.to(torch.int32).reshape(1, 1, 1, -1), self.d1_ids, 3)
        for k in range(2, self.K + 1):  # level k of the next iteration sits at p + m + k - 1 = next anchor + k - 2
            _set_i32(self.mesh, [p + int(mu) + k - 1 for p, mu in zip(positions, m)], self.level_pos[k])

    def draft_chain(self, m: torch.Tensor, d1: torch.Tensor, positions: list[int]) -> torch.Tensor:
        """MTP levels ``2 .. K`` from verify row ``m_u`` of every user (its level-1 hidden and draft ``d1``), at the
        anchor positions ``positions`` of the verify that produced them -> ``[N, K - 1]`` draft ids."""
        if self.K < 2:
            return torch.zeros(self.N, 0, dtype=torch.long)
        self._set_draft(m, d1, positions)
        self._run(self.trace_d, self._draft_graph)
        return torch.stack([self._ids_host(self.out_draft[k]) for k in range(2, self.K + 1)], dim=1)

    # ------------------------------------------------------------------------ the served ordinary step (P / D0)
    def _set_draft_plain(self, positions: list[int]):
        """The D0 chain's per-level positions after an ordinary step at ``positions`` (its level-1 row sat at the
        step's position ``p``; level ``k`` at ``p + k - 1``: ``_set_draft`` with ``m = 0``)."""
        for k in range(2, self.K + 1):
            _set_i32(self.mesh, [p + k - 1 for p in positions], self.level_pos[k])

    def plain_step(self, eager: bool = False) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One ordinary step (P) on the model's persistent inputs (loaded by the caller: ``load_decode_inputs`` +
        ``set_mtp_positions``): the captured trace's ``(logits, ids)`` after ``capture_plain``, else (or with
        ``eager``) an eager run's transient outputs (the caller deallocates those)."""
        if self.trace_p is not None and not eager:
            ttnn.execute_trace(self.mesh.device, self.trace_p, cq_id=0, blocking=True)
            return self.plain_outputs
        return self._plain_graph()

    def read_plain_d1(self) -> torch.Tensor:
        """``[N]`` long: the MTP level-1 draft of every user slot after the last P step (its prediction of the token
        after the step's greedy id)."""
        return self._ids_host(self.plain_d1)

    def draft_chain_plain(self, positions: list[int]) -> torch.Tensor:
        """MTP levels ``2 .. K`` of every user slot from the last P step's level-1 output (``plain_d1`` /
        ``plain_hidden``) at the step positions ``positions`` -> ``[N, K - 1]`` draft ids (the drafts after
        ``read_plain_d1``'s)."""
        if self.K < 2:
            return torch.zeros(self.N, 0, dtype=torch.long)
        self._set_draft_plain(positions)
        self._run(self.trace_d0, lambda: self._draft_graph("plain"))
        return torch.stack([self._ids_host(self.out_draft[k]) for k in range(2, self.K + 1)], dim=1)

    # ------------------------------------------------------------------------------------------ prefill
    def prefill(self, prompt_ids: torch.Tensor, user: int = 0) -> tuple[int, int]:
        """``prefill_with_mtp`` on this decoder's model and drafter."""
        return prefill_with_mtp(self.model, self.mtp, prompt_ids, user=user)

    # ----------------------------------------------------------------------------------------- generation
    def loop(self) -> GreedySpecLoop:
        return GreedySpecLoop(
            self.B, self.K, verify=self.verify_forward, commit=self.commit, draft_chain=self.draft_chain, rows=self.rows
        )

    def generate(
        self, anchors: list[int], positions: list[int], max_new_tokens: int, eos: int | None = None
    ) -> tuple[list[list[int]], SpecStats]:
        """Lossless greedy speculative generation for every user slot from its committed ``anchors`` at
        ``positions`` (``prefill`` gives both). The caller keeps ``max(positions) + max_new_tokens + K + 1``
        below the dense range / ``max_seq_len``."""
        return self.loop().run(anchors, positions, max_new_tokens, eos=eos)
