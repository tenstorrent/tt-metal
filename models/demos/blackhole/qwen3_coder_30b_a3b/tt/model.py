# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full 48-layer Qwen3-Coder-30B-A3B-Instruct on the 4-die P300_X2 mesh.

Stage 05. This module is the *wrapper* around the stage-04 optimized multichip
decoder layer and it deliberately changes nothing about that layer's strategy:

* attention TP=4 (8 Q heads, 1 K head, 1 V head per die), experts EP=4 (32 of
  128 per die), router and both residual RMSNorms and the residual replicated;
* two all-reduces per layer, ``FABRIC_1D_RING``, 2 links prefill / 1 decode;
* expert weights ``bfloat4_b`` at LoFi with ``in0_block_w`` 16/12, attention
  projections ``bfloat8_b`` DRAM-sharded, paged KV cache ``bfloat16``;
* router top-k in fp32 logit space;
* **the inter-layer residual layout contract**: every layer takes and returns a
  replicated ``[1, 1, B, 2048]`` bfloat16 ``TILE`` ``DRAM_MEMORY_CONFIG``
  tensor, and there is no collective, gather, reshard or layout conversion
  between layers. ``prefill_hidden`` and ``decode_hidden`` below are literally a
  ``for`` loop over 48 layers with the residual threaded straight through.

What the wrapper adds, and where each new boundary lives:

``embed_tokens``
    **Replicated**, bf16, so the embedding output *is* the residual contract
    with no collective at all. A hidden-sharded embedding would be 4x smaller
    per die but would owe an all-gather on every prefill chunk and every decode
    token; at 0.622 GB/die against 22.35 GB of measured headroom
    (``config/context_contract.json``) the replicated table is free and the
    collective is not. This is also the shape the stage-03 footprint probe
    allocated, so the published capacity numbers describe what actually runs.

``model.norm`` (final RMSNorm)
    Replicated, and shares the layer code: decode uses
    ``multichip_decoder.decode_residual_norm`` (width-sharded over 8 L1 cores,
    the same kernel and compute config as the two residual norms), prefill uses
    the interleaved ``ttnn.rms_norm``.

``lm_head``
    **Column-parallel over the vocabulary**: die *d* owns columns
    ``37984*d .. 37984*d+37983`` of ``[2048, 151936]``. 151936 = 4 * 37984 and
    37984 = 32 * 1187, so the split is exact and needs no vocabulary padding.
    **Logits never reach the host on the token-out path, and neither strategy
    all-gathers them.** Both reduce first and gather the survivors: greedy takes
    a per-die argmax and all-gathers four candidate values and indices
    (``_WatcherCleanSampling1D._sample_argmax``), top-k/top-p takes a per-die
    top-32 and all-gathers 32 values and indices. Which of the two is faster here
    was measured, not assumed -- see ``sample_greedy_argmax``.

``rotary`` (decode only)
    ``ttnn.experimental.rotary_embedding_hf(is_decode_mode=True)`` reading a
    per-user cos/sin pair **gathered on device** by ``ttnn.embedding`` from a
    position tensor the trace advances with ``ttnn.plus_one``. The layer's
    shipped spelling, ``ttnn.experimental.rotary_embedding``, takes the position
    as a **Python int** compile-time argument and therefore cannot be replayed:
    a captured trace would rotate every subsequent token at the position it was
    captured at. Note this is the *HF* rotary, same ``rotate_half`` channel
    convention -- so unlike stage 04's rejected ``rotary_embedding_llama`` lever
    (README limitation 4) it needs no weight permutation, changes no KV-cache
    channel convention and leaves prefill untouched.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig

from .functional_decoder import DecoderLayerConfig, KVCache
from .multichip_decoder import (
    MESH_SHAPE,
    NUM_DEVICES,
    TOPOLOGY,
    MeshContext,
    MeshDecoderConfig,
    MultichipWeights,
    _head_shard,
    _norm_compute_config,
    build_local_sparsity,
    decode_residual_norm,
    decoder_layer_decode_multichip,
    decoder_layer_prefill_multichip,
    fallback_audit,
    mesh_context,
    upload_multichip_weights,
)
from .precision import DEFAULT_PRECISION, PrecisionConfig, dtype_to_name

HF_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
HF_REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"

HIDDEN_SIZE = 2048
VOCAB_SIZE = 151936
NUM_LAYERS = 48
HEAD_DIM = 128
MAX_CONTEXT = 262144
DEFAULT_PAGE_BLOCK_SIZE = 32
DEFAULT_MAX_BATCH_SIZE = 1
#: ``ttnn.sampling`` and ``nlp_create_qkv_heads_decode`` both address 32 fixed
#: user slots; decode is always one 32-row tile regardless of the active batch.
SAMPLING_SLOTS = 32
#: Trace region per device. Two traces (model decode + sampling) over 48 layers.
DEFAULT_TRACE_REGION_SIZE = 300_000_000
#: RoPE table rows materialised at construction; grown on demand to the request
#: horizon by ``ensure_rope_capacity`` so a short request never pays for 262144
#: rows (which would be 64 MB/die of cos plus 64 MB of sin).
DEFAULT_ROPE_CACHE_LEN = 8192

#: ``lm_head`` weight dtype. bfloat8_b halves the 155 MB/die bf16 read that a
#: decode step would otherwise make against a 2048x37984 weight.
#:
#: Since stage 07 this is an **alias** for ``DEFAULT_PRECISION.lm_head_dtype``,
#: not the source of truth: a model built at a non-default ``PrecisionConfig``
#: does not read it. See ``tt/precision.py``.
LM_HEAD_WEIGHT_DTYPE = DEFAULT_PRECISION.lm_head_dtype
#: The embedding table stays bf16: it is a gather, not a matmul, and bfloat8_b
#: would quantise every token's hidden state at the very top of the stack.
#: Alias for ``DEFAULT_PRECISION.embedding_dtype``, as above.
EMBED_WEIGHT_DTYPE = DEFAULT_PRECISION.embedding_dtype


#: ``_WatcherCleanSampling1D._sample_argmax``'s "not a winner" sentinel. Any value
#: strictly greater than the vocabulary works; 2**20 is exact in int32 and leaves
#: ``idx - BIG`` far from overflow.
_DIST_ARGMAX_BIG = 1 << 20


class _WatcherCleanSampling1D(Sampling1D):
    """``Sampling1D`` with the force-argmax gather spelled the way this layer spells it.

    Two overrides, for two different reasons.

    ------------------------------------------------------------------------
    ``_sample_argmax`` -- reduce first, gather second
    ------------------------------------------------------------------------

    ``Sampling1D._sample_argmax`` all-gathers the whole column-parallel logit
    shard (37984 bf16 columns per die) up to the full 151936 on **every** die,
    untilizes 151936 columns and runs one ``ttnn.argmax`` over them.
    ``doc/full_model/tt_perf_report_full_model_decode.txt`` shows those two ops
    at ``AllGatherAsync 889 us`` and ``ArgMax 859 us``. **Neither number is a
    share of a token-out step, and the two must not be summed against one.**
    That report is stage 05's **2-layer** window, which charges the terminal
    path against two layers instead of 48 and so over-weights it by
    construction -- the two rows are 27.5% and 26.5% of *that* window. And the
    column is per-op device-kernel time summed over the op's own cores (2 for
    the gather, 110 for the argmax), which is a different accounting from the
    wall clock of a decode step. An earlier revision of this docstring set
    ``889 + 859`` against "the 1.87 ms of non-layer work in a 22.079 ms
    token-out step"; the near-agreement was a coincidence between two
    incommensurable measurements and the claim is withdrawn.

    The full 48-layer profile is the accounting that means something. On the
    shipped tree ``doc/optimized_full_model/probes/profile_summary_decode.json``
    puts the *whole* terminal block -- final norm, LM head, this sampler and the
    token feedback -- at **366.5 us of an 18889.5 us decode iteration, 1.94%**,
    of which this sampler is **126.2 us**. The baseline path was replaced before
    that profile was taken and so has no 48-layer op row of its own; its
    in-model price is a token-out delta and is quoted as one under
    ``sample_greedy_argmax``.

    The override computes the same token by reducing on each die first and
    all-gathering only the four survivors::

        rm         = untilize(local_shard)          # bf16   [1,1,32,37984]
        rm         = rm[:, :, :B, :]                # bf16   [1,1,B,37984]
        local_idx  = argmax(rm, -1, keepdim)        # uint32 [1,1,B,1]
        local_max  = gather(rm, -1, local_idx)      # bf16   [1,1,B,1]
        global_idx = local_idx + rank*37984         # int32, sharded constant
        vals4      = all_gather(local_max)          # bf16   [1,1,B,4]
        idx4       = all_gather(global_idx)         # int32  [1,1,B,4]
        gmax       = max(vals4, -1, keepdim)
        mask       = (vals4 == gmax)                # int32 0/1
        token      = min(BIG + mask*(idx4 - BIG), -1)
        token      = pad(token, to=32, value=0)     # uint32 [1,1,32]

    ``doc/optimized_full_model/probes/distributed_argmax_probe.py`` measures the
    two against each other at the shipped shape, trace-captured, median of 100:
    **1.1432 ms baseline against 0.6275 ms, 1.82x**. Five things in that spelling
    are load-bearing and were each established on the device, not assumed:

    * **The local maximum must come from ``ttnn.gather``, not ``ttnn.max``.**
      ``ttnn.max`` over the 37984-wide shard costs 0.494 ms -- more than the
      ``ttnn.argmax`` over the same tensor (0.371 ms). ``ttnn.gather`` at the
      index the argmax already produced costs 0.059 ms. That single substitution
      is the difference between 1.05x and 1.82x.
    * **Only the live user rows are reduced.** The logit tile is logically 32
      rows because ``ttnn.sampling`` addresses 32 slots, but at batch ``B`` the
      other ``32-B`` are zero-logit padding and reduce to token 0 by
      construction. ``ttnn.argmax``'s kernel compares scalar-wise on a
      data-movement RISC, so the cost is linear in rows: the whole reduction is
      **631.6 us over 32 rows and 250.8 us over 1**, and the ``ttnn.pad(value=0)``
      that restores the 32 slots writes back exactly the values the 32-row
      reduction produced. ``argmax_outer_dim_probe.py`` checks that on the
      device rather than asserting it.
    * **Untilize before the argmax.** ``ttnn.argmax``'s multicore path needs
      ROW_MAJOR; the TILE path is single-core and the whole leg becomes 23.25 ms.
      The untilize itself is 0.075 ms.
    * **Indices are INT32 end to end.** FLOAT32 elementwise rounds an index
      through bf16 (36885 -> 36864), and ``ttnn.where`` on int32 operands returns
      bit garbage -- hence the arithmetic select ``BIG + mask*(idx-BIG)`` rather
      than a ``where``. ``ttnn.gather`` in turn demands a UINT32 index, which is
      exactly what ``ttnn.argmax`` emits, so no cast happens on that edge.
    * **The cross-die reduction is a ``min`` over masked indices, never a sum.**
      On an exact tie both lanes survive the mask; ``sum(mask*idx)`` would add
      the two indices together, ``min`` keeps the lower one. Because the dies own
      contiguous ascending vocabulary ranges and ``ttnn.argmax`` returns the
      first occurrence within a die, that is precisely ``torch.argmax``'s
      first-maximal rule. The probe checks it with crafted cross-die, within-die
      and triple ties, and checks the first-occurrence property of ``ttnn.argmax``
      itself.

    **Output contract.** The base writes the token into the caller's
    ``tt_out_tok`` via ``ttnn.argmax(output_tensor=...)`` and returns it. The
    traced decode loop feeds that same buffer back as the next token's input, so
    returning a *new* tensor would silently break token feedback
    (``models/common/sampling/generator.py::_validate_trace_inputs`` checks
    identity, and this model's trace binds ``token`` as both sampler output and
    model input). The override therefore ends in ``ttnn.copy`` into the caller's
    buffer and returns that exact object -- same dtype (uint32), layout
    (ROW_MAJOR), shape and buffer address.

    **Fallback.** The fast path is only taken when the reduction it performs is
    provably the same function as the base's. It falls back to
    ``super()._sample_argmax`` whenever ``valid_vocab_size < vocab_size`` (a
    padded vocabulary needs the invalid tail masked *before* the local argmax,
    which ``_mask_invalid_vocab_logits`` /
    ``_can_slice_valid_vocab_for_argmax`` do around the base's gather and this
    path does not reproduce), whenever any invalid-vocab mask buffer is present,
    on a single device, or when the logits do not arrive as an exact even shard.
    For this model ``valid_vocab_size == vocab_size == 151936 == 4*37984``, so
    the fast path is what runs -- but a token id >= the real vocabulary stays
    impossible either way, because a padded vocabulary never reaches it.

    ------------------------------------------------------------------------
    ``_argmax_all_gather`` -- no ``Topology::Linear`` + ``num_workers_per_link=1``
    ------------------------------------------------------------------------

    Still overridden, still needed: the split top-k/top-p path is live for any
    request with ``top_k > 1`` or ``top_p > 0`` (``sample_split``), and
    ``_sample_argmax``'s fallback branch above uses it too.

    ``ttnn.experimental.all_gather_async`` trips a
    BRISC ``ASSERT`` in ``minimal_default_writer.cpp`` when it is given
    ``topology=Topology::Linear`` **together with** ``num_workers_per_link=1``.
    Neither alone does it; the pair does, at any width. The full A/B matrix is
    ``doc/full_model/watcher_ab.log`` and the model-free reproducer is
    ``doc/full_model/probes/ccl_watcher_ab.py --leg linear_workers1``.

    ``Sampling1D._argmax_all_gather`` walks straight into that pair on any mesh
    smaller than T3K. Its first branch -- Ring, no barrier -- is guarded by
    ``default_topology(mesh) == Topology.Ring``, which is **False** on this 1x4
    Blackhole mesh, so the branch is unreachable here. The fallback then runs
    ``_get_argmax_all_gather_config``, which forces ``Topology.Linear`` for any
    mesh under 8 devices, and the call below it hardcodes
    ``num_workers_per_link=1``. Linear + 1 worker: exactly the tripping pair.

    The decoder layer's own two all-reduces have been watcher-clean for four
    stages, and the reason is visible in the same matrix: the layer never passes
    ``num_workers_per_link`` at all, so the op picks its default. This override
    does the same thing -- same op, same ``dim``, same semaphores, same
    ``Topology.Ring`` the layer uses, and **no tuning knobs pinned**. The
    matrix's ``sampler_shape_default_knobs`` leg is this exact call at this exact
    shape, and it is clean.

    This is a local workaround for an upstream bug, not a fix for it. Both
    reports (the op, and ``sampling_1d.py``'s unreachable Ring branch) still
    stand and should still be filed; this subclass just means stage 05 does not
    ship an unchecked-but-violated device invariant while they are open. When
    the op is fixed, delete this class and pass ``Sampling1D`` directly.

    Subclassing is the seam because ``Sampling1D.from_config`` builds through
    ``object.__new__(cls)`` and ``_bind_strategy`` binds
    ``self._pre_argmax_gather = self._argmax_all_gather`` by attribute lookup on
    the instance -- so the override is what gets bound. **No shared code is
    edited.**
    """

    def _argmax_all_gather(self, logits):
        cfg = self.config
        return ttnn.experimental.all_gather_async(
            logits,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_argmax_gather_links,
            memory_config=logits.memory_config(),
            topology=cfg.ag_topology,
            # Deliberately NOT passing chunks_per_sync / num_workers_per_link /
            # num_buffers_per_channel. Pinning num_workers_per_link=1 is the half
            # of the tripping pair we control. See the class docstring.
        )

    # -- distributed argmax ---------------------------------------------------

    def _distributed_argmax_local_vocab(self):
        """Per-die vocabulary width if the distributed argmax applies, else ``None``.

        Every condition here is a condition under which the reduction below is
        *provably* the same function as ``Sampling1D._sample_argmax``'s. Anything
        else falls back to the base implementation rather than being approximated.
        """
        cfg = self.config
        if getattr(self, "_invalid_vocab_mask", None) is not None:
            return None
        if getattr(self, "_invalid_vocab_tail_mask", None) is not None:
            return None
        valid = cfg.valid_vocab_size if cfg.valid_vocab_size is not None else cfg.vocab_size
        if valid != cfg.vocab_size:
            # A padded vocabulary needs the invalid tail masked before the *local*
            # argmax, which this path does not do. The base masks/slices around
            # its full gather and stays correct; use it.
            return None
        num_devices = cfg.mesh_device.get_num_devices()
        if num_devices < 2 or cfg.vocab_size % num_devices != 0:
            return None
        local = cfg.vocab_size // num_devices
        if local % ttnn.TILE_SIZE != 0:
            return None
        return local

    #: Live user rows in the sampler's 32-slot logit tile. ``None`` means "all 32"
    #: and reproduces the pre-row-slicing behaviour exactly. The model sets it to
    #: its own ``max_batch_size`` (1 by default) -- see ``_sample_argmax``, and
    #: ``doc/optimized_full_model/probes/argmax_outer_dim_probe.py`` for why it is
    #: worth 2.5x on the whole sampler.
    _dist_active_rows = None

    def _distributed_argmax_active_rows(self, slots: int) -> int:
        rows = self._dist_active_rows
        if rows is None:
            return int(slots)
        return max(1, min(int(rows), int(slots)))

    # -- sampling penalties ---------------------------------------------------
    #
    # ``Sampling1D`` has no penalty stage at all, and the vLLM TT plugin does not
    # route penalised requests to host sampling (``platform.py`` sends ``min_p``,
    # ``bad_words``, ``logit_bias``, ``allowed_token_ids``, ``min_tokens``,
    # ``prompt_logprobs`` and structured output to the host sampler -- penalties
    # are deliberately *not* in that list). It packs all three into
    # ``TTSamplingParams`` and hands the model the token history it needs
    # (``model_runner.py``: ``prompt_tokens`` / ``output_tokens`` are populated
    # "if penalties are needed (decode only)"), expecting the model's on-device
    # sampler to apply them. This is that stage.
    #
    # ------------------------------------------------------------------------
    # The shard-boundary problem, and why this spelling cannot get it wrong
    # ------------------------------------------------------------------------
    #
    # Logits are column-parallel: die ``d`` holds vocabulary ids
    # ``d*37984 .. d*37984+37983`` of the 151936, contiguous and ascending -- the
    # same decomposition ``load_device_buffers`` above builds ``_dist_die_offset``
    # from, and ``_dist_local_vocab`` is reused here rather than recomputed. A
    # penalty is keyed by a **global** token id, so for id ``t`` only die
    # ``t // 37984`` may touch column ``t % 37984``; penalising local index
    # ``t % 37984`` on the *other three* dies would silently penalise three
    # unrelated tokens and produce plausible-looking wrong output rather than an
    # error.
    #
    # This stage never does that arithmetic in a kernel. The penalty operands are
    # built on the host as **full-vocabulary** ``[1, 1, 32, 151936]`` tensors --
    # indexed by global id, which is the only frame in which a penalty is
    # defined -- and handed to the device through
    # ``ttnn.ShardTensorToMesh(dim=-1)``, the *same* mapper and the same even
    # 4-way split the logits themselves were produced under by the
    # column-parallel LM head. Column ``t`` of the host tensor therefore lands on
    # exactly the die and exactly the local column that holds logit ``t``, by
    # construction rather than by a computed index. Every op below is
    # elementwise between two tensors with identical per-die shapes, so no op
    # ever needs to know a global id.
    #
    # The identity is *checked* rather than assumed:
    # ``probes/penalty_shard_boundary_probe.py`` penalises one token in die 0's
    # range and one in die 3's, and asserts both moved and that the same local
    # index on the other dies did not.
    #
    # ------------------------------------------------------------------------
    # The arithmetic
    # ------------------------------------------------------------------------
    #
    # vLLM's ``model_executor/layers/utils.py::apply_penalties`` is the contract,
    # and its order is load-bearing -- repetition first, on the raw logit:
    #
    #     repetition p (over prompt+output): x = x/p if x > 0 else x*p
    #     frequency  f (over output):        x -= f * count(t in output)
    #     presence   q (over output):        x -= q * (count(t in output) > 0)
    #
    # The repetition rule is sign-dependent, so it is *not* expressible as an
    # additive delta. It is spelled as a per-column multiplicative factor whose
    # two branches are both uploaded:
    #
    #     pos    = gtz(x)                       # 1.0 where x > 0, else 0.0
    #     factor = rep_neg + pos * rep_dif      # rep_neg = p, rep_dif = 1/p - p
    #     x      = x * factor
    #     x      = x - add_delta                # f*count + q*presence, host-summed
    #
    # For a column no row penalises, the host writes ``rep_neg = 1.0``,
    # ``rep_dif = 0.0``, ``add_delta = 0.0``: ``x * 1.0 - 0.0`` is **bit-exact**
    # in bf16, so an unpenalised token is not merely close to unchanged, it is
    # unchanged. That is what makes the cross-die non-perturbation claim a
    # property of the arithmetic and not of a tolerance.
    #
    # Per-row isolation is likewise structural: the operands are ``[1,1,32,V]``
    # and every op is elementwise, so row *i*'s columns are only ever combined
    # with row *i*'s logits. Padding slots get the neutral row and are untouched.
    #
    # Baking the per-row scalars (p, 1/p, f, q) into the full-width tensors on
    # the host, rather than broadcasting a ``[1,1,32,1]`` scalar column on
    # device, costs one more upload but removes every H-broadcast from the traced
    # graph -- and the host is rebuilding these rows anyway, because vLLM re-sends
    # the whole token history each step.
    #
    # ------------------------------------------------------------------------
    # Fast path
    # ------------------------------------------------------------------------
    #
    # ``_penalty_mode`` is a *graph* property, not a value: 0 means the ops below
    # are not in the captured trace at all, so an unpenalised request pays
    # nothing -- no op, no buffer, no upload. Bit 0 is the repetition stage and
    # bit 1 the additive stage, and they are independent, so a repetition-only
    # request never pays for the additive tensor. The generator releases and
    # re-captures the decode traces when the mode changes, exactly as it already
    # does when ``_sampling_stochastic`` flips between the argmax and split
    # strategies.

    #: Bitmask: 1 = repetition stage in the graph, 2 = frequency/presence stage.
    _penalty_mode = 0
    _penalty_rep_neg = None
    _penalty_add = None

    def penalty_buffer_shape(self) -> tuple[int, int]:
        """``(slots, vocab_size)`` the host-side penalty operands must have."""
        cfg = self.config
        return int(cfg.max_batch_size), int(cfg.vocab_size)

    def penalty_shard_geometry(self) -> tuple[int, int]:
        """``(num_devices, local_vocab)`` -- the split the operands must be staged in.

        The **same** decomposition ``load_device_buffers`` builds
        ``_dist_die_offset`` from, read off the same config rather than
        recomputed, so the staging path and the distributed argmax cannot drift
        apart.
        """
        cfg = self.config
        devices = cfg.mesh_device.get_num_devices()
        vocab = int(cfg.vocab_size)
        if vocab % devices:
            raise RuntimeError(f"penalties need an even column-parallel split; {vocab} % {devices} != 0")
        return devices, vocab // devices

    def allocate_penalty_buffers(self, mode: int) -> None:
        """Allocate/free the per-stage operands for ``mode``.

        Called by the generator **outside** any trace capture -- ``ttnn.from_torch``
        inside ``begin_trace_capture`` raises and leaves the capture open (stage-04
        ``work_log.md`` §6), which is the same reason ``load_device_buffers``
        builds ``_dist_die_offset`` eagerly.
        """
        mode = int(mode)
        if mode == self._penalty_mode:
            return
        cfg = self.config
        slots, vocab = self.penalty_buffer_shape()
        num_devices = cfg.mesh_device.get_num_devices()
        if mode and vocab % num_devices != 0:
            raise RuntimeError(f"penalties need an even column-parallel vocabulary split; {vocab} % {num_devices} != 0")

        def _alloc(fill: float):
            return ttnn.from_torch(
                torch.full((1, 1, slots, vocab), fill, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=cfg.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(cfg.mesh_device, dim=-1),
            )

        for want, names, fills in (
            (mode & 1, ("_penalty_rep_neg",), (1.0,)),
            (mode & 2, ("_penalty_add",), (0.0,)),
        ):
            for name, fill in zip(names, fills):
                current = getattr(self, name, None)
                if want and current is None:
                    setattr(self, name, _alloc(fill))
                elif not want and current is not None:
                    ttnn.deallocate(current, True)
                    setattr(self, name, None)
        self._penalty_mode = mode

    def penalty_device_buffers(self) -> dict:
        """The live operands, keyed by name; the generator uploads into these."""
        return {"rep_neg": self._penalty_rep_neg, "add": self._penalty_add}

    def _apply_penalties(self, logits):
        """``logits`` -> penalised logits, or ``logits`` itself when the mode is 0.

        Returns ``(tensor, is_new)``; the caller deallocates when ``is_new``.
        """
        mode = self._penalty_mode
        if not mode:
            return logits, False
        if int(logits.shape[-1]) != self.penalty_buffer_shape()[1] // self.config.mesh_device.get_num_devices():
            # Already gathered, or some shape this stage was not built for. The
            # penalty operands are per-die shards; refusing is the only safe
            # answer, because applying them at the wrong width would penalise
            # the wrong tokens.
            raise RuntimeError(
                f"penalty operands are per-die shards of width "
                f"{self.penalty_buffer_shape()[1] // self.config.mesh_device.get_num_devices()}, "
                f"got logits of width {int(logits.shape[-1])}"
            )
        out = logits
        if mode & 1:
            # ``rep_dif`` (= 1/p - p) is derived **on device** rather than
            # uploaded. It used to be a second full-width operand, and staging one
            # of those costs 2.049 ms of host time per decode step -- more than
            # every device op in this stage put together. ``ttnn.reciprocal`` of
            # the operand gives the same thing for free, because the operand is
            # ``p`` at penalised columns and exactly ``1.0`` everywhere else.
            #
            # This is only allowed to be here because ``reciprocal(1.0)`` is
            # **exactly** 1.0 on this device -- checked, not assumed
            # (``penalty_shard_boundary_probe.py``'s reference and
            # bit-identity legs both fail if it is not). That is what keeps the
            # unpenalised column at ``x * 1.0 - 0.0``, i.e. bit-exact, which is
            # the whole cross-die non-perturbation argument. At *penalised*
            # columns the LLK reciprocal differs from a host-computed ``1/p`` by
            # up to about one bf16 ulp (p=1.05: 0.95703 against 0.95313), which is
            # inside the accuracy the bf16 operand already has.
            inv = ttnn.reciprocal(self._penalty_rep_neg)
            rep_dif = ttnn.subtract(inv, self._penalty_rep_neg)
            ttnn.deallocate(inv)
            # gtz, not a where: ttnn.where on this path is the op the argmax
            # override already avoids, and gtz is a single unary.
            pos = ttnn.gtz(out)
            scaled = ttnn.multiply(pos, rep_dif)
            ttnn.deallocate(pos)
            ttnn.deallocate(rep_dif)
            factor = ttnn.add(scaled, self._penalty_rep_neg)
            ttnn.deallocate(scaled)
            out = ttnn.multiply(out, factor)
            ttnn.deallocate(factor)
        if mode & 2:
            penalised = ttnn.subtract(out, self._penalty_add)
            if out is not logits:
                ttnn.deallocate(out)
            out = penalised
        return out, True

    def decode_forward(self, logits, **kwargs):
        """Penalty stage, then ``Sampling1D``'s own routing -- unchanged.

        Overriding here rather than in each strategy means both the argmax path
        and the top-k/top-p split path get penalties from one place, applied
        **before** any selection, which is the only order that is correct.
        """
        penalised, is_new = self._apply_penalties(logits)
        try:
            return super().decode_forward(penalised, **kwargs)
        finally:
            if is_new:
                ttnn.deallocate(penalised)

    def load_device_buffers(self):
        """Base buffers, plus the per-die vocabulary offset the reduction adds.

        Built here rather than lazily in ``_sample_argmax`` because the first
        ``_sample_argmax`` may already be inside ``begin_trace_capture``, and
        ``ttnn.from_torch`` inside a capture raises and leaves the capture open.
        """
        already_loaded = self._device_buffers_loaded
        super().load_device_buffers()
        if already_loaded and getattr(self, "_dist_die_offset", None) is not None:
            return
        local_vocab = self._distributed_argmax_local_vocab()
        if local_vocab is None:
            self._dist_die_offset = None
            return
        cfg = self.config
        num_devices = cfg.mesh_device.get_num_devices()
        # ``_dist_active_rows`` rows, not ``cfg.max_batch_size``: the reduction
        # below runs over the live user rows only and the shapes must match
        # exactly, because an H-broadcast here would silently re-expand the
        # result back to 32 rows. See ``_sample_argmax``.
        rows = self._distributed_argmax_active_rows(cfg.max_batch_size)
        offsets = (
            (
                torch.arange(num_devices, dtype=torch.int64)
                .reshape(1, 1, 1, num_devices)
                .expand(1, 1, rows, num_devices)
                * local_vocab
            )
            .contiguous()
            .to(torch.int32)
        )
        # Sharded on the last dim: die d holds the single column ``d*local_vocab``.
        self._dist_die_offset = ttnn.from_torch(
            offsets,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=cfg.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(cfg.mesh_device, dim=-1),
        )
        self._dist_local_vocab = local_vocab

    def _sample_argmax(self, logits, tt_out_tok):
        """Distributed argmax: reduce per die, all-gather 4 candidates, reduce again.

        Honours ``Sampling1D._sample_argmax``'s contract exactly -- writes the
        caller's ``tt_out_tok`` in place and returns ``(tt_out_tok, None)``; the
        argmax path never emits logprobs. See the class docstring for why each
        step is spelled the way it is and for the measured 1.82x.
        """
        self.load_device_buffers()
        die_offset = getattr(self, "_dist_die_offset", None)
        if die_offset is None or int(logits.shape[-1]) != self._dist_local_vocab:
            # Not an even per-die shard (already gathered, padded vocab, 1x1
            # mesh, ...): the base path is the one that is still correct.
            return super()._sample_argmax(logits, tt_out_tok)

        # -- per-die reduction, over this die's own columns only ---------------
        # ROW_MAJOR because ttnn.argmax's TILE path is single-core (23 ms).
        rm = ttnn.untilize(logits, use_multicore=True)
        # **Reduce the live user rows, not the padding.** ``decode_terminal``
        # hands the sampler a logically-32-row tile because ``ttnn.sampling``
        # addresses 32 fixed slots, but at batch B only the first B rows carry a
        # user: the rest are the zero rows ``ttnn.pad(..., value=0.0)`` put on the
        # pre-head hidden, and ``lm_head`` has no bias, so their logits are exactly
        # zero. ``ttnn.argmax``'s multicore kernel does the comparison as a scalar
        # C++ ``>`` loop on the RISCV_1 *data-movement* core -- 32 x 37984 values
        # over 110 cores is ~11k compares each, and that, not the 32-round
        # semaphore barrier, is why the op costs 366 us and sits 75x off
        # bandwidth. Dropping the padding rows drops the work proportionally.
        #
        # Measured standalone at the shipped shape
        # (``doc/optimized_full_model/probes/argmax_outer_dim_probe.py``,
        # trace-captured, median of 60; the harness floor is ~58 us):
        #
        #   argmax over 32 rows                        371.1 us
        #   argmax over 32 rows, keepdim=False         309.3    (one barrier, not 32)
        #   ROW_MAJOR slice to 1 row + argmax           58.0    (i.e. at the floor)
        #   whole reduction, 32 rows                   631.6
        #   whole reduction, 1 row                     250.8    **2.52x**
        #
        # ``keepdim=False`` is a real but small effect and is *not* taken: it buys
        # 62 us on its own and nothing at all once the rows are sliced (251.0 vs
        # 250.8), while costing a ``[1,1,B] -> [1,1,B,1]`` reshape.
        #
        # The substitution is exact, not an approximation. The probe's
        # ``padding_rows_produce_token_zero`` leg checks on the device that a
        # zero logit row reduces to token **0** on the shipped 32-row path -- all
        # four dies tie at 0.0, so the masked ``min`` keeps global index 0 -- which
        # is precisely the value the ``ttnn.pad`` below writes back.
        slots = int(rm.shape[-2])
        active = self._distributed_argmax_active_rows(slots)
        if active < slots:
            live = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, active, self._dist_local_vocab])
            ttnn.deallocate(rm)
            rm = live
        local_idx = ttnn.argmax(rm, dim=-1, keepdim=True)  # uint32 RM [1,1,B,1]
        # ttnn.gather, NOT ttnn.max: 0.059 ms against 0.494 ms. This is the win.
        local_max = ttnn.to_layout(ttnn.gather(rm, dim=-1, index=local_idx), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        # INT32, not FLOAT32: fp32 elementwise rounds the index through bf16.
        local_idx_i32 = ttnn.to_layout(ttnn.typecast(local_idx, ttnn.int32), ttnn.TILE_LAYOUT)
        ttnn.deallocate(local_idx)
        global_idx = ttnn.add(local_idx_i32, die_offset)
        ttnn.deallocate(local_idx_i32)

        # -- gather 4 candidates, not the whole vocabulary ---------------------
        vals4 = self._argmax_all_gather(local_max)  # bf16  [1,1,B,4]
        idx4 = self._argmax_all_gather(global_idx)  # int32 [1,1,B,4]
        ttnn.deallocate(local_max)
        ttnn.deallocate(global_idx)

        # -- cross-die reduction ----------------------------------------------
        gmax = ttnn.max(vals4, dim=-1, keepdim=True)
        mask = ttnn.typecast(ttnn.eq(vals4, gmax), ttnn.int32)  # 0/1
        # NOT sum(mask*idx): on a tie that adds the tied indices together.
        # BIG + mask*(idx-BIG) sends losers to BIG and leaves every tied winner at
        # its own global index, so min() keeps the lowest -- the first-maximal one,
        # because die ranges ascend.
        sel = ttnn.add(ttnn.multiply(mask, ttnn.subtract(idx4, _DIST_ARGMAX_BIG)), _DIST_ARGMAX_BIG)
        token = ttnn.min(sel, dim=-1, keepdim=False)  # int32 TILE
        for scratch in (vals4, idx4, gmax, mask, sel):
            ttnn.deallocate(scratch)

        # -- match ttnn.argmax's output contract: UINT32 / ROW_MAJOR ------------
        token = ttnn.typecast(ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32)
        if active < slots:
            # Restore the 32-slot vector ``tt_out_tok`` is. 0 is not a convenient
            # filler, it is the token the shipped 32-row reduction *already*
            # produces for a padding row (see the comment above the slice), so the
            # buffer's contents are unchanged slot for slot. It also keeps every
            # slot a valid id: ``embed_decode`` runs ``ttnn.embedding`` over all 32
            # before slicing to ``batch``, and an out-of-vocabulary id there would
            # be an out-of-bounds table read.
            padded = ttnn.pad(token, [(0, 0), (0, 0), (0, slots - active)], value=0)
            ttnn.deallocate(token)
            token = padded
        if tt_out_tok is None:
            return token, None
        # Write **into the caller's buffer**. The traced decode loop feeds this
        # exact tensor back as the next token, so the object and its address must
        # survive; returning a new tensor breaks token feedback silently.
        ttnn.copy(ttnn.reshape(token, tt_out_tok.shape), tt_out_tok)
        ttnn.deallocate(token)
        return tt_out_tok, None


def _resolve_precision(precision) -> PrecisionConfig:
    """Accept a ``PrecisionConfig``, a dict, a path to JSON, or ``None``.

    ``None`` is ``DEFAULT_PRECISION``, so every existing caller keeps the
    shipped policy. The dict and path forms exist so a sweep runner -- and,
    later, the vLLM construction path -- can pass the *artifact*
    (``selected_precision_config.json``) rather than importing the dataclass,
    which is what makes the artifact something the model consumes rather than
    something written next to it.
    """
    if precision is None:
        return DEFAULT_PRECISION
    if isinstance(precision, PrecisionConfig):
        return precision
    if isinstance(precision, dict):
        return PrecisionConfig.from_dict(precision)
    if isinstance(precision, (str, Path)):
        return PrecisionConfig.read_json(precision)
    # A ``PrecisionConfig`` from a *duplicate copy* of ``tt.precision``, which is
    # a real hazard in this tree and not a hypothetical one: ``tt/generator.py``
    # imports ``tt.model`` by absolute path while tests and probes import it
    # relatively, and under pytest's ``--import-mode=importlib`` (this repo's
    # ``addopts``) with no ``models/__init__.py`` the two spellings produce two
    # distinct module objects and therefore two distinct classes. ``isinstance``
    # is then False for an object that is, by every meaning that matters, the
    # right one. Rebuild it through the serialised form rather than refusing it.
    if type(precision).__name__ == "PrecisionConfig" and hasattr(precision, "to_dict"):
        return PrecisionConfig.from_dict(precision.to_dict())
    raise TypeError(f"precision must be a PrecisionConfig, dict, path or None; got {type(precision).__name__}")


def _lm_head_compute_config(device, precision: PrecisionConfig = DEFAULT_PRECISION):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=precision.lm_head_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


class ShardedCheckpoint:
    """Read named tensors out of a sharded safetensors checkpoint on demand.

    The full checkpoint is 30.5B parameters, ~61 GB in bf16. Materialising it as
    one ``state_dict`` to build a model that uploads it layer by layer would
    need that whole 61 GB of host RAM at once; this reads only the tensors asked
    for, from only the shards that hold them, and holds nothing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        index_path = self.path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"checkpoint index is missing: {index_path}")
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(name)
        with safe_open(self.path / shard, framework="pt") as f:
            return f.get_tensor(name)

    def layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Every ``model.layers.<i>.*`` tensor, keyed layer-relative."""
        prefix = f"model.layers.{layer_idx}."
        by_shard: dict[str, list[str]] = {}
        for name, shard in self.weight_map.items():
            if name.startswith(prefix):
                by_shard.setdefault(shard, []).append(name)
        if not by_shard:
            raise KeyError(f"no tensors for layer {layer_idx}")
        out: dict[str, torch.Tensor] = {}
        for shard, names in by_shard.items():
            with safe_open(self.path / shard, framework="pt") as f:
                for name in names:
                    out[name[len(prefix) :]] = f.get_tensor(name)
        return out


def _validate_mesh(mesh_device) -> None:
    shape = tuple(int(v) for v in mesh_device.shape)
    if shape != MESH_SHAPE:
        raise ValueError(f"Qwen3CoderModel requires mesh {MESH_SHAPE}, got {shape}")
    if mesh_device.get_num_devices() != NUM_DEVICES:
        raise ValueError(f"Qwen3CoderModel requires exactly {NUM_DEVICES} devices")


def _rope_parameters(hf_config) -> dict:
    """``rope_parameters`` on current transformers, ``rope_theta`` on older ones.

    ``Qwen3MoeConfig`` no longer exposes a top-level ``rope_theta`` attribute --
    reading it raises ``AttributeError`` rather than returning ``None`` -- so
    the dict is the only spelling that works on both.
    """
    params = getattr(hf_config, "rope_parameters", None)
    if params:
        return dict(params)
    return {"rope_theta": hf_config.rope_theta, "rope_type": "default"}


def _rope_type(hf_config) -> str:
    return str(_rope_parameters(hf_config).get("rope_type", "default"))


def _rope_theta(hf_config) -> float:
    return float(_rope_parameters(hf_config)["rope_theta"])


def _rope_tables(hf_config, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The HF ``(cos, sin)`` tables for positions ``0..capacity-1``.

    Built here rather than through ``Qwen3MoeRotaryEmbedding`` so that no
    transformers model object is constructed at load time; the formula is the
    default rope (``rope_scaling`` is null in this checkpoint, which
    ``from_checkpoint`` asserts).
    """
    head_dim = int(getattr(hf_config, "head_dim", HEAD_DIM))
    theta = _rope_theta(hf_config)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = torch.outer(torch.arange(capacity, dtype=torch.float64), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().float(), angles.sin().float()


class Qwen3CoderModel:
    """The 48-layer causal LM over the stage-04 multichip decoder layer."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        checkpoint: ShardedCheckpoint,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_cache_len: int = MAX_CONTEXT,
        num_layers: int = NUM_LAYERS,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        rope_cache_len: int = DEFAULT_ROPE_CACHE_LEN,
        precision: "PrecisionConfig | dict | str | Path | None" = None,
    ) -> None:
        _validate_mesh(mesh_device)
        if not 1 <= int(max_batch_size) <= 32:
            # nlp_create_qkv_heads_decode_device_operation.cpp:51 asserts
            # num_users <= 32; a TTNN op limit, unchanged by TP.
            raise ValueError(f"max_batch_size must be in [1,32], got {max_batch_size}")
        if not 1 <= int(num_layers) <= int(hf_config.num_hidden_layers):
            raise ValueError(f"num_layers must be in [1,{hf_config.num_hidden_layers}]")
        if not 1 <= int(max_cache_len) <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_cache_len must be in [1,{hf_config.max_position_embeddings}]")
        if int(hf_config.hidden_size) != HIDDEN_SIZE or int(hf_config.vocab_size) != VOCAB_SIZE:
            raise ValueError("HF config does not match the Qwen3-Coder-30B-A3B full-model contract")
        if bool(hf_config.tie_word_embeddings):
            raise ValueError("this checkpoint has an untied lm_head; tied weights would be a different contract")
        if _rope_type(hf_config) != "default":
            raise ValueError(f"rope_type {_rope_type(hf_config)!r} is not supported by this port's rotary tables")

        # The precision policy, resolved once and then read by every builder and
        # every forward below. ``None`` -> ``DEFAULT_PRECISION``, the shipped
        # stage-06 policy, so a caller that says nothing gets exactly the model
        # it got before this parameter existed. A ``dict`` or a path is accepted
        # too, so a sweep runner can hand over a ``selected_precision_config.json``
        # without importing the dataclass.
        self.precision = _resolve_precision(precision)

        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.max_batch_size = int(max_batch_size)
        self.max_cache_len = int(max_cache_len)
        self.num_layers = int(num_layers)
        self.page_block_size = int(page_block_size)
        self.hidden_size = HIDDEN_SIZE
        self.vocab_size = VOCAB_SIZE
        self.head_dim = int(getattr(hf_config, "head_dim", HEAD_DIM))
        self.rms_norm_eps = float(hf_config.rms_norm_eps)
        # Exact: 151936 = 4 * 37984 and 37984 = 32 * 1187.
        assert self.vocab_size % (32 * NUM_DEVICES) == 0, self.vocab_size
        self.local_vocab_size = self.vocab_size // NUM_DEVICES

        #: Skip the expert work of decode rows that hold no live request. On by
        #: default and a no-op at ``max_batch_size == 1``; see
        #: ``_decode_active_mask``. ``QWEN3_DECODE_ACTIVE_ROW_GATING=0`` restores
        #: the stage-08 graph exactly, which is what
        #: ``doc/optimized_vllm/probes/inactive_row_gating_probe.py`` A/Bs
        #: against for the token-equality leg.
        self.active_row_gating = os.getenv("QWEN3_DECODE_ACTIVE_ROW_GATING", "1") not in ("0", "", "false", "no")

        #: Width of the decode graph currently being built or captured.
        #: Equal to ``max_batch_size`` everywhere except inside
        #: ``decode_width_scope``, which the generator opens to capture a
        #: **narrower** decode graph than the configured slot count -- see
        #: ``doc/batch_scaling/README.md``. Every decode-path use of the row
        #: count reads this, not ``max_batch_size``: the embedding slice, the
        #: rotary gather and shard, and the active-row mask. Prefill and the
        #: sampler are untouched -- ``decode_terminal`` pads to the 32 fixed
        #: ``SAMPLING_SLOTS`` regardless, so the sampler never sees the width.
        self.decode_width = self.max_batch_size

        self.ctx: MeshContext = mesh_context(mesh_device)
        self.config = MeshDecoderConfig.from_hf(hf_config)
        self.global_config: DecoderLayerConfig = self.config.global_config

        self.embed_tokens = self._build_embedding(checkpoint)
        self.layers: list[MultichipWeights] = self._build_layers(checkpoint)
        self.final_norm, self.final_norm_rm = self._build_final_norm(checkpoint)
        self.lm_head = self._build_lm_head(checkpoint)

        self.sparsity = build_local_sparsity(mesh_device, self.config.local_moe)
        self.lm_head_compute_config = _lm_head_compute_config(mesh_device, self.precision)
        self.norm_compute_config = _norm_compute_config(mesh_device, self.precision)
        # Set by ``local_logits`` / the sampler-input path once a forward has
        # run, so ``runtime_fallback_audit`` can report the dtypes the terminal
        # path *produced* rather than the ones the config asked for.
        self._observed_logits_dtype = None
        self._observed_sampling_dtype = None

        self.rope_cache_len = 0
        self.cos_table = None
        self.sin_table = None
        self.ensure_rope_capacity(min(int(rope_cache_len), self.max_cache_len))

        # ``_WatcherCleanSampling1D`` rather than ``Sampling1D``: same module,
        # same strategies, the force-argmax gather spelled without the pinned
        # ``num_workers_per_link`` that trips the watcher on this mesh. See the
        # class docstring above and ``doc/full_model/watcher_ab.log``.
        self.sampler = _WatcherCleanSampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.vocab_size,
                valid_vocab_size=self.vocab_size,
                mesh_device=mesh_device,
                tt_ccl=self.ctx.ccl,
                max_batch_size=32,
                max_top_k=32,
                num_gather_links=1,
                sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                allow_force_argmax=True,
                num_argmax_gather_links=1,
                ag_topology=TOPOLOGY,
                # **False, and that is a measurement.** ``Sampling1D``'s comment
                # calls the power-of-two pad a "big device-perf win for
                # non-power-of-2 vocab on the multi-device path". For a per-die
                # shard of 37984 it is the opposite: the pad is to 65536, a 1.73x
                # blow-up of the tensor ``ttnn.topk`` then scans, and
                # ``probes/sampler_probe.py`` measures the whole split path at
                # **11.006 ms padded against 6.151 ms unpadded**, 1.79x, at the
                # shipped logits shape with the sampled token unchanged.
                pad_to_power_of_2=False,
            )
        )
        # ``max_batch_size=32`` above is the *slot* count ``ttnn.sampling`` and
        # ``decode_terminal`` address; this is how many of those slots carry a
        # user. The distributed argmax reduces only those rows -- the rest are the
        # zero-logit padding ``decode_terminal`` adds -- which is worth 2.52x on
        # the whole sampler at batch 1. Set before ``load_device_buffers`` because
        # the per-die offset constant is built to this row count.
        self.sampler._dist_active_rows = self.max_batch_size
        self.sampler.load_device_buffers()
        self.kv_cache: list[KVCache] | None = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        mesh_device,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_cache_len: int = MAX_CONTEXT,
        num_layers: int = NUM_LAYERS,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        rope_cache_len: int = DEFAULT_ROPE_CACHE_LEN,
        precision: "PrecisionConfig | dict | str | Path | None" = None,
    ) -> "Qwen3CoderModel":
        checkpoint_path = Path(checkpoint_path)
        hf_config = AutoConfig.from_pretrained(checkpoint_path)
        checkpoint = ShardedCheckpoint(checkpoint_path)
        model = cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            checkpoint=checkpoint,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            num_layers=num_layers,
            page_block_size=page_block_size,
            rope_cache_len=rope_cache_len,
            precision=precision,
        )
        gc.collect()
        return model

    def _build_embedding(self, checkpoint: ShardedCheckpoint) -> ttnn.Tensor:
        host = checkpoint.get("model.embed_tokens.weight").float()
        tensor = ttnn.from_torch(
            host,
            dtype=self.precision.embedding_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        del host
        gc.collect()
        return tensor

    def _build_layers(self, checkpoint: ShardedCheckpoint) -> list[MultichipWeights]:
        from .weight_mapping import convert_layer_weights

        layers = []
        for layer_idx in range(self.num_layers):
            sd = checkpoint.layer(layer_idx)
            torch_weights = convert_layer_weights(sd, self.hf_config)
            del sd
            layers.append(
                upload_multichip_weights(torch_weights, self.mesh_device, self.config, precision=self.precision)
            )
            del torch_weights
            gc.collect()
        return layers

    def _build_final_norm(self, checkpoint: ShardedCheckpoint):
        host = checkpoint.get("model.norm.weight").float().reshape(-1)
        tiled = ttnn.from_torch(
            host.reshape(1, 1, 1, -1),
            dtype=self.precision.norm_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # The layout the sharded rms_norm program factory reads; see
        # ``multichip_decoder.upload_multichip_weights.norm_row_major``.
        row_major = ttnn.from_torch(
            host.reshape(1, 1, host.numel() // 32, 32).contiguous(),
            dtype=self.precision.norm_weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tiled, row_major

    def _build_lm_head(self, checkpoint: ShardedCheckpoint) -> ttnn.Tensor:
        host = checkpoint.get("lm_head.weight").float().transpose(-2, -1).contiguous()
        assert tuple(host.shape) == (self.hidden_size, self.vocab_size), tuple(host.shape)
        tensor = ttnn.from_torch(
            host.reshape(1, 1, self.hidden_size, self.vocab_size),
            dtype=self.precision.lm_head_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )
        del host
        gc.collect()
        return tensor

    # -- rotary ---------------------------------------------------------------

    def ensure_rope_capacity(self, required_len: int) -> bool:
        """Grow the device cos/sin tables to cover ``required_len`` positions."""
        required_len = int(required_len)
        if required_len <= self.rope_cache_len:
            return False
        if required_len > self.max_cache_len:
            raise ValueError(f"rotary capacity {required_len} exceeds context {self.max_cache_len}")
        capacity = min(self.max_cache_len, max(32, 1 << (required_len - 1).bit_length()))
        cos, sin = _rope_tables(self.hf_config, capacity)
        new = []
        for host in (cos, sin):
            new.append(
                ttnn.from_torch(
                    host.reshape(1, 1, capacity, self.head_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        old = (self.cos_table, self.sin_table)
        self.cos_table, self.sin_table = new
        for tensor in old:
            if tensor is not None:
                ttnn.deallocate(tensor, True)
        self.rope_cache_len = capacity
        return True

    def rope_decode_tables(self, rotary_position: ttnn.Tensor):
        """Per-user ``(cos, sin)`` for one decode step, gathered **on device**.

        ``rotary_position`` is a ``[1, batch]`` uint32 device tensor. The gather
        is ``ttnn.embedding`` against the replicated cos/sin tables, so the
        position never leaves the device and the whole thing is capturable; the
        trace advances ``rotary_position`` itself with ``ttnn.plus_one``.

        Returns the height-sharded ``[1, batch, 1, head_dim]`` pair that
        ``rotary_embedding_hf(is_decode_mode=True)`` requires -- one core per
        user, the same ``_head_shard`` layout ``nlp_create_qkv_heads_decode``
        emits for Q and K.
        """
        batch = self.decode_width
        shard = _head_shard(32, self.head_dim, batch)
        out = []
        for table in (self.cos_table, self.sin_table):
            # [1, batch] -> [1, batch, head_dim] -> [1, 1, batch, head_dim]
            # -> [1, batch, 1, head_dim], the layout rotary_embedding_hf's decode
            # factory reads. Same sequence as ``RotarySetup1D.decode_forward``.
            gathered = ttnn.unsqueeze_to_4D(ttnn.embedding(rotary_position, table, layout=ttnn.TILE_LAYOUT))
            transposed = ttnn.transpose(gathered, 1, 2)
            if int(transposed.shape[1]) != batch:
                trimmed = ttnn.slice(transposed, [0, 0, 0, 0], [1, batch, 1, self.head_dim])
                ttnn.deallocate(transposed, True)
                transposed = trimmed
            out.append(ttnn.interleaved_to_sharded(transposed, shard))
            ttnn.deallocate(transposed, True)
        return out[0], out[1]

    def _rope_decode(self, tensor: ttnn.Tensor, cos_sharded, sin_sharded, _token_index):
        """The ``rope=`` seam handed to ``decoder_layer_decode_multichip``.

        ``_token_index`` is accepted and ignored: the position lives in the
        cos/sin pair, which is what makes this spelling replayable where the
        layer's default one is not.
        """
        shard = _head_shard(32, self.head_dim, self.decode_width)
        staged = ttnn.to_memory_config(tensor, shard)
        rotated = ttnn.experimental.rotary_embedding_hf(staged, cos_sharded, sin_sharded, is_decode_mode=True)
        ttnn.deallocate(staged, True)
        out = ttnn.to_memory_config(rotated, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rotated, True)
        return out

    # -- KV cache -------------------------------------------------------------

    def allocate_kv_cache(
        self,
        *,
        max_cache_len: int | None = None,
        num_blocks: int | None = None,
        page_table: ttnn.Tensor | None = None,
    ) -> list[KVCache]:
        """One paged ``KVCache`` per layer, 1 local KV head per die.

        512 B per token per layer per die -- a quarter of the single-die 2048 --
        which is what makes the advertised 262144 context fit; see
        ``config/context_contract.json``.
        """
        cache_len = self.max_cache_len if max_cache_len is None else int(max_cache_len)
        blocks_per_seq = math.ceil(cache_len / self.page_block_size)
        total_blocks = self.max_batch_size * blocks_per_seq if num_blocks is None else int(num_blocks)
        local = self.config.local_attention
        caches = []
        for _ in range(self.num_layers):
            k, v = (
                ttnn.from_torch(
                    torch.zeros(total_blocks, local.num_key_value_heads, self.page_block_size, local.head_dim),
                    dtype=self.precision.kv_cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for _ in range(2)
            )
            caches.append(KVCache(k=k, v=v, page_table=page_table, block_size=self.page_block_size))
        return caches

    def ensure_internal_kv_cache(self, page_table: ttnn.Tensor | None = None) -> list[KVCache]:
        if self.kv_cache is None:
            self.kv_cache = self.allocate_kv_cache(page_table=page_table)
        return self.kv_cache

    @staticmethod
    def bind_page_table(kv_cache: Sequence[KVCache], page_table: ttnn.Tensor | None) -> list[KVCache]:
        """Point every layer's cache at ``page_table`` in place.

        The page table is a *persistent device tensor* owned by the caller (the
        generator, or vLLM later). Rebinding mutates the ``KVCache`` records
        rather than reallocating, so the tensor identity a captured trace
        recorded is preserved and an unchanged page table costs nothing.
        """
        for cache in kv_cache:
            cache.page_table = page_table
        return list(kv_cache)

    def reset_kv_cache(self, kv_cache: Sequence[KVCache] | None = None) -> None:
        selected = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        for cache in selected:
            ttnn.fill(cache.k, 0.0, memory_config=cache.k.memory_config(), output_tensor=cache.k)
            ttnn.fill(cache.v, 0.0, memory_config=cache.v.memory_config(), output_tensor=cache.v)

    # -- forward: prefill -----------------------------------------------------

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, S]`` uint32 -> replicated ``[1, 1, S, 2048]``, no collective."""
        hidden = ttnn.embedding(
            tokens,
            self.embed_tokens,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.precision.activation_dtype,
        )
        hidden = ttnn.unsqueeze_to_4D(hidden)
        return ttnn.reshape(hidden, (1, 1, int(hidden.shape[-2]), self.hidden_size))

    def prefill_hidden(
        self,
        tokens: ttnn.Tensor,
        *,
        kv_cache: Sequence[KVCache] | None = None,
        user_id: int = 0,
        start_pos: int = 0,
        chunk_page_table=None,
        fill_page_table=None,
    ) -> ttnn.Tensor:
        """Run the whole stack over one user's prompt. ``S`` is arbitrary.

        Nothing here constrains ``S``: the collectives scatter on dim 3 (hidden,
        2048, fixed), ``attention_prefill`` slices RoPE's tile padding back, and
        ``moe_prefill_optimized`` pads to its chunk internally and slices back.
        """
        caches = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        if len(caches) != self.num_layers:
            raise ValueError(f"kv_cache has {len(caches)} layers, expected {self.num_layers}")
        hidden = self.embed_prefill(tokens)
        seq_len = int(hidden.shape[-2])
        # A split prefill's suffix occupies absolute positions
        # [start_pos, start_pos + seq_len), so the tables must cover the END of
        # the range, not its length.
        self.ensure_rope_capacity(start_pos + seq_len)
        # Exactly ``seq_len`` rows, including non-tile-aligned lengths -- the
        # same shape the single-layer prefill gates pass at S = 33/100/257.
        # RoPE is applied at ABSOLUTE positions: the suffix of a split prefill
        # must rotate at [start_pos, start_pos + seq_len), not from 0, or its keys
        # disagree with the ones already in the cache. Identical to the shipped
        # slice when start_pos == 0.
        cos = ttnn.slice(self.cos_table, [0, 0, start_pos, 0], [1, 1, start_pos + seq_len, self.head_dim])
        sin = ttnn.slice(self.sin_table, [0, 0, start_pos, 0], [1, 1, start_pos + seq_len, self.head_dim])
        for layer_idx in range(self.num_layers):
            hidden = decoder_layer_prefill_multichip(
                hidden,
                self.layers[layer_idx],
                self.config,
                self.ctx,
                cos,
                sin,
                self.sparsity,
                kv_cache=caches[layer_idx],
                user_id=user_id,
                precision=self.precision,
                start_pos=start_pos,
                chunk_page_table=chunk_page_table,
                fill_page_table=fill_page_table,
            )
        ttnn.deallocate(cos, True)
        ttnn.deallocate(sin, True)
        return hidden

    def prefill_norm(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            hidden,
            weight=self.final_norm,
            epsilon=self.rms_norm_eps,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def select_prefill_rows(self, hidden: ttnn.Tensor, rows: Sequence[int]) -> ttnn.Tensor:
        """Keep only ``rows`` of a ``[1, 1, S, H]`` prefill result."""
        seq_len = int(hidden.shape[-2])
        pieces = []
        for row in rows:
            if not 0 <= int(row) < seq_len:
                raise ValueError(f"prefill row {row} is outside [0,{seq_len})")
            if seq_len == 1:
                # At a **one-token prompt** the requested slice covers the whole
                # tensor, and ``ttnn.slice`` then hands back a view of its input
                # rather than a copy -- as a *different* Python object, so an
                # ``is`` guard does not catch it. The caller deallocates
                # ``hidden`` immediately afterwards, leaving the retained row
                # pointing at freed DRAM; that does not raise, it **segfaults**
                # in whatever reads it next (the final norm here). Copy instead.
                # `probes/prompt_len_1_repro.py` is the four-line reproduction.
                pieces.append(ttnn.clone(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                continue
            pieces.append(
                ttnn.slice(
                    hidden,
                    [0, 0, int(row), 0],
                    [1, 1, int(row) + 1, self.hidden_size],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        if len(pieces) == 1:
            return pieces[0]
        out = ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for piece in pieces:
            ttnn.deallocate(piece, True)
        return out

    def local_logits(self, normed: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, rows, 2048]`` -> this die's ``[1, 1, rows, 37984]`` logits."""
        out = ttnn.linear(
            normed,
            self.lm_head,
            compute_kernel_config=self.lm_head_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.precision.logits_dtype,
        )
        # Observed, not asserted: the dtype the produced tensor actually carries.
        # ``runtime_fallback_audit`` reports it so ``logits_dtype`` is verified
        # off a real tensor rather than echoed back out of the config -- see the
        # ``*_observed`` entries there.
        self._observed_logits_dtype = out.dtype
        return out

    def gather_logits_to_torch(self, local_logits: ttnn.Tensor, *, valid_rows: int | None = None) -> torch.Tensor:
        """Host-side full-vocabulary logits. **Not** on the token-out path.

        Used by ``return_all_logits`` prefill checks and the host-sampling
        compatibility mode only; the measured decode path never calls this.
        """
        gathered = ttnn.all_gather(
            local_logits,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPOLOGY,
        )
        host = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
        ttnn.deallocate(gathered, True)
        if valid_rows is not None:
            host = host[..., : int(valid_rows), :]
        return host[..., : self.vocab_size]

    # -- forward: decode ------------------------------------------------------

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, 1, 32]`` uint32 -> replicated ``[1, 1, batch, 2048]``."""
        hidden = ttnn.embedding(
            tokens,
            self.embed_tokens,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.precision.activation_dtype,
        )
        hidden = ttnn.unsqueeze_to_4D(hidden)
        flat = ttnn.reshape(hidden, (1, 1, int(hidden.shape[-2]), self.hidden_size))
        if int(flat.shape[-2]) == self.decode_width:
            return flat
        sliced = ttnn.slice(
            flat, [0, 0, 0, 0], [1, 1, self.decode_width, self.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(flat, True)
        return sliced

    @contextlib.contextmanager
    def decode_width_scope(self, width: int):
        """Build the decode graph ``width`` rows wide instead of ``max_batch_size``.

        The narrow graph is legal because **nothing in decode binds a user to a
        slot index except the three per-row inputs** -- ``current_pos``, the
        rotary position and the page-table row. The KV cache is fully paged:
        ``paged_update_cache`` and ``paged_scaled_dot_product_attention_decode``
        both reach the cache only through ``page_table_tensor`` rows and
        ``cur_pos_tensor`` entries, and neither takes a ``batch_offset``. So a
        request can be decoded in *any* row provided its page-table row, its
        position and its token travel with it; no cache page moves.

        What the width actually changes is the amount of work: the expert
        ``ttnn.sparse_matmul`` visits ``width x local_experts`` slots per layer
        with ``nnz=None``, the expert tail is dense over the same product, the
        router runs one ``topk`` per row, and paged SDPA reads one window per
        row. That cost is paid per row *configured*, not per row live -- see
        ``doc/optimized_vllm/README.md``'s control curve -- so narrowing the
        graph is the only lever that removes it.

        Sampling is deliberately outside the scope: ``decode_terminal`` pads to
        the 32 fixed ``SAMPLING_SLOTS`` whatever the width is, so ``tt_out_tok``
        keeps its ``[1,1,1,32]`` shape and the sampler's per-slot parameters
        keep their meaning. Row *i* of a narrow graph is sampling slot *i*.
        """
        width = int(width)
        if not 1 <= width <= self.max_batch_size:
            raise ValueError(f"decode width must be in [1,{self.max_batch_size}], got {width}")
        previous = self.decode_width
        self.decode_width = width
        try:
            yield width
        finally:
            self.decode_width = previous

    def _decode_active_mask(self, current_pos: ttnn.Tensor):
        """``[1, 1, batch, 1]`` of 1.0 for live slots and 0.0 for inactive ones.

        A serving decode batch is always the configured ``max_num_seqs`` rows --
        vLLM pads it so the trace shape is constant -- with inactive slots
        carrying ``current_pos = -1``. Those rows still embed a token, still run
        attention, and, critically, still route to a full top-8 of experts, so
        their ``(row, expert)`` pairs land in ``sparse_matmul``'s sparsity and
        cost real expert weight reads and real math. Multiplying the routing
        vector by this mask takes them out of the sparsity instead
        (``decoder_layer_decode_multichip``).

        **Why it is derived on device rather than passed in.** ``current_pos`` is
        already a persistent trace input, and the traced graph advances it with
        ``ttnn.plus_one(..., skip_negative_entries=True)`` -- an inactive row
        stays at ``-1`` through any number of replays, and a slot only becomes
        active through a host reinstall of ``current_pos``. So a mask computed
        from it inside the same graph is correct by construction on every replay,
        with no extra trace input to refresh and no way for it to go stale. A
        host-supplied mask would be one more thing that has to be right.

        Returns ``None`` at ``max_batch_size == 1``, where there is no inactive
        row to skip: the graph is then byte-for-byte the one stage 08 shipped and
        the single-user headline cannot be perturbed by this change.
        """
        if self.decode_width <= 1 or not self.active_row_gating:
            return None
        row = ttnn.to_layout(ttnn.reshape(current_pos, (1, 1, 1, self.decode_width)), ttnn.TILE_LAYOUT)
        # bf16 cannot represent every position exactly at 262144, but it
        # represents every position's *sign* exactly, and ``gez`` only reads the
        # sign. -1 -> 0.0, everything >= 0 -> 1.0.
        as_float = ttnn.typecast(row, ttnn.bfloat16)
        ttnn.deallocate(row, True)
        live_row = ttnn.gez(as_float)
        ttnn.deallocate(as_float, True)
        mask = ttnn.transpose(live_row, -2, -1)
        ttnn.deallocate(live_row, True)
        return mask

    def decode_hidden(
        self,
        tokens: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rotary_position: ttnn.Tensor,
        kv_cache: Sequence[KVCache] | None = None,
    ) -> ttnn.Tensor:
        caches = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        if len(caches) != self.num_layers:
            raise ValueError(f"kv_cache has {len(caches)} layers, expected {self.num_layers}")
        hidden = self.embed_decode(tokens)
        cos, sin = self.rope_decode_tables(rotary_position)
        # Computed once per decode step and shared by all 48 layers.
        active_mask = self._decode_active_mask(current_pos)
        for layer_idx in range(self.num_layers):
            hidden = decoder_layer_decode_multichip(
                hidden,
                self.layers[layer_idx],
                self.config,
                self.ctx,
                cos,
                sin,
                caches[layer_idx],
                current_pos,
                0,  # token_index: unused by the rope seam below, see _rope_decode
                rope=self._rope_decode,
                precision=self.precision,
                active_mask=active_mask,
            )
        ttnn.deallocate(cos, True)
        ttnn.deallocate(sin, True)
        if active_mask is not None:
            ttnn.deallocate(active_mask, True)
        return hidden

    def decode_terminal(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Final norm + column-parallel ``lm_head``, sampler-ready local logits.

        The norm is the layer's own width-sharded decode kernel, and the shard
        it emits is exactly the width-sharded L1 config the projections read, so
        crossing into the head costs one sharded-to-interleaved.
        """
        normed_sharded = decode_residual_norm(hidden, self.final_norm_rm, self.rms_norm_eps, self.precision)
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed_sharded, True)
        # ``ttnn.sampling`` addresses 32 fixed user slots, and it compares the
        # *logical* shapes of its values and indices, so the logits handed to it
        # must be logically 32 rows and not ``batch`` rows padded to a tile.
        # The rows are already physically there -- ``batch <= 32`` and decode is
        # one 32-row tile -- so this only rewrites the logical shape.
        rows = int(normed.shape[-2])
        if rows < SAMPLING_SLOTS:
            padded = ttnn.pad(
                normed,
                [(0, 0), (0, 0), (0, SAMPLING_SLOTS - rows), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(normed, True)
            normed = padded
        logits = self.local_logits(normed)
        ttnn.deallocate(normed, True)
        if self.precision.sampling_dtype != logits.dtype:
            # Equal on the shipped path, so this is dead code at the default and
            # the traced decode graph is byte-for-byte what stage 06 captured.
            cast = ttnn.typecast(logits, self.precision.sampling_dtype)
            ttnn.deallocate(logits, True)
            logits = cast
        self._observed_sampling_dtype = logits.dtype
        return logits

    def decode_forward_from_ttnn_inputs(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        rotary_position: ttnn.Tensor,
        kv_cache: Sequence[KVCache] | None = None,
        advance_position: bool = True,
    ) -> ttnn.Tensor:
        """Token in -> sampler-ready local logits out, entirely on device.

        With ``advance_position`` the two position tensors are incremented
        **inside** this graph, so a captured trace steps its own positions on
        replay and the host never refreshes them per token.
        """
        hidden = self.decode_hidden(
            tokens,
            current_pos=current_pos,
            rotary_position=rotary_position,
            kv_cache=kv_cache,
        )
        logits = self.decode_terminal(hidden)
        if advance_position:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rotary_position)
        return logits

    # -- sampling -------------------------------------------------------------

    def sample_split(self, logits, *, k, p, temp, seeds=None, tt_out_tok=None):
        """Canonical split sampling: local top-32 -> all-gather -> ``ttnn.sampling``.

        ``k=1, p=0, temp=1`` is **semantically greedy**: the global argmax is by
        construction inside some die's local top-32, and the all-gather makes
        all four dies' candidates visible before the top-1 is taken.
        """
        return self.sampler.decode_forward(
            logits, k=k, p=p, temp=temp, seeds=seeds, tt_out_tok=tt_out_tok, enable_log_probs=False
        )[0]

    def sample_greedy_argmax(self, logits, *, tt_out_tok=None):
        """``Sampling1D``'s force-argmax path, on this model's distributed override.

        Still the common module, still on device, still traced, still writes the
        sampled token straight into ``tt_out_tok`` -- it is a different strategy
        inside the same implementation, not a custom sampler. The strategy body
        is ``_WatcherCleanSampling1D._sample_argmax``: reduce on each die, then
        all-gather the four survivors instead of all-gathering the vocabulary.

        **This is what greedy uses**, because at this vocabulary it is 6.6x
        faster than the top-k/top-p split path (0.928 ms against 6.155 ms in the
        48-layer model, both rows of
        ``doc/optimized_full_model/probes/perf_full_model.csv``, which is the
        **shipped** measurement) and produces the same token -- both rows sample
        token 16 on that run. Whole-model ``token_out`` on that same run is
        **19.693 ms, 50.78 t/s/u**.

        Stage 05 shipped the same choice at 1.125 ms against 6.155 ms, and two
        changes inside this override moved the greedy row since, each with its
        own token-out delta at a like-for-like context:

        * the **distributed reduction** above -- 22.079 ms to 21.461 ms
          (45.29 -> 46.60 t/s/u), both at ``context`` 4096,
          ``../full_model/probes/perf_full_model.json`` against
          ``doc/optimized_full_model/probes/perf_full_model_part1_preadoption.json``;
        * the **live-row slice** (reduce ``max_batch_size`` rows, not 32) --
          20.146 ms to 19.693 ms at ``context`` 8192,
          ``doc/optimized_full_model/probes/perf_full_model_p128_after.json``
          against
          ``doc/optimized_full_model/probes/perf_full_model_p128_argmaxrows.json``.

        The remaining step between them is the paged SDPA program config in
        ``tt/multichip_decoder.py`` and is not this sampler's. The moment any
        slot asks for ``top_k > 1`` or ``top_p > 0`` the generator switches back
        to ``sample_split``.
        """
        return self.sampler.decode_forward(logits, tt_out_tok=tt_out_tok, enable_log_probs=False)[0]

    # -- audit ----------------------------------------------------------------

    def runtime_fallback_audit(self, batch: int | None = None) -> dict:
        """The layer audit, plus the boundaries this wrapper owns."""
        batch = self.max_batch_size if batch is None else int(batch)
        audit = fallback_audit(self.layers[0], self.config, batch, self.precision)
        audit.update(
            {
                "num_layers": self.num_layers,
                "embedding": "replicated_bf16_no_collective",
                "residual_contract": "replicated [1,1,B,2048] bf16 TILE DRAM, no inter-layer collective",
                "final_norm": "replicated, width-sharded decode kernel",
                "lm_head_parallelism": "column_parallel_over_vocab",
                "lm_head_local_vocab": self.local_vocab_size,
                "lm_head_weight_dtype": str(self.lm_head.dtype),
                "embedding_weight_dtype": str(self.embed_tokens.dtype),
                "precision": self.precision.to_dict(),
                "vocab_padding": 0,
                "decode_rope": "rotary_embedding_hf(is_decode_mode=True), device position gather",
                "decode_rope_position_source": "device tensor advanced by ttnn.plus_one inside the trace",
                "sampling_greedy": (
                    "Sampling1D force-argmax, distributed: per-die untilize/argmax/gather -> "
                    "all-gather 4 candidates -> masked-min, traced, writes tt_out_tok"
                ),
                "sampling_topk_topp": "Sampling1D split (local topk -> all-gather 32 candidates -> ttnn.sampling)",
                "sampling_pad_to_power_of_2": False,
                "host_logit_readback_on_token_out_path": False,
                "host_argmax_on_token_out_path": False,
                # Read off the allocated cache when one exists, so a swept
                # ``kv_cache_dtype`` is *observed* rather than asserted. This
                # was a hard-coded "bfloat16" until stage 07's sweep, which
                # would have silently mislabelled every non-default KV row.
                # Falls back to the configured value before allocation.
                # Emitted as the PLAIN name ("bfloat16"), not ``str(dtype)``
                # ("DataType.BFLOAT16"), because that is the existing contract:
                # doc/optimized_full_model's committed runtime_fallback_audit.json
                # and check_published_figures.py both pin the plain spelling, and
                # they are stage evidence that must keep passing. The sibling
                # ``device_*`` fields use str(dtype) and are left alone.
                "kv_cache_dtype": dtype_to_name(
                    self.kv_cache[0].k.dtype if self.kv_cache else self.precision.kv_cache_dtype
                ),
                "kv_cache_dtype_source": "device_readback" if self.kv_cache else "config_not_yet_allocated",
                # -- the four fields stage 07's selection proof could not check --
                #
                # Before the stage-07 review these were the only swept fields
                # with no audit entry at all, so ``R03_lmhead_lofi``,
                # ``R21_norm_hifi2`` and ``R22_logits_sampling_bfp8`` produced
                # ``device_audit`` blocks byte-identical to the baseline's and
                # "this lever does nothing" was indistinguishable from "this
                # lever is not wired up". For ``norm_fidelity`` it was the
                # second: ``decode_residual_norm`` built its compute config from
                # the module default and never saw ``self.precision``.
                #
                # The two fidelities are read off the ``compute_kernel_config``
                # objects the ops are actually handed (built here, passed at the
                # call site), so they verify the config -> compute-config
                # threading. The two dtypes are read off the **produced
                # tensors** and are ``None`` until a forward has run.
                "lm_head_math_fidelity": str(self.lm_head_compute_config.math_fidelity),
                "norm_math_fidelity": str(self.norm_compute_config.math_fidelity),
                "logits_dtype_observed": (
                    None if self._observed_logits_dtype is None else dtype_to_name(self._observed_logits_dtype)
                ),
                "sampling_dtype_observed": (
                    None if self._observed_sampling_dtype is None else dtype_to_name(self._observed_sampling_dtype)
                ),
                "terminal_dtype_source": ("device_readback" if self._observed_logits_dtype else "no_forward_yet"),
                "kv_cache_paged": True,
                "page_block_size": self.page_block_size,
                "collective_topology": str(TOPOLOGY),
                "prefill_num_links": self.ctx.num_links,
                "decode_num_links": self.ctx.decode_num_links,
            }
        )
        return audit

    def teardown(self) -> None:
        if self.kv_cache is not None:
            for cache in self.kv_cache:
                ttnn.deallocate(cache.k, True)
                ttnn.deallocate(cache.v, True)
            self.kv_cache = None


__all__ = [
    "DEFAULT_MAX_BATCH_SIZE",
    "DEFAULT_PAGE_BLOCK_SIZE",
    "DEFAULT_ROPE_CACHE_LEN",
    "DEFAULT_TRACE_REGION_SIZE",
    "HF_MODEL_ID",
    "HF_REVISION",
    "MAX_CONTEXT",
    "NUM_LAYERS",
    "Qwen3CoderModel",
    "ShardedCheckpoint",
]
