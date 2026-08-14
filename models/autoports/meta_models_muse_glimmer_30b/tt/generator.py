# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serving-shaped generator for the ``meta-models/Muse-Glimmer-30B`` full model.

Two API levels, as the readiness contract requires
(``models/common/readiness_check/contract.py``):

* **low level** -- :meth:`MuseGlimmerGenerator.prefill_forward` and
  :meth:`MuseGlimmerGenerator.decode_forward`.  The caller owns the KV cache, the
  page table, per-user prompt lengths and per-user decode positions, and threads
  them through every call.  Mixed-length prompts, fixed request slots and
  inactive rows (position ``-1``) are all part of that contract, because a later
  vLLM adapter has to drive exactly this path without re-implementing the model;
* **high level** -- :meth:`MuseGlimmerGenerator.generate`, a deterministic loop
  over the low-level pieces that owns cache and page-table state itself.

Token-out decode
----------------

The measured decode path is **canonical split sampling**: two traces, no host
work between them.

1. the model decode trace runs token-ids -> embedding -> 52 layers -> terminal
   norm -> LM head -> softcap, and returns **vocab-sharded** logits.  There is no
   logits gather and no readback inside it;
2. the sampling trace is
   :class:`~models.common.sampling.generator.SamplingGenerator`'s own, captured
   over that exact logits tensor, with ``tt_out_tok`` pointing at the persistent
   decode **token input**.  ``ttnn.sampling`` writes the sampled token in place,
   so the token feedback never leaves the device;
3. the decode position and the RoPE index are advanced by ``ttnn.plus_one``
   *inside* the model trace, after every read of them
   (:meth:`MuseGlimmerModel.ttnn_decode_forward`);
4. the page table is copied only when it changes.

So the steady-state free-running step is: replay, replay, read 32 uint32.  No
token staging, no position staging, no page-table copy, no mask rebuild, no cache
reset, no host argmax and no full-vocab readback.  ``self.counters`` records all
of those so the claim is falsifiable rather than asserted.

Greedy decode goes through the same top-k op path with semantically greedy
parameters (``k=1``, ``p=0``, ``temp=1``) rather than through force-argmax.  That is
a contract decision: force-argmax needs a full-vocab ``all_gather`` (12.9 MB per step
across this mesh), and that gather goes through
``self.tt_ccl.get_and_cycle_ag_semaphore_handles(...)`` while this generator
constructs ``SamplingGenerator`` with ``tt_ccl=None`` on purpose -- a ``TT_CCL``
would put 36 more global semaphores in the main L1 pool, and the decode step has
7,296 B of headroom there.  With ``tt_ccl=None`` the path does not error, it hangs.
(An earlier version of this docstring rejected force-argmax because ``ttnn.argmax``
returns rank-3 and the token buffer is rank-4.  That reason was wrong -- upstream
passes ``output_tensor=tt_out_tok`` straight into ``ttnn.argmax`` -- and is
withdrawn; the ``tt_ccl`` requirement above is the real blocker.)

The sampler's *shape* is measured rather than inherited, and the measurement that
matters is the **program factory**, not the padding.  ``ttnn.topk`` reaches its
multi-core factory only when the reduced width is a power of two **below 65535**
and at least 8192; a 50688-column shard is neither a power of two nor, once padded
to 65536, under the bound, so both settings of ``pad_logits_to_power_of_2`` run the
same single-core kernel, whose cost scales with the width.  Shipped instead:
``topk_split_to_power_of_2``, which pads to 65536 **and splits into 2 x 32768** --
every call multi-core, and the whole sampling trace drops by more than an order of
magnitude (`doc/full_model/sampler_ab.json`).  Note that this
therefore turns the pad **on** (``TTSampling`` forces it when the split is
requested); the port-level ``pad_logits_to_power_of_2`` knob stays False and is
only meaningful with the split off.  ``max_top_k=8`` is measurable again with the
split (slightly slower than 32; see `sampler_ab.json`) -- without it, 8 candidates pad to a
32-column tile and drop ``ttnn.all_gather`` onto its composite path, where it stops
making progress.  See the sampler A/B in ``doc/full_model/README.md``.

Host sampling is available as an explicit compatibility mode
(``host_sampling=True``) for tests that require it.  It gathers logits and takes
the argmax on the host, and it is never the measured path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import torch
from loguru import logger

import ttnn

# Absolute imports, not relative: the readiness runners load this file with
# ``importlib.util.spec_from_file_location`` under a synthetic module name and no
# package, so ``from .model import ...`` raises *"attempted relative import with no
# known parent package"* before the generator is ever constructed.
from models.autoports.meta_models_muse_glimmer_30b.tt import model as model_mod
from models.autoports.meta_models_muse_glimmer_30b.tt import optimized_decoder as dec_mod
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import TILE_SIZE
from models.autoports.meta_models_muse_glimmer_30b.tt.model import (
    DECODE_ROWS,
    GENERATOR_CACHE,
    HF_MODEL_ID,
    LM_HEAD_CORES,
    LM_HEAD_DTYPE,
    LM_HEAD_FIDELITY,
    LM_HEAD_FP32_ACC,
    LM_HEAD_IN0_BLOCK_W,
    LM_HEAD_MATMUL,
    LM_HEAD_OUTPUT_DTYPE,
    MuseGlimmerModel,
    dram_capacity_bytes,
    weights_snapshot_dir,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_MESH_SHAPE,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.common.readiness_check.contract import Generator
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params

#: Trace region a full-model decode trace needs: the model decode trace plus the
#: sampler's, each over ~2400 ops for the 52-layer stack.  Generous rather than
#: tuned -- it is DRAM reserved once at mesh open, and weights, caches and tables
#: together use 7.18 GB of the 31.5 GiB this part has.
DEFAULT_TRACE_REGION_SIZE = 400_000_000

#: Greedy sampling parameters.  ``temperature=0`` is
#: ``format_sampling_params``' canonical greedy encoding: it rewrites to
#: ``(temp=1, k=1, p=0)``, which is argmax expressed through the top-k op.
GREEDY = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)


@dataclass
class GeneratorConfig:
    """Port-local knobs.  Defaults are the shipped configuration."""

    model_id: str = HF_MODEL_ID
    #: Cache slots, i.e. how many users the paged pool is sized for.  One by
    #: default because the primary target is batch-1 single-user latency at the
    #: full advertised context: 52 layers x 2048 blocks is 1.854 GB/device, and 32
    #: of those would be 59 GB against 31.5 GB of DRAM.  Larger batches are built
    #: at a proportionally smaller ``max_seq_len``; see doc/full_model/README.md.
    max_batch_size: int = 1
    max_seq_len: int | None = None  # None -> the HF-advertised context
    page_block_size: int = 64
    prefill_chunk_size: int | None = None
    layer_indices: Sequence[int] | None = None
    lm_head_dtype: Any = LM_HEAD_DTYPE
    lm_head_matmul: str = LM_HEAD_MATMUL
    lm_head_cores: int = LM_HEAD_CORES
    lm_head_in0_block_w: int = LM_HEAD_IN0_BLOCK_W
    lm_head_fidelity: Any = LM_HEAD_FIDELITY
    lm_head_fp32_acc: bool = LM_HEAD_FP32_ACC
    lm_head_output_dtype: Any = LM_HEAD_OUTPUT_DTYPE
    allow_force_argmax: bool = False
    #: Sampler shape knobs, measured rather than assumed; see the sampler A/B.
    max_top_k: int = 32
    pad_logits_to_power_of_2: bool = False
    #: Split the padded vocab shard so ``ttnn.topk`` reaches its multi-core factory.
    #: Default **on**. What it is worth is a measurement, not a constant, so the number
    #: lives in `doc/full_model/sampler_ab.json` rather than here -- a figure in a comment
    #: goes stale silently and forces a source edit every time it is re-measured, which
    #: then puts this file's mtime after the artifacts it describes.
    topk_split_to_power_of_2: bool = True
    #: Trace the prefill, keyed by padded prompt length.  **Off** by default, and the
    #: default is the interesting part.
    #:
    #: Batch-1 prefill on this mesh is *host-issue* bound, not device bound: 4122 ttnn
    #: dispatches at 9-60 us of host issue each, measured at 54.9 ms of issue against
    #: 55.1 ms to drain, and no collective implementation or persistent-buffer variant
    #: moves the per-call cost (12 arms, ``doc/optimized_full_model/ccl_host_probe.json``).
    #: Tracing is the only mechanism that removes host issue, and it works: on the real
    #: 52-layer build a warmed replay is **44.96 ms against 59.80 ms eager (1.33x)** and
    #: **bit-identical** to it, coexisting with the decode and sampling traces and
    #: retaining 3.3 MB of DRAM per device at 128 rows
    #: (``doc/optimized_full_model/prefill_trace_probe.json``).
    #:
    #: It is off by default because the graph is shaped by the *padded* row count, so one
    #: trace serves one 32-row bucket, and capture costs ~98 ms against a ~15 ms
    #: per-replay saving -- payback after ~7 prefills of the same bucket. That is a win
    #: for a caller whose prompt lengths repeat or are bucketed and a loss for one whose
    #: lengths do not, and the generator cannot tell which it is. A serving stage that
    #: buckets prompt lengths should turn it on and raise
    #: :attr:`prefill_trace_max_entries` to its bucket count; DRAM per entry scales with
    #: the padded row count.
    prefill_trace: bool = False
    #: How many padded-length buckets may hold a prefill trace at once.  Lengths beyond
    #: this fall back to the eager path rather than evicting -- releasing a trace and its
    #: retained intermediates mid-request is a bigger hazard than a slower prefill.
    prefill_trace_max_entries: int = 1
    decoder_kwargs: dict = field(default_factory=dict)


class _SamplingArgs:
    """The attribute bag ``TTSampling``/``TTPenalties`` read.

    Two widths, and they do different jobs -- conflating them is a real defect this
    stage shipped and the stage review caught:

    * ``padded_vocab_size`` (202752) is what the **index arithmetic** uses. The
      sampler turns a per-device top-k index into a global token id by adding
      ``device * padded_vocab_size / tp``, so an unpadded value here would shift
      every token id on devices 1-3 by ``d * 176`` and produce fluent garbage
      rather than an error;
    * ``vocab_size`` (202048) is what the **invalid-vocab mask** uses.
      ``TTSampling._create_invalid_vocab_mask`` builds a mask for the columns
      between the two, and ``models/common/sampling/vocab_padding.py`` returns
      ``None`` -- i.e. **no mask at all** -- when the two are equal. Setting this
      to the padded width therefore silently disabled the mask and made the 704
      padded ids drawable: the LM head zero-fills those columns, so each carries
      logit exactly ``20 * tanh(0) = 0.0``, and at positions where only a handful
      of real logits exceed 0 they enter the local top-k ahead of real tokens.
      Greedy survives it whenever the argmax is positive, but a top-k/top-p
      request can draw one, and an id >= 202048 is outside the tokenizer and
      outside the 202049-row embedding table it would be fed back into.
    """

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        vocab_size: int,
        padded_vocab_size: int,
        max_batch_size: int,
        allow_force_argmax: bool,
        max_top_k: int = 32,
        pad_logits_to_power_of_2: bool = False,
        topk_split_to_power_of_2: bool = True,
    ):
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.cluster_shape = list(mesh_device.shape)
        self.max_batch_size = max_batch_size
        #: Candidates kept per device shard **per topk call** before the gather.
        #: ``ttnn.sampling`` requires the *gathered* candidate width to give a
        #: power-of-two tile count.  With the multi-core split each device contributes
        #: ``pieces * max_top_k``, so on this 4-device mesh 32 gathers to 256 (8 tiles)
        #: and 8 gathers to 64 (2 tiles); with the split off they gather to 128 and 32.
        #: Which one to ship is a measurement -- see the sampler A/B in
        #: ``doc/full_model/README.md``.
        self.max_top_k = max_top_k
        self.sampling_dp = 1
        self.sub_core_grids = None
        # 202752 / 4 = 50688 is not a power of two.  Padding it to 65536 does *not*
        # buy the bitonic fast path -- 65536 is over the multi-core factory's uint16
        # index bound -- so on its own this knob only widens a single-core kernel.  It
        # is kept because ``topk_split_to_power_of_2`` needs the pad, and because the
        # measurement that established both is worth being able to repeat.
        self.pad_logits_to_power_of_2 = pad_logits_to_power_of_2
        #: The one that matters.  Padding 50688 to 65536 keeps ``ttnn.topk`` on its
        #: **single-core** factory (65536 exceeds the multi-core uint16 index bound),
        #: which is why the pad alone measured as a pure loss.  Splitting the padded
        #: shard into 2 x 32768 clears every multi-core condition and collapses the op's
        #: cost; the measured widths are in ``doc/full_model/topk_geometry_probe.json``.
        self.topk_split_to_power_of_2 = topk_split_to_power_of_2
        self.use_topk_logprobs = False
        self.model_config = {
            # Two ethernet links between adjacent Blackhole dies, which is what the
            # decoder's collectives already use.
            "GALAXY_NUM_LINKS": 2,
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": allow_force_argmax,
                "num_links": 2,
                "chunks_per_sync": 10,
                # ``Topology.Ring`` in the sampler's own all-gather is gated to
                # 8-device meshes upstream; the gather it would apply to is the
                # 32-candidate tuple, not a payload, so Linear costs nothing here.
                "topology": ttnn.Topology.Linear,
            },
        }


class MuseGlimmerGenerator(Generator):
    """Readiness/vLLM-shaped generator over :class:`MuseGlimmerModel`."""

    def __init__(
        self,
        model: MuseGlimmerModel,
        *,
        tokenizer: Any = None,
        gen_config: GeneratorConfig | None = None,
    ) -> None:
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.gen_config = gen_config or GeneratorConfig()
        config = model.config

        self.sampling = SamplingGenerator(
            args=_SamplingArgs(
                mesh_device=self.mesh_device,
                vocab_size=config.vocab_size,
                padded_vocab_size=config.padded_vocab_size,
                max_batch_size=DECODE_ROWS,
                allow_force_argmax=self.gen_config.allow_force_argmax,
                max_top_k=self.gen_config.max_top_k,
                pad_logits_to_power_of_2=self.gen_config.pad_logits_to_power_of_2,
                topk_split_to_power_of_2=self.gen_config.topk_split_to_power_of_2,
            ),
            mesh_device=self.mesh_device,
            # The top-k sampling path takes no semaphores of its own -- its two
            # fixed-shape all-gathers go through ``ttnn.all_gather``, which owns
            # them.  A ``TT_CCL`` would put 36 global semaphores in the *main* L1
            # pool, and the decode step has 7,296 B of headroom there; twelve of
            # them already broke the decoder's sharded norm.
            tt_ccl=None,
        )
        self._sampling_params: SamplingParams | None = None
        self._sampling_captured = False

        # One decode trace, always advancing position and RoPE index on device.
        #
        # One rather than two (a device-advancing graph and a host-stepped one)
        # because the in-trace ``plus_one`` runs *after* every read of those
        # tensors, so a caller that restages them from the host simply overwrites
        # the increment -- correct in both modes.  Two graphs would also mean two
        # sampling traces, and ``SamplingGenerator`` validates its trace by tensor
        # identity: its slot is keyed on (penalties, logprobs, force_argmax), not on
        # which logits tensor it was captured over, so a second decode trace's
        # logits raises *"The provided logits tensor does not match the tensor used
        # during trace capture"*.
        self._trace_id: int | None = None
        self._trace_logits: ttnn.Tensor | None = None
        #: ``padded_len -> {"id", "tokens", "page_table", "logits", "page_rows"}`` for the
        #: opt-in prefill trace; see :attr:`GeneratorConfig.prefill_trace`.
        self._prefill_traces: dict[int, dict] = {}
        #: KV-cache buffer addresses the prefill traces were captured over; see
        #: :meth:`_kv_cache_signature`.
        self._prefill_trace_cache_sig: tuple = ()
        #: The same, for the **decode** trace.  ``ttnn_decode_forward`` calls
        #: ``paged_update_cache(layer.k_cache, layer.v_cache, ...)``, so the captured
        #: decode graph bakes those buffer addresses exactly as a prefill trace does, and
        #: replaying it after a rebind to *different* buffers would read and write the old
        #: ones -- silently wrong tokens rather than an error.  Round 4 of the stage review
        #: found that: the invalidation covered only the opt-in prefill trace, while the
        #: decode trace runs on every token of the shipped default.
        #: See :meth:`_invalidate_traces_if_cache_moved`.
        self._decode_trace_cache_sig: tuple = ()
        #: How many times a cache move has forced a release, for the log line and for the
        #: stage's evidence; see :meth:`_invalidate_traces_if_cache_moved`.
        self._prefill_trace_releases = 0
        self._decode_trace_releases = 0
        #: Traces whose ``ttnn.release_trace`` *raised*.  Round 8 of the stage review found
        #: that round 7's retain-in-place policy fixed the leak by reintroducing round 4's
        #: bug: the handle stayed in ``_trace_id`` / ``_prefill_traces``, so the very next
        #: ``decode_forward`` (or ``_prefill_traced``) found it and replayed it -- against the
        #: cache the caller had just rebound away from.  Silently wrong tokens, which is worse
        #: than the leak.
        #:
        #: So a failed release *fails closed*: the handle and every tensor the trace may still
        #: read move here, out of the lookup paths, and the id/logits slots are cleared so the
        #: next call recaptures.  Nothing is deallocated -- a possibly-live trace still holds
        #: those addresses -- and ``note_trace_released()`` is **not** called, so
        #: ``live_traces_over_kv_cache`` stays raised and ``deallocate()`` keeps warning.
        #: :meth:`_retry_orphaned_traces` retries them at every release and at teardown.
        #: Entries are ``{"what": str, "id": int, "tensors": [ttnn.Tensor, ...]}``.
        self._orphaned_traces: list[dict] = []
        self._device_inputs: dict[str, ttnn.Tensor] = {}
        self._prev_page_table: torch.Tensor | None = None
        self._needs_reseed = True
        self._eos_ids = self._resolve_eos(config.eos_token_id)
        self._staged_tokens: list[int] = [0] * DECODE_ROWS
        self._staged_positions: torch.Tensor = torch.zeros(DECODE_ROWS, dtype=torch.int64)
        self.counters = model.counters
        self.reset_counters()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_eos(eos: Any) -> tuple[int, ...]:
        if eos is None:
            return ()
        if isinstance(eos, (list, tuple)):
            return tuple(int(e) for e in eos)
        return (int(eos),)

    def reset_counters(self) -> None:
        self.model.reset_counters()
        self.counters = self.model.counters

    def _allocate_device_inputs(self) -> None:
        """The four persistent decode trace inputs, allocated once.

        Every replay reads exactly these tensors, and every refresh writes into
        them with ``copy_host_to_device_tensor``; nothing in the decode loop
        allocates.  ``tokens`` doubles as the sampler's ``tt_out_tok``.
        """
        if self._device_inputs:
            return
        model = self.model
        config = model.config
        tokens_host = model.tokens_to_device([0], device=False)
        current_host, rope_host = model.positions_to_device(torch.zeros(1), device=False)
        page_host = model.page_table_to_device(model.normalize_page_table(None), device=False)
        # ``tokens`` is already [1, 1, 1, 32]: ``ttnn.sampling``'s preallocated
        # output must be rank 4, and ``ttnn.embedding`` collapses the leading unit
        # dims, so one buffer is both the sampler's output and the next decode's
        # input.
        self._device_inputs = {
            "tokens": ttnn.to_device(tokens_host, self.mesh_device),
            "current_pos": ttnn.to_device(current_host, self.mesh_device),
            "rope_pos_ids": ttnn.to_device(rope_host, self.mesh_device),
            "page_table": ttnn.to_device(page_host, self.mesh_device),
        }
        logger.info(
            "MuseGlimmerGenerator: persistent decode inputs allocated "
            f"(batch={config.max_batch_size}, blocks={config.blocks_per_seq})"
        )

    # ------------------------------------------------------- input staging

    def _stage(
        self,
        *,
        tokens: Sequence[int] | None = None,
        positions: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
    ) -> None:
        """Refresh the named persistent trace inputs from the host.

        Each branch bumps its own counter, which is what makes the
        "steady-state decode does no host staging" claim measurable rather than
        rhetorical.
        """
        model = self.model
        if tokens is not None:
            host = model.tokens_to_device(tokens, device=False)
            ttnn.copy_host_to_device_tensor(host, self._device_inputs["tokens"])
            self.counters["token_refreshes"] += 1
        if positions is not None:
            current_host, rope_host = model.positions_to_device(positions, device=False)
            ttnn.copy_host_to_device_tensor(current_host, self._device_inputs["current_pos"])
            ttnn.copy_host_to_device_tensor(rope_host, self._device_inputs["rope_pos_ids"])
            self.counters["position_refreshes"] += 1
        if page_table is not None:
            rows = model.normalize_page_table(page_table)
            if self._prev_page_table is not None and torch.equal(self._prev_page_table, rows):
                return
            host = model.page_table_to_device(rows, device=False)
            ttnn.copy_host_to_device_tensor(host, self._device_inputs["page_table"])
            self._prev_page_table = rows.clone()
            self.counters["page_table_refreshes"] += 1

    # -------------------------------------------------------------- sampling

    def _apply_sampling_params(self, sampling_params: SamplingParams | None) -> None:
        params = format_sampling_params(sampling_params or GREEDY, DECODE_ROWS)
        self._sampling_params = params
        self.sampling.reset_sampling_params(params)
        self.sampling.seed_manager.reset_seed(params.seed, list(range(DECODE_ROWS)))
        self.sampling.seed_manager.get_new_values()

    def _sample_eager(self, logits: ttnn.Tensor, *, into_tokens: bool) -> torch.Tensor:
        """One untraced sampling call, for prefill's first token.

        Untraced because it runs once per request over a *different* logits
        tensor than the decode trace's; the sampling trace is validated by tensor
        identity, so reusing it here would be a contract violation rather than an
        optimization.  Writing the result straight into the persistent decode
        token input still keeps the prefill -> decode hand-off on device.
        """
        out = self.sampling.sample(
            logits,
            enable_trace=False,
            tt_out_tok=self._device_inputs["tokens"] if into_tokens else None,
        )
        host = self._read_tokens(out)
        if not into_tokens:
            # The sampler allocated its own output; nothing else holds it.
            tensor = out[0] if isinstance(out, tuple) else out
            ttnn.deallocate(tensor)
        return host

    def _read_tokens(self, sampled: Any) -> torch.Tensor:
        tensor = sampled[0] if isinstance(sampled, tuple) else sampled
        host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1)
        self.counters["readbacks"] += 1
        return host[:DECODE_ROWS].to(torch.int64)

    # --------------------------------------------------------------- prefill

    def _prefill_user(
        self,
        token_ids: Sequence[int],
        *,
        user_id: int,
        page_table: torch.Tensor,
        return_all_logits: bool = False,
    ):
        """Embed + 52 layers + terminal path for one user's prompt.

        ``token_ids`` is the *logical* prompt, any length up to the supported
        context.  Nothing here rounds it: the layer pads to a tile internally,
        masks the padded tail and slices its output back, so a prompt length that
        is not a multiple of the tile, the page block or the prefill chunk is an
        ordinary input.
        """
        model = self.model
        config = model.config
        prompt_len = len(token_ids)
        if prompt_len < 1:
            raise ValueError("prefill needs at least one token")
        if prompt_len > config.max_seq_len:
            raise ValueError(f"prompt of {prompt_len} tokens exceeds the supported context {config.max_seq_len}")

        # The generator owns prompt padding: the ids are padded to a tile boundary
        # with the zero-embedding pad id, so the layer stack sees an aligned prompt
        # (its own internal pad is a no-op) and every padded row is exactly zero
        # rather than uninitialised DRAM.  The junk-free K/V those rows write past
        # ``prompt_len`` is never read: decode starts at ``cur_pos = prompt_len``.
        if (
            self.gen_config.prefill_trace
            and not return_all_logits
            and user_id == 0
            and prompt_len <= config.prefill_chunk_size
        ):
            traced = self._prefill_traced(token_ids, page_table=page_table)
            if traced is not None:
                return traced

        tt_tokens, padded_len = model.prefill_tokens_to_device(token_ids)
        tt_page_table = model.page_table_to_device(page_table)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=user_id)
        if return_all_logits:
            rows = model.prefill_all_logits(hidden, prompt_len=prompt_len)
            ttnn.deallocate(hidden)
            ttnn.deallocate(tt_page_table)
            return rows
        logits = model.prefill_logits(hidden, last_token_index=prompt_len - 1)
        ttnn.deallocate(hidden)
        ttnn.deallocate(tt_page_table)
        return logits, model.row_within_tile(prompt_len - 1)

    # --------------------------------------------------- the opt-in prefill trace

    def _prefill_traced(self, token_ids: Sequence[int], *, page_table: torch.Tensor):
        """``(logits, row_in_tile)`` from a replayed prefill trace, or ``None``.

        Returns ``None`` -- so the caller takes the eager path -- when this padded
        length has no trace and the cache is full.

        The graph bakes in the padded row count, ``user_id=0``, ``start_pos=0`` and the
        last-token tile-row slice.  That slice is a property of the *bucket* rather than
        the prompt: for a prompt of length ``L`` in ``(R-32, R]`` the tile row holding
        ``L-1`` starts at ``R-32`` for every ``L`` in the bucket, so one trace serves all
        32 of them and only ``row_within_tile`` differs, which is host arithmetic:
        ``row_within_tile(L-1) = (L-1) % 32`` and the tile the trace returns starts at
        ``R-32``, so ``(L-1) % 32 == (L-1) - (R-32)`` for every ``L`` in the bucket.  The
        ids past ``L`` are the zero-embedding pad id, exactly as on the eager path, and
        the junk-free K/V they write past the logical length is never read because decode
        starts at ``cur_pos = L``.

        The returned logits are a **clone**, not the trace's persistent output: callers
        deallocate what ``_prefill_user`` hands them, and handing over a buffer the next
        replay writes into would be a use-after-free waiting for a second request.
        """
        model = self.model
        length = len(token_ids)
        padded_len = ((length + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        rows = model.normalize_page_table(page_table)
        entry = self._prefill_traces.get(padded_len)
        if entry is None:
            if len(self._prefill_traces) >= self.gen_config.prefill_trace_max_entries:
                return None
            entry = self._capture_prefill_trace(token_ids, padded_len=padded_len, page_rows=rows)

        # These two counters are the *prefill*'s, and with this path on they are no
        # longer zero-per-decode-token: one token refresh and one trace replay per
        # request.  The steady-state decode counter assertions
        # (``test_steady_state_decode_does_no_per_token_host_work``, the fallback audit)
        # are stated for the default arm, where this path is off; nothing in the decode
        # loop changes either way.
        host_tokens, _ = model.prefill_tokens_to_device(token_ids, device=False)
        ttnn.copy_host_to_device_tensor(host_tokens, entry["tokens"])
        self.counters["token_refreshes"] += 1
        if not torch.equal(entry["page_rows"], rows):
            ttnn.copy_host_to_device_tensor(model.page_table_to_device(rows, device=False), entry["page_table"])
            entry["page_rows"] = rows.clone()
            self.counters["page_table_refreshes"] += 1
        ttnn.execute_trace(self.mesh_device, entry["id"], cq_id=0, blocking=True)
        self.counters["trace_replays"] += 1
        logits = ttnn.clone(entry["logits"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits, model.row_within_tile(length - 1)

    def _capture_prefill_trace(self, token_ids: Sequence[int], *, padded_len: int, page_rows: torch.Tensor) -> dict:
        """Allocate this bucket's persistent inputs, warm every program, then capture."""
        model = self.model
        tokens, _ = model.prefill_tokens_to_device(token_ids)
        tt_page_table = model.page_table_to_device(page_rows)
        last = padded_len - 1

        def forward():
            hidden = model.embed_prefill(tokens)
            out = model.prefill_forward(hidden, page_table=tt_page_table, user_id=0)
            logits = model.prefill_logits(out, last_token_index=last)
            ttnn.deallocate(out)
            return logits

        # Warm compile: trace capture cannot compile programs, and this path's
        # ``ttnn.slice`` offsets and program configs are baked into the program hash.
        ttnn.deallocate(forward())
        ttnn.synchronize_device(self.mesh_device)
        self.counters["synchronizations"] += 1
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits = forward()
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.counters["synchronizations"] += 1
        entry = {
            "id": trace_id,
            "tokens": tokens,
            "page_table": tt_page_table,
            "logits": logits,
            "page_rows": page_rows.clone(),
        }
        self._prefill_traces[padded_len] = entry
        self._prefill_trace_cache_sig = self._kv_cache_signature()
        self.model.note_trace_captured()
        logger.info(
            f"MuseGlimmerGenerator: captured a prefill trace for padded_len={padded_len} "
            f"({len(self._prefill_traces)}/{self.gen_config.prefill_trace_max_entries} buckets)"
        )
        return entry

    def _kv_cache_signature(self) -> tuple:
        """Identity of the *currently bound* KV cache, for prefill-trace invalidation.

        A prefill trace bakes in the device addresses of the cache buffers it writes, so
        it survives a rebind to the **same** buffers and must not survive a rebind to
        different ones.  ``kv_cache is not None`` cannot tell those apart, and getting
        that wrong is expensive in the direction that matters: a serving caller threading
        the same external handles on every request would recapture per request (98 ms
        capture plus a 45 ms replay against a 60 ms eager prefill), i.e. the flag would
        make the path it is advertised for ~83 ms/request *slower*.  Addresses rather
        than ``id()`` because a freed Python wrapper's id can be reused.
        """
        sig = []
        for layer in self.model.layers:
            for cache in (layer.k_cache, layer.v_cache):
                try:
                    sig.append(int(cache.buffer_address()))
                except Exception:  # noqa: BLE001 -- no address exposed: fall back to identity
                    sig.append(id(cache))
        return tuple(sig)

    def _invalidate_traces_if_cache_moved(self) -> None:
        """Release **every** trace whose graph baked the old cache's buffer addresses.

        Both the opt-in prefill trace and the always-on decode trace write the KV cache
        through ``paged_fill_cache`` / ``paged_update_cache``, so both bake
        ``layer.k_cache`` / ``layer.v_cache`` addresses at capture time.  Rebinding to
        *different* buffers and replaying either one reads and writes the buffers the
        caller no longer owns: wrong tokens, no error, and -- before round 4 of the stage
        review caught it -- a log line about releasing the *prefill* traces that read as if
        the rebind had been handled.  The decode trace is the one that matters most here,
        because it runs on every token of the shipped default while the prefill trace is
        opt-in and off.

        Each trace is compared against its own recorded signature, so a caller that binds
        one cache, decodes, and then binds a second is served correctly by recapture
        against whichever buffers are bound at the time.

        The comparison is the whole fix.  ``kv_cache is not None`` -- what this used to
        test -- releases on *every* call, so a serving caller threading the same external
        handles per request would recapture per request and pay ~83 ms/request more than
        the eager path the flag is supposed to beat.  Comparing addresses releases only on
        a genuine rebind, which is a once-per-adapter event.

        An earlier revision also *retired* tracing permanently after the first release, as
        a mitigation for the fabric ERISC assert in the stage README's limitation 6.  It
        was never a mitigation for it.  That assert fires at **process teardown**, and
        ``teardown()`` releases the prefill traces whether or not tracing was retired, so
        retirement removed no exposure at all; what it did remove was the caller's flag —
        one cache rebind and prefill fell back to the eager path for the life of the
        generator, silently.  (The sequence retirement was aimed at, release-then-build-
        another-model, is separately clean under watcher: see
        ``prefill_trace_release_probe.py --arm rebuild`` and the two opt-in cases run
        together in one process.)  So: release on a genuine move, and let the next
        eligible prefill recapture.  That costs one 98 ms capture at the rebind and
        nothing after it.
        """
        signature = self._kv_cache_signature()
        stale_prefill = bool(self._prefill_traces) and signature != self._prefill_trace_cache_sig
        stale_decode = self._trace_id is not None and signature != self._decode_trace_cache_sig
        if not (stale_prefill or stale_decode):
            return
        released = []
        if stale_prefill:
            self._prefill_trace_releases += 1
            released.append(f"prefill x{len(self._prefill_traces)}")
        if stale_decode:
            self._decode_trace_releases += 1
            released.append("decode" + (" + sampling" if self._sampling_captured else ""))
        logger.warning(
            "MuseGlimmerGenerator: the KV cache was rebound to different buffers; releasing the "
            f"{' and '.join(released)} trace(s). The next call recaptures against the new buffers; "
            "bind the cache before the first prefill to avoid paying for that."
        )
        if stale_prefill:
            self._release_prefill_traces()
        if stale_decode:
            self._release_decode_trace()
        # A rebind is the natural retry point for anything an earlier rebind could not free.
        self._retry_orphaned_traces()

    def _release_decode_trace(self) -> None:
        """Drop the decode trace, and with it the sampling trace captured over its logits.

        The sampling trace has to go too: ``SamplingGenerator`` validates by tensor
        identity against the logits it was captured over, so leaving it alive across a
        decode recapture raises *"The provided logits tensor does not match the tensor used
        during trace capture"* on the next sampled step.  ``decode_forward`` re-captures
        both lazily, against whichever cache is bound then.

        The drains are the same async-CCL hygiene as in :meth:`_release_prefill_traces`,
        and equally outside any steady-state loop -- this runs at a cache rebind, not per
        token.
        """
        ttnn.synchronize_device(self.mesh_device)
        self.counters["synchronizations"] += 1
        # Unconditional: ``reset_trace()`` iterates its own slot table and skips empty
        # slots, so it is a no-op when nothing was captured, and calling it on the strength
        # of *our* flag would trust a flag to describe someone else's state.  The old
        # ``teardown()`` called it unconditionally for that reason and this keeps the
        # property.
        try:
            self.sampling.reset_trace()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to release the sampling trace: {exc}")
        self._sampling_captured = False
        if self._trace_id is not None:
            released = False
            try:
                ttnn.release_trace(self.mesh_device, self._trace_id)
                released = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"failed to release the decode trace: {exc}")
            if not released:
                # Round 7 kept the id in place so a retry was possible.  Round 8 showed what
                # that cost: ``decode_forward`` tests ``self._trace_id is None`` to decide
                # whether to recapture, so a retained id is a *replayed* id, and on the rebind
                # path the buffers it was captured over are no longer the ones bound.  Fail
                # closed instead -- the handle and the logits move to ``_orphaned_traces``,
                # which no lookup path consults, and the slots are cleared so the next call
                # recaptures against the live cache.  Nothing is deallocated and the live-trace
                # count stays raised; :meth:`_retry_orphaned_traces` retries.
                self._orphaned_traces.append(
                    {
                        "what": "decode",
                        "id": self._trace_id,
                        "tensors": [t for t in (self._trace_logits,) if t is not None],
                    }
                )
                logger.warning(
                    "MuseGlimmerGenerator: the decode trace failed to release; it is retained "
                    "unusable (never replayed again) with its logits, the live-trace count stays "
                    "raised, and the release is retried at the next release and at teardown()."
                )
                self._trace_id = None
                self._trace_logits = None
                self._decode_trace_cache_sig = ()
                ttnn.synchronize_device(self.mesh_device)
                return
            self._trace_id = None
            self.model.note_trace_released()
        ttnn.synchronize_device(self.mesh_device)
        if self._trace_logits is not None:
            try:
                ttnn.deallocate(self._trace_logits)
            except Exception:  # noqa: BLE001
                pass
            self._trace_logits = None
        self._decode_trace_cache_sig = ()

    def _release_prefill_traces(self) -> None:
        """Drop every prefill trace.  Their KV-cache addresses are baked in, so this is
        mandatory whenever the cache is rebound to *different* buffers.

        A prefill trace contains fabric collectives, so the drains around the release are
        ordinary async-CCL hygiene: nothing is deallocated while the fabric may still have
        work in flight.  They are outside any steady-state loop -- releases happen at a
        cache rebind or at teardown -- so they cost nothing per token.

        What they are *not* is the fix for the one watcher trip this path ever produced
        (``doc/optimized_full_model/logs/watcher_bisect_rebind.log``: *"subordinate_erisc
        detected invalid NOC command buffer state ... fabric_erisc_router.cpp"* on
        ``acteth core virtual(x=29,y=25)``).  Draining first was tried and did not move it.
        The cause was in the test: it freed a *cloned* KV cache while a trace holding those
        addresses was still alive.  Releasing the trace before freeing the cache fixed it
        (``logs/watcher_bisect_rebind_fixed.log``), and the release path is watcher-clean
        in the gated suite, in both opt-in cases run together in one process, and in
        ``prefill_trace_release_probe.py --arm rebuild``.  A *separate* teardown-time trip
        does survive when those opt-in cases share a process with the other ten gated
        cases; it is README limitation 6, it happens after every test has passed, and it
        is not this function's ordering.
        """
        if self._prefill_traces:
            ttnn.synchronize_device(self.mesh_device)
            self.counters["synchronizations"] += 1
        orphaned = 0
        for padded_len, entry in list(self._prefill_traces.items()):
            released = False
            try:
                ttnn.release_trace(self.mesh_device, entry["id"])
                released = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"failed to release the prefill trace for {padded_len}: {exc}")
            if not released:
                # Fails closed exactly as the decode trace does, and for a sharper reason: the
                # bucket dict *is* the lookup, so a retained entry is one ``_prefill_traced``
                # will find and replay.  Worse, with ``prefill_trace_max_entries > 1`` -- which
                # the stage README recommends for serving -- a partial failure used to return
                # before clearing ``_prefill_trace_cache_sig``, so the next capture on another
                # bucket re-stamped the signature to the *new* cache and the stale entry could
                # never be invalidated again.  Moving it out of the dict removes both.
                self._orphaned_traces.append(
                    {
                        "what": f"prefill[{padded_len}]",
                        "id": entry["id"],
                        "tensors": [entry[key] for key in ("tokens", "page_table", "logits")],
                    }
                )
                orphaned += 1
                continue
            ttnn.synchronize_device(self.mesh_device)
            for key in ("tokens", "page_table", "logits"):
                try:
                    ttnn.deallocate(entry[key])
                except Exception:  # noqa: BLE001
                    pass
            self.model.note_trace_released()
        # Unconditional on both counts: every entry was either released or orphaned, so no
        # bucket is left replayable, and the signature must be cleared even when something
        # was orphaned or a later capture would stamp the new cache over the old comparison.
        self._prefill_traces = {}
        self._prefill_trace_cache_sig = ()
        if orphaned:
            logger.warning(
                f"MuseGlimmerGenerator: {orphaned} prefill trace(s) failed to release; they are "
                "retained unusable (never replayed again) with their buffers, the live-trace count "
                "stays raised, and the release is retried at the next release and at teardown()."
            )

    def _retry_orphaned_traces(self) -> int:
        """Retry every trace whose release raised, and free its tensors once it lands.

        Called after a rebind-driven release and from :meth:`teardown`, so a transient
        failure is retried at the next natural boundary, not only at process exit.  A trace that
        still refuses to release stays orphaned: it is never replayed, its buffers are never
        freed, and it keeps ``live_traces_over_kv_cache`` raised so
        :meth:`MuseGlimmerModel.deallocate` still warns.  Returns how many are still held.
        """
        if not self._orphaned_traces:
            return 0
        ttnn.synchronize_device(self.mesh_device)
        still_held = []
        for orphan in self._orphaned_traces:
            try:
                ttnn.release_trace(self.mesh_device, orphan["id"])
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"retrying the release of the {orphan['what']} trace failed again: {exc}")
                still_held.append(orphan)
                continue
            ttnn.synchronize_device(self.mesh_device)
            for tensor in orphan["tensors"]:
                try:
                    ttnn.deallocate(tensor)
                except Exception:  # noqa: BLE001
                    pass
            self.model.note_trace_released()
            logger.info(f"MuseGlimmerGenerator: the retained {orphan['what']} trace released on retry.")
        self._orphaned_traces = still_held
        return len(still_held)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: Any = None,
        kv_cache: Any = None,
        prompt_lens: List[int] | None = None,
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Low-level prefill.  Updates ``kv_cache`` in place; returns host logits.

        ``[batch, padded_len]`` tokens in, ``[batch, 1, vocab]`` logits out (or
        ``[batch, prompt_len, vocab]`` with ``return_all_logits``).  Per-user
        ``prompt_lens`` may differ; each user is prefilled into its own cache
        slot, which is what makes mixed-length batched prefill work without a
        common padded length.

        ``page_table`` may be a torch tensor, a ttnn tensor, or ``None``; the
        readiness prefill check hands in a ttnn tensor of a different width than
        the model's ``blocks_per_seq``, so it is normalised rather than trusted.
        """
        self.model.set_kv_cache(kv_cache)
        # Bind first, then invalidate only if the buffers actually moved: a serving
        # caller threading the *same* external cache every request must keep its traces.
        if kv_cache is not None:
            self._invalidate_traces_if_cache_moved()
        tokens = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.reshape(1, -1)
        batch = int(tokens.shape[0])
        if prompt_lens is None:
            prompt_lens = [int(tokens.shape[1])] * batch
        prompt_lens = [int(p) for p in prompt_lens]
        table = self._coerce_page_table(page_table)
        # ``**kwargs`` exists because the readiness runners pass keywords this signature
        # does not name, and swallowing them silently is the right default for those.
        # It is *not* right for the ones that change what prefill does: limitation 2
        # used to advertise the per-layer sliding-tail hand-off as "implemented and
        # exposed" (it now records that this guard refuses it, and points a caller at
        # ``MuseGlimmerModel.prefill_forward``), and before this guard existed a caller
        # writing ``prefill_forward(..., continuation=True)`` was getting an
        # ordinary non-continuation prefill with no error. Those are named and refused
        # here rather than dropped, because the failure is silent and the caller is the
        # vLLM stage.
        # ``start_pos`` is a *required* keyword of the serving prefill signature
        # (``models/common/readiness_check/contract_vllm.py``), and the ordinary
        # single-chunk case passes 0 -- which is exactly what ``_prefill_user`` does. So
        # 0 is accepted; only a non-zero value, which would mean chunked continuation,
        # is refused.
        start_pos = kwargs.pop("start_pos", 0)
        if start_pos:
            raise NotImplementedError(
                f"prefill_forward() starts every user at position 0, got start_pos={start_pos}; "
                "drive MuseGlimmerModel.prefill_forward directly for chunked continuation"
            )
        for owned in ("continuation", "keep_sliding_tails"):
            if owned in kwargs:
                raise NotImplementedError(
                    f"prefill_forward() does not thread {owned!r} through to the layer stack yet; "
                    "drive MuseGlimmerModel.prefill_forward directly for chunked continuation"
                )

        outputs: list[torch.Tensor] = []
        for user in range(batch):
            ids = tokens[user, : prompt_lens[user]].tolist()
            if return_all_logits:
                rows = self._prefill_user(ids, user_id=user, page_table=table, return_all_logits=True)
                host_rows = [self.model.logits_to_torch(row) for row in rows]
                for row in rows:
                    ttnn.deallocate(row)
                outputs.append(torch.cat(host_rows, dim=0)[: prompt_lens[user]].unsqueeze(0))
            else:
                logits, row_in_tile = self._prefill_user(ids, user_id=user, page_table=table)
                host = self.model.logits_to_torch(logits)
                ttnn.deallocate(logits)
                outputs.append(host[row_in_tile : row_in_tile + 1].unsqueeze(0))
        if return_all_logits:
            width = max(int(o.shape[1]) for o in outputs)
            if len({int(o.shape[1]) for o in outputs}) != 1:
                outputs = [torch.nn.functional.pad(o, (0, 0, 0, width - int(o.shape[1]))) for o in outputs]
        return torch.cat(outputs, dim=0)

    def _coerce_page_table(self, page_table: Any) -> torch.Tensor:
        if page_table is None:
            return self.model.normalize_page_table(None)
        if isinstance(page_table, torch.Tensor):
            return self.model.normalize_page_table(page_table)
        # A ttnn tensor: read device 0's copy.  The page table is replicated, so
        # any device would do.
        host = ttnn.to_torch(ttnn.get_device_tensors(page_table)[0])
        return self.model.normalize_page_table(host.reshape(host.shape[-2], host.shape[-1]) if host.dim() > 2 else host)

    # ---------------------------------------------------------------- decode

    def _capture_decode_trace(self) -> None:
        """Compile, then capture, the device-only decode graph.

        The warm-compile pass *executes* -- including the in-trace
        ``ttnn.plus_one`` and the KV-cache write -- so every persistent input it
        mutated is restaged to its intended capture state immediately before
        ``begin_trace_capture``.  Skipping that restage is the classic
        off-by-one: the captured graph would be correct and the first replay
        would decode at the wrong position.
        """
        advance_positions = True
        model = self.model
        inputs = self._device_inputs
        staged_tokens = self._staged_tokens
        staged_positions = self._staged_positions

        warm = model.ttnn_decode_forward(
            inputs["tokens"],
            inputs["current_pos"],
            inputs["rope_pos_ids"],
            inputs["page_table"],
            advance_positions=advance_positions,
        )
        ttnn.deallocate(warm)
        ttnn.synchronize_device(self.mesh_device)
        self.counters["synchronizations"] += 1
        self._stage(tokens=staged_tokens, positions=staged_positions)

        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        except RuntimeError as exc:
            raise RuntimeError(
                "decode trace capture failed to start; the mesh must be opened with a non-zero "
                f"trace_region_size (open_multichip_mesh(trace_region_size={DEFAULT_TRACE_REGION_SIZE})). "
                f"Underlying error: {exc}"
            ) from exc
        logits = model.ttnn_decode_forward(
            inputs["tokens"],
            inputs["current_pos"],
            inputs["rope_pos_ids"],
            inputs["page_table"],
            advance_positions=advance_positions,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.counters["synchronizations"] += 1
        self._trace_id = trace_id
        self._trace_logits = logits
        self._decode_trace_cache_sig = self._kv_cache_signature()
        model.note_trace_captured()
        logger.info("MuseGlimmerGenerator: captured decode trace (positions advance on device)")

    def _capture_sampling_trace(self, logits: ttnn.Tensor) -> None:
        """Capture the sampler's own trace over the decode trace's logits tensor.

        ``tt_out_tok`` is the persistent decode token input, so replay writes the
        sampled token exactly where the next decode replay reads it.  The
        precompile inside ``capture_trace`` runs sampling eagerly and therefore
        clobbers that buffer, so the caller restages afterwards.
        """
        self.sampling.capture_trace(logits, tt_out_tok=self._device_inputs["tokens"])
        self._sampling_captured = True
        logger.info("MuseGlimmerGenerator: captured sampling trace (tt_out_tok -> decode token input)")

    def _decode_step_traced(self, *, host_sampling: bool) -> torch.Tensor:
        """One traced decode step; returns the sampled token per batch row."""
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        self.counters["trace_replays"] += 1
        logits = self._trace_logits
        if host_sampling:
            return self._host_argmax(logits)
        self.sampling.seed_manager.get_new_values()
        sampled = self.sampling.sample(logits, enable_trace=True, tt_out_tok=self._device_inputs["tokens"])
        return self._read_tokens(sampled)

    def _host_argmax(self, logits: ttnn.Tensor) -> torch.Tensor:
        """Explicit host-sampling compatibility mode.

        Gathers the full vocab and takes the argmax on the host.  This exists for
        tests that require host sampling and is never the measured path; see the
        module docstring.
        """
        gathered = self.model.gather_and_untilize_logits(logits)
        host = self.model.logits_to_torch(gathered, gathered=True)
        ttnn.deallocate(gathered)
        return torch.argmax(host, dim=-1).to(torch.int64)

    def _decode_step_eager(self, *, host_sampling: bool, want_logits: bool = False) -> torch.Tensor:
        """Untraced decode, for debugging only.  Not readiness evidence."""
        inputs = self._device_inputs
        logits = self.model.ttnn_decode_forward(
            inputs["tokens"], inputs["current_pos"], inputs["rope_pos_ids"], inputs["page_table"]
        )
        if want_logits:
            gathered = self.model.gather_and_untilize_logits(logits)
            host = self.model.logits_to_torch(gathered, gathered=True)
            ttnn.deallocate(gathered)
            ttnn.deallocate(logits)
            return host
        if host_sampling:
            tokens = self._host_argmax(logits)
        else:
            tokens = self._sample_eager(logits, into_tokens=True)
        ttnn.deallocate(logits)
        return tokens

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: Any = None,
        kv_cache: Any = None,
        sample_on_device: bool = False,
        sampling_params: SamplingParams | None = None,
        enable_trace: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Low-level decode: one step for every active row, caller-owned state.

        ``tokens`` is ``[batch, 1]``, ``start_pos`` is ``[batch]``.  Rows beyond
        ``batch`` are padded to ``max_batch_size`` with position ``-1``, the
        inactive-slot sentinel the paged attention op skips and
        ``plus_one(skip_negative_entries=True)`` preserves -- so a fixed 32-slot
        batch with only some rows running is a supported input, not a special
        case.

        Positions come from the caller every step, and that is compatible with the
        single decode trace: the in-trace ``plus_one`` runs after every read of
        them, so the next call's restage simply overwrites the increment.
        """
        self.model.set_kv_cache(kv_cache)
        if kv_cache is not None:
            self._invalidate_traces_if_cache_moved()
        self._allocate_device_inputs()
        if self._sampling_params is None:
            self._apply_sampling_params(sampling_params)
        elif sampling_params is not None:
            self._apply_sampling_params(sampling_params)

        token_list = [int(t) for t in torch.as_tensor(tokens).reshape(-1).tolist()]
        positions = torch.as_tensor(start_pos).reshape(-1)
        self._staged_tokens = token_list
        self._staged_positions = positions
        self._stage(tokens=token_list, positions=positions, page_table=self._coerce_page_table(page_table))

        if not enable_trace:
            out = self._decode_step_eager(host_sampling=False, want_logits=not sample_on_device)
            return out[: len(token_list)]

        if self._trace_id is None:
            self._capture_decode_trace()
            self._stage(tokens=token_list, positions=positions)
        logits = self._trace_logits
        if sample_on_device and not self._sampling_captured:
            self._capture_sampling_trace(logits)
            self._stage(tokens=token_list, positions=positions)
        if sample_on_device:
            sampled = self._decode_step_traced(host_sampling=False)
            return sampled[: len(token_list)]
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        self.counters["trace_replays"] += 1
        gathered = self.model.gather_and_untilize_logits(logits)
        host = self.model.logits_to_torch(gathered, gathered=True)
        ttnn.deallocate(gathered)
        return host[: len(token_list)]

    # ----------------------------------------------------------- generation

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[Callable[[int, int], int]] = None,
        enable_trace: bool = True,
        sampling_params: SamplingParams | None = None,
        host_sampling: bool = False,
        stop_on_eos: bool | None = None,
        user_id: int = 0,
        page_table: Any = None,
        **kwargs: Any,
    ) -> List[int]:
        """High-level greedy/sampled generation for one user.

        Returns the model's **own** predictions, one per requested token, even
        when ``next_input`` overrides what is fed back -- that is what the
        readiness check scores.

        ``next_input is None`` is the free-running path and the one the reported
        token-out decode numbers come from: the sampled token stays on device and
        the position advances on device, so the steady-state step stages nothing.
        Teacher forcing necessarily restages the **token** each step, but not the
        position: the in-trace ``plus_one`` has already advanced ``current_pos`` and
        the RoPE index to the value a restage would write.  One position refresh is
        needed and taken, for the first step after prefill, whose position is the
        prompt length rather than a +1.  Both paths still run through the traced
        decode and the traced sampler.
        """
        if max_new_tokens < 1:
            return []
        if user_id != 0:
            # The prefill below fills cache slot ``user_id``, but this loop stages all
            # 32 decode rows at the same position and reads the prediction out of row 0,
            # whose page-table row is slot 0.  At ``max_batch_size=1`` that is invisible
            # (every row aliases slot 0), which is exactly why it would be a silent
            # wrong-slot decode on a multi-slot generator.  The low-level
            # ``prefill_forward``/``decode_forward`` pair is the API for driving a
            # specific slot; this convenience loop owns slot 0 only.
            raise ValueError(
                f"generate() drives cache slot 0 only, got user_id={user_id}; use "
                "prefill_forward/decode_forward to drive a specific slot"
            )
        config = self.model.config
        prompt_len = len(prompt_token_ids)
        if prompt_len + max_new_tokens > config.max_seq_len:
            raise ValueError(
                f"prompt {prompt_len} + generation {max_new_tokens} exceeds the supported context "
                f"{config.max_seq_len}"
            )
        stop_on_eos = (next_input is None) if stop_on_eos is None else stop_on_eos
        device_loop = enable_trace and next_input is None and not host_sampling

        self._allocate_device_inputs()
        self._apply_sampling_params(sampling_params)
        table = self._coerce_page_table(page_table)
        self._stage(page_table=table)

        # ------------------------------------------------------- prefill
        logits, row_in_tile = self._prefill_user(prompt_token_ids, user_id=user_id, page_table=table)
        if host_sampling:
            gathered = self.model.gather_and_untilize_logits(logits)
            host = self.model.logits_to_torch(gathered, gathered=True)
            ttnn.deallocate(gathered)
            predicted = int(torch.argmax(host[row_in_tile]).item())
        else:
            # The sampler runs on all 32 rows; the prompt's last token lives in
            # row ``row_in_tile`` of the tile the LM head was given.
            # ``into_tokens=False``: the sampler samples all 32 rows of the tile
            # the LM head was given, and the prompt's last token is row
            # ``row_in_tile`` of it -- writing that whole vector into the decode
            # token buffer would put the wrong row in slot 0.  The first decode
            # step stages the chosen token explicitly (``_needs_reseed``).
            sampled = self._sample_eager(logits, into_tokens=False)
            predicted = int(sampled[row_in_tile].item())
        ttnn.deallocate(logits)

        predictions = [predicted]
        fed = int(next_input(0, predicted)) if next_input is not None else predicted

        # -------------------------------------------------------- decode
        position = prompt_len
        self._staged_tokens = [fed] * DECODE_ROWS
        self._staged_positions = torch.full((DECODE_ROWS,), position, dtype=torch.int64)
        self._needs_reseed = True

        for step in range(1, max_new_tokens):
            if stop_on_eos and predictions[-1] in self._eos_ids:
                break
            if enable_trace:
                if self._trace_id is None:
                    self._stage(tokens=self._staged_tokens, positions=self._staged_positions)
                    self._capture_decode_trace()
                if not host_sampling and not self._sampling_captured:
                    self._capture_sampling_trace(self._trace_logits)
                if self._needs_reseed:
                    self._stage(tokens=self._staged_tokens, positions=self._staged_positions)
                    self._needs_reseed = False
                elif not device_loop:
                    # A caller-driven step (teacher forcing, or an explicit
                    # ``next_input``) changes the **token** and nothing else: the
                    # in-trace ``plus_one`` has already advanced ``current_pos`` and
                    # the RoPE index to exactly the values a restage would write, and
                    # it did so after every read of them.  Restaging positions here as
                    # well -- which this did until the stage review caught it -- costs
                    # two more host tensors and two more host-to-device copies per
                    # token for no change in the values.  No end-to-end figure is
                    # attached to this deliberately: the only teacher-forcing rates in
                    # the evidence tree also span the topk split, which moved that
                    # number by ~9 ms/token, so nothing in the artifacts isolates this
                    # change and quoting a delta from them would be attribution by
                    # coincidence.  It is removed because it is provably redundant work
                    # on the serving path -- the values written are the values already
                    # there -- and ``test_caller_driven_decode_restages_only_the_token``
                    # pins that on the counters rather than on milliseconds.
                    # ``_needs_reseed`` still covers the one step that does need both:
                    # the first decode after prefill, whose position is the prompt
                    # length rather than a +1.
                    self._stage(tokens=self._staged_tokens)
                sampled = self._decode_step_traced(host_sampling=host_sampling)
            else:
                self._stage(tokens=self._staged_tokens, positions=self._staged_positions)
                sampled = self._decode_step_eager(host_sampling=host_sampling)
            predicted = int(sampled[0].item())
            predictions.append(predicted)
            fed = int(next_input(step, predicted)) if next_input is not None else predicted
            position += 1
            self._staged_tokens = [fed] * DECODE_ROWS
            self._staged_positions = torch.full((DECODE_ROWS,), position, dtype=torch.int64)
        return predictions

    # -------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Wipe per-prompt state; keep weights, traces and device buffers.

        Traces survive on purpose: they are keyed on the graph, not on the
        prompt, and re-capturing per prompt would put ~2400 ops of compile back
        into every request.  What has to go is the cache contents, the
        page-table memo and the decode-position/token staging.
        """
        self.model.reset_kv_cache()
        self._prev_page_table = None
        self._needs_reseed = True
        self._staged_tokens = [0] * DECODE_ROWS
        self._staged_positions = torch.zeros(DECODE_ROWS, dtype=torch.int64)

    def teardown(self) -> None:
        """Release every trace.  One release path, shared with the cache-rebind one.

        This used to inline its own copy of the decode/sampling release, and round 5 of the
        stage review enumerated the four ways that copy had drifted from
        :meth:`_release_decode_trace`: it never deallocated ``_trace_logits`` (3.24 MB per
        device, left to Python GC of the dead generator), never cleared
        ``_sampling_captured``, released the decode trace *before* the sampling trace
        captured over its logits rather than after, and -- on the shipped default, where
        there are no prefill traces to release -- reached ``ttnn.release_trace`` on a graph
        full of fabric collectives with **no drain anywhere in the path**, which is exactly
        the hygiene both release functions document as mandatory.

        So there is one release path now, and it is the deterministic one.  The drain is
        unconditional here because the shipped default is the case that had none.
        """
        ttnn.synchronize_device(self.mesh_device)
        self.counters["synchronizations"] += 1
        self._release_prefill_traces()
        self._release_decode_trace()
        # Last chance for anything a previous release could not free.  Both release calls
        # above may themselves have orphaned something, so this runs after them.
        still_held = self._retry_orphaned_traces()
        if still_held:
            logger.warning(
                f"MuseGlimmerGenerator: teardown() leaves {still_held} trace(s) unreleased after a "
                "retry; their KV-cache buffers must not be freed while this device stays open."
            )

    # ------------------------------------------------------------- reporting

    def capability_report(self) -> dict:
        config = self.model.config
        report = {
            "hf_model": config_model_id(self.gen_config),
            "supported_context": config.max_seq_len,
            "cache_slots": config.max_batch_size,
            "decode_rows": DECODE_ROWS,
            "max_num_blocks": config.max_num_blocks,
            "page_block_size": config.page_block_size,
            "blocks_per_seq": config.blocks_per_seq,
            "prefill_chunk_size": config.prefill_chunk_size,
            "vocab_size": config.vocab_size,
            "padded_vocab_size": config.padded_vocab_size,
            "num_layers": config.num_layers,
            "layer_kinds": list(config.layer_kinds),
            "force_argmax": bool(self.sampling.tt_sampling.force_argmax_sampling),
            "sampling_implementation": "models.common.sampling.SamplingGenerator",
            # This stage's own three decode-path flags plus the opt-in prefill trace, read off
            # the *built model* rather than from prose.  Round 7 pointed out that without them
            # the ``capacity`` blocks of the baseline and shipped evidence arms are
            # byte-identical on exactly the settings that separate them, so the only thing
            # distinguishing the arms was ``performance.baseline_arm``.  Same reasoning as the
            # carried-forward decoder contract above: a build that silently flipped one of these
            # would be a different model and should say so in its own evidence.
            #
            # Round 8: the softcap flag is the one of the three that is *snapshotted* --
            # ``_LMHead.__init__`` copies the module constant into ``self.softcap_in_l1`` and
            # ``forward`` consults the instance (``tt/model.py``), and both the L1 probe and the
            # acceptance test drive the instance -- so reading the module global here described
            # the import, not the build, and a model constructed with ``softcap_in_l1=False``
            # reported ``true``.  The other two are read per-forward off the module, so there
            # the global *is* the built state.
            # (The reported *value* is unchanged for all three shipped evidence arms -- the
            # harness mutates the global before the build and never restores -- so no
            # committed artifact goes stale on this fix; what changes is that a build which
            # overrode the constructor argument can no longer misreport itself.)
            "lm_head_softcap_in_l1": bool(self.model.lm_head.softcap_in_l1),
            "embed_decode_gather_sharded": bool(model_mod.EMBED_DECODE_GATHER_SHARDED),
            "decode_swiglu_mul_cores": dec_mod.DECODE_SWIGLU_MUL_CORES,
            "prefill_trace": bool(self.gen_config.prefill_trace),
            "prefill_trace_max_entries": int(self.gen_config.prefill_trace_max_entries),
            "lm_head_dtype": str(self.gen_config.lm_head_dtype),
            "lm_head_matmul": self.gen_config.lm_head_matmul,
            "lm_head_cores": self.gen_config.lm_head_cores,
            "lm_head_in0_block_w": self.gen_config.lm_head_in0_block_w,
            "lm_head_fidelity": str(self.gen_config.lm_head_fidelity),
            "lm_head_fp32_acc": self.gen_config.lm_head_fp32_acc,
            "lm_head_output_dtype": str(self.gen_config.lm_head_output_dtype),
            "sampler_max_top_k": self.gen_config.max_top_k,
            # Read off the **built sampler**, not off the port knob. Requesting the
            # multi-core split forces the pad on inside ``TTSampling``, so reporting
            # ``gen_config.pad_logits_to_power_of_2`` alone would tell a serving
            # integrator the shard is 50688 wide when the op sees 65536 -- which is
            # exactly what the committed evidence files said until the stage review
            # caught it. Both are reported: what was asked for, and what is running.
            "sampler_pad_logits_to_power_of_2_requested": self.gen_config.pad_logits_to_power_of_2,
            "sampler_pad_logits_to_power_of_2_effective": bool(self.sampling.tt_sampling.pad_to_power_of_2),
            "sampler_topk_split_to_power_of_2": self.gen_config.topk_split_to_power_of_2,
            "sampler_topk_pieces": int(getattr(self.sampling.tt_sampling, "topk_pieces", 1)),
            # The invalid-vocab mask shipped **absent** for three review rounds and no
            # artifact would have shown it: reported here so a regression is visible in
            # every evidence file rather than only in a unit test.
            "sampler_invalid_vocab_tail_width": int(getattr(self.sampling.tt_sampling, "_invalid_vocab_tail_width", 0)),
            "sampler_invalid_vocab_mask_built": bool(
                getattr(self.sampling.tt_sampling, "tt_invalid_vocab_mask", None) is not None
                or getattr(self.sampling.tt_sampling, "tt_invalid_vocab_tail_mask", None) is not None
            ),
            "sampler_candidates_per_device": int(
                getattr(self.sampling.tt_sampling, "candidates_per_device", self.gen_config.max_top_k)
            ),
            "counters": dict(self.counters),
        }
        # The carried-forward decoder contract, read off the built layers rather
        # than restated: a full model that silently turned one of these back on
        # would be a different model from the one the decoder stage measured.
        layer = self.model.layers[0]
        precision = self.model.precision
        report["carried_forward_decoder_contract"] = {
            "activation_dtype": str(precision.activation_dtype),
            "kv_cache_dtype": str(precision.kv_cache_dtype),
            "weight_dtype_wqkv": str(precision.weight_dtype("wqkv")),
            "weight_dtype_o_proj": str(precision.weight_dtype("o_proj")),
            "weight_dtype_mlp_gate": str(precision.weight_dtype("mlp_gate")),
            "weight_dtype_mlp_down": str(precision.weight_dtype("mlp_down")),
            "prefill_ccl_impl": layer.prefill_ccl_impl,
            "decode_ccl_impl": layer.decode_ccl_impl,
            "prefill_ccl_mode": layer.prefill_ccl_mode,
            "decode_ccl_mode": layer.decode_ccl_mode,
            "prefill_ccl_dtype": str(layer.prefill_ccl_dtype),
            "decode_ccl_dtype": str(layer.decode_ccl_dtype),
            "prefill_ccl_rs_workers": layer.prefill_ccl_rs_workers,
            "decode_ccl_rs_workers": layer.decode_ccl_rs_workers,
            "ccl_ag_barrier": layer.ccl_ag_barrier,
            "ccl_persistent_buffers_rejected_and_off": layer.ccl_persistent_buffers is False,
            "prefill_fractured_norm": layer.prefill_fractured_norm,
            "prefill_fractured_norm_min_rows": layer.prefill_fractured_norm_min_rows,
            "sharded_decode_io": layer.sharded_decode_io,
            "boundary_cores": layer.boundary_cores,
            "decode_matmul_o_proj": list(layer.decode_matmul["o_proj"]),
            "max_cores_per_head_batch": layer.max_cores_per_head_batch,
        }
        report.update(self.model.dram_report())
        report["per_device_dram_capacity_bytes"] = dram_capacity_bytes(self.mesh_device)
        return report


def config_model_id(gen_config: GeneratorConfig) -> str:
    return gen_config.model_id


# --------------------------------------------------------- factory convention

#: Built generators, keyed by mesh and configuration.
#:
#: Loading the 52-layer stack takes ~160 s: ~484 M parameters per layer have to be
#: read, transposed and packed into BFP4/BFP8 on the host.  Each readiness runner
#: calls ``build_generator`` itself, so four runners in four processes repeat that
#: work four times.  Returning the same generator for the same (mesh, config) lets
#: one process drive every runner over one build, which is what
#: ``doc/full_model/bench/evidence.py`` does.  ``reset()`` is called on the way out
#: so a reused generator starts from the state a fresh one would.
#:
#: The dict itself lives in ``tt/model.py`` on purpose.  The readiness runners load
#: *this* file by path under a synthetic module name, so a module-level dict here
#: would be a second, unrelated copy -- and the whole point is that the driver's
#: build and the runner's build are the same object.  ``tt.model`` is imported
#: normally, so there is exactly one of it.


def _resolve_max_seq_len(gen_config: GeneratorConfig) -> int:
    """``max_seq_len=None`` means the HF-advertised context; resolve it before keying.

    The readiness runners call ``build_generator(model_dir, mesh_device)`` with no
    knobs at all, so their config has ``max_seq_len=None`` while a driver that spells
    the same number out has ``131072``.  Keying on the unresolved value makes those
    two look like different models and rebuilds the 52-layer stack -- which is not
    just slow, it is 7.2 GB/device of DRAM per redundant copy.
    """
    if gen_config.max_seq_len is not None:
        return int(gen_config.max_seq_len)
    from transformers import AutoConfig

    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import _text_config

    config = AutoConfig.from_pretrained(str(weights_snapshot_dir(gen_config.model_id)), local_files_only=True)
    return int(_text_config(config).max_position_embeddings)


def _cache_key(mesh_device: ttnn.MeshDevice, gen_config: GeneratorConfig) -> tuple:
    return (
        id(mesh_device),
        gen_config.model_id,
        gen_config.max_batch_size,
        _resolve_max_seq_len(gen_config),
        gen_config.page_block_size,
        gen_config.prefill_chunk_size,
        None if gen_config.layer_indices is None else tuple(gen_config.layer_indices),
        str(gen_config.lm_head_dtype),
        gen_config.lm_head_matmul,
        gen_config.lm_head_cores,
        gen_config.lm_head_in0_block_w,
        str(gen_config.lm_head_fidelity),
        gen_config.lm_head_fp32_acc,
        str(gen_config.lm_head_output_dtype),
        gen_config.allow_force_argmax,
        gen_config.max_top_k,
        gen_config.pad_logits_to_power_of_2,
        gen_config.topk_split_to_power_of_2,
        gen_config.prefill_trace,
        gen_config.prefill_trace_max_entries,
        tuple(sorted((str(k), str(v)) for k, v in gen_config.decoder_kwargs.items())),
    )


def clear_generator_cache() -> None:
    GENERATOR_CACHE.clear()


def build_generator(model_dir: str | Path, mesh_device: ttnn.MeshDevice, **kwargs: Any) -> MuseGlimmerGenerator:
    """The readiness/vLLM factory: ``build_generator(model_dir, mesh_device, **kwargs)``.

    ``model_dir`` is accepted for the convention's sake; the weights come from the
    HF cache (``model_id``), which is what every earlier stage of this port also
    reads.  Everything else is a knob on :class:`GeneratorConfig`.
    """
    gen_config = GeneratorConfig(
        model_id=kwargs.pop("model_id", HF_MODEL_ID),
        max_batch_size=int(kwargs.pop("max_batch_size", 1)),
        max_seq_len=kwargs.pop("max_seq_len", None),
        page_block_size=int(kwargs.pop("page_block_size", 64)),
        prefill_chunk_size=kwargs.pop("prefill_chunk_size", None),
        layer_indices=kwargs.pop("layer_indices", None),
        lm_head_dtype=kwargs.pop("lm_head_dtype", LM_HEAD_DTYPE),
        lm_head_matmul=str(kwargs.pop("lm_head_matmul", LM_HEAD_MATMUL)),
        lm_head_cores=int(kwargs.pop("lm_head_cores", LM_HEAD_CORES)),
        lm_head_in0_block_w=int(kwargs.pop("lm_head_in0_block_w", LM_HEAD_IN0_BLOCK_W)),
        lm_head_fidelity=kwargs.pop("lm_head_fidelity", LM_HEAD_FIDELITY),
        lm_head_fp32_acc=bool(kwargs.pop("lm_head_fp32_acc", LM_HEAD_FP32_ACC)),
        lm_head_output_dtype=kwargs.pop("lm_head_output_dtype", LM_HEAD_OUTPUT_DTYPE),
        allow_force_argmax=bool(kwargs.pop("allow_force_argmax", False)),
        max_top_k=int(kwargs.pop("max_top_k", 32)),
        pad_logits_to_power_of_2=bool(kwargs.pop("pad_logits_to_power_of_2", False)),
        topk_split_to_power_of_2=bool(kwargs.pop("topk_split_to_power_of_2", True)),
        prefill_trace=bool(kwargs.pop("prefill_trace", False)),
        prefill_trace_max_entries=int(kwargs.pop("prefill_trace_max_entries", 1)),
        decoder_kwargs=dict(kwargs.pop("decoder_kwargs", {})),
    )
    reuse = bool(kwargs.pop("reuse", True))
    key = _cache_key(mesh_device, gen_config)
    cached = GENERATOR_CACHE.get(key) if reuse else None
    if cached is not None:
        logger.info("MuseGlimmerGenerator: reusing the generator already built for this mesh and config")
        cached.reset()
        cached.reset_counters()
        return cached

    tokenizer = kwargs.pop("tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer

        # The default revision for this repo is metadata-only, so resolve the
        # snapshot that actually holds the shards; its tokenizer is the one every
        # stage of this port has used.
        tokenizer = AutoTokenizer.from_pretrained(str(weights_snapshot_dir(gen_config.model_id)), local_files_only=True)

    started = time.perf_counter()
    model = MuseGlimmerModel.from_pretrained(
        mesh_device,
        model_id=gen_config.model_id,
        max_batch_size=gen_config.max_batch_size,
        max_seq_len=gen_config.max_seq_len,
        page_block_size=gen_config.page_block_size,
        prefill_chunk_size=gen_config.prefill_chunk_size,
        layer_indices=gen_config.layer_indices,
        lm_head_dtype=gen_config.lm_head_dtype,
        lm_head_matmul=gen_config.lm_head_matmul,
        lm_head_cores=gen_config.lm_head_cores,
        lm_head_in0_block_w=gen_config.lm_head_in0_block_w,
        lm_head_fidelity=gen_config.lm_head_fidelity,
        lm_head_fp32_acc=gen_config.lm_head_fp32_acc,
        lm_head_output_dtype=gen_config.lm_head_output_dtype,
        **gen_config.decoder_kwargs,
        **kwargs,
    )
    logger.info(f"MuseGlimmerModel built in {time.perf_counter() - started:.1f}s")
    generator = MuseGlimmerGenerator(model, tokenizer=tokenizer, gen_config=gen_config)
    if reuse:
        GENERATOR_CACHE[key] = generator
    return generator


def open_generator_mesh(
    *,
    mesh_shape: tuple[int, int] = DEFAULT_MESH_SHAPE,
    trace_region_size: int = DEFAULT_TRACE_REGION_SIZE,
) -> ttnn.MeshDevice:
    """Open the mesh this generator needs: ring fabric plus a trace region."""
    return open_multichip_mesh(mesh_shape, trace_region_size=trace_region_size)


def close_generator_mesh(mesh_device: ttnn.MeshDevice) -> None:
    close_multichip_mesh(mesh_device)


__all__ = [
    "DEFAULT_TRACE_REGION_SIZE",
    "GREEDY",
    "GeneratorConfig",
    "MuseGlimmerGenerator",
    "build_generator",
    "clear_generator_cache",
    "close_generator_mesh",
    "open_generator_mesh",
]
