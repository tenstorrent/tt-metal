# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full TTNN autoregressive model for ``meta-models/Muse-Glimmer-30B``.

This is the whole text path: token embeddings, the 52-layer decoder stack, the
terminal RMSNorm, the LM head and the tanh logit softcap -- everything the
per-layer decoder does not own.  The same wrapper serves all qualified meshes:
``OptimizedDecoder`` on P150 (1x1), and ``MultichipDecoder`` on P150x2/P150x4
(1x2/1x4).  All retain the selected precision policy, paged KV cache, and
width-sharded-L1 inter-layer decode residual.  Multi-chip profiles add the
measured collective split; P150 has no fabric or collectives.  No profile falls
back to host execution or replicated projection weights.

Parallelisation
---------------

The decoder's contract is a **replicated** residual stream with column-parallel
and row-parallel projections inside each layer, so this wrapper has to produce a
replicated hidden state and consume one.  Both terminal weights are therefore
column-parallel in the only direction that leaves the residual replicated (the
single P150 shard is the degenerate, communication-free case):

============  ==============================  ==========================================
tensor        fracture                        what it costs
============  ==============================  ==========================================
embed_tokens  hidden dim, ``ShardTensorToMesh(-1)``   one ``all_gather`` of the embedded
                                              rows -- ``202048 x 1664`` per device
                                              instead of ``202048 x 6656``
lm_head       vocab dim, ``ShardTensorToMesh(-1)``    nothing: the sampler consumes
                                              vocab-sharded logits directly, so there is
                                              no logits gather on the token-out path
norm.weight   replicated                      nothing (one tile row)
============  ==============================  ==========================================

On P150x2/P150x4 the embedding all-gather uses the *async* primitive with
semaphores this module owns, not ``ttnn.all_gather``: the composite wrapper
creates one global semaphore per program and never releases it, so a wrapper
gather would spend 256 B of the 6144 B ``L1_SMALL`` region on **every distinct
prompt length** the model ever sees.  See
``MultichipDecoder._ccl_semaphores`` for the measurement that bounds that
region.  P150 converts the local embedding directly to the requested layout.

The vocab is padded up so each device's shard is tile-aligned; the padded ids are
never valid tokens, and the sampler masks them out by being told **both** widths --
the padded one for its index arithmetic and the real one for its invalid-vocab mask
(see ``_SamplingArgs``, which shipped with the two conflated and the mask therefore
absent).  Getting the padded width wrong does not fail loudly either -- it shifts
every token id on device *d* by ``d * (pad / tp)`` -- so :func:`padded_vocab_size`
is the single place it is computed and the generator reads it from here.

Terminal path shapes
--------------------

Decode arrives at the terminal path in the decoder's boundary layout
(``[1, 1, 32, 6656]`` width-sharded L1 over 16 cores, replicated) and stays
sharded through the norm and into the DRAM-sharded LM-head matmul, so the whole
token-out tail is one norm, one matmul and two elementwise ops with no
interleaved round trip.  Prefill arrives DRAM-interleaved, is sliced to the tile
row that holds the last prompt token *before* the norm, and then takes the same
32-row path -- the LM head is never run over a whole prompt unless the caller
explicitly asks for all logits.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any, Sequence

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

from .functional_decoder import LAYER_KIND_SLIDING, TILE_SIZE, _rope_cos_sin, _text_config, resolve_layer_kind
from .fused_decoder import norm_compute_kernel_config
from .multichip_decoder import CCL_TOPOLOGY, MeshPlan, MultichipDecoder, mesh_plan
from .optimized_decoder import (
    BOUNDARY_CORES,
    DEFAULT_PRECISION,
    OptimizedDecoder,
    PrecisionPolicy,
    dram_sharded_weight_memcfg,
    width_sharded_l1,
)

HF_MODEL_ID = "meta-models/Muse-Glimmer-30B"
SUPPORTED_TP = (1, 2, 4)

# Precision artifacts carry the CCL payload policy used by tensor-parallel
# builds.  It is still useful provenance for P150, but OptimizedDecoder has no
# collectives and correctly rejects these kwargs.  Filter only this closed set
# on the 1x1 path; every other decoder override remains checked by its builder.
_MULTICHIP_ONLY_DECODER_KWARGS = {
    "ccl_dtype",
    "prefill_ccl_dtype",
    "decode_ccl_dtype",
    "ccl_mode",
    "prefill_ccl_mode",
    "decode_ccl_mode",
    "ccl_rs_workers",
    "prefill_ccl_rs_workers",
    "decode_ccl_rs_workers",
    "ccl_impl",
    "prefill_ccl_impl",
    "decode_ccl_impl",
    "ccl_ag_workers",
    "prefill_ccl_ag_workers",
    "decode_ccl_ag_workers",
    "ccl_persistent_buffers",
    "ccl_chunks_per_sync",
    "ccl_buffers_per_channel",
    "ccl_num_links",
    "ccl_ag_barrier",
    "prefill_fractured_norm",
    "prefill_fractured_norm_min_rows",
}

#: Canonical HF key prefixes.  ``model.language_model`` is the text tower of the
#: multimodal ``MuseGlimmerForConditionalGeneration``; the vision tower and the
#: projector are not part of the text-only path this port implements.
TEXT_PREFIX = "model.language_model"
EMBED_KEY = f"{TEXT_PREFIX}.embed_tokens.weight"
FINAL_NORM_KEY = f"{TEXT_PREFIX}.norm.weight"
LM_HEAD_KEY = "lm_head.weight"

#: Matmul contract for the LM head, and the geometry that goes with it.
#:
#: Two contracts are legal and both are measured at the real 32-row decode payload
#: by ``doc/full_model/bench/terminal_probe.py``
#: (``doc/full_model/logs/lm_head_sweep.log``):
#:
#: ============  =========  =====================  =========  =========
#: contract      dtype      geometry               ms/step    weight/dev
#: ============  =========  =====================  =========  =========
#: dram_sharded  BFP4       cores=52, in0=2        **0.6029**  190 MB
#: mcast1d       BFP4       in0=8                  0.6765     190 MB
#: mcast1d       BFP8       in0=8                  0.9779     359 MB
#: dram_sharded  BFP8       cores=16, in0=1        1.0396     359 MB
#: ============  =========  =====================  =========  =========
#:
#: The DRAM-sharded matmul requires ``K_tiles % cores == 0`` (*"in DRAM sharded
#: Matmul we don't have support for un-even sharding currently"*), and K is
#: ``6656 / 32 = 208`` tiles, so its legal core counts are the divisors of 208 that
#: fit an 11x10 grid: 8, 13, 16, 26, 52, 104.  Core count is worth nothing here
#: (1.0107-1.0147 ms across all six at BFP4/in0=1; the BFP8 family at the same
#: ``in0_block_w`` is a separate, disjoint band at 1.0396-1.0426 ms) and
#: ``in0_block_w`` is worth
#: everything (1.013 -> 0.603 ms at 1 -> 2).  Above those values the op fails with
#: an exact L1 blocker, *"Statically allocated circular buffers ... grow to
#: 1821824 B which is beyond max L1 size of 1572864 B"* -- which is also why BFP8
#: cannot take ``in0_block_w=2`` on this contract and loses to ``mcast1d``.
LM_HEAD_MATMUL = "dram_sharded"
LM_HEAD_CORES = 52
LM_HEAD_IN0_BLOCK_W = 2

#: Weight dtype for the LM head.  BFP4 is the same format the decoder's MLP
#: projections use, and this is the same kind of tensor: one big matmul whose
#: weight is streamed once per token.  It is 1.62x faster than the best BFP8
#: contract and 169 MB/device smaller.  Selected on the **real-weight** accuracy
#: gate rather than on synthetic PCC -- see ``doc/full_model/README.md``.
LM_HEAD_DTYPE = ttnn.bfloat4_b

#: Math fidelity, accumulate precision and output dtype for the LM-head matmul.
#:
#: LoFi/bf16 matches the decoder's DRAM-sharded decode projections, which is the
#: same op on the same weight dtype.  All three are knobs because the LM head is
#: the last thing between the layer stack and a token id, so when a top-k gate
#: misses it has to be possible to *exonerate* it rather than argue about it -- see
#: the LM-head precision ladder in ``doc/full_model/README.md``.
LM_HEAD_FIDELITY = ttnn.MathFidelity.LoFi
LM_HEAD_FP32_ACC = False
LM_HEAD_OUTPUT_DTYPE = ttnn.bfloat16

#: Run the tanh softcap in the layout the LM-head matmul produced, rather than
#: after converting the logits to DRAM interleaved.
#:
#: The full-model stage's order was matmul -> ``sharded_to_interleaved`` -> ``tanh``
#: -> ``multiply``, which put both elementwise ops on a DRAM-interleaved
#: ``[1, 1, 32, 50688]`` bf16 tensor: 3.24 MB read and written twice over, for
#: **17.7 us** (``tanh``) plus **19.1 us** (``multiply``) in the committed
#: full-model decode profile.  Doing them on the matmul's own width-sharded L1
#: output and converting **once**, at the end, is the same three tensors of
#: arithmetic with the DRAM round trip removed.
#:
#: The shard is padded -- ``50688 / 52 = 975`` columns is not a tile multiple, so
#: ``width_sharded_l1`` rounds each core to 992 and the last 896 columns of the
#: shard set are pad.  That is safe for this pair specifically: ``tanh`` is bounded
#: on every input including NaN-free garbage, the scalar multiply keeps it bounded,
#: and ``sharded_to_interleaved`` reconstructs the logical width, so no padded lane
#: can reach the sampler.  ``test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form`` pins the
#: equivalence on the device rather than on this argument.
LM_HEAD_SOFTCAP_IN_L1 = True

#: Gather the *decode* embedding straight into the decoder's boundary layout.
#:
#: A decode step's embedding is one tile row, and its all-gather wrote DRAM
#: interleaved and was immediately followed by ``interleaved_to_sharded`` into the
#: 16-core width-sharded L1 boundary spec.  ``all_gather_async`` takes an output
#: ``memory_config``, so the conversion can be the collective's own output layout.
#: Prefill is untouched: it needs the interleaved form, and its gather is the
#: chunk-and-clone path of :data:`EMBED_GATHER_CHUNK_ROWS`.
EMBED_DECODE_GATHER_SHARDED = True

#: Embedding table dtype.  ``ttnn.embedding`` needs a ROW_MAJOR table, which rules
#: out the block-float formats: they only exist in TILE layout.
EMBED_DTYPE = ttnn.bfloat16

#: One extra embedding row, all zeros, at index ``vocab_size``.
#:
#: This exists to make a non-tile-aligned prompt length *reproducible*.
#: ``ttnn.embedding`` writes only the rows its input asks for, so embedding a
#: 37-token prompt leaves rows 37..63 of the (tile-padded) output holding whatever
#: was in that DRAM page -- and those rows are real query rows to the prefill
#: attention, whose K/V lands in the cache and, because the paged SDPA reads a
#: *rounded* window rather than exactly ``seq_len`` keys, perturbs the logits of
#: the real rows.  Measured: a 37-token prompt returned three different top1/top2
#: gaps (0.25 / 0.375 / 0.75) across three identical prefills, while 64- and
#: 128-token prompts were bit-identical
#: (``doc/full_model/logs/prefill_repeat_probe.log``).
#:
#: Padding the *token ids* with an id whose embedding row is exactly zero fixes it
#: at the source: RMSNorm of a zero row is zero (``0 * (0 + eps) ** -0.5``), so the
#: padded rows are exactly the zeros every earlier decoder stage validated its
#: non-aligned prefill PCC against.  Costs one 3.3 KB table row per device.
EMBED_PAD_ROWS = 1

#: Rows per chunk of the prefill embedding all-gather, and the reason there is one.
#:
#: Gathering the embedding's output directly is **not bit-reproducible**: the same
#: prompt prefilled twice returns different logits, sporadically (roughly one run in
#: three), moving the argmax, with per-logit differences up to ~8.  It was found by
#: ``test_logits_are_reproducible_across_batch_positions`` and localised in
#: ``doc/full_model/bench/``:
#:
#: * ``prefill_divergence_probe.py`` -- the *first* stage that moves is the
#:   embedding, before any layer; the layers only carry it forward;
#: * ``embedding_gather_probe.py`` -- the local ``ttnn.embedding`` lookup is stable
#:   across runs in every arm, so the values are right; only the gather's output
#:   moves, and at small payloads the deviation is confined to a single remote
#:   device's 1664-column shard;
#: * ``ccl_reproducibility_probe.py`` -- gathering a **host-staged constant** of the
#:   identical shape is reproducible *and exactly correct* from 32 to 8192 rows,
#:   both with ``all_gather_async`` and with the composite ``ttnn.all_gather``,
#:   standalone and inside the built model.  So the collective is not broken and the
#:   memory context is not the cause: the embedding's own output is a bad gather
#:   input.
#:
#: What is reproducible over 25 repeats per arm (``embedding_gather_fix.json``):
#: a *freshly allocated* buffer, gathered at no more than 1024 rows.  A single
#: 4096-row gather fails through every variant tried, including a clone; 1024 and
#: 128 pass with one.  Hence chunk-and-clone at this width rather than one dispatch.
#:
#: This is a mitigation with a measured envelope, not a root cause.  ``ttnn.embedding``
#: returns rank 3 whatever its input rank, so the ``unsqueeze_to_4D`` view that
#: distinguishes this input from a staged one cannot simply be avoided; that is the
#: remaining lead, and it belongs upstream in TTNN rather than here.
EMBED_GATHER_CHUNK_ROWS = 1024

#: Global semaphores for this module's async all-gather, per open mesh.  Two plus a
#: barrier, created once and shared by every shape, mirroring
#: ``MultichipDecoder._ccl_semaphores`` (which is where the ``L1_SMALL`` budget is
#: documented).  They are safe to share across shapes *and* with the layer stack
#: because every collective in a forward pass consumes the previous one's output:
#: the embedding gather's result is layer 0's input.
_MODEL_CCL_SEMAPHORES: dict[int, dict] = {}

#: Built generators, keyed by (mesh, config).  Lives here rather than in
#: ``tt/generator.py`` because the readiness runners load that file by path under a
#: synthetic module name -- a dict there would be a second copy, and sharing one
#: build between the driver and the runners is the entire point.  See
#: ``build_generator``.
GENERATOR_CACHE: dict[tuple, Any] = {}

#: Rows the terminal path runs at.  ``nlp_create_qkv_heads_decode`` caps decode at
#: 32 users and the DRAM-sharded matmul requires exactly one M tile, so both the
#: decode step and the prefill last-token slice are one tile row.
TERMINAL_ROWS = TILE_SIZE

#: Physical width of every decode-step tensor: the token buffer, the two position
#: tensors, the page table's row count and the sampler's parameter/seed rows.
#:
#: It is **not** the batch size and it is deliberately independent of it.  The
#: decode step is one tile row whatever the batch is (the activation is tile-padded
#: and ``nlp_create_qkv_heads_decode`` caps ``num_users`` at 32), ``ttnn.sampling``
#: requires exactly a ``[1, 1, 1, 32]`` preallocated output, and ``TTSampling``
#: floors its own batch to a multiple of 32.  So decode always runs 32 rows and
#: inactive rows carry ``current_pos = -1``, the sentinel both
#: ``paged_update_cache`` and ``paged_scaled_dot_product_attention_decode`` skip
#: and ``plus_one(skip_negative_entries=True)`` preserves.  ``max_batch_size`` is a
#: separate thing: the number of *cache slots*, which is what the paged pool is
#: sized for.
DECODE_ROWS = 32


def padded_vocab_size(vocab_size: int, tp: int, *, tile: int = TILE_SIZE, cores: int | None = None) -> int:
    """Total vocab width whose per-device shard is legal for the LM-head matmul.

    Two constraints, and the second is the binding one:

    * every device's shard must be **tile-aligned**, or the gather that
      reconstructs the vocab leaves a gap per device and every token id on device
      *d* is shifted by ``d * (tile - shard % tile)``.  ``202048 / 4 = 50512`` is
      *not* a multiple of 32, so this alone already forces padding;
    * ``dram_sharded_weight_memcfg`` pads the per-device width up to
      ``tile * dram_banks`` so there is one shard per DRAM bank, which the
      DRAM-sharded matmul requires of ``input_tensor_b``.  A weight whose real
      width is below that pad would be described by a shard spec wider than the
      tensor, so the pad has to be materialised in the weight instead.

    With ``cores=None`` only the tile constraint applies (202112); with the 8-bank
    DRAM grid it is 202752, i.e. 704 padded ids, which the sampler masks because it
    is given the real vocab size alongside this one.
    """
    step = tile * tp if cores is None else tile * cores * tp
    return int(math.ceil(vocab_size / step) * step)


def build_rope_cache(
    mesh_device: ttnn.MeshDevice,
    text_config: Any,
    *,
    max_seq_len: int,
    plan: MeshPlan,
) -> dict[str, ttnn.Tensor]:
    """The four RoPE tables, built once for the whole stack.

    Every sliding layer in this checkpoint has the same ``layer_rope_theta``
    (500000.0; the 13 full-attention layers are NoPE and use no table at all), so
    one set of tables serves all 39 of them.  Built per layer they would be
    134 MB x 39 = 5.2 GB of device DRAM holding 39 copies of one tensor, which is
    more than the entire 52-layer weight footprint.  The uniform-theta assumption
    is *checked* here rather than assumed.
    """
    thetas = {
        float(text_config.layer_rope_theta[idx])
        for idx in range(text_config.num_hidden_layers)
        if resolve_layer_kind(text_config, idx) == LAYER_KIND_SLIDING
    }
    if len(thetas) != 1:
        raise ValueError(
            f"a shared RoPE cache needs one theta across the sliding layers, got {sorted(thetas)}; "
            "build the tables per layer instead (pass rope_cache=None)"
        )
    theta = thetas.pop()
    cos, sin = _rope_cos_sin(max_seq_len, plan.head_dim, theta)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_mesh(tensor: torch.Tensor, *, layout):
        return ttnn.from_torch(
            tensor,
            device=mesh_device,
            layout=layout,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

    return {
        "theta": theta,
        # ROW_MAJOR 2-D for the decode-time per-user ttnn.embedding gather.
        "cos": to_mesh(cos.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT),
        "sin": to_mesh(sin.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT),
        # TILE 4-D for the prefill rotary_embedding_hf slice.
        "cos_tile": to_mesh(cos.to(torch.bfloat16).reshape(1, 1, max_seq_len, plan.head_dim), layout=ttnn.TILE_LAYOUT),
        "sin_tile": to_mesh(sin.to(torch.bfloat16).reshape(1, 1, max_seq_len, plan.head_dim), layout=ttnn.TILE_LAYOUT),
    }


@dataclass(frozen=True)
class ModelConfig:
    """Everything the runtime path needs that is not a tensor."""

    hidden_size: int
    vocab_size: int
    padded_vocab_size: int
    num_hidden_layers: int
    layer_indices: tuple[int, ...]
    layer_kinds: tuple[str, ...]
    max_seq_len: int
    max_batch_size: int
    page_block_size: int
    max_num_blocks: int
    prefill_chunk_size: int
    sliding_window: int
    rms_norm_eps: float
    final_logit_softcapping: float
    output_multiplier: float
    tp: int
    eos_token_id: int | tuple[int, ...]
    bos_token_id: int | None

    @property
    def decode_rows(self) -> int:
        return DECODE_ROWS

    @property
    def blocks_per_seq(self) -> int:
        return (self.max_seq_len + self.page_block_size - 1) // self.page_block_size

    @property
    def local_vocab_size(self) -> int:
        return self.padded_vocab_size // self.tp

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)


# --------------------------------------------------------------------- weights


def weights_snapshot_dir(model_id: str = HF_MODEL_ID) -> pathlib.Path:
    """Cache snapshot that actually holds the safetensors shards.

    ``refs/main`` for this repo points at a **metadata-only** revision -- config,
    tokenizer and the weight index, but no shards -- so neither the default
    revision nor "the first snapshot with an index" is a safe answer.  A snapshot
    only counts here when every shard its own index names is present, which is
    the condition the loader actually needs.  Picking the wrong one fails late,
    inside ``safe_open``, with a missing-file error that reads like a broken cache
    rather than a resolution bug.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = pathlib.Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    indexes = sorted(repo.glob("snapshots/*/model.safetensors.index.json"))
    if not indexes:
        raise FileNotFoundError(f"no cached safetensors index for {model_id} under {repo}")
    incomplete = []
    for index_path in indexes:
        snapshot = index_path.parent
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        missing = sorted(shard for shard in shards if not (snapshot / shard).exists())
        if not missing:
            return snapshot
        incomplete.append(f"{snapshot.name} (missing {len(missing)}/{len(shards)} shards)")
    raise FileNotFoundError(
        f"no cached snapshot of {model_id} under {repo} holds all its safetensors shards: " + ", ".join(incomplete)
    )


class LazyCheckpoint:
    """Per-tensor safetensors reader, so 56 GB never lands in host RAM at once.

    ``from_pretrained`` builds one layer at a time and this hands back exactly the
    keys that layer asks for; the peak host cost is one layer's FP32 working set
    (~1.9 GB) rather than the whole checkpoint.
    """

    def __init__(self, snapshot: pathlib.Path) -> None:
        self.snapshot = snapshot
        index_path = snapshot / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        self._handles: dict[str, Any] = {}

    def _handle(self, shard: str):
        from safetensors import safe_open

        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.snapshot / shard), framework="pt")
            self._handles[shard] = handle
        return handle

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"{key} is not in {self.snapshot / 'model.safetensors.index.json'}")
        return self._handle(self.weight_map[key]).get_tensor(key)

    def layer_state_dict(self, layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f"{TEXT_PREFIX}.layers.{layer_idx}."
        return {key: self.get(key) for key in self.weight_map if key.startswith(prefix)}

    def close(self) -> None:
        self._handles.clear()


# ------------------------------------------------------------------- submodules


class _TerminalNorm(LightweightModule):
    """The model-level RMSNorm, in both the interleaved and width-sharded forms.

    Two of these exist and they are *not* the same math as the decoder's four:

    * the embedding norm is ``MuseGlimmerRMSNorm(with_scale=False)`` -- no weight
      at all.  It is given a weight of ones rather than ``weight=None`` so the
      op contract is identical in both forms and there is no second kernel path;
    * the final norm is ``MuseGlimmerRMSNorm(with_scale=True)``, which multiplies
      by ``w`` -- **not** by ``1 + w`` the way the decoder's centered norms do.
      The checkpoint's ``norm.weight`` is O(3), so folding a ``+1`` in here would
      be a ~30 % error on every channel rather than a subtle one.
    """

    def __init__(self, weight: ttnn.Tensor, weight_rm: ttnn.Tensor, eps: float, compute_kernel_config: Any) -> None:
        super().__init__()
        self.weight = weight
        self.weight_rm = weight_rm
        self.eps = eps
        self.compute_kernel_config = compute_kernel_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

    def sharded_forward(self, x_sharded: ttnn.Tensor, program_config, memory_config) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x_sharded,
            weight=self.weight_rm,
            epsilon=self.eps,
            program_config=program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )


class _LMHead(LightweightModule):
    """Column-parallel vocab projection plus the tanh logit softcap.

    HF computes ``T * tanh(lm_head(h) * m / T)`` with ``m = output_multiplier``
    and ``T = final_logit_softcapping``.  The ``m / T`` factor is folded into the
    weight at setup, so the runtime path is one matmul, one ``tanh`` and one
    scalar ``mul``.  Folding is exact in spirit for a block-float weight: scaling
    every element by a constant moves the per-block shared exponent and leaves the
    mantissas alone, so it costs no precision and saves an op on the hot path.

    The trailing ``* T`` is kept even though it cannot change an argmax or a
    top-k *ordering*: it is what makes the returned values real logits, which
    temperature, top-p and any later log-prob path all depend on.
    """

    def __init__(
        self,
        weight: ttnn.Tensor,
        *,
        local_vocab_size: int,
        hidden_size: int,
        softcap: float,
        matmul: str,
        cores: int,
        in0_block_w: int,
        device_grid: ttnn.CoreCoord,
        compute_kernel_config: Any,
        output_dtype: ttnn.DataType = LM_HEAD_OUTPUT_DTYPE,
        softcap_in_l1: bool | None = None,
    ) -> None:
        super().__init__()
        self.output_dtype = output_dtype
        # Read at construction rather than bound as a default argument, so an A/B
        # harness can flip the module constant between builds.
        self.softcap_in_l1 = LM_HEAD_SOFTCAP_IN_L1 if softcap_in_l1 is None else softcap_in_l1
        if matmul not in ("dram_sharded", "mcast1d"):
            raise ValueError(f"lm_head matmul must be 'dram_sharded' or 'mcast1d', got {matmul!r}")
        self.weight = weight
        self.local_vocab_size = local_vocab_size
        self.hidden_size = hidden_size
        self.softcap = softcap
        self.matmul = matmul
        self.cores = cores
        self.in0_block_w = in0_block_w
        self.device_grid = device_grid
        self.compute_kernel_config = compute_kernel_config
        if matmul == "dram_sharded":
            if (hidden_size // TILE_SIZE) % cores:
                raise ValueError(
                    f"the DRAM-sharded matmul needs K_tiles ({hidden_size // TILE_SIZE}) divisible by "
                    f"cores ({cores}); legal values are its divisors that fit the grid"
                )
            self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=math.ceil(local_vocab_size / (TILE_SIZE * cores)),
            )
            self.input_memcfg = width_sharded_l1(TERMINAL_ROWS, hidden_size, cores, device_grid)
            self.output_memcfg = width_sharded_l1(TERMINAL_ROWS, local_vocab_size, cores, device_grid)
        else:
            per_core_n = math.ceil(local_vocab_size / TILE_SIZE / (device_grid.x * device_grid.y))
            out_subblock_w = min(per_core_n, 4)
            while out_subblock_w > 1 and per_core_n % out_subblock_w:
                out_subblock_w -= 1
            self.program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(device_grid.x, device_grid.y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            self.input_memcfg = ttnn.DRAM_MEMORY_CONFIG
            self.output_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def _as_input(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, bool]:
        """Put ``hidden`` in the layout this matmul contract needs."""
        if self.matmul == "mcast1d":
            if hidden.is_sharded():
                return ttnn.sharded_to_interleaved(hidden, ttnn.DRAM_MEMORY_CONFIG), True
            return hidden, False
        if hidden.memory_config() == self.input_memcfg:
            return hidden, False
        if hidden.is_sharded():
            return ttnn.to_memory_config(hidden, self.input_memcfg), True
        return ttnn.interleaved_to_sharded(hidden, self.input_memcfg), True

    def forward(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, 32, hidden]`` -> vocab-sharded ``[1, 1, 32, local_vocab]``.

        Returns DRAM-interleaved bf16 TILE logits, which is what the sampler
        requires of its input and what its per-device index offsets assume the
        shard layout to be.
        """
        hidden_in, owned = self._as_input(hidden)
        logits = ttnn.linear(
            hidden_in,
            self.weight,
            dtype=self.output_dtype,
            memory_config=self.output_memcfg,
            program_config=self.program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        if owned:
            ttnn.deallocate(hidden_in)
        if not self.softcap_in_l1 and logits.is_sharded():
            interleaved = ttnn.sharded_to_interleaved(logits, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(logits)
            logits = interleaved
        memcfg = logits.memory_config()
        capped = ttnn.tanh(logits, memory_config=memcfg)
        ttnn.deallocate(logits)
        scaled = ttnn.multiply(capped, self.softcap, memory_config=memcfg)
        ttnn.deallocate(capped)
        if scaled.is_sharded():
            # The sampler's per-device index arithmetic assumes DRAM-interleaved
            # vocab shards, so the conversion is a contract rather than a choice.
            # It is one op here instead of one op plus two DRAM-interleaved
            # elementwise passes; see :data:`LM_HEAD_SOFTCAP_IN_L1`.
            interleaved = ttnn.sharded_to_interleaved(scaled, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(scaled)
            scaled = interleaved
        return scaled


# ----------------------------------------------------------------- the model


class MuseGlimmerModel(LightweightModule):
    """The full text model: embeddings, 52 decoder layers, final norm, LM head."""

    def __init__(
        self,
        *,
        config: ModelConfig,
        mesh_device: ttnn.MeshDevice,
        plan: MeshPlan,
        layers: list[OptimizedDecoder],
        embed_weight: ttnn.Tensor,
        embed_norm: _TerminalNorm,
        final_norm: _TerminalNorm,
        lm_head: _LMHead,
        rope_cache: dict[str, ttnn.Tensor] | None,
        precision: PrecisionPolicy,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.plan = plan
        self.layers = layers
        self.embed_weight = embed_weight
        self.embed_norm = embed_norm
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rope_cache = rope_cache
        self.precision = precision
        self.device_grid = mesh_device.compute_with_storage_grid_size()
        self.boundary_cores = layers[0].boundary_cores if layers else BOUNDARY_CORES
        #: Per-layer sliding-window K/V tails, for continuation prefill.  ``None``
        #: until a caller asks for them; see :meth:`prefill_forward`.
        self._sliding_tails: list[tuple[ttnn.Tensor, ttnn.Tensor] | None] | None = None
        self._decode_norm_cache: dict[int, tuple] = {}
        #: Host-side counters for the trace-loop audit; the generator reads them.
        self.counters: dict[str, int] = {}
        self.reset_counters()

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        *,
        model_id: str = HF_MODEL_ID,
        hf_config: Any = None,
        max_batch_size: int = 32,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        prefill_chunk_size: int | None = None,
        precision: PrecisionPolicy = DEFAULT_PRECISION,
        layer_indices: Sequence[int] | None = None,
        lm_head_dtype: ttnn.DataType = LM_HEAD_DTYPE,
        lm_head_matmul: str = LM_HEAD_MATMUL,
        lm_head_cores: int = LM_HEAD_CORES,
        lm_head_in0_block_w: int = LM_HEAD_IN0_BLOCK_W,
        lm_head_fidelity: ttnn.MathFidelity = LM_HEAD_FIDELITY,
        lm_head_fp32_acc: bool = LM_HEAD_FP32_ACC,
        lm_head_output_dtype: ttnn.DataType = LM_HEAD_OUTPUT_DTYPE,
        share_rope_cache: bool = True,
        checkpoint: LazyCheckpoint | None = None,
        **decoder_kwargs: Any,
    ) -> "MuseGlimmerModel":
        """Build the model on ``mesh_device`` from the cached HF checkpoint.

        ``layer_indices`` selects a subset of the real layers, keeping their real
        weights, kinds and shapes.  It exists for the reduced full-model probe the
        ``$full-model`` skill asks for (one layer of each kind, real terminal
        path) and must not be used for accuracy or performance evidence.
        """
        tp = int(mesh_device.get_num_devices())
        if tp not in SUPPORTED_TP:
            raise ValueError(
                f"MuseGlimmerModel supports P150/P150x2/P150x4 tensor parallelism "
                f"({SUPPORTED_TP} devices); got a {tp}-device mesh."
            )
        if hf_config is None:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        text_config = _text_config(hf_config)
        plan = mesh_plan(text_config, tp, dram_banks=mesh_device.dram_grid_size().x)

        max_seq_len = int(max_seq_len or text_config.max_position_embeddings)
        if prefill_chunk_size is None:
            prefill_chunk_size = min(8192, max(page_block_size, ((max_seq_len + 31) // 32) * 32))
            prefill_chunk_size = max(page_block_size, (prefill_chunk_size // page_block_size) * page_block_size)
        blocks_per_seq = (max_seq_len + page_block_size - 1) // page_block_size
        if max_num_blocks is None:
            max_num_blocks = max_batch_size * blocks_per_seq

        all_indices = tuple(range(text_config.num_hidden_layers))
        indices = tuple(int(i) for i in (layer_indices if layer_indices is not None else all_indices))
        for idx in indices:
            if idx not in all_indices:
                raise ValueError(f"layer index {idx} outside the checkpoint's {len(all_indices)} layers")

        vocab_size = int(text_config.vocab_size)
        padded_vocab = padded_vocab_size(
            vocab_size,
            tp,
            cores=mesh_device.dram_grid_size().x if lm_head_matmul == "dram_sharded" else None,
        )
        config = ModelConfig(
            hidden_size=int(text_config.hidden_size),
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab,
            num_hidden_layers=int(text_config.num_hidden_layers),
            layer_indices=indices,
            layer_kinds=tuple(resolve_layer_kind(text_config, idx) for idx in indices),
            max_seq_len=max_seq_len,
            max_batch_size=int(max_batch_size),
            page_block_size=int(page_block_size),
            max_num_blocks=int(max_num_blocks),
            prefill_chunk_size=int(prefill_chunk_size),
            sliding_window=int(text_config.sliding_window),
            rms_norm_eps=float(text_config.rms_norm_eps),
            final_logit_softcapping=float(text_config.final_logit_softcapping),
            output_multiplier=float(text_config.output_multiplier),
            tp=tp,
            eos_token_id=text_config.eos_token_id,
            bos_token_id=getattr(text_config, "bos_token_id", None),
        )

        owns_checkpoint = checkpoint is None
        checkpoint = checkpoint or LazyCheckpoint(weights_snapshot_dir(model_id))
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        norm_ck = norm_compute_kernel_config(mesh_device.arch())

        rope_cache = (
            build_rope_cache(mesh_device, text_config, max_seq_len=max_seq_len, plan=plan)
            if share_rope_cache and any(kind == LAYER_KIND_SLIDING for kind in config.layer_kinds)
            else None
        )

        try:
            # ---------------------------------------------------- embeddings
            embed = checkpoint.get(EMBED_KEY).to(torch.bfloat16)
            if tuple(embed.shape) != (vocab_size, config.hidden_size):
                raise ValueError(f"{EMBED_KEY} is {tuple(embed.shape)}, expected {(vocab_size, config.hidden_size)}")
            # See EMBED_PAD_ROWS: one zero row at index ``vocab_size``, used to pad
            # a prompt out to a tile boundary with rows that are exactly zero.
            embed = torch.cat([embed, torch.zeros(EMBED_PAD_ROWS, config.hidden_size, dtype=torch.bfloat16)], dim=0)
            embed_weight = ttnn.from_torch(
                embed,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=EMBED_DTYPE,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            del embed
            embed_norm = cls._build_norm(
                torch.ones(config.hidden_size, dtype=torch.float32),
                mesh_device=mesh_device,
                eps=config.rms_norm_eps,
                compute_kernel_config=norm_ck,
                replicate=replicate,
            )

            # --------------------------------------------------- final norm
            final_norm = cls._build_norm(
                checkpoint.get(FINAL_NORM_KEY).to(torch.float32),
                mesh_device=mesh_device,
                eps=config.rms_norm_eps,
                compute_kernel_config=norm_ck,
                replicate=replicate,
            )

            # ------------------------------------------------------ LM head
            lm_head = cls._build_lm_head(
                checkpoint,
                mesh_device=mesh_device,
                config=config,
                dtype=lm_head_dtype,
                matmul=lm_head_matmul,
                cores=lm_head_cores,
                in0_block_w=lm_head_in0_block_w,
                fidelity=lm_head_fidelity,
                fp32_acc=lm_head_fp32_acc,
                output_dtype=lm_head_output_dtype,
            )

            # ------------------------------------------------------- layers
            layers: list[OptimizedDecoder] = []
            decoder_class = OptimizedDecoder if tp == 1 else MultichipDecoder
            layer_decoder_kwargs = dict(decoder_kwargs)
            if tp == 1:
                for key in _MULTICHIP_ONLY_DECODER_KWARGS:
                    layer_decoder_kwargs.pop(key, None)
                # The full model has a stable 16-core boundary contract across
                # layers; retain it on P150 instead of round-tripping every layer
                # through DRAM interleaved form.
                layer_decoder_kwargs.setdefault("sharded_decode_io", True)
            for position, layer_idx in enumerate(indices):
                state_dict = checkpoint.layer_state_dict(layer_idx)
                layers.append(
                    decoder_class.from_state_dict(
                        state_dict,
                        hf_config=hf_config,
                        layer_idx=layer_idx,
                        mesh_device=mesh_device,
                        max_batch_size=max_batch_size,
                        max_seq_len=max_seq_len,
                        page_block_size=page_block_size,
                        max_num_blocks=max_num_blocks,
                        prefill_chunk_size=prefill_chunk_size,
                        # ``for_layer`` resolves the policy's layer exceptions --
                        # e.g. "BFP4 attention weights everywhere but the first
                        # and last layer".  It returns the policy unchanged when
                        # no exception names this index, so the common case is
                        # the same object every layer gets today.
                        precision=precision.for_layer(layer_idx),
                        rope_cache=rope_cache,
                        **layer_decoder_kwargs,
                    )
                )
                del state_dict
                if (position + 1) % 8 == 0 or position + 1 == len(indices):
                    logger.info(f"MuseGlimmerModel: built {position + 1}/{len(indices)} layers")
        finally:
            if owns_checkpoint:
                checkpoint.close()

        return cls(
            config=config,
            mesh_device=mesh_device,
            plan=plan,
            layers=layers,
            embed_weight=embed_weight,
            embed_norm=embed_norm,
            final_norm=final_norm,
            lm_head=lm_head,
            rope_cache=rope_cache,
            precision=precision,
        )

    @staticmethod
    def _build_norm(
        weight: torch.Tensor,
        *,
        mesh_device: ttnn.MeshDevice,
        eps: float,
        compute_kernel_config: Any,
        replicate: Any,
    ) -> _TerminalNorm:
        hidden = int(weight.numel())
        folded = weight.to(torch.bfloat16)
        tile = ttnn.from_torch(
            folded.reshape(1, 1, 1, hidden),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        row_major = ttnn.from_torch(
            folded.reshape(1, 1, hidden // TILE_SIZE, TILE_SIZE),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        return _TerminalNorm(tile, row_major, eps, compute_kernel_config)

    @staticmethod
    def _build_lm_head(
        checkpoint: LazyCheckpoint,
        *,
        mesh_device: ttnn.MeshDevice,
        config: ModelConfig,
        dtype: ttnn.DataType,
        matmul: str,
        cores: int,
        in0_block_w: int,
        fidelity: ttnn.MathFidelity = LM_HEAD_FIDELITY,
        fp32_acc: bool = LM_HEAD_FP32_ACC,
        output_dtype: ttnn.DataType = LM_HEAD_OUTPUT_DTYPE,
    ) -> _LMHead:
        # ``lm_head.weight`` and ``embed_tokens.weight`` are separate tensors in
        # this checkpoint (``text_config.tie_word_embeddings`` is False and the two
        # differ elementwise), so the tie in ``_tied_weights_keys`` is inactive and
        # the real head is loaded.  Falling back to the embedding would still
        # produce plausible text, so this is checked rather than assumed.
        if checkpoint.has(LM_HEAD_KEY):
            head = checkpoint.get(LM_HEAD_KEY)
            tied = False
        else:
            head = checkpoint.get(EMBED_KEY)
            tied = True
        head = head.to(torch.float32)
        if tuple(head.shape) != (config.vocab_size, config.hidden_size):
            raise ValueError(f"LM head is {tuple(head.shape)}, expected {(config.vocab_size, config.hidden_size)}")
        # HF: logits = T * tanh(h @ W.T * m / T).  Fold m/T into the weight.
        scale = config.output_multiplier / config.final_logit_softcapping
        padded = torch.zeros(1, 1, config.hidden_size, config.padded_vocab_size, dtype=torch.bfloat16)
        padded[0, 0, :, : config.vocab_size] = (head.transpose(0, 1) * scale).to(torch.bfloat16)
        del head
        weight_memcfg = (
            dram_sharded_weight_memcfg(config.hidden_size, config.local_vocab_size, mesh_device)
            if matmul == "dram_sharded"
            else ttnn.DRAM_MEMORY_CONFIG
        )
        weight = ttnn.from_torch(
            padded,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=weight_memcfg,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        del padded
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            # Defaults match the decoder's DRAM-sharded decode projections; the LM
            # head is the same op on the same weight dtype.
            math_fidelity=fidelity,
            math_approx_mode=fidelity == ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=not fp32_acc,
        )
        module = _LMHead(
            weight,
            local_vocab_size=config.local_vocab_size,
            hidden_size=config.hidden_size,
            softcap=config.final_logit_softcapping,
            matmul=matmul,
            cores=cores,
            in0_block_w=in0_block_w,
            output_dtype=output_dtype,
            device_grid=mesh_device.compute_with_storage_grid_size(),
            compute_kernel_config=compute_kernel_config,
        )
        module.tied_to_embedding = tied
        return module

    # -------------------------------------------------------------- teardown

    def deallocate(self) -> None:
        """Free the device tensors this model owns and drop its layers.

        Freeing the KV cache while a trace still holds its buffer addresses is a
        use-after-free, and on this fabric it reports as an ERISC assert rather than as a
        wrong number -- which is what this stage's watcher run caught in a test
        (``doc/optimized_full_model/logs/watcher_bisect_rebind.log``).  This function cannot
        release the traces itself: the generator owns them and the model does not know its
        generator.  So it *checks*, and says so loudly, because round 5 of the stage review
        was right that the shipped class silently permitted the exact ordering the stage
        had already been burned by.  Callers release first --
        ``MuseGlimmerGenerator.teardown()`` then ``model.deallocate()``.

        "Weights included" was not true and is not claimed: the per-layer weights go when
        ``self.layers`` is dropped and Python frees the wrappers.  What is explicit here is
        the KV cache and the sliding tails.
        """
        if self.live_traces_over_kv_cache:
            logger.warning(
                "MuseGlimmerModel.deallocate(): a trace is still registered over this model's KV "
                "cache. Call the generator's teardown() first; freeing the cache under a live trace "
                "is a use-after-free that reports as a fabric ERISC assert."
            )
        for layer in self.layers:
            for tensor in (layer.k_cache, layer.v_cache):
                if tensor is not None:
                    ttnn.deallocate(tensor)
        self.release_sliding_tails()
        self.layers = []

    # ------------------------------------------------------------ properties

    @property
    def kv_cache(self) -> list[list[ttnn.Tensor]]:
        """``[[k, v], ...]`` per layer, in the readiness/vLLM cache-handle shape."""
        return [[layer.k_cache, layer.v_cache] for layer in self.layers]

    def set_kv_cache(self, kv_cache: Sequence[Sequence[ttnn.Tensor]] | None) -> None:
        """Bind an externally owned cache, or restore nothing when ``None``.

        Standalone generation lets each layer keep the cache it allocated; a
        serving caller owns the cache and threads it through every call, so both
        have to work.  The shapes are checked because a silently mismatched cache
        reads zeros rather than failing.
        """
        if kv_cache is None:
            return
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} entries for {len(self.layers)} layers")
        # Validate **every** layer before binding any of them.  Round 9 of the stage review
        # found the interleaved form: a mismatch at layer i left layers 0..i-1 already rebound
        # and raised, and because the caller's ``_invalidate_traces_if_cache_moved`` runs after
        # this returns, the exception skipped it -- leaving a half-rebound cache under traces
        # whose recorded signature still described the old one.  Two passes make the failure
        # atomic: either the whole cache is bound or none of it is.
        for layer, pair in zip(self.layers, kv_cache):
            k, v = pair
            for name, tensor, current in (("k", k, layer.k_cache), ("v", v, layer.v_cache)):
                if tuple(tensor.shape) != tuple(current.shape):
                    raise ValueError(
                        f"external {name} cache for layer {layer.config.layer_idx} is {tuple(tensor.shape)}, "
                        f"expected {tuple(current.shape)}"
                    )
        for layer, pair in zip(self.layers, kv_cache):
            layer.k_cache, layer.v_cache = pair

    def adopt_external_kv_cache(
        self,
        kv_cache: Sequence[Sequence[ttnn.Tensor]],
        *,
        cache_slots: int | None = None,
        free_existing: bool = True,
    ) -> int:
        """Bind a serving-owned paged pool whose **block count** differs from the built one.

        :meth:`set_kv_cache` is the readiness contract's rebind: it requires the external
        buffers to have exactly the shape this model allocated, which is the right check for
        a caller that hands back a cache of the same geometry.  A serving caller is a
        different case, and the difference is not cosmetic.  The standalone build sizes the
        pool as ``max_batch_size x blocks_per_seq`` -- every user simultaneously at the full
        advertised context -- while vLLM owns one *shared* pool sized by its own token
        budget and hands out block ids from it.  Those two numbers are unrelated, and at
        this model's geometry the standalone rule is not even satisfiable for a serving
        batch (32 users x 2048 blocks x 905,216 B/block is 59 GB against 31.5 GiB).

        So this method checks everything that makes the cache *interpretable* by the paged
        ops -- rank, local KV head count, block size, head dim and dtype -- and lets the
        block count be whatever the owner allocated.  It then updates the model and layer
        configs so ``normalize_page_table``'s bounds check and ``dram_report`` describe the
        pool that is actually bound.  ``max_num_blocks`` is a construction-time input to the
        cache shape only; the runtime path reads ``block_size`` from the same config and the
        block ids from the page table, so nothing else has to move.

        ``cache_slots`` raises the number of *request slots* the model will accept, which
        is the other half of the same contract: the build-time pool is sized for one
        sequence, so the build also has ``max_batch_size = 1``, and the layer's
        ``user_id >= max_batch_size`` guard would then reject every serving request past
        the first.  It is a bound on the page table's row index and on nothing else -- the
        decode tensors are 32 rows wide regardless -- so it is safe to state it here, once
        the pool that has to back those slots is the one being adopted.

        ``free_existing`` releases the buffers this model allocated at build time.  A
        serving process wants that -- they are dead weight the moment the external pool is
        bound -- and it is only safe here because a rebind happens before any trace is
        captured over them; ``deallocate()``'s live-trace warning covers the other order.

        Returns the adopted block count.
        """
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} entries for {len(self.layers)} layers")
        expected_heads = self.plan.local_kv_heads
        expected_block = self.config.page_block_size
        expected_head_dim = self.plan.head_dim
        expected_dtype = self.precision.kv_cache_dtype
        blocks: set[int] = set()
        for layer, pair in zip(self.layers, kv_cache):
            if len(pair) != 2:
                raise ValueError(f"layer {layer.config.layer_idx} cache entry must be (k, v), got {len(pair)} tensors")
            for name, tensor in (("k", pair[0]), ("v", pair[1])):
                shape = tuple(tensor.shape)
                if len(shape) != 4:
                    raise ValueError(
                        f"external {name} cache for layer {layer.config.layer_idx} must be rank 4, got {shape}"
                    )
                if shape[1:] != (expected_heads, expected_block, expected_head_dim):
                    raise ValueError(
                        f"external {name} cache for layer {layer.config.layer_idx} is {shape}; this model needs "
                        f"(num_blocks, {expected_heads}, {expected_block}, {expected_head_dim}) -- one local KV head "
                        "per device, the model's page block size and its head dim"
                    )
                if tensor.dtype != expected_dtype:
                    raise ValueError(
                        f"external {name} cache for layer {layer.config.layer_idx} is {tensor.dtype}; the selected "
                        f"precision policy's kv_cache_dtype is {expected_dtype}"
                    )
                blocks.add(int(shape[0]))
        if len(blocks) != 1:
            raise ValueError(f"every layer's external cache must hold the same number of blocks, got {sorted(blocks)}")
        num_blocks = blocks.pop()
        if num_blocks < self.config.blocks_per_seq:
            raise ValueError(
                f"the external cache holds {num_blocks} blocks, which cannot hold one sequence at the supported "
                f"context ({self.config.blocks_per_seq} blocks of {expected_block} tokens)"
            )
        slots = self.config.max_batch_size if cache_slots is None else int(cache_slots)
        if slots < 1 or slots > DECODE_ROWS:
            raise ValueError(f"cache_slots must be within 1..{DECODE_ROWS} decode rows, got {slots}")
        if free_existing:
            for layer in self.layers:
                for tensor in (layer.k_cache, layer.v_cache):
                    if tensor is not None:
                        try:
                            ttnn.deallocate(tensor)
                        except Exception:  # noqa: BLE001 -- an already-freed buffer is not a failure here
                            pass
        for layer, pair in zip(self.layers, kv_cache):
            layer.k_cache, layer.v_cache = pair[0], pair[1]
            layer.config = dataclass_replace(
                layer.config,
                max_batch_size=slots,
                paged_attention_config=dataclass_replace(
                    layer.config.paged_attention_config, max_num_blocks=num_blocks
                ),
            )
        self.config = dataclass_replace(self.config, max_num_blocks=num_blocks, max_batch_size=slots)
        logger.info(
            f"MuseGlimmerModel: adopted an externally owned paged KV cache of {num_blocks} blocks "
            f"({num_blocks * expected_block} tokens across all users) for {slots} request slot(s); "
            f"the build-time pool was {'freed' if free_existing else 'retained'}."
        )
        return num_blocks

    def reset_counters(self) -> None:
        self.counters = {
            "trace_replays": 0,
            "token_refreshes": 0,
            "position_refreshes": 0,
            "page_table_refreshes": 0,
            "synchronizations": 0,
            "readbacks": 0,
            "device_position_advances": 0,
        }

    #: How many traces are currently captured over this model's KV cache.  The generator
    #: owns the traces and the model owns the cache, so neither can enforce the ordering
    #: alone; this is the one bit of shared state that lets :meth:`deallocate` refuse to be
    #: silent about a use-after-free.  Maintained by
    #: ``MuseGlimmerGenerator._capture_*``/``_release_*``.
    @property
    def live_traces_over_kv_cache(self) -> int:
        return int(getattr(self, "_live_traces_over_kv_cache", 0))

    def note_trace_captured(self) -> None:
        self._live_traces_over_kv_cache = self.live_traces_over_kv_cache + 1

    def note_trace_released(self) -> None:
        self._live_traces_over_kv_cache = max(0, self.live_traces_over_kv_cache - 1)

    def reset_kv_cache(self) -> None:
        """Zero the paged cache in place, without freeing it (``Generator.reset``).

        In place because the contract says ``reset()`` must not free device
        buffers: a serving caller may hold the same handles.  One zeros tensor is
        allocated and reused across all 104 cache tensors -- every layer's cache has
        the same shape -- rather than one per tensor.
        """
        zeros: ttnn.Tensor | None = None
        try:
            for layer in self.layers:
                for cache in (layer.k_cache, layer.v_cache):
                    if zeros is None or tuple(zeros.shape) != tuple(cache.shape) or zeros.dtype != cache.dtype:
                        if zeros is not None:
                            ttnn.deallocate(zeros)
                        zeros = ttnn.zeros(
                            cache.shape,
                            dtype=cache.dtype,
                            layout=cache.layout,
                            device=self.mesh_device,
                            memory_config=cache.memory_config(),
                        )
                    ttnn.copy(zeros, cache)
        finally:
            if zeros is not None:
                ttnn.deallocate(zeros)
        self.release_sliding_tails()

    def release_sliding_tails(self) -> None:
        if not self._sliding_tails:
            self._sliding_tails = None
            return
        for tail in self._sliding_tails:
            if tail is not None:
                ttnn.deallocate(tail[0])
                ttnn.deallocate(tail[1])
        self._sliding_tails = None

    # ------------------------------------------------------------------- CCL

    def _ccl_semaphores(self) -> dict:
        key = id(self.mesh_device)
        sems = _MODEL_CCL_SEMAPHORES.get(key)
        if sems is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

            def sem():
                return ttnn.create_global_semaphore(self.mesh_device, crs, 0, ttnn.BufferType.L1_SMALL)

            sems = {"ag": [sem() for _ in range(2)], "ag_barrier": sem()}
            _MODEL_CCL_SEMAPHORES[key] = sems
        return sems

    def _all_gather_async(
        self, tensor: ttnn.Tensor, *, dim: int = 3, memory_config: ttnn.MemoryConfig | None = None
    ) -> ttnn.Tensor:
        """All-gather with semaphores this model owns rather than per-program ones.

        ``ttnn.all_gather`` leaves one global semaphore in ``L1_SMALL`` per
        distinct program for the life of the program cache, and the embedding
        gather has one program per prompt length -- a test session would exhaust
        the 6144 B region.  These three are created once per mesh.
        """
        if self.config.tp == 1:
            # A one-device "gather" is identity, but the decode embedding asks
            # the collective to write directly into the boundary L1 layout.  Keep
            # that output-layout contract on P150 with an ordinary conversion.
            if memory_config is None or tensor.memory_config() == memory_config:
                return tensor
            converted = ttnn.to_memory_config(tensor, memory_config)
            ttnn.deallocate(tensor)
            return converted
        sems = self._ccl_semaphores()
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=sems["ag"],
            barrier_semaphore=sems["ag_barrier"],
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
        )
        ttnn.deallocate(tensor)
        return gathered

    # ------------------------------------------------------------ embeddings

    def _embed(self, tokens: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor:
        """``[1, n]`` uint32 token ids -> replicated ``[1, 1, n, hidden]`` bf16.

        Each device looks its own quarter of the hidden dimension up locally, so
        the gather moves ``n x 1664`` per device instead of the whole table being
        replicated (2.7 GB/device against 672 MB, both decimal).

        Above a decode step's single tile row the gather is **chunked into freshly
        allocated buffers** rather than issued once over the embedding's own output.
        That is not an optimisation; it is what makes prefill reproducible.  See
        :data:`EMBED_GATHER_CHUNK_ROWS` for the measurement.
        """
        local = ttnn.embedding(tokens, self.embed_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        local4 = ttnn.unsqueeze_to_4D(local)
        if local4 is not local:
            ttnn.deallocate(local)

        rows = int(local4.shape[-2])
        width = int(local4.shape[-1])
        # A decode step is one tile row, where the gather is reproducible as issued
        # and the step is traced and at its latency floor: leave it exactly alone.
        if rows <= TERMINAL_ROWS:
            return self._all_gather_async(local4, memory_config=memory_config)

        pieces: list[ttnn.Tensor] = []
        for offset in range(0, rows, EMBED_GATHER_CHUNK_ROWS):
            length = min(EMBED_GATHER_CHUNK_ROWS, rows - offset)
            if length == rows:
                # ``ttnn.slice`` hands back the input itself when the range covers
                # the whole tensor, which would defeat the point; clone instead.
                piece = ttnn.clone(local4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                piece = ttnn.slice(local4, [0, 0, offset, 0], [1, 1, offset + length, width])
            pieces.append(self._all_gather_async(piece))
        ttnn.deallocate(local4)
        if len(pieces) == 1:
            return pieces[0]
        gathered = ttnn.concat(pieces, dim=2)
        for piece in pieces:
            ttnn.deallocate(piece)
        return gathered

    @property
    def embed_pad_id(self) -> int:
        """Token id whose embedding row is exactly zero; see :data:`EMBED_PAD_ROWS`."""
        return self.config.vocab_size

    def prefill_tokens_to_device(self, token_ids: Sequence[int], *, device: bool = True) -> tuple[ttnn.Tensor, int]:
        """``(ids, padded_len)`` for a prompt of any logical length.

        The ids are padded up to a tile boundary with :attr:`embed_pad_id`, so the
        embedded tensor has no uninitialised rows and the whole prefill is
        reproducible.  The generator owns what that padding then means -- the
        junk-free K/V it writes past the logical length is never read, because
        decode starts at ``cur_pos = logical length``.
        """
        length = len(token_ids)
        padded_len = ((length + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        ids = torch.full((1, padded_len), self.embed_pad_id, dtype=torch.int32)
        ids[0, :length] = torch.tensor(list(token_ids), dtype=torch.int32)
        kwargs: dict[str, Any] = {
            "layout": ttnn.ROW_MAJOR_LAYOUT,
            "dtype": ttnn.uint32,
            "mesh_mapper": ttnn.ReplicateTensorToMesh(self.mesh_device),
        }
        if device:
            # ``device=False`` returns the host form, which is what a traced prefill
            # copies into its persistent token input before replay.
            kwargs["device"] = self.mesh_device
            kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        return ttnn.from_torch(ids, **kwargs), padded_len

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Prefill embeddings: DRAM-interleaved, which is layer 0's prefill input."""
        embedded = self._embed(tokens)
        normed = self.embed_norm.forward(embedded)
        ttnn.deallocate(embedded)
        return normed

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Decode embeddings, already in the decoder's boundary layout.

        The embedding norm runs *sharded* here for the same reason the decoder's
        four decode norms do: the interleaved RMSNorm kernel parallelises over
        rows and a decode step is one tile row, so it would land on a single core.
        Handing layer 0 the boundary layout also skips its entry
        ``interleaved_to_sharded``.
        """
        program_config, memory_config = self._decode_norm_configs(TERMINAL_ROWS)
        if EMBED_DECODE_GATHER_SHARDED:
            # The gather's own output layout *is* the boundary spec, so the
            # ``interleaved_to_sharded`` the DRAM-interleaved form needed is gone.
            sharded = self._embed(tokens, memory_config=memory_config)
        else:
            embedded = self._embed(tokens)
            sharded = ttnn.interleaved_to_sharded(embedded, memory_config)
            ttnn.deallocate(embedded)
        normed = self.embed_norm.sharded_forward(sharded, program_config, memory_config)
        ttnn.deallocate(sharded)
        return normed

    def _decode_norm_configs(self, rows: int):
        """The decoder's boundary norm configs, for the two model-level norms.

        Delegated to the layer rather than recomputed so the terminal norms
        consume and produce *exactly* the inter-layer residual contract; a second
        derivation of the same spec is how that contract drifts.
        """
        cached = self._decode_norm_cache.get(rows)
        if cached is None:
            cached = self.layers[0]._decode_norm_configs(rows)
            self._decode_norm_cache[rows] = cached
        return cached

    def boundary_memcfg(self, rows: int = TERMINAL_ROWS) -> ttnn.MemoryConfig:
        return self._decode_norm_configs(rows)[1]

    # --------------------------------------------------------------- prefill

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        user_id: int = 0,
        start_pos: int = 0,
        continuation: bool = False,
        keep_sliding_tails: bool = False,
    ) -> ttnn.Tensor:
        """Run the layer stack over a prompt; returns the final hidden state.

        ``continuation`` threads each sliding layer's K/V window from the previous
        call, which the paged chunked SDPA op cannot recover from the cache
        because it has no sliding-window mask (see
        ``FunctionalDecoder.prefill_forward``).  The tails are per layer and owned
        by this model, so a caller doing chunked prefill only has to say whether
        this call continues the last one.
        """
        if continuation and start_pos == 0:
            raise ValueError("continuation prefill needs start_pos > 0")
        if continuation and self._sliding_tails is None:
            raise ValueError(
                "continuation prefill needs the previous call's sliding K/V tails; "
                "pass keep_sliding_tails=True to the call that precedes it"
            )
        if not continuation:
            self.release_sliding_tails()
        tails = self._sliding_tails or [None] * len(self.layers)
        next_tails: list[tuple[ttnn.Tensor, ttnn.Tensor] | None] = [None] * len(self.layers)

        hidden = hidden_states
        for position, layer in enumerate(self.layers):
            sliding = layer.config.is_sliding
            want_tail = keep_sliding_tails and sliding
            result = layer.prefill_forward(
                hidden,
                page_table=page_table,
                user_id=user_id,
                start_pos=start_pos,
                sliding_kv_tail=tails[position] if sliding else None,
                return_sliding_kv_tail=want_tail,
            )
            if want_tail:
                out, next_tails[position] = result
            else:
                out = result
            if tails[position] is not None:
                # Consumed by the layer's first internal chunk; the layer does not
                # free a tail it was handed.
                ttnn.deallocate(tails[position][0])
                ttnn.deallocate(tails[position][1])
                tails[position] = None
            ttnn.deallocate(hidden)
            hidden = out
        self._sliding_tails = next_tails if keep_sliding_tails else None
        return hidden

    def prefill_logits(self, hidden: ttnn.Tensor, *, last_token_index: int) -> ttnn.Tensor:
        """Vocab-sharded logits for one prompt position, from the prefill hidden state.

        The LM head runs on the single tile row that holds ``last_token_index``,
        not on the prompt: at 202752 vocab columns a whole-prompt projection is
        the largest matmul in the model and nothing in a token-out path needs it.
        """
        row = self._slice_rows(hidden, last_token_index)
        normed = self.final_norm.forward(row)
        ttnn.deallocate(row)
        logits = self.lm_head.forward(normed)
        ttnn.deallocate(normed)
        return logits

    def prefill_all_logits(self, hidden: ttnn.Tensor, *, prompt_len: int) -> list[ttnn.Tensor]:
        """Vocab-sharded logits for every prompt position, one tile row at a time.

        Only the readiness prefill check needs this.  It is deliberately a list of
        32-row tensors rather than one ``[1, 1, S, vocab]`` tensor: the whole thing
        is ``S x 202752`` and the DRAM-sharded LM-head matmul requires exactly one
        M tile anyway.
        """
        outputs = []
        for offset in range(0, prompt_len, TILE_SIZE):
            row = self._slice_rows(hidden, offset, aligned=True)
            normed = self.final_norm.forward(row)
            ttnn.deallocate(row)
            outputs.append(self.lm_head.forward(normed))
            ttnn.deallocate(normed)
        return outputs

    def _slice_rows(self, hidden: ttnn.Tensor, index: int, *, aligned: bool = False) -> ttnn.Tensor:
        """The tile row of ``hidden`` containing row ``index``, as ``[1, 1, 32, hidden]``.

        ``aligned`` means ``index`` is already the tile-row start.  Slicing on a
        tile boundary is what keeps this one op: an unaligned slice of a TILE
        tensor is a gather.
        """
        rows = int(hidden.shape[-2])
        start = index if aligned else (index // TILE_SIZE) * TILE_SIZE
        end = min(start + TILE_SIZE, rows)
        if start == 0 and end == rows and rows == TILE_SIZE:
            return ttnn.clone(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sliced = ttnn.slice(hidden, [0, 0, start, 0], [1, 1, end, self.config.hidden_size])
        if end - start == TILE_SIZE:
            return sliced
        padded = ttnn.pad(sliced, [(0, 0), (0, 0), (0, TILE_SIZE - (end - start)), (0, 0)], value=0.0)
        ttnn.deallocate(sliced)
        return padded

    def row_within_tile(self, index: int) -> int:
        """Where row ``index`` lands inside the tile row :meth:`_slice_rows` returns."""
        return index % TILE_SIZE

    # ---------------------------------------------------------------- decode

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        rope_pos_ids: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """One paged decode step through the whole stack; returns the final hidden.

        The residual crosses every layer boundary width-sharded in L1 on the
        16-core boundary grid -- no conversion, no gather, no all-reduce between
        layers.  The layer does not free a sharded input it was handed, so this
        method owns and frees each intermediate.
        """
        hidden = hidden_states
        for layer in self.layers:
            out = layer.decode_forward(
                hidden,
                current_pos=current_pos,
                page_table=page_table,
                rope_pos_ids=rope_pos_ids if layer.config.uses_rope else None,
            )
            ttnn.deallocate(hidden)
            hidden = out
        return hidden

    def decode_logits(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Terminal norm + LM head on the decode boundary layout."""
        program_config, memory_config = self._decode_norm_configs(TERMINAL_ROWS)
        if hidden.is_sharded() and hidden.memory_config() == memory_config:
            normed = self.final_norm.sharded_forward(hidden, program_config, memory_config)
        else:
            sharded = ttnn.interleaved_to_sharded(hidden, memory_config)
            normed = self.final_norm.sharded_forward(sharded, program_config, memory_config)
            ttnn.deallocate(sharded)
        ttnn.deallocate(hidden)
        logits = self.lm_head.forward(normed)
        ttnn.deallocate(normed)
        return logits

    def ttnn_decode_forward(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rope_pos_ids: ttnn.Tensor,
        page_table: ttnn.Tensor,
        *,
        advance_positions: bool = False,
    ) -> ttnn.Tensor:
        """Device-only token-in / sampler-ready-logits-out decode step.

        This is the method a decode trace captures.  It reads only the four
        persistent device tensors it is handed and returns vocab-sharded logits;
        there is no host work, no readback and no logits gather inside it.

        ``advance_positions`` increments the decode position and the RoPE index
        **on device, inside the captured graph**, after every read of them.  With
        the sampled token written straight back into ``tokens`` by the sampler
        (``tt_out_tok``), a fixed-step decode loop then needs no host staging at
        all between replays.

        The shipped generator captures **one** decode trace, always with
        ``advance_positions=True``, and that is deliberate: because the increment
        runs *after* every read, a caller that restages positions from the host
        simply overwrites it, so the same graph serves free-running, teacher-forced
        and caller-driven decode.  An earlier version of this docstring said teacher
        forcing "must leave it off"; that is wrong and would be actively harmful,
        because the caller-driven path in ``generate()`` restages only the token --
        with the increment off, ``current_pos`` and ``rope_pos_ids`` would freeze at
        the prompt length and every forced step would attend the same position and
        overwrite the same cache block, with nothing raising.
        """
        embedded = self.embed_decode(tokens)
        hidden = self.decode_forward(
            embedded,
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        logits = self.decode_logits(hidden)
        if advance_positions:
            # ``skip_negative_entries`` keeps the -1 sentinel of an inactive user
            # slot at -1 instead of walking it into a real cache position.
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rope_pos_ids)
            self.counters["device_position_advances"] += 1
        return logits

    # ------------------------------------------------------ host-side logits

    def gather_and_untilize_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """Vocab-sharded logits -> replicated full-vocab ROW_MAJOR, for host sampling.

        Only the explicit host-sampling compatibility mode and the readiness
        prefill check use this; the measured token-out path never gathers logits.
        """
        gathered = self._all_gather_async(ttnn.clone(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        untilized = ttnn.untilize(gathered, use_multicore=True)
        ttnn.deallocate(gathered)
        return untilized

    def logits_to_torch(self, logits: ttnn.Tensor, *, gathered: bool = False) -> torch.Tensor:
        """``[rows, vocab_size]`` float32 on the host, padded ids removed."""
        if gathered:
            local = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
            flat = local.reshape(-1, self.config.padded_vocab_size)
        else:
            shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(logits)]
            width = self.config.local_vocab_size
            flat = torch.cat([s.reshape(-1, s.shape[-1])[:, :width] for s in shards], dim=-1)
        self.counters["readbacks"] += 1
        return flat[:, : self.config.vocab_size].to(torch.float32)

    # ------------------------------------------------------------ host input

    def _replicated(self, tensor: torch.Tensor, dtype: ttnn.DataType, *, device: bool) -> ttnn.Tensor:
        kwargs: dict[str, Any] = {
            "layout": ttnn.ROW_MAJOR_LAYOUT,
            "dtype": dtype,
            "mesh_mapper": ttnn.ReplicateTensorToMesh(self.mesh_device),
        }
        if device:
            kwargs["device"] = self.mesh_device
            kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        return ttnn.from_torch(tensor, **kwargs)

    def tokens_to_device(self, token_ids: Sequence[int], *, device: bool = True) -> ttnn.Tensor:
        """``[1, 1, 1, 32]`` uint32 ROW_MAJOR token-id tensor, replicated.

        Rank 4 with the batch on the last axis is exactly the shape
        ``ttnn.sampling`` requires of a preallocated ``output_tensor``, which is
        what lets the sampled token *be* the next decode input with no
        reallocation.  ``ttnn.embedding`` collapses the leading unit dims and
        returns ``[1, 32, hidden]`` from it, so one buffer serves both ends of the
        feedback loop.
        """
        padded = torch.zeros(1, 1, 1, DECODE_ROWS, dtype=torch.int32)
        padded[0, 0, 0, : len(token_ids)] = torch.tensor(list(token_ids), dtype=torch.int32)
        return self._replicated(padded, ttnn.uint32, device=device)

    def positions_to_device(self, positions: torch.Tensor, *, device: bool = True):
        """``(current_pos, rope_pos_ids)`` for a decode step, replicated.

        ``current_pos`` is ``[batch]`` int32 (what
        ``paged_scaled_dot_product_attention_decode`` and ``paged_update_cache``
        take) and ``rope_pos_ids`` is ``[1, batch]`` uint32 (what the per-user
        on-device cos/sin ``ttnn.embedding`` gather takes).  Both are padded to
        ``max_batch_size``: ``current_pos`` with -1, the inactive-slot sentinel the
        attention op skips and ``plus_one(skip_negative_entries=True)`` preserves,
        and the RoPE index with 0 because it is unsigned and its row is unused.

        **-1 is the only legal negative.**  The paged ops take this tensor as
        ``update_idxs`` / ``cur_pos`` and their skip test is an exact comparison
        against ``(uint32_t)-1``
        (``paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp``);
        any other negative is reinterpreted as a huge unsigned index, so
        ``virtual_block_id = update_idx / block_size`` reads far past the page-table
        circular buffer and the op issues a NOC transaction to whatever physical
        block that garbage names.  The transaction never retires, ``PagedUpdateCache``
        never completes, and the whole mesh -- including the fabric routers, which is
        where ``check_noc_status`` first reports it -- hangs until a ``tt-smi -r``.
        There is no in-band error, so this is checked here rather than left to the
        device: an out-of-range position is a caller bug and must fail as one.
        Positions at or past ``max_seq_len`` are refused for the same reason, since
        they index past the caller's page table.
        """
        batch = DECODE_ROWS
        padded = torch.full((batch,), -1, dtype=torch.int32)
        supplied = positions.reshape(-1).to(torch.int32)
        if supplied.numel() > batch:
            raise ValueError(f"start_pos has {supplied.numel()} entries; the decode batch is {batch} rows")
        bad = (supplied < -1) | (supplied >= self.config.max_seq_len)
        if bool(bad.any()):
            rows = bad.nonzero().reshape(-1).tolist()
            raise ValueError(
                "decode start_pos must be in [0, "
                f"{self.config.max_seq_len}) or exactly -1 for an inactive slot; row(s) "
                f"{rows} carry {supplied[bad].tolist()}. Any other negative is read by "
                "paged_update_cache / paged_scaled_dot_product_attention_decode as a huge "
                "unsigned index and hangs the mesh with an unretired NOC transaction."
            )
        padded[: supplied.numel()] = supplied
        rope = torch.clamp(padded, min=0).reshape(1, batch)
        return (
            self._replicated(padded, ttnn.int32, device=device),
            self._replicated(rope, ttnn.uint32, device=device),
        )

    def page_table_to_device(self, page_table: torch.Tensor, *, device: bool = True) -> ttnn.Tensor:
        """``[batch, blocks_per_seq]`` int32, **replicated** across the mesh.

        Replicated rather than fractured because every device holds the same
        *logical* sequence: KV parallelism splits the head dimension, never the
        blocks, so all four devices index their own cache with the same rows.
        """
        return self._replicated(self.normalize_page_table(page_table), ttnn.int32, device=device)

    def page_table_row(self, page_table: torch.Tensor | None, user_id: int) -> torch.Tensor:
        """``[1, blocks_per_seq]`` int32: one cache slot's row of the normalised table.

        A *prefill* writes exactly one cache slot, and both places the layer stack
        reads the table in prefill -- the ``paged_fill_cache`` chunk row and the
        chunked-SDPA prefix row -- want that single row.  Handing the stack this row
        with ``user_id=0`` therefore computes exactly what handing it the full
        ``[32, blocks]`` table with ``user_id=slot`` computed, and it removes a
        ``ttnn.slice`` whose row *offsets* are baked into the program hash: against the
        full table each serving slot compiles its own slice program, so a request
        landing in slot 7 paid a program-cache miss that a warmup driving slot 0 could
        not cover.  Against the row form there is one program for every slot, which is
        also what lets a single prefill trace serve every slot.
        """
        rows = self.normalize_page_table(page_table)
        if not 0 <= int(user_id) < int(rows.shape[0]):
            raise ValueError(f"user_id={user_id} outside the {int(rows.shape[0])} rows of the page table")
        return rows[int(user_id) : int(user_id) + 1].contiguous()

    def page_table_row_to_device(self, row: torch.Tensor, *, device: bool = True) -> ttnn.Tensor:
        """Replicate a :meth:`page_table_row` result onto the mesh.

        Replicated for the same reason the full table is: KV parallelism splits the
        head dimension, never the blocks, so every device indexes its own cache with
        the same row.
        """
        blocks = self.config.blocks_per_seq
        if tuple(row.shape) != (1, blocks):
            raise ValueError(f"a page-table row must be shaped (1, {blocks}), got {tuple(row.shape)}")
        return self._replicated(row.to(torch.int32), ttnn.int32, device=device)

    def normalize_page_table(self, page_table: torch.Tensor | None) -> torch.Tensor:
        """Coerce a caller's page table to ``[max_batch_size, blocks_per_seq]`` int32.

        Callers legitimately disagree about the width: the readiness prefill check
        hands in ``[1, 1024]``, vLLM hands in whatever its block manager allocated,
        and the layer needs exactly ``blocks_per_seq`` columns for the sequence it
        was built for.  Short tables are extended with blocks no active row uses,
        so an over-long decode against a short table faults on an unmapped page
        rather than silently reading another user's cache.
        """
        batch = DECODE_ROWS
        blocks = self.config.blocks_per_seq
        slots = self.config.max_batch_size
        if page_table is None:
            # One contiguous run of blocks per *cache slot*; rows past the last
            # slot alias the last one below, which is safe because they are the
            # inactive rows (current_pos == -1) that no op reads or writes.
            #
            # How many slots get their own run is bounded by the pool, not only by
            # the slot count.  A standalone build sizes the pool at
            # ``max_batch_size x blocks_per_seq`` so the bound never binds and this
            # is exactly the old behaviour.  A serving build does not: vLLM owns one
            # shared pool and hands out ids from it, so ``max_batch_size`` slots at
            # the full context is a number the pool is deliberately smaller than.
            # Without the bound this produced a default table naming blocks that do
            # not exist -- the *first* thing the vLLM adapter hit, because
            # ``_allocate_device_inputs`` builds this table before any request has
            # supplied one.
            private = max(1, min(slots, self.config.max_num_blocks // blocks))
            rows = torch.arange(min(batch, private) * blocks, dtype=torch.int32).reshape(min(batch, private), blocks)
            out = torch.zeros(batch, blocks, dtype=torch.int32)
            out[: rows.shape[0]] = rows
            if rows.shape[0] < batch:
                out[rows.shape[0] :] = rows[-1:]
            return out
        table = page_table.detach().to(torch.int32).cpu()
        if table.dim() == 1:
            table = table.reshape(1, -1)
        if table.dim() != 2:
            raise ValueError(f"page_table must be 2-D [batch, blocks], got {tuple(table.shape)}")
        if int(table.max()) >= self.config.max_num_blocks:
            raise ValueError(
                f"page_table references block {int(table.max())} but the cache holds "
                f"{self.config.max_num_blocks} blocks"
            )
        out = torch.zeros(batch, blocks, dtype=torch.int32)
        rows = min(table.shape[0], batch)
        cols = min(table.shape[1], blocks)
        out[:rows, :cols] = table[:rows, :cols]
        if cols < blocks:
            # Extend only the rows the caller actually supplied, and only with
            # blocks none of them already uses, so a decode that runs past the
            # caller's table faults on an unmapped page rather than silently
            # reading another row's cache.  The remaining rows are the inactive
            # ones and are aliased below.  (The readiness prefill check hands in a
            # [1, 1024] table against blocks_per_seq = 2048, so this path is
            # exercised by the gate, not hypothetical.)
            used = set(out[:rows, :cols].reshape(-1).tolist())
            spare = [b for b in range(self.config.max_num_blocks) if b not in used]
            need = (blocks - cols) * rows
            if len(spare) < need:
                raise ValueError(
                    f"page_table is {table.shape[1]} blocks wide for {rows} row(s) but the model needs "
                    f"{blocks}, and only {len(spare)} of the cache's {self.config.max_num_blocks} blocks "
                    "are unused"
                )
            out[:rows, cols:] = torch.tensor(spare[:need], dtype=torch.int32).reshape(rows, blocks - cols)
        if rows < batch:
            out[rows:, :] = out[rows - 1 : rows, :]
        return out

    # ---------------------------------------------------------------- report

    def precision_report(self) -> dict[str, Any]:
        """The whole model's realised precision policy, read off the build.

        Every field is derived from a tensor or a compute-kernel config that is
        already on the device, so this is what ``$datatype-sweep`` needs to prove
        that a ``selected_precision_config.json`` field was consumed rather than
        merely recorded.  Per-layer reports are folded to the distinct ones with
        the layer indices that produced each, so a 52-layer stack with a
        first/last-layer exception reports three groups rather than 52 copies.
        """
        groups: list[dict[str, Any]] = []
        overrides: dict[str, Any] = {}
        for layer in self.layers:
            report = layer.precision_report()
            layer_idx = report.pop("layer_idx")
            # Companion settings are layer-uniform; hoist them so the model-level
            # report carries them once rather than inside every layer group.
            overrides = report.pop("decoder_overrides", None) or overrides
            report.pop("layer_kind", None)
            key = json.dumps(report, sort_keys=True)
            for group in groups:
                if group["_key"] == key:
                    group["layers"].append(layer_idx)
                    break
            else:
                groups.append({"_key": key, "layers": [layer_idx], "precision": report})
        for group in groups:
            del group["_key"]

        head = self.lm_head
        return {
            "policy_name": self.precision.name,
            "decoder_overrides": overrides or {},
            "num_layers": len(self.layers),
            "layer_groups": groups,
            "embedding": {"weight_dtype": str(self.embed_weight.dtype)},
            "lm_head": {
                "weight_dtype": str(head.weight.dtype),
                "fidelity": str(head.compute_kernel_config.math_fidelity),
                "fp32_dest_acc_en": bool(head.compute_kernel_config.fp32_dest_acc_en),
                "output_dtype": str(head.output_dtype),
                "matmul": head.matmul,
                "cores": head.cores,
                "in0_block_w": head.in0_block_w,
                "softcap_in_l1": bool(head.softcap_in_l1),
            },
            "terminal_norms": {
                "embed_norm_weight_dtype": str(self.embed_norm.weight.dtype),
                "final_norm_weight_dtype": str(self.final_norm.weight.dtype),
            },
        }

    def dram_report(self) -> dict[str, Any]:
        """Measured per-device DRAM footprint of everything long-lived.

        Recomputed from the tensors actually on the device rather than from a
        formula, so the context contract cannot drift from the build.
        """

        def nbytes(tensor: ttnn.Tensor | None) -> int:
            """One device's share, from the tensor's padded shape and dtype."""
            if tensor is None:
                return 0
            return _tensor_bytes(ttnn.get_device_tensors(tensor)[0])

        layer_weights = 0
        kv_cache = 0
        for layer in self.layers:
            for tensor in (layer.wqkv, layer.w_attn_gate, layer.wo, layer.mlp.gate, layer.mlp.up, layer.mlp.down):
                layer_weights += nbytes(tensor)
            for norm in (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.pre_feedforward_layernorm,
                layer.post_feedforward_layernorm,
            ):
                layer_weights += nbytes(norm.weight) + nbytes(norm.weight_rm) + nbytes(norm.local_weight)
            kv_cache += nbytes(layer.k_cache) + nbytes(layer.v_cache)
        rope = 0
        if self.rope_cache is not None:
            for key in ("cos", "sin", "cos_tile", "sin_tile"):
                rope += nbytes(self.rope_cache[key])
        terminal = (
            nbytes(self.embed_weight)
            + nbytes(self.lm_head.weight)
            + nbytes(self.final_norm.weight)
            + nbytes(self.final_norm.weight_rm)
            + nbytes(self.embed_norm.weight)
            + nbytes(self.embed_norm.weight_rm)
        )
        return {
            "per_device_layer_weight_bytes": layer_weights,
            "per_device_kv_cache_bytes": kv_cache,
            "per_device_rope_table_bytes": rope,
            "per_device_terminal_weight_bytes": terminal,
            "per_device_total_bytes": layer_weights + kv_cache + rope + terminal,
            "per_device_dram_capacity_bytes": dram_capacity_bytes(self.mesh_device),
        }


def _tensor_bytes(tensor: ttnn.Tensor) -> int:
    """Padded on-device byte size of one device shard."""
    shape = tensor.padded_shape
    elements = 1
    for dim in shape:
        elements *= int(dim)
    per_element = {
        ttnn.bfloat16: 2.0,
        ttnn.float32: 4.0,
        ttnn.uint32: 4.0,
        ttnn.int32: 4.0,
        ttnn.uint16: 2.0,
        ttnn.uint8: 1.0,
        # Block float: mantissa bytes plus one shared-exponent byte per 16 rows of
        # a 32x32 tile, i.e. 1088 B / 1024 elements and 576 B / 1024 elements.
        ttnn.bfloat8_b: 1088.0 / 1024.0,
        ttnn.bfloat4_b: 576.0 / 1024.0,
    }.get(tensor.dtype)
    if per_element is None:
        raise ValueError(f"unknown element size for {tensor.dtype}")
    return int(elements * per_element)


def dram_capacity_bytes(mesh_device: ttnn.MeshDevice) -> int:
    """Allocatable DRAM per device, from the allocator rather than a data sheet."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return int(view.total_bytes_per_bank) * int(view.num_banks)


__all__ = [
    "EMBED_DTYPE",
    "HF_MODEL_ID",
    "LM_HEAD_CORES",
    "LM_HEAD_DTYPE",
    "LM_HEAD_FIDELITY",
    "LM_HEAD_FP32_ACC",
    "LM_HEAD_OUTPUT_DTYPE",
    "LM_HEAD_IN0_BLOCK_W",
    "DECODE_ROWS",
    "LazyCheckpoint",
    "ModelConfig",
    "MuseGlimmerModel",
    "build_rope_cache",
    "dram_capacity_bytes",
    "padded_vocab_size",
    "weights_snapshot_dir",
]
