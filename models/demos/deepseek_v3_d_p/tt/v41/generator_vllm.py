# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash for vllm-tt-plugin (harness): the V4.1 prefill and the decode step behind the plugin's
model contract (``initialize_vllm_model``, ``allocate_kv_cache``, ``prefill_forward``, ``decode_forward``,
``read_decode_output``, ``warmup_model_prefill`` / ``warmup_model_decode``; ``models/demos/deepseek_v3/tt/generator_vllm.py``
is the precedent).

Requests and state. The plugin hands prefill and decode rows with vLLM's block tables; this model keeps its own
caches and ignores the KV tensors it allocates for the plugin, so a request is identified by the first block of its
block table (block 0 is vLLM's null block, never a request's). A prefill runs into one of ``TT_V41_PREFILL_SLOTS``
prefill states (``V41PrefillState``; traced per state and prompt length, else eager) and joins the request's decode
slot (``TtV41BatchDecoder``: ``max_batch_size`` slots in a pool sharded over the mesh rows; the slot of that block,
or the least recently used one); a decode call steps every row's slot at once (one replayed capture) and reads the
gathered logits once. Every row is text, single-chunk prefill (``start_pos`` 0), host sampling (the logits of the
full vocabulary come to the host). Prompts need at least ``SLIDING_WINDOW`` tokens (the decode step starts after a
full window) and prompt plus output at most ``max_model_len``.

Environment: ``TT_V41_CHECKPOINT`` (the pinned snapshot; ``weights.resolve_checkpoint``), ``TT_V41_WEIGHT_CACHE``
(the device MoE cache, ``tests/v41/weight_cache.py``), ``TT_V41_ENGRAM_MMAP`` (the Engram tables mapped from disk).
``TT_V41_CHUNK`` (default 8192) is the prefill chunk; ``max_model_len`` is rounded up to a multiple of it.
``TT_V41_LAYERS`` (default ``all``) selects the layer schedule, for checks on shallower models.
"""

import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41ChunkTables, V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.decode import ROWS
from models.demos.deepseek_v3_d_p.tt.v41.dspark_drafter import TtDSparkDrafter
from models.demos.deepseek_v3_d_p.tt.v41.dspark_weights import seeding_weights
from models.demos.deepseek_v3_d_p.tt.v41.engram import V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules, unswap
from models.demos.deepseek_v3_d_p.tt.v41.pool import RING
from models.demos.deepseek_v3_d_p.tt.v41.spec_verify import TtV41VerifyDecoder
from models.demos.deepseek_v3_d_p.tt.v41.transformer import (
    TtV41Transformer,
    V41GenericPrefillTrace,
    V41PrefillTrace,
    prefill_generic,
)
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

WARMUP_PROMPT = 255  # tokens of the warm-up prompt (a full window; odd, so the ratio-2 carry path runs first)
# TT_V41_SPEC=K (speculative decode): the decode step is the verify decoder (tt/v41/spec_verify)
# with the DSpark drafter in its capture; a greedy request's step verifies K drafts and the round's
# accepted tokens are handed to the plugin one per decode_forward call (the pool is ahead of the
# plugin's position by the buffered tokens; the plugin's contract is unchanged, the calls that drain
# the buffer touch no device). Speculation needs one live slot a mesh row and the drafter's rings
# hold one request's state (the last joined): the other slots' rows run plain. A sampling request's
# round draws the served sampler on every verify row and accepts while the draw equals the next
# draft (the match rule preserves the sequential served sampler's law); TT_V41_SPEC_SAMPLED=0 makes
# such a request run plain instead (the greedy-only path). TT_V41_SPEC_GREEDY_ALL=1 (a smoke's
# knob): every request is verified greedily whatever it asked.
SPEC = "TT_V41_SPEC"
SPEC_GREEDY_ALL = "TT_V41_SPEC_GREEDY_ALL"
SPEC_SAMPLED = "TT_V41_SPEC_SAMPLED"
# TT_V41_SPEC_RINGS=slot (the served c32 round): the drafter keeps a ring per slot, so every
# request speculates: a step of one live slot is the fused c1 round above; a step of several runs every slot of the
# pool as a block of 1 + K rows (the requests' drafts padded to K, a pad row rejected by construction; the slots
# without a request as dummies at their last position, whose writes sit past their frontier: one capture whatever
# the live set) and then one standalone draft call for the four slots of every rank at once (the accepted rows'
# taps written first). "one" (default): one ring set, the last joined request speculates, the others run plain.
SPEC_RINGS = "TT_V41_SPEC_RINGS"
SPEC_PAD = 0  # the pad draft and the dummy slots' token: any id; a pad row's token is never committed
WARMUP_DECODE_STEPS = 8  # at most; done once the step is captured and replayed once
PREFILL_SLOTS = "TT_V41_PREFILL_SLOTS"  # prefill states (default 2): requests prefill through them into the pool
# Traced prefill: ``TT_V41_TRACE_PREFILL_LENGTHS`` lists prompt lengths (tokens, comma-separated) whose prefill is
# captured per slot at warm-up (the trace phase) and replayed for every prompt of exactly that length; other
# lengths run eagerly, or are captured on their first request when ``TT_V41_TRACE_PREFILL_ON_MISS=1``.
TRACE_PREFILL_LENGTHS = "TT_V41_TRACE_PREFILL_LENGTHS"
TRACE_PREFILL_ON_MISS = "TT_V41_TRACE_PREFILL_ON_MISS"
TRACE_PREFILL_SLACK = "TT_V41_TRACE_PREFILL_SLACK"  # also capture each listed length +- this many tokens (1)
# ``TT_V41_PREFILL_GENERIC=1`` (transformer.PREFILL_GENERIC): the prefill states are position-generic and the trace
# phase captures one ``V41GenericPrefillTrace`` per state (one capture per score-width bucket), so prompts of any
# length longer than one chunk replay captures for every chunk but the final one. The per-length captures above
# still apply to prompts of exactly those lengths (they replay the final chunk too).
# ``TT_V41_SHORT_CHUNK``: smaller prefill chunks (tokens, comma-separated, ascending, each below TT_V41_CHUNK; 0 or
# unset: none) for the short prompts. Each is a view of the model with that chunk (``TtV41Transformer.with_chunk``)
# and one prefill state whose context is the first score-width bucket; the listed lengths that fit a short chunk
# are captured on the smallest one that holds them instead of the full-chunk states, and a whole prompt of such a
# length prefills there (a 1,028-token prompt paid a 5,120-row chunk's padded rows).
SHORT_CHUNK = "TT_V41_SHORT_CHUNK"
# deferred final chunks: a request parked after its bulk chunks
# (joined into its slot, its carries in the slot's parking buffers) has its prefix restored into a prefill state when
# its final chunk is called; off (unset or "0") is the serial path, bit for bit.
DEFER_FINAL = "TT_V41_DEFER_FINAL"


def defer_final() -> bool:
    return os.environ.get(DEFER_FINAL, "0") == "1"


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _dram_bytes(mesh_device) -> int:
    """Bytes allocated in DRAM per chip (0 when the statistics are unavailable: a report, not a gate)."""
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return int(view.total_bytes_allocated_per_bank * view.num_banks)
    except Exception:
        return 0


def _l1_probe(mesh_device, tag: str) -> None:
    """``TT_V41_L1_PROBE=1``: log main L1's allocation (per bank) and its block table under ``tag``, at the warm-up's
    stages. A program-cache entry first built inside a step (a collective's global semaphores: ``ttnn.all_gather``,
    ``ttnn.reduce_scatter``) keeps its L1 blocks where the step's transients left room, and a program whose static
    CBs reach past them is then rejected at compile or enqueue ("Statically allocated circular buffers ... clash
    with L1 buffers ... allocated at N"): the owner is found by the stage its blocks first appear in. The table's
    addresses are allocator-relative (absolute = address + the L1 unreserved base, 111,616 on Blackhole)."""
    if os.environ.get("TT_V41_L1_PROBE", "0") != "1":
        return
    try:
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
        rows = [str(r) for r in v.block_table]
        logger.info(
            f"L1 PROBE [{tag}] allocated/bank {v.total_bytes_allocated_per_bank} largest free "
            f"{v.largest_contiguous_bytes_free_per_bank} blocks {len(rows)}: " + " | ".join(rows)
        )
    except Exception as e:  # a report, not a gate
        logger.warning(f"L1 PROBE [{tag}] failed: {e}")


def _layers(spec: str) -> list[int]:
    """``all`` or a layer list such as ``0,2-13,20-39`` (a harness knob: shallower models for checks)."""
    if spec.strip() == "all":
        return list(range(C.NUM_LAYERS))
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return sorted(out)


class DeepseekV41ForCausalLM:
    # device sampling of the decode step (TT_V41_SAMPLE_ON_DEVICE, default on) with the plugin's
    # sample_on_device_mode "decode_only": the step ends in ttnn.sampling over each row's top-128 candidates and
    # returns token ids; prefill returns logits (host sampling once per request). The host sampler over 32 x
    # 129,280 logits took ~100 ms a step.
    SAMPLE_ON_DEVICE = os.environ.get("TT_V41_SAMPLE_ON_DEVICE", "1") == "1"
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_sample_on_device": SAMPLE_ON_DEVICE,
        "supports_async_decode": False,
    }

    def __init__(
        self,
        mesh_device,
        model: TtV41Transformer,
        decoder: TtV41BatchDecoder,
        tokenizer,
        max_batch_size: int,
        prefill_slots: int = 2,
        vllm_config=None,
    ):
        """``vllm_config``: unused; vLLM's registry classifies a model as generative by the shape of ``__init__``,
        ``embed_input_ids``, ``forward`` and ``compute_logits`` (interfaces_base.py), none of which the TT plugin
        calls: it runs ``prefill_forward`` / ``decode_forward``."""
        self.mesh_device, self.model, self.decoder, self.tokenizer = mesh_device, model, decoder, tokenizer
        self.config = model.config
        self.max_batch_size = max_batch_size
        assert decoder.num_slots == max_batch_size, (decoder.num_slots, max_batch_size)
        # decode slots: first block id -> pool slot, least recently used first; the free slots
        self.slots: OrderedDict[int, int] = OrderedDict()
        self.free: list[int] = list(range(max_batch_size))
        # prefill states, in use round-robin (a request's rows are in its decode slot once its prefill returns)
        self.generic = prefill_generic()
        self.prefill_states: list[V41PrefillState] = [
            model.new_state(generic=self.generic) for _ in range(prefill_slots)
        ]
        self.generic_traces: dict[int, V41GenericPrefillTrace] = {}  # id(state) -> its bucket captures
        # the short-chunk geometries (TT_V41_SHORT_CHUNK): per chunk its model view and its one prefill state;
        # ``models`` maps every prefill state (by id) to the model it prefills through
        self.models: dict[int, TtV41Transformer] = {id(st): model for st in self.prefill_states}
        self.short: list[tuple[int, TtV41Transformer, V41PrefillState]] = []
        spec = os.environ.get(SHORT_CHUNK, "0").strip()
        for chunk in sorted({int(x) for x in spec.split(",") if x.strip() and int(x) > 0}):
            assert chunk < model.chunk, f"{SHORT_CHUNK} {chunk}: not below the chunk ({model.chunk})"
            t0, dram0 = time.time(), _dram_bytes(mesh_device)
            # the state's context: the first score-width bucket, so a short prompt scores at the width the
            # full-chunk state would score it at (the same selection, up to the MoE's row split)
            _l1_probe(mesh_device, f"before chunk {chunk} geometry")
            view = model.with_chunk(chunk, _round_up(V41ChunkTables.MIN_BUCKET, chunk))
            state = view.new_state(generic=self.generic)
            _l1_probe(mesh_device, f"after chunk {chunk} geometry")
            self.models[id(state)] = view
            self.short.append((chunk, view, state))
            logger.info(
                f"DeepSeek-V4.1-Flash: chunk {chunk} geometry (max_seq_len {view.max_seq_len}) built in "
                f"{time.time() - t0:.1f} s, {(_dram_bytes(mesh_device) - dram0) / 2**20:.0f} MB of DRAM a chip"
            )
        # requests prefilled (wholly or up to a chunk boundary) but not yet joined into a decode slot:
        # request key -> (its prefill state, the prefix so far); oldest first
        self.pending: OrderedDict[int, tuple[V41PrefillState, torch.Tensor, dict]] = OrderedDict()
        self.next_prefill_state = 0
        # speculative decode (TT_V41_SPEC): the draft length, the drafter (attached to the verify decoder), the slot
        # whose state the drafter's rings hold, per slot the next round's drafts and the round's tokens not yet
        # handed to the plugin
        self.spec_k = int(os.environ.get(SPEC, "0"))
        self.spec_greedy_all = os.environ.get(SPEC_GREEDY_ALL, "0") == "1"
        self.spec_sampled = os.environ.get(SPEC_SAMPLED, "1") == "1"
        self.spec_rings_per_slot = os.environ.get(SPEC_RINGS, "one") == "slot"
        self.sp = tuple(mesh_device.shape)[0]
        self.users_per_rank = max_batch_size // self.sp
        self.drafter: TtDSparkDrafter | None = None
        self.ring_owner: int | None = None
        self.next_drafts: dict[int, list[int]] = {}
        self.buffered: dict[int, list[int]] = {}  # slot -> the round's tokens not yet handed to the plugin
        self.handed: dict[int, int] = {}  # slot -> the token last handed (the plugin feeds it back)
        self.last_drafts: dict[int, list[int]] = {}  # the ragged contract: row -> the next drafts (``take_drafts``)
        self.spec_stats = {"rounds": 0, "accepted": 0, "emitted": 0, "buffered_steps": 0}
        self.warmed_up = False
        # traced prefill per (slot, prompt length): the fixed input buffers are shared by every trace of a model
        self.prefill_traces: dict[tuple[int, int], V41PrefillTrace] = {}
        self.prefill_buffers: dict[int, dict] = {id(m): {} for m in [model] + [v for _, v, _ in self.short]}
        self.prefill_traced = False
        self.prefill_last: dict = {}  # the last prefill_forward row: tokens, start, mode, chunk (the tests read it)
        spec = os.environ.get(TRACE_PREFILL_LENGTHS, "").strip()
        listed = sorted({int(x) for x in spec.split(",") if x.strip()})
        # each listed length and its neighbours: the benchmark's synthetic prompts miss a registered length by a
        # token now and then (a quarter of the requests at some points, each paying an eager
        # final chunk of 1.7 s), and a capture is exact in its valid length
        slack = int(os.environ.get(TRACE_PREFILL_SLACK, "1"))
        self.trace_lengths = sorted({n + d for n in listed for d in range(-slack, slack + 1)})
        # with the generic trace, lengths past one chunk get their final chunk captured (generic in its start)
        # instead of a per-length capture of every chunk
        self.trace_on_miss = os.environ.get(TRACE_PREFILL_ON_MISS, "0") == "1"
        bad = [n for n in self.trace_lengths if n + ROWS > model.max_seq_len or n < C.SLIDING_WINDOW]
        assert not bad, f"{TRACE_PREFILL_LENGTHS} {bad}: outside [{C.SLIDING_WINDOW}, max_seq_len - {ROWS}]"

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("the TT plugin runs prefill_forward / decode_forward")

    def forward(self, input_ids, positions):
        raise NotImplementedError("the TT plugin runs prefill_forward / decode_forward")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("the TT plugin runs prefill_forward / decode_forward")

    # --- construction (the plugin's loader) --------------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        assert tt_data_parallel == 1, f"tt_data_parallel {tt_data_parallel}: one request pool on the whole mesh"
        ckpt = resolve_checkpoint()
        assert ckpt is not None, "the V4.1 checkpoint is not available (TT_V41_CHECKPOINT or the HF cache)"
        chunk = int(os.environ.get("TT_V41_CHUNK", 8192))
        max_seq = _round_up(max_seq_len, chunk)
        layers = _layers(os.environ.get("TT_V41_LAYERS", "all"))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ckpt.root)  # the snapshot holds the tokenizer
        t0 = time.time()
        spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
        root = weight_cache_dir(spec, mesh_device.shape)
        engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
        top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
        spec_k = int(os.environ.get(SPEC, "0"))
        model = TtV41Transformer(
            mesh_device,
            C,
            layers,
            lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
            top["embed.weight"],
            top["norm.weight"],
            top["head.weight"],
            max_seq_len=max_seq,
            chunk=chunk,
            engram=engram,
            engram_hash=engram_hash,
            weight_cache_path=root,
            dspark_weights=seeding_weights(ckpt, C) if spec_k else None,
        )
        sp = tuple(mesh_device.shape)[0]
        assert (
            max_batch_size % sp == 0
        ), f"max_num_seqs {max_batch_size}: the pool holds the same slots per mesh row ({sp})"
        users = max_batch_size // sp
        rings_per_slot = os.environ.get(SPEC_RINGS, "one") == "slot"
        if spec_k:
            # rings per slot: every slot of a rank holds a block of 1 + K rows in the multi-slot step (the MoE's
            # real rows decoupled from the slots)
            decoder = TtV41VerifyDecoder(
                model,
                root,
                users_per_rank=users,
                trace=True,
                sample_on_device=cls.SAMPLE_ON_DEVICE,
                real_rows=users * (1 + spec_k) if rings_per_slot else None,
                parking=defer_final(),
            )
        else:
            decoder = TtV41BatchDecoder(
                model,
                root,
                users_per_rank=users,
                trace=True,
                sample_on_device=cls.SAMPLE_ON_DEVICE,
                parking=defer_final(),
            )
        drafter = None
        if spec_k:
            from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import WEIGHT_CACHE, routed_expert_dtype

            assert 1 <= spec_k <= decoder.C1_ROWS - 1, f"{SPEC} {spec_k}: 1 to {decoder.C1_ROWS - 1} drafts"
            t1, dram0 = time.time(), _dram_bytes(mesh_device)
            drafter = TtDSparkDrafter(
                mesh_device,
                C,
                ckpt,
                model.embedding,
                None,
                WEIGHT_CACHE,
                routed_expert_dtype(),
                shared_head=model.head,
                trace=True,
                ring_slots=decoder.ring_slots,
                users=users if rings_per_slot else 1,
            )
            decoder.attach_drafter(drafter, users=list(range(max_batch_size // sp)), max_k=spec_k)
            if rings_per_slot:
                # the multi-slot step's merged index selection over every slot's rows, and the standalone call's
                # input buffers, before any capture (a persistent tensor allocated after a
                # capture may sit in its freed intermediates)
                decoder.prepare_merge(
                    [tuple(row for u in range(users) for row in decoder.block_rows(u, spec_k, one_slot=False))]
                )
                dummy = [{"user": u, "token": SPEC_PAD, "position": RING - 1, "taps": []} for u in range(users)]
                drafter.stage(drafter.host_inputs_blocks([dummy] * sp))
            decoder.draft_plain = True
            logger.info(
                f"DeepSeek-V4.1-Flash: DSpark drafter built in {time.time() - t1:.1f} s, "
                f"{(_dram_bytes(mesh_device) - dram0) / 2**20:.0f} MB of DRAM a chip; K = {spec_k} drafts a round"
            )
        prefill_slots = int(os.environ.get(PREFILL_SLOTS, 2))
        logger.info(
            f"DeepSeek-V4.1-Flash: {len(layers)} layers built in {time.time() - t0:.0f} s; max_seq_len {max_seq} "
            f"(chunk {chunk}), {max_batch_size} decode slots ({max_batch_size // sp} per mesh row), {prefill_slots} "
            f"prefill states, mesh {tuple(mesh_device.shape)}"
        )
        gen = cls(mesh_device, model, decoder, tokenizer, max_batch_size, prefill_slots)
        gen.drafter = drafter
        gen.weight_cache_root = Path(root)
        gen._unswap("after the build")  # the layers' weight files pushed the tables into swap (TT_V41_UNSWAP)
        return gen

    def _unswap(self, where: str) -> None:
        """``TT_V41_UNSWAP=1``: release the weight cache's page cache and swap the resident
        Engram tables back in (a swapped table row is a 5 ms major fault on the step's host side)."""
        # the resident tables (TT_V41_ENGRAM_MMAP=0: anonymous memory, swappable); a mapped table is page cache
        tables = [e.table.rows for e in self.model.engram.values() if isinstance(e.table, V41EngramTable)]
        if tables:
            unswap(tables, getattr(self, "weight_cache_root", None), where)

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """vLLM's KV block pool: every slot can hold a full context (the plugin's blocks are bookkeeping here)."""
        assert max_model_len is not None and max_num_seqs is not None
        return int(max_model_len) * int(max_num_seqs)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """The plugin's per-layer KV tensors: not used (the request slots hold the caches); placeholders."""
        logger.info(
            f"DeepSeek-V4.1-Flash: vLLM KV cache {kv_cache_shape} x {num_layers} layers not allocated (slot states)"
        )
        return [torch.zeros(1) for _ in range(num_layers)]

    # --- requests -------------------------------------------------------------------------------------------------
    @staticmethod
    def _request_key(page_table, row: int) -> int:
        key = int(page_table[row, 0])
        assert key != 0, f"row {row}: block table starts with the null block"
        return key

    def _slot_for_prefill(self, key: int) -> int:
        if key in self.slots:
            slot = self.slots.pop(key)
        elif self.free:
            slot = self.free.pop()
        else:
            _, slot = self.slots.popitem(last=False)  # the least recently used request is done or preempted
        self.slots[key] = slot
        return slot

    def _slot_for_decode(self, key: int) -> int:
        slot = self.slots.pop(key)
        self.slots[key] = slot  # most recently used
        return slot

    def release_request(self, key: int) -> None:
        """The plugin's notice that request ``key`` (its first block id) finished or was preempted: its decode
        slot goes back to the free list and its pending prefill state is dropped. Without the notice a new
        request's join evicts the least recently used slot, which is a finished request's only while the join
        runs after every live slot's row of the same step; a join from ``_state_for`` (a prefill step that
        needs a pending request's state) evicts whichever slot decoded first in the previous step, a live
        one when several requests finished together (the served c32 round's KeyError)."""
        slot = self.slots.pop(key, None)
        if slot is not None:
            self.free.append(slot)
            if self.drafter is not None:
                self.next_drafts.pop(slot, None)
                self.buffered.pop(slot, None)
                self.handed.pop(slot, None)
        self.pending.pop(key, None)

    def _prefill_state(self) -> V41PrefillState:
        state = self.prefill_states[self.next_prefill_state]
        self.next_prefill_state = (self.next_prefill_state + 1) % len(self.prefill_states)
        return state

    def _short_state(self, n: int | None) -> V41PrefillState | None:
        """The short-chunk state a whole prompt of ``n`` tokens prefills through: the smallest short chunk that
        holds it and has its capture (``n`` None: a chunk of a longer prompt; no short state)."""
        if n is None:
            return None
        for chunk, _, state in self.short:
            if n <= chunk and (id(state), n) in self.prefill_traces:
                return state
        return None

    def _state_for(self, key: int, start: int, whole: int | None = None) -> V41PrefillState:
        """The prefill state of request ``key``: at ``start`` 0 a free one (a short-chunk state when ``whole``, the
        length of a prompt that is one call, fits one with a capture), else the one holding the request's
        earlier chunks. States are owned from a request's first chunk to its join into a decode slot; the join
        happens at the request's first decode step, or here, when another request needs its state first (the
        scheduler prefills the waiting requests before it decodes, so a request's prefix is complete by then: it
        admits one new prefill a step and advances a partial one alone)."""
        if start == 0:
            short = self._short_state(whole)
            if key in self.pending:  # the same request restarted (preempted and resumed from position 0)
                state = self.pending.pop(key)[0]
                if short is None or short is state:
                    return state
            candidates = [short] if short is not None else self.prefill_states
            owned = {id(st) for st, _, _ in self.pending.values()}
            free = [st for st in candidates if id(st) not in owned]
            if free:
                return free[0]
            for old_key, (state, _, _) in self.pending.items():  # the oldest pending request holding one
                if any(state is c for c in candidates):
                    self._join_pending(old_key)
                    return state
            raise AssertionError(f"no prefill state for request {key}")
        if key not in self.pending and defer_final() and key in self.slots:
            return self._restore(key, start)
        assert key in self.pending, f"request {key}: a chunk at {start} without its earlier chunks"
        state, _, _ = self.pending[key]
        assert state.start == start, f"request {key}: chunk at {start}, its state is at {state.start}"
        return state

    def _restore(self, key: int, start: int) -> V41PrefillState:
        """TT_V41_DEFER_FINAL: request ``key``, parked (joined into its slot after its bulk chunks), gets a free
        prefill state with its ``start``-token prefix restored from the slot; the caller
        re-creates the pending entry."""
        slot = self.slots[key]
        owned = {id(st) for st, _, _ in self.pending.values()}
        free = [st for st in self.prefill_states if id(st) not in owned]
        if not free:
            old_key = next(iter(self.pending))  # the oldest pending request holding a state: parked now
            self._join_pending(old_key)
            owned = {id(st) for st, _, _ in self.pending.values()}
            free = [st for st in self.prefill_states if id(st) not in owned]
        state = free[0]
        t0 = time.time()
        if os.environ.get("TT_V41_RESTORE_SYNC", "1") == "1":
            # a join's replay issued just before (non-blocking) reads the state the restore overwrites, and the
            # restore's transients may land on a replay's freed intermediates (the second
            # restore of a cohort hung in its gather without this wait)
            ttnn.synchronize_device(self.mesh_device)
        self.decoder.restore(slot, state, start)
        logger.info(
            f"DeepSeek-V4.1-Flash: request restored from slot {slot} ({start} tokens) in {time.time() - t0:.3f} s "
            f"(issued {self.decoder.last_restore.get('issue_s')} s, {self.decoder.last_restore.get('pieces')} pieces)"
        )
        return state

    def prefill_forward(self, **kwargs) -> torch.Tensor:
        """``tokens`` [B, S] (padded), ``prompt_lens`` [B], ``page_table`` [B, blocks], ``start_pos`` [B] ->
        fp32 logits ``[B, 1, vocab]`` of each row's last token (host sampling reads row -1). A row is a prompt
        (``start_pos`` 0, ``prompt_lens`` its length) or one chunk of it (chunked prefill: ``start_pos`` a
        multiple of the chunk, ``prompt_lens`` the end of the chunk, ``tokens`` the whole prefix, or the whole
        prompt with ``prompt_total_lens`` its length, so the next chunk's host inputs are prepared ahead);
        either way the row's state keeps the prefix and the request joins its decode slot at its first
        ``decode_forward``. Rows resuming a prefill run before rows starting one (one state can finish before
        the next takes it)."""
        tokens, lengths, page_table = kwargs["tokens"], kwargs["prompt_lens"], kwargs["page_table"]
        start_pos = kwargs.get("start_pos")
        total_lens = kwargs.get("prompt_total_lens")  # the whole prompt's length per row, when ``tokens`` has it
        assert "sampling_params" not in kwargs or kwargs["sampling_params"] is None, "host sampling only"
        out = torch.zeros(tokens.shape[0], 1, self.config.VOCAB_SIZE, dtype=torch.float32)
        rows = [i for i in range(tokens.shape[0]) if int(lengths[i]) > 0]
        rows.sort(key=lambda i: 0 if start_pos is not None and int(start_pos[i]) > 0 else 1)
        for i in rows:
            n = int(lengths[i])
            start = 0 if start_pos is None else int(start_pos[i])
            assert start % self.model.chunk == 0, f"row {i}: chunk at {start}, not a multiple of {self.model.chunk}"
            assert (
                n >= self.config.SLIDING_WINDOW
            ), f"prompt of {n} tokens: decode needs a full window ({self.config.SLIDING_WINDOW})"
            assert (
                n + ROWS <= self.model.max_seq_len
            ), f"prompt of {n} tokens beyond max_seq_len {self.model.max_seq_len}"
            total = n if total_lens is None else max(n, int(total_lens[i]))
            prompt = tokens[i, :total].to(torch.int64)  # the whole prompt when given, else the prefix so far
            key = self._request_key(page_table, i)
            state = self._state_for(key, start, n if start == 0 and n == total else None)
            model = self.models[id(state)]  # the full-chunk model, or a short chunk's view
            resume = self.pending[key][2] if key in self.pending else {}
            if key not in self.pending and start > 0:  # restored (TT_V41_DEFER_FINAL): the host pipeline rebuilt
                self.pending[key] = (state, prompt, resume)
            t0 = time.time()
            trace = self.prefill_traces.get((id(state), n)) if start == 0 and n == total else None
            if trace is None and start == 0 and n == total and self.trace_on_miss:
                trace = self._capture_prefill(state, n)
                logger.info(f"DeepSeek-V4.1-Flash: prefill of {n} tokens captured in {time.time() - t0:.1f} s")
                t0 = time.time()
            generic = self.generic_traces.get(id(state)) if trace is None and n - start > 0 else None
            if generic is not None and n - start <= model.chunk and generic.key_of(start // model.chunk, total) is None:
                generic = None  # one chunk without a capture: eager either way
            if trace is not None:
                logits, _ = trace.run(prompt)
                mode = "replay"
            elif generic is not None:
                # an intermediate chunk (n < total) returns no logits and does not wait for the device: the
                # plugin discards its row, and the next call's copies and replay queue behind this chunk's
                logits, _ = generic.run(prompt, start=start, end=n, resume=resume)
                mode = f"generic {generic.last['replayed']}+{generic.last['eager']}"
            else:
                logits = model.prefill_into(prompt, state, start=start, end=n, resume=resume)
                mode = "eager"
            self.pending[key] = (state, prompt, resume)
            self.pending.move_to_end(key)
            phases = trace if trace is not None else generic
            logger.info(
                f"DeepSeek-V4.1-Flash: prefill of {n - start} tokens at {start} (prefix {n}) in {time.time() - t0:.2f} s "
                f"({mode}{'' if phases is None else ' ' + ' '.join(f'{k} {v:.3f}' for k, v in phases.last.items() if isinstance(v, float))}"
                f"{'' if model is self.model else f' chunk {model.chunk}'})"
            )
            self.prefill_last = {"tokens": n - start, "start": start, "mode": mode, "chunk": model.chunk}
            if logits is not None:
                out[i, 0] = logits[-1]
        return out

    def _join_pending(self, key: int) -> None:
        """Move request ``key``'s prefilled state into its decode slot (its first decode step)."""
        state, prompt, _ = self.pending.pop(key)
        slot = self._slot_for_prefill(key)
        t0 = time.time()
        self.decoder.join(slot, prompt[: state.start], state)
        if self.drafter is not None:
            # the drafter's rings take this request's seeding: the slot's own ring on its rank (rings per slot), or
            # the one set of rings (the last joined request speculates)
            assert state.dspark_rings is not None, "the prefill did not seed the drafter's rings"
            if self.spec_rings_per_slot:
                r, u = divmod(slot, self.users_per_rank)
                self.drafter.load_rings_from_device(state.dspark_rings, start=state.start, user=u, rank=r)
            else:
                self.drafter.load_rings_from_device(state.dspark_rings, start=state.start)
            self.ring_owner = slot
            self.next_drafts.pop(slot, None)
            self.buffered.pop(slot, None)
            self.handed.pop(slot, None)
        if defer_final() and state.start < prompt.numel():  # a bulk's join: the request is parked, its carries kept
            self.decoder.park(slot, state)
        logger.info(
            f"DeepSeek-V4.1-Flash: request of {prompt.numel()} tokens joined slot {slot} in {time.time() - t0:.3f} s "
            f"({self.decoder.last_join.get('mode')} width {self.decoder.last_join.get('width')})"
        )

    def _capture_prefill(self, state: V41PrefillState, n: int) -> V41PrefillTrace:
        """Capture the prefill of ``n``-token prompts into ``state`` (the slot's own cache tensors, so the slot's
        decode captures read what the replay wrote). The first capture at a length compiles its kernels (two
        untraced runs); later slots at that length need one untraced run for the steady-state resources."""
        model = self.models[id(state)]
        first = not any(k[1] == n and self.models[k[0]] is model for k in self.prefill_traces)
        trace = V41PrefillTrace(
            model,
            self._warmup_prompt(n),
            1,
            state=state,
            buffers=self.prefill_buffers[id(model)],
            warmup=2 if first else 1,
        )
        self.prefill_traces[(id(state), n)] = trace
        return trace

    def decode_forward(self, **kwargs) -> torch.Tensor:
        """``tokens`` [B, 1], ``start_pos`` [B], ``page_table`` [B, blocks] -> fp32 logits ``[B, 1, vocab]``, or with
        ``sampling_params`` (device sampling: per-row ``temperature``, ``top_k``, ``top_p`` lists) the sampled token
        ids ``[B]`` int32; rows whose block table starts with the null block are idle. One step of the pool for
        every active row."""
        tokens, positions, page_table = kwargs["tokens"], kwargs["start_pos"], kwargs["page_table"]
        params = kwargs.get("sampling_params")
        assert (params is not None) == self.decoder.sample_on_device, (
            f"the decode step samples {'on device' if self.decoder.sample_on_device else 'on the host'} "
            f"(TT_V41_SAMPLE_ON_DEVICE); sampling_params {'given' if params is not None else 'missing'}"
        )
        if params is not None:
            assert (
                float(params.repetition_penalty if isinstance(params.repetition_penalty, (int, float)) else 1.0) == 1.0
            )
            out = torch.zeros(tokens.shape[0], dtype=torch.int32)
        else:
            out = torch.zeros(tokens.shape[0], 1, self.config.VOCAB_SIZE, dtype=torch.float32)
        if self.drafter is not None:
            return self._decode_forward_spec(tokens, positions, page_table, params, out)
        step, rows, sampling = {}, {}, {}
        for i in range(tokens.shape[0]):
            if int(page_table[i, 0]) == 0:
                continue
            key = self._request_key(page_table, i)
            if key in self.pending:
                self._join_pending(key)
            slot = self._slot_for_decode(key)
            p = int(positions[i])
            assert p == self.decoder.position(
                slot
            ), f"row {i}: vLLM position {p}, slot {slot} at {self.decoder.position(slot)}"
            step[slot] = int(tokens[i, 0])
            rows[slot] = i
            if params is not None:
                sampling[slot] = (int(params.top_k[i]), float(params.top_p[i]), float(params.temperature[i]))
        if not step:
            return out
        t0 = time.time()
        result = self.decoder.decode(step, sampling if params is not None else None)
        for slot, i in rows.items():
            if params is not None:
                out[i] = result[slot]
            else:
                out[i, 0] = result[slot]
        last = self.decoder.last
        logger.debug(f"DeepSeek-V4.1-Flash: decode step of {len(step)} rows in {time.time() - t0:.3f} s {last}")
        return out

    def _decode_forward_spec(self, tokens, positions, page_table, params, out) -> torch.Tensor:
        """``decode_forward`` under ``TT_V41_SPEC``: the verify decoder's round for the slot whose rows the
        drafter holds (greedy or sampled: the request's ``(top_k, top_p, temperature)`` on every one of its rows
        and the match rule; under ``TT_V41_SPEC_SAMPLED=0`` greedy requests only), plain rows for the
        others (every slot under ``TT_V41_SPEC_GREEDY_ALL`` is verified greedily). Two
        contracts, told apart by the tokens' width: ``[B, 1]`` (the plugin unchanged): the
        round's tokens are handed over one per call, a slot with buffered tokens from its last round hands the
        next one over without a device step; ``[B, 1 + K]`` (the plugin's ragged contract): column 0 is
        the last token, columns ``1 ..`` the drafts the scheduler scheduled (-1 past them: the ones this generator
        proposed last step, as the scheduler clipped them), the result ``[B, 1 + K]`` int32 holds every row's
        emitted tokens (the accepted drafts and the bonus; -1 past them) and ``take_drafts`` the next drafts per
        row."""
        assert params is not None, "speculative decode samples on device"
        ragged = tokens.shape[1] > 1
        if ragged:
            assert tokens.shape[1] == 1 + self.spec_k, (tuple(tokens.shape), self.spec_k)
            out = torch.full((tokens.shape[0], 1 + self.spec_k), -1, dtype=torch.int32)
            self.last_drafts = {}
        blocks, rows, sampling = {}, {}, {}
        speculating = False  # a live slot proposes this step (the drafter in the graph: the ``vd:`` variants)
        t0 = time.time()
        for i in range(tokens.shape[0]):
            if int(page_table[i, 0]) == 0:
                continue
            key = self._request_key(page_table, i)
            if key in self.pending:
                self._join_pending(key)
            slot = self._slot_for_decode(key)
            token, p = int(tokens[i, 0]), int(positions[i])
            buffered = None if ragged else self.buffered.get(slot)
            if buffered:
                # the plugin feeds back the token we handed it; the pool processed it in the last round and the
                # next of the round's tokens follows without a step
                assert (
                    token == self.handed[slot]
                ), f"slot {slot}: the plugin's token {token}, handed {self.handed[slot]}"
                out[i] = self.handed[slot] = buffered.pop(0)
                self.spec_stats["buffered_steps"] += 1
                continue
            assert p == self.decoder.position(
                slot
            ), f"row {i}: vLLM position {p}, slot {slot} at {self.decoder.position(slot)}"
            temperature = float(params.temperature[i])
            greedy = temperature <= 0 or self.spec_greedy_all
            proposes = (self.spec_rings_per_slot or slot == self.ring_owner) and (greedy or self.spec_sampled)
            if ragged:
                given = [int(t) for t in tokens[i, 1:].tolist() if t >= 0]
                drafts = given if proposes else []
                if given and not proposes:  # verified by nobody: the scheduler counts them rejected
                    logger.warning(f"DeepSeek-V4.1-Flash: slot {slot} runs plain; {len(given)} drafts dropped")
            else:
                drafts = self.next_drafts.get(slot, []) if proposes else []
            speculating = speculating or proposes
            blocks[slot] = [token] + drafts[: self.spec_k]
            rows[slot] = i
            sampling[slot] = (int(params.top_k[i]), float(params.top_p[i]), 0.0 if greedy else temperature)
        if not blocks:
            return out
        if self.spec_rings_per_slot and len(blocks) > 1:
            return self._spec_step_slots(blocks, rows, sampling, out, ragged, t0)
        # a step whose live slots all run plain (sampled, or not the rings' owner) is the plain variant (``v:``,
        # captured at warm-up), not a round proposing drafts nobody verifies
        self.decoder.draft_plain = speculating
        result = self.decoder.decode(blocks, sampling)
        for slot, i in rows.items():
            block, argmax = blocks[slot], result[slot]
            a, emitted = self.decoder.accept(block, argmax)
            self.decoder.commit(slot, a)
            if slot in self.decoder.drafts:
                self.next_drafts[slot] = self.decoder.drafts[slot][a][: self.spec_k]
            else:
                self.next_drafts.pop(slot, None)
            if ragged:
                out[i, : len(emitted)] = torch.tensor(emitted, dtype=torch.int32)
                self.last_drafts[i] = list(self.next_drafts.get(slot, []))
                self.spec_stats["emitted_steps"] = self.spec_stats.get("emitted_steps", 0) + len(emitted)
            else:
                out[i] = self.handed[slot] = emitted[0]
                self.buffered[slot] = list(emitted[1:])
            if len(block) > 1:
                self.spec_stats["rounds"] += 1
                self.spec_stats["accepted"] += a
                self.spec_stats["emitted"] += len(emitted)
        last = self.decoder.last
        accepted = [int((out[i] >= 0).sum()) - 1 if ragged else len(self.buffered[s]) for s, i in rows.items()]
        logger.debug(
            f"DeepSeek-V4.1-Flash: spec step of {len(blocks)} slots ({[len(b) for b in blocks.values()]} rows) in "
            f"{time.time() - t0:.3f} s {last}; accepted {accepted} stats {self.spec_stats}"
        )
        return out

    def _uniform_blocks(self, blocks: dict) -> tuple[dict, dict]:
        """``TT_V41_SPEC_RINGS=slot``'s multi-slot step: every slot of the pool as a block of ``1 + K`` rows (the
        requests' drafts padded, the slots without a request as dummies at their position; a slot whose block would
        not fit its pool is left out, at the cost of a capture) -> (the blocks, the real draft count per slot)."""
        K, decoder = self.spec_k, self.decoder
        real = {slot: len(block) - 1 for slot, block in blocks.items()}
        uniform = {}
        for slot in range(self.max_batch_size):
            if slot in blocks:
                block = list(blocks[slot])
            elif slot in decoder.slots and decoder.position(slot) + 1 + K <= decoder.pool.max_seq_len:
                block = [SPEC_PAD]
            else:
                logger.warning(f"DeepSeek-V4.1-Flash: slot {slot} cannot pad the multi-slot step")
                continue
            uniform[slot] = block + [SPEC_PAD] * (1 + K - len(block))
        return uniform, real

    def _spec_step_slots(self, blocks: dict, rows: dict, sampling: dict, out, ragged: bool, t0: float):
        """The multi-slot round with rings per slot (``TT_V41_SPEC_RINGS=slot``): the uniform
        step over every slot, the accept capped at each request's real drafts, then one standalone draft call for
        the live slots of every rank (their accepted rows' taps written to their rings first), whose drafts are the
        slots' next."""
        K, decoder, drafter, users = self.spec_k, self.decoder, self.drafter, self.users_per_rank
        uniform, real = self._uniform_blocks(blocks)
        full = dict(sampling)
        for slot in uniform:
            full.setdefault(slot, (1, 0.0, 0.0))
        decoder.draft_plain = False
        result = decoder.decode(uniform, full)
        entries: list[list[dict]] = [[] for _ in range(self.sp)]
        where: dict[int, tuple[int, int]] = {}  # slot -> (rank, block index)
        for slot, i in rows.items():
            m = real[slot]
            block, argmax = uniform[slot][: 1 + m], result[slot][: 1 + m]
            a, emitted = decoder.accept(block, argmax)
            p = decoder.position(slot)
            decoder.commit(slot, a)
            r, u = divmod(slot, users)
            taps = [(decoder.rows[slot][j], p + j) for j in range(a + 1)]
            where[slot] = (r, len(entries[r]))
            entries[r].append({"user": u, "token": emitted[-1], "position": p + a, "taps": taps})
            if ragged:
                out[i, : len(emitted)] = torch.tensor(emitted, dtype=torch.int32)
                self.spec_stats["emitted_steps"] = self.spec_stats.get("emitted_steps", 0) + len(emitted)
            else:
                out[i] = self.handed[slot] = emitted[0]
                self.buffered[slot] = list(emitted[1:])
            if m:
                self.spec_stats["rounds"] += 1
                self.spec_stats["accepted"] += a
                self.spec_stats["emitted"] += len(emitted)
        # one block count on every rank: the ranks short of blocks get dummies (no tap write, their drafts unread)
        n = max(len(e) for e in entries)
        for r in range(self.sp):
            while len(entries[r]) < n:
                taken = {e["user"] for e in entries[r]}
                u = next(u for u in range(users) if u not in taken)
                entries[r].append({"user": u, "token": SPEC_PAD, "position": RING - 1, "taps": []})
        t1 = time.time()
        drafter.stage(drafter.host_inputs_blocks(entries), tap32=decoder.taps)
        drafter.run_blocks([e["token"] for e in entries[0]], read=False)
        by_rank = drafter.read_ranks(drafter.out)
        for slot, i in rows.items():
            r, b = where[slot]
            self.next_drafts[slot] = by_rank[r][b][:K]
            if ragged:
                self.last_drafts[i] = list(self.next_drafts[slot])
        t2 = time.time()
        last = decoder.last
        accepted = [int((out[i] >= 0).sum()) - 1 if ragged else len(self.buffered[s]) for s, i in rows.items()]
        # the same line as the one-slot round's; the draft call's
        # own line after it
        logger.debug(
            f"DeepSeek-V4.1-Flash: spec step of {len(blocks)} slots ({[1 + K] * len(uniform)} rows) in "
            f"{t2 - t0:.3f} s {last}; accepted {accepted} stats {self.spec_stats}"
        )
        logger.debug(f"DeepSeek-V4.1-Flash: draft call of {n} blocks a rank in {t2 - t1:.3f} s {drafter.last}")
        return out

    def take_drafts(self) -> dict[int, list[int]]:
        """The ragged contract's drafts after a ``decode_forward``: row index -> the drafts proposed for the row's
        next step (empty for a plain row), once."""
        drafts, self.last_drafts = self.last_drafts, {}
        return drafts

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out

    # --- warm-up (the plugin's model_runner.warmup_model) ---------------------------------------------------------------
    def warmup_model_prefill(self, enable_trace: bool, **kwargs) -> None:
        """Compile the prefill (one prompt per prefill state, each joining every decode slot) on the first call;
        the second (trace) call, which the plugin makes after the decode capture, captures the prefill of every
        ``TT_V41_TRACE_PREFILL_LENGTHS`` length on every prefill state (the captures' persistent tensors are
        consumed right after each replay, so they may share addresses with the decode capture's intermediates; the
        pool and the prefill states predate every capture), a length that fits a short chunk on that chunk's state
        only (``TT_V41_SHORT_CHUNK``; its joins captured at the chunk's width). Other lengths keep running eagerly."""
        if enable_trace:
            if self.prefill_traced:
                return
            t0 = time.time()
            for state in self.prefill_states:
                for slot in range(self.max_batch_size):
                    self.decoder.capture_join(state, slot)
            for chunk, _, state in self.short:  # a short state's prompts join at the chunk's width
                for slot in range(self.max_batch_size):
                    self.decoder.capture_join(state, slot, self.decoder.join_width(chunk))
            logger.info(
                f"DeepSeek-V4.1-Flash warm-up: {len(self.decoder.join_traces)} joins captured in {time.time() - t0:.1f} s"
            )
            if defer_final() and self.decoder.restore_traced:
                # TT_V41_RESTORE_TRACE=1: the restores captured per (state, slot, width) beside the joins, on the
                # states before the prefill captures (the eager compile runs and the captures write the states)
                t0 = time.time()
                for state in self.prefill_states:
                    for slot in range(self.max_batch_size):
                        self.decoder.capture_restore(state, slot)
                logger.info(
                    f"DeepSeek-V4.1-Flash warm-up: {len(self.decoder.restore_traces)} restores captured"
                    f" in {time.time() - t0:.1f} s"
                )
            chunk = self.model.chunk
            for n in self.trace_lengths:
                if self.generic and n > chunk:
                    continue  # the generic captures below serve it (its final chunk captured per length)
                short = self._short_chunk(n)
                states = [self.short[short][2]] if short is not None else self.prefill_states
                for state in states:
                    t0 = time.time()
                    self._capture_prefill(state, n)
                    logger.info(
                        f"DeepSeek-V4.1-Flash warm-up: prefill of {n} tokens captured in {time.time() - t0:.1f} s"
                        f"{'' if short is None else f' (chunk {self.short[short][0]})'}"
                    )
            if self.generic:
                for k, state in enumerate(self.prefill_states):
                    t0 = time.time()
                    trace = V41GenericPrefillTrace(
                        self.model, state, self.prefill_buffers[id(self.model)], warmup=2 if k == 0 else 1
                    )
                    self.generic_traces[id(state)] = trace
                    # the final chunks of the listed prompt lengths (a partial chunk's capture is exact in its
                    # valid length, generic in its start within the bucket)
                    for n in self.trace_lengths:
                        if n > chunk and trace.final_key(n)[0] is not None:
                            trace.capture_final(n)
                    logger.info(
                        f"DeepSeek-V4.1-Flash warm-up: generic prefill of state {k} captured in {time.time() - t0:.1f} s"
                        f" ({len(trace.controllers)} captures {sorted(map(str, trace.captured_at))}, "
                        f"{trace.num_segments} segments)"
                    )
            self.prefill_traced = True
            return
        if self.warmed_up:
            return
        self._warmup_join()

    def _short_chunk(self, n: int) -> int | None:
        """The index in ``short`` of the smallest short chunk that holds a whole prompt of ``n`` tokens (None: the
        full chunk's states serve it)."""
        for k, (chunk, _, _) in enumerate(self.short):
            if n <= chunk:
                return k
        return None

    def _padding_lengths(self, model: TtV41Transformer | None = None) -> list[int]:
        """Every chunk valid length the served prefills will hand ``model``'s MoE (the full-chunk model, or a short
        chunk's view): the chunk, the warm-up prompt's, and each listed prompt length's single chunk or final chunk
        (a short chunk: the listed lengths it serves)."""
        model = self.model if model is None else model
        chunk = model.chunk
        lengths = {chunk, WARMUP_PROMPT}
        for n in self.trace_lengths:
            if model is self.model:
                lengths.add(n if n <= chunk else n - (n - 1) // chunk * chunk)
            elif self._short_chunk(n) is not None and self.short[self._short_chunk(n)][1] is model:
                lengths.add(n)
        return sorted(lengths)

    def _prebuild_padding_configs(self) -> None:
        """Build the MoE gates' memoized padding configs (``TtMoEGatePrefill.build_padding_config``, one device
        tensor per chunk valid length) before any capture, for the full-chunk model and every short chunk's view.
        Built lazily, a length's config was allocated the first time it ran, during the per-length captures after
        the decode capture, where a persistent buffer can take the addresses of a capture's freed intermediates:
        the decode replay then overwrote the config and the next prefill of that length dispatched pad rows (a
        one-chunk replay 0.26 to 0.36 s by what the decode had written, the logits unchanged)."""
        for model in [self.model] + [view for _, view, _ in self.short]:
            t0 = time.time()
            gates = [block.ffn.moe.gate for block in model.blocks if hasattr(getattr(block, "ffn", None), "moe")]
            lengths = self._padding_lengths(model)
            for gate in gates:
                for n in lengths:
                    gate.build_padding_config(n, "right", 0)
            logger.info(
                f"DeepSeek-V4.1-Flash warm-up: padding configs of {len(gates)} MoE gates (chunk {model.chunk}) for "
                f"{len(lengths)} lengths {lengths} built in {time.time() - t0:.1f} s"
            )

    def _warmup_join(self) -> None:
        """Prefill the warm-up prompt into every prefill state (the full-chunk states first: the shared expert's
        reduce-scatter intermediate is allocated at the first input's shape, ``with_chunk``) and join every decode
        slot from them."""
        self._prebuild_padding_configs()
        prompt = self._warmup_prompt()
        for state in self.prefill_states + [state for _, _, state in self.short]:
            model = self.models[id(state)]
            t0 = time.time()
            _l1_probe(self.mesh_device, f"before warm-up prefill (chunk {model.chunk})")
            model.prefill_into(prompt, state)
            logger.info(
                f"DeepSeek-V4.1-Flash warm-up: prefill of {prompt.numel()} tokens (chunk {model.chunk}) in "
                f"{time.time() - t0:.1f} s"
            )
            _l1_probe(self.mesh_device, f"after warm-up prefill (chunk {model.chunk})")
        t0 = time.time()
        for slot in range(self.max_batch_size):
            self.decoder.join(slot, prompt, self.prefill_states[slot % len(self.prefill_states)])
        if self.drafter is not None:  # the drafter's rings: slot 0's (its warm-up rounds, ``_warmup_spec``)
            state = self.prefill_states[0]
            for u in range(self.drafter.users):  # every ring with rings per slot (every slot joined the prompt)
                self.drafter.load_rings_from_device(state.dspark_rings, start=state.start, user=u)
            self.ring_owner = 0
        logger.info(f"DeepSeek-V4.1-Flash warm-up: {self.max_batch_size} slots joined in {time.time() - t0:.1f} s")
        _l1_probe(self.mesh_device, "after warm-up joins")
        if defer_final():
            self._prewarm_restore()

    def _prewarm_restore(self) -> None:
        """Run one eager restore of every width bucket (``join_widths``) into a prefill state before any capture
        (TT_V41_DEFER_FINAL). The restore's row-major gathers are ``all_gather_async``'s
        composite path, ``all_broadcast`` + ``concat``, and ``all_broadcast`` creates two global semaphores in L1
        per cached workload, resident for the program cache's lifetime: created lazily, at the first restore of a
        shape after the captures, they took L1 addresses a capture's freed intermediates had baked, the next
        replay overwrote them with tensor rows, and the next restore of that shape waited forever on its
        out-ready semaphore (``broadcast_rm_writer.cpp:212``, every device). Created here, the captures allocate
        below them. A served restore uses the bucket's shapes only (``TtV41BatchDecoder.restore``)."""
        state = self.prefill_states[0]
        t0 = time.time()
        for width in self.decoder.join_widths:
            self.decoder.pool.restore_ops(0, state, width)
        ttnn.synchronize_device(self.mesh_device)
        logger.info(
            f"DeepSeek-V4.1-Flash warm-up: the restore's {len(self.decoder.join_widths)} width buckets run once"
            f" in {time.time() - t0:.1f} s"
        )
        _l1_probe(self.mesh_device, "after warm-up restores")

    def warmup_model_decode(self, enable_trace: bool, **kwargs) -> None:
        """Run the decode step on every slot until it is captured and replayed once (the first call compiles, the
        capture follows; the second call finds it captured)."""
        if self.warmed_up:
            return
        if not self.decoder.slots:  # the prefill warm-up did not run
            self._warmup_join()
        prompt = self._warmup_prompt()
        tokens = {slot: int(prompt[-1]) for slot in range(self.max_batch_size)}
        sampling = {slot: (1, 0.0, 0.0) for slot in tokens} if self.decoder.sample_on_device else None
        replays = 0
        for step in range(WARMUP_DECODE_STEPS):
            t0 = time.time()
            try:
                out = self.decoder.decode(tokens, sampling)
            except Exception:
                _l1_probe(self.mesh_device, f"decode step {step} raised")
                raise
            tokens = {slot: (int(out[slot]) if sampling else int(out[slot].argmax())) for slot in tokens}
            mode = self.decoder.last.get("mode")
            replays += mode == "replay"
            logger.info(
                f"DeepSeek-V4.1-Flash warm-up: decode step {step} ({mode}) in {time.time() - t0:.2f} s {self.decoder.last}"
            )
            _l1_probe(self.mesh_device, f"after decode step {step} ({mode})")
            if self.decoder.capture is not None and replays >= 1:
                break
        if self.drafter is not None:
            self._warmup_spec(prompt)
            self.warmed_up = True
            self._unswap("after the warm-up")
            return
        assert self.decoder.capture is not None, "the decode step was not captured"
        if self.decoder.c1:
            # the one-live-row variant (V41_DECODE_C1_MOE=1): its own eager steps and capture, on slot 0
            tokens = {0: int(prompt[-1])}
            sampling = {0: (1, 0.0, 0.0)} if self.decoder.sample_on_device else None
            replays = 0
            for step in range(WARMUP_DECODE_STEPS):
                t0 = time.time()
                out = self.decoder.decode(tokens, sampling)
                tokens = {0: int(out[0]) if sampling else int(out[0].argmax())}
                mode = self.decoder.last.get("mode")
                replays += mode == "replay"
                logger.info(
                    f"DeepSeek-V4.1-Flash warm-up: c1 decode step {step} ({mode}) in {time.time() - t0:.2f} s "
                    f"{self.decoder.last}"
                )
                if "c1" in self.decoder.captures and replays >= 1:
                    break
            assert "c1" in self.decoder.captures, "the c1 decode step was not captured"
        self.warmed_up = True
        self._unswap("after the warm-up")

    def _warmup_spec(self, prompt: torch.Tensor) -> None:
        """The verify decoder's captures: the plain 32-slot step (``v:``) is captured by ``warmup_model_decode``'s
        loop; here, per user index (the tile row a request's slot takes: the capture is per (user, rows), not per
        rank), slot ``u``'s rounds at 1 .. K + 1 rows (``vd:``: the drafter in the graph), each variant run eagerly
        twice, captured and replayed once, the drafts of each step feeding the next (every slot was joined from the
        warm-up prompt, whose seeding the drafter's rings hold), then its plain one-slot step (``v:``, the drafter
        off: a sampled request's, or one whose rings another request holds; captured on its first served call
        otherwise, 1.3 s)."""
        decoder, K = self.decoder, self.spec_k
        assert any(k.startswith("v:") for k in decoder.captures), "the 32-slot step was not captured"
        assert self.ring_owner == 0, f"the drafter's rings hold slot {self.ring_owner}, not slot 0"
        users = self.max_batch_size // tuple(self.mesh_device.shape)[0]
        for u in range(users):
            self.ring_owner = u  # the same seeding in every slot
            token, drafts = int(prompt[-1]), []
            seen: dict[str, int] = {}
            c1 = "c" if decoder.c1 else ""
            keys = [f"vd:{c1}1:{u}.{n}" for n in range(1, K + 2)]
            decoder.draft_plain = True
            for step in range(WARMUP_DECODE_STEPS * (K + 1)):
                # grow the block only once the shorter variants replayed: 1 row, then 2, .. K + 1
                want = 1 + min(len(drafts), K)
                for n in range(1, want + 1):
                    if seen.get(keys[n - 1], 0) < 1:
                        want = n
                        break
                block = [token] + drafts[: want - 1]
                t0 = time.time()
                out = decoder.decode({u: block}, {u: (1, 0.0, 0.0)})
                a, emitted = decoder.accept(block, out[u])
                decoder.commit(u, a)
                variant, mode = decoder.last.get("variant"), decoder.last.get("mode")
                seen[variant] = seen.get(variant, 0) + (mode == "replay")
                token, drafts = emitted[-1], decoder.drafts[u][a][:K]
                logger.info(
                    f"DeepSeek-V4.1-Flash warm-up: spec step {step} {variant} ({mode}) in {time.time() - t0:.2f} s "
                    f"{decoder.last}; accepted {a}"
                )
                if all(seen.get(key, 0) >= 1 for key in keys):
                    break
            missing = [key for key in keys if seen.get(key, 0) < 1]
            assert not missing, f"spec variants not captured and replayed: {missing}"
            # the plain one-slot step of this user index (the drafter off)
            decoder.draft_plain = False
            plain, replayed = f"v:{c1}1:{u}.1", False
            for step in range(WARMUP_DECODE_STEPS):
                t0 = time.time()
                out = decoder.decode({u: [token]}, {u: (1, 0.0, 0.0)})
                decoder.commit(u, 0)
                token = out[u][0]
                variant, mode = decoder.last.get("variant"), decoder.last.get("mode")
                assert variant == plain, (variant, plain)
                logger.info(
                    f"DeepSeek-V4.1-Flash warm-up: plain step {step} {variant} ({mode}) in {time.time() - t0:.2f} s "
                    f"{decoder.last}"
                )
                if mode == "replay":
                    replayed = True
                    break
            assert replayed, f"plain variant not captured and replayed: {plain}"
            decoder.draft_plain = True
        self.ring_owner = None  # the next joined request's
        if self.spec_rings_per_slot:
            self._warmup_spec_slots(prompt)

    def _warmup_spec_slots(self, prompt: torch.Tensor) -> None:
        """``TT_V41_SPEC_RINGS=slot``: the multi-slot round's two captures, the uniform step of every slot
        (``1 + K`` rows each, one variant) and the standalone draft call of ``users`` blocks a rank, each run
        eagerly, captured and replayed once."""
        decoder, drafter, K, users = self.decoder, self.drafter, self.spec_k, self.users_per_rank
        token = int(prompt[-1])
        blocks = {slot: [token] + [SPEC_PAD] * K for slot in range(self.max_batch_size)}
        sampling = {slot: (1, 0.0, 0.0) for slot in blocks}
        decoder.draft_plain = False
        replayed = False
        for step in range(WARMUP_DECODE_STEPS):
            t0 = time.time()
            out = decoder.decode(blocks, sampling)
            for slot in blocks:
                decoder.commit(slot, 0)
                blocks[slot][0] = out[slot][0]
            variant, mode = decoder.last.get("variant"), decoder.last.get("mode")
            logger.info(
                f"DeepSeek-V4.1-Flash warm-up: multi-slot step {step} {variant} ({mode}) in {time.time() - t0:.2f} s "
                f"{decoder.last}"
            )
            if mode == "replay":
                replayed = True
                break
        assert replayed, "the multi-slot step was not captured and replayed"
        p = decoder.position(0)
        entries = [
            [
                {"user": u, "token": token, "position": p - 1, "taps": [(decoder.rows[u][0], p - 1)]}
                for u in range(users)
            ]
        ] * self.sp
        for step in range(WARMUP_DECODE_STEPS):
            t0 = time.time()
            drafter.stage(drafter.host_inputs_blocks(entries), tap32=decoder.taps)
            drafter.run_blocks([token] * users, read=False)
            drafts = drafter.read_ranks(drafter.out)
            logger.info(
                f"DeepSeek-V4.1-Flash warm-up: draft call {step} ({drafter.last}) in {time.time() - t0:.2f} s; "
                f"rank 0 {drafts[0]}"
            )
            if drafter.last.get("mode") == "replay":
                break
        assert users in drafter.captures, "the multi-slot draft call was not captured"
        decoder.draft_plain = True

    def _warmup_prompt(self, n: int = WARMUP_PROMPT) -> torch.Tensor:
        """The first ``n`` tokens of the text the tests use (``oracle.text_tokens``, with this tokenizer), the text
        repeated when ``n`` outruns it."""
        import bz2

        book = bz2.open(orc._HERE.parents[3] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2", "rt").read()
        text = book[book.index("It was the best of times") :]
        text = (text * (12 * n // len(text) + 1))[: 12 * n]
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n]
        assert len(ids) == n, f"warm-up text yields {len(ids)} tokens, {n} needed"
        return torch.tensor(ids, dtype=torch.int64)
