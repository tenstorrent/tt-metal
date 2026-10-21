# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM serving adapter for the ``google/gemma-4-12B-it`` autoport on a QB2 (``tenstorrent/vllm-tt-plugin``).

The adapter only translates the plugin's model interface (``docs/MODEL_INTERFACE.md`` of the plugin) to the
low-level API of ``tt/generator.py``; every device program, trace, cache layout and the sampler are the generator's:

* ``initialize_vllm_model`` builds ``Gemma4Model`` with the selected precision config
  (``doc/datatype_sweep/selected_precision_config.json``, verified against the built model) and **no** KV cache.
* KV cache: vLLM owns it. ``get_kv_cache_spec`` declares vLLM hybrid KV groups: sliding-window layers
  ``SlidingWindowSpec(1024)``, full-attention layers ``FullAttentionSpec``, block 64. The full layers' single KV head is
  replicated on the 4 chips, so their spec declares 4 KV heads: one per chip, which is both the physical per-mesh
  footprint and the head count that makes every layer's page the same size (vLLM then keeps one block size for all
  groups instead of enlarging the full layers' blocks). vLLM groups the layers into 1 full + 5 sliding groups and
  lets 6 layers (one per group) share one buffer (``tensor_idx``). ``allocate_kv_cache_per_layer`` allocates one
  BFP8 K and V buffer per ``tensor_idx`` (the selected KV dtype; vLLM's dtype is only its accounting unit), with one
  extra physical block as the model's padding scratch block, and gives each layer a zero-copy
  ``ttnn.experimental.view`` in its own shape (sliding ``[n + 1, 2, 64, 256]``, full ``[n + 1, 1, 64, 512]``:
  32 tiles per block either way, so block b occupies the same bytes in both views). The caches are attached to the
  model (``Gemma4Model.attach_kv_cache``) and the generator is built on them with one page-table group per vLLM KV
  group; the ``kv_cache`` argument of every call is checked to be that object.
* Prefill: ``Gemma4Generator.prefill_forward`` per scheduled chunk (single-chunk prompts at position 0 replay the
  bucket's prefill trace), first token sampled on device by the sampling trace when the plugin samples on device,
  host logits ``[B, 1, V]`` otherwise. Chunk continuations must start on the model's 8192-token chunk grid: serve with
  ``--max-num-batched-tokens 8192`` and ``"prefill_chunk_alignment": 8192`` in the TT config.
* Decode: ``Gemma4Generator.decode_forward`` (traced model replay + traced split sampling with on-device token
  feedback, non-blocking), split readback through ``read_decode_output`` / ``process_decode_output_host``. The
  decode reload contract version 1 commands map to the generator's resident state: ``reload_inputs`` passes the
  plugin's tokens, positions and page tables; otherwise the token buffer and the positions the trace advanced stay
  resident (``tokens=None``, ``resident_positions``) and the page tables are copied only with
  ``reload_page_table``. ``reload_sampling_params`` rewrites the per-slot k / p / 1/T; ``reset_sampling_state``
  re-seeds every row (seeded rows aligned to their input position, unseeded rows randomly).
* Host sampling (the plugin's compatibility path for logprobs, penalties, host-only parameters, ``top_k`` > 32 and
  intermediate prefill chunks) returns logits; it is never used by greedy or plain top-k / top-p / temperature
  requests, which run the traced on-device sampler.

Registration (per launch): ``TT_MODEL_CLASS_OVERRIDES="Gemma4UnifiedForConditionalGeneration=
models.autoports.google_gemma_4_12b_it.tt.generator_vllm:Gemma4AutoportForCausalLM"`` (the checkpoint's architecture;
the plugin registers both the bare and the ``TT``-prefixed name, ahead of its built-in ``models.demos.gemma4`` class).
"""

import atexit
import copy
import json
import os
import threading
import time
from collections import Counter, deque

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import MAX_DECODE_BATCH
from models.autoports.google_gemma_4_12b_it.tt.generator import DECODE_BUCKETS, Gemma4Generator, SamplingParams
from models.autoports.google_gemma_4_12b_it.tt.model import BLOCK_SIZE, HF_MODEL, Gemma4Model, text_config
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import MESH_SHAPE, TP, fabric_router_config

# KV token budget handed to the plugin (it adds one block per user and 8 sliding windows per user of headroom).
# 1048576 tokens = 4 full-context requests of full-attention blocks: 20512 blocks x 557056 B (8 shared buffers x K, V
# x 32 BFP8 tiles) = 11.4 GB per chip, next to ~5.1 GB of weights, activations and traces. Override with
# GEMMA4_12B_KV_TOKENS.
DEFAULT_KV_TOKENS = 1 << 20
LAYERS_ENV = "GEMMA4_12B_VLLM_LAYERS"  # debugging only: comma-separated layer indices of a reduced model
# serving evidence: write the generator's trace registry / replay counts and the adapter's execution signatures
# (prefill chunks, decode buckets and reload commands) to this JSON file after every prefill and every 64 decodes
REPORT_ENV = "GEMMA4_12B_VLLM_TRACE_REPORT"
# benchmark evidence: append a host phase timeline (JSONL) for full-phase roofline accounting to this file
PHASE_TIMING_ENV = "GEMMA4_12B_PHASE_TIMING"
# stored bytes per element of a TILE tensor (BFP8: 1024 mantissa bytes + 64 exponent bytes per tile; BFP4: 512 + 64)
_TILE_BYTES_PER_ELEMENT = {"BFLOAT16": 2.0, "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024, "FLOAT32": 4.0}


def _stored_bytes(tensor):
    """Per-device stored DRAM bytes of a mesh tensor: ``[(mesh position, dtype, padded shape, bytes)]``."""
    out = []
    for i, t in enumerate(ttnn.get_device_tensors(tensor)):
        dtype = str(t.dtype).split(".")[-1]
        shape = [int(v) for v in t.padded_shape]
        n = 1
        for v in shape:
            n *= v
        out.append((i, dtype, shape, int(round(n * _TILE_BYTES_PER_ELEMENT[dtype]))))
    return out


class _PhaseTimeline:
    """Host phase timeline for the benchmark stage's roofline accounting (``GEMMA4_12B_PHASE_TIMING=<jsonl>``).

    The serving thread only appends small records to a deque (``time.perf_counter``, i.e. CLOCK_MONOTONIC, the clock
    ``vllm bench serve`` stamps request start times with). A daemon thread writes them to the file once a second, so
    no file I/O, device synchronization or extra device read happens on the serving path. Records:
    ``p`` prefill call (entry, return after the blocking first-token read), ``d`` decode submission (entry, return,
    padded rows, active positions), ``c`` decode completion (the host has the step's tokens)."""

    def __init__(self, path, meta):
        self.path = path
        self.records = deque()
        with open(path, "w") as f:
            f.write(json.dumps({"k": "meta", **meta}) + "\n")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gemma4-phase-timeline", daemon=True)
        self._thread.start()
        atexit.register(self.flush)

    def add(self, record):
        self.records.append(record)

    def flush(self):
        lines = []
        while self.records:
            lines.append(json.dumps(self.records.popleft()))
        if lines:
            with open(self.path, "a") as f:
                f.write("\n".join(lines) + "\n")

    def _run(self):
        while not self._stop.wait(1.0):
            self.flush()


class _DecodeOutput:
    """A submitted decode step whose sampled tokens are still on device (``read_from_device=False``)."""

    def __init__(self, rows):
        self.rows = rows
        self.step = None  # the adapter's decode step index (phase timeline)
        self.read = None  # Gemma4Generator.read_decode_output result (a DeferredRead when overlapped)


class Gemma4AutoportForCausalLM:
    model_capabilities = {
        "supports_sample_on_device": True,
        "max_device_top_k": 32,  # SplitTopKSampling keeps 32 candidates per row
        "supports_device_penalties": False,  # penalties sample on the host (logits path)
        "supports_async_decode": True,
        "supports_chunked_prefill": True,  # continuations on the 8192-token chunk grid (prefill_chunk_alignment)
        "supports_prefix_caching": False,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D_RING, "router_config": fabric_router_config()},
    }
    decode_input_update_contract = 1
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True
    _spec_vllm_config = None  # the config get_kv_cache_spec saw, to rebuild vLLM's KV grouping at allocation

    def __init__(self, model, max_batch_size, **kwargs):  # **kwargs: vLLM's model-class protocol (vllm_config=)
        self.model = model
        self.mesh = model.mesh
        self.max_batch_size = max_batch_size
        # decode batch buckets the plugin may pad to: every captured decode trace, capped at max_num_seqs
        self.tt_supported_decode_batch_sizes = tuple(
            sorted({b for b in DECODE_BUCKETS if b < max_batch_size} | {max_batch_size})
        )
        self.generator = None
        self.kv_caches = None
        self.layer_groups = None
        self._groups_checked = False
        self.report_path = os.environ.get(REPORT_ENV)
        # bounded (a long-running server must not grow host memory per request)
        self.prefill_events = deque(maxlen=20000)  # (time, start, end, slot, on-device sampled) per prefill row
        self.decode_signatures = Counter()  # (bucket, active rows, reload_inputs, reload_page_table, reset) per step
        self.decode_steps = 0
        self.prefill_call_ms = deque(maxlen=2000)  # adapter prefill_forward wall time per call (TTFT breakdown)
        self.timeline = None
        if os.environ.get(PHASE_TIMING_ENV):
            self.timeline = _PhaseTimeline(os.environ[PHASE_TIMING_ENV], self._timeline_meta())

    def _timeline_meta(self):
        """What the accounting needs from the running server: the imported implementation, the built layers and
        precision, the participating chips and every decode-path weight's stored bytes per chip."""
        import hashlib
        import sys

        from models.autoports.google_gemma_4_12b_it.tt.precision import built_precision_summary

        m = self.model
        cfg = text_config()
        weights = []
        for layer in m.layers:
            for role, t in (
                ("qkv", layer.wqkv),
                ("o", layer.wo),
                ("gate", layer.w_gate),
                ("up", layer.w_up),
                ("down", layer.w_down),
            ):
                weights.append({"layer": layer.layer_idx, "role": role, "devices": _stored_bytes(t)})
        weights.append({"layer": None, "role": "lm_head", "devices": _stored_bytes(m.lm_head)})
        summary = built_precision_summary(m)
        return {
            "clock": "time.perf_counter (CLOCK_MONOTONIC)",
            "perf_counter": time.perf_counter(),
            "unix_time": time.time(),
            "pid": os.getpid(),
            "adapter_module": __name__,
            "adapter_file": os.path.abspath(__file__),
            "model_module_file": os.path.abspath(sys.modules[type(m).__module__].__file__),
            "hf_model": HF_MODEL,
            "configured_layer_count": int(cfg.num_hidden_layers),
            "layer_count": len(m.layers),
            "layer_indices": [layer.layer_idx for layer in m.layers],
            "layer_types": [layer.layer_type for layer in m.layers],
            "sliding_window": int(cfg.sliding_window),
            "max_batch_size": int(self.max_batch_size),
            "decode_buckets": list(self.tt_supported_decode_batch_sizes),
            "max_context": int(m.max_context),
            "mesh_shape": list(MESH_SHAPE),
            "tp": TP,
            "device_ids": sorted(int(d) for d in self.mesh.get_device_ids()),
            "precision": m.precision.describe(),
            "precision_summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest(),
            "precision_summary": summary,
            "kv_dtype": str(m.layers[0].kv_dtype).split(".")[-1],
            "decode_weights": weights,
        }

    # ------------------------------------------------------------------ vLLM model-class protocol
    # vLLM's ``is_text_generation_model`` inspects the registered class for these methods (no upstream Gemma 4 class
    # exists to inspect instead); the TT runner only calls prefill_forward / decode_forward.
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT adapter: embeddings run on device inside prefill_forward / decode_forward")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("TT adapter: the TT runner calls prefill_forward / decode_forward")

    def compute_logits(self, hidden_states, **kwargs):
        raise NotImplementedError("TT adapter: logits are produced on device by prefill_forward / decode_forward")

    # ------------------------------------------------------------------ construction
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None
    ):
        assert tt_data_parallel == 1, "one model on the whole [1, 4] QB2 ring (no lane data parallelism)"
        assert tuple(mesh_device.shape) == MESH_SHAPE, f"Gemma 4 12B serves on a {MESH_SHAPE} QB2 mesh"
        assert 1 <= max_batch_size <= MAX_DECODE_BATCH, f"max_num_seqs must be in [1, {MAX_DECODE_BATCH}]"
        cfg = text_config()
        served = getattr(hf_config, "text_config", hf_config)
        for key in ("num_hidden_layers", "hidden_size", "vocab_size", "sliding_window", "head_dim", "global_head_dim"):
            assert getattr(served, key) == getattr(cfg, key), f"served config {key} differs from {HF_MODEL}"
        max_seq_len = max_seq_len or cfg.max_position_embeddings
        assert max_seq_len <= cfg.max_position_embeddings
        # the model's context: max_model_len rounded up to the prefill chunk grid (RoPE tables, page-table width)
        chunk = 8192
        max_context = -(-max_seq_len // chunk) * chunk
        layers = os.environ.get(LAYERS_ENV)
        model = Gemma4Model(
            mesh_device,
            num_blocks=0,  # vLLM allocates the KV cache (allocate_kv_cache_per_layer)
            max_context=max_context,
            layer_indices=[int(v) for v in layers.split(",")] if layers else None,
        )
        return cls(model, max_batch_size)

    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=4, tt_data_parallel=1, max_model_len=None, **kwargs):
        tokens = int(os.environ.get("GEMMA4_12B_KV_TOKENS", DEFAULT_KV_TOKENS))
        return max(tokens, max_model_len or 0)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        cfg = getattr(vllm_config.model_config.hf_config, "text_config", vllm_config.model_config.hf_config)
        block = vllm_config.cache_config.block_size
        assert block == BLOCK_SIZE, f"serve with --block-size {BLOCK_SIZE} (the model's paged-cache block)"
        cls._check_scheduler(vllm_config)
        dtype = vllm_config.model_config.dtype
        specs = {}
        for i, kind in enumerate(cfg.layer_types):
            name = f"model.layers.{i}.self_attn"
            if kind == "sliding_attention":
                specs[name] = SlidingWindowSpec(
                    block_size=block,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_size=cfg.head_dim,
                    dtype=dtype,
                    sliding_window=cfg.sliding_window,
                )
            else:
                # the single KV head is replicated on every chip: TP copies per mesh
                assert cfg.num_global_key_value_heads == 1
                specs[name] = FullAttentionSpec(
                    block_size=block, num_kv_heads=TP, head_size=cfg.global_head_dim, dtype=dtype
                )
        cls._spec_vllm_config = vllm_config
        return specs

    @staticmethod
    def _check_scheduler(vllm_config):
        """Chunked-prefill continuations must start on the model's 8192-token chunk grid."""
        sched = vllm_config.scheduler_config
        if not sched.enable_chunked_prefill:
            return
        tt = (vllm_config.additional_config or {}).get("tt", {})
        align = tt.get("prefill_chunk_alignment", 128)
        assert align and align % 8192 == 0, (
            "chunked prefill needs partial chunks on the 8192-token grid: pass "
            '"prefill_chunk_alignment": 8192 in --additional-config tt (and --max-num-batched-tokens 8192)'
        )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate the vLLM-owned caches (``(shape, dtype, tensor_idx)`` per attention layer, vLLM layer order) and
        build the generator on them. Returns ``[[k, v] per model layer]``."""
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        m = self.model
        cfg = type(self)._spec_vllm_config
        assert cfg is not None, "get_kv_cache_spec must run before allocation"
        if cfg.cache_config.kv_cache_layout is None:
            # the engine core resolves the layout; the TT worker supports only LBNHC (the grouping of uniform-page
            # specs does not depend on it, but vLLM reads it first)
            cfg = copy.copy(cfg)
            cfg.cache_config = copy.copy(cfg.cache_config)
            cfg.cache_config.kv_cache_layout = "LBNHC"
        # vLLM's own grouping of these specs, as the engine computed it for the block tables
        groups = get_kv_cache_groups(cfg, type(self).get_kv_cache_spec(cfg))
        group_of = {}
        for g, group in enumerate(groups):
            for name in group.layer_names:
                group_of[int(name.split("layers.")[1].split(".")[0])] = g
        num_blocks = per_layer_specs[0][0][0]
        buffers = {}
        caches = []
        for layer in m.layers:
            shape, _, tensor_idx = per_layer_specs[layer.layer_idx]
            assert shape[0] == num_blocks and tuple(shape[1:]) == tuple(
                layer.kv_block_shape
            ), f"layer {layer.layer_idx}: vLLM KV shape {shape} vs model block shape {layer.kv_block_shape}"
            if tensor_idx not in buffers:
                buffers[tensor_idx] = [
                    ttnn.zeros(
                        [num_blocks + 1, *shape[1:]],
                        dtype=layer.kv_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                ]
            kv = []
            for buf in buffers[tensor_idx]:
                want = [num_blocks + 1, *shape[1:]]
                if list(buf.shape) != want:
                    # another group's layer of this buffer: same bytes per block (vLLM's uniform page size), own view
                    have = list(buf.shape)
                    assert buf.dtype == layer.kv_dtype and have[1] * have[2] * have[3] == want[1] * want[2] * want[3]
                    buf = ttnn.experimental.view(buf, want)
                kv.append(buf)
            caches.append(kv)
        m.attach_kv_cache(caches, num_blocks)
        # page-table groups of the model's layers, renumbered compactly (a reduced model may skip groups)
        present = sorted({group_of[layer.layer_idx] for layer in m.layers})
        compact = {g: i for i, g in enumerate(present)}
        self.layer_groups = [compact[group_of[layer.layer_idx]] for layer in m.layers]
        # the vLLM layer index whose page table stands for each compact group
        self.group_layer = {}
        for layer, g in zip(m.layers, self.layer_groups):
            self.group_layer.setdefault(g, layer.layer_idx)
        self.generator = Gemma4Generator(
            m,
            decode_buckets=self.tt_supported_decode_batch_sizes,
            sampler=m.precision.sampler,
            prepare=False,
            kv_groups=self.layer_groups,
            windowed_groups=True,
        )
        self.kv_caches = caches
        print(
            f"Gemma4 autoport vLLM KV: {num_blocks} blocks (+1 scratch) x {len(buffers)} shared buffers x [k, v], "
            f"{len(groups)} vLLM groups ({len(present)} used), group windows {self.generator.group_window}"
        )
        return caches

    def allocate_kv_cache(self, *args, **kwargs):
        raise NotImplementedError("hybrid KV groups: the plugin calls allocate_kv_cache_per_layer")

    # ------------------------------------------------------------------ warmup / lifecycle
    def warmup_model_prefill(self, kv_cache=None, enable_trace=True, can_sample_on_device=True, **kwargs):
        """Generator-owned two-phase warmup: the eager call compiles every program, the traced call captures."""
        self.generator.prepare(capture=bool(enable_trace))

    def warmup_model_decode(self, kv_cache=None, enable_trace=True, max_batch_size=None, **kwargs):
        self.generator.prepare(capture=bool(enable_trace))
        self._report()  # the pre-workload baseline: every trace captured at warmup

    def release_persistent_capture(self):
        if self.generator is not None:
            self.generator.teardown()
            self._report()

    def _report(self):
        if not self.report_path or self.generator is None:
            return
        out = self.generator.trace_report()
        out["prefill_events"] = list(self.prefill_events)
        out["decode_signatures"] = {
            f"bucket{b}/active{a}/reload{int(r)}/page{int(p)}/reset{int(z)}": n
            for (b, a, r, p, z), n in sorted(self.decode_signatures.items())
        }
        out["decode_steps"] = self.decode_steps
        out["prefill_call_ms"] = list(self.prefill_call_ms)
        out["time"] = time.time()
        tmp = self.report_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, self.report_path)

    # ------------------------------------------------------------------ helpers
    def _group_tables(self, page_tables_per_layer, page_table):
        """vLLM tables -> one table per (compact) page-table group of the generator."""
        if page_tables_per_layer is None:
            assert self.generator.num_groups == 1
            return [page_table]
        tables = [page_tables_per_layer[self.group_layer[g]] for g in range(self.generator.num_groups)]
        if not self._groups_checked:  # once: every layer of a group carries its group's table
            for layer, g in zip(self.model.layers, self.layer_groups):
                assert torch.equal(page_tables_per_layer[layer.layer_idx], tables[g]), "KV group mapping mismatch"
            self._groups_checked = True
        return tables

    @staticmethod
    def _params(sampling_params, rows):
        """The plugin's per-row device sampling parameters -> the generator's ``SamplingParams``."""
        return SamplingParams(
            temperature=[float(v) for v in sampling_params.temperature[:rows]],
            top_k=[int(v) for v in sampling_params.top_k[:rows]],
            top_p=[float(v) for v in sampling_params.top_p[:rows]],
            seed=list(sampling_params.seed[:rows]),
        )

    def _check_cache(self, kv_cache):
        assert kv_cache is None or kv_cache is self.kv_caches, "kv_cache must be the allocate_kv_cache_per_layer result"

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        tokens,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache=None,
        enable_trace=True,
        empty_slots=None,
        page_tables_per_layer=None,
        sampling_params=None,
        **kwargs,
    ):
        """Prefill ``tokens[i, start_pos[i]:prompt_lens[i]]`` per row (``prompt_lens`` is the scheduled chunk's end)."""
        self._check_cache(kv_cache)
        t0 = time.perf_counter()
        starts = [int(v) for v in start_pos]
        ends = [int(v) for v in prompt_lens]
        B = len(ends)
        chunk = self.model.prefill_chunk
        for s in starts:
            assert s % chunk == 0, (
                f"prefill continuation at {s} is off the {chunk}-token chunk grid: serve with "
                f'"prefill_chunk_alignment": {chunk} and --max-num-batched-tokens {chunk}'
            )
        lens = [e - s for s, e in zip(starts, ends)]
        toks = torch.zeros(B, max(lens), dtype=torch.int64)
        for i, (s, e) in enumerate(zip(starts, ends)):
            toks[i, : e - s] = tokens[i, s:e]
        tables = [t[:B] for t in self._group_tables(page_tables_per_layer, page_table)]
        slots = list(range(B)) if empty_slots is None else [int(v) for v in empty_slots]
        g = self.generator
        now = time.time()
        self.prefill_events.extend(
            [now, s, e, slot, sampling_params is not None] for s, e, slot in zip(starts, ends, slots)
        )
        if sampling_params is None:  # host sampling: logits at each row's last position
            out = g.prefill_forward(
                toks, page_table=tables, prompt_lens=lens, start_pos=starts, slots=slots, enable_trace=enable_trace
            )
            t1 = time.perf_counter()
            self.prefill_call_ms.append([B, sum(lens), (t1 - t0) * 1000])
            if self.timeline is not None:
                self.timeline.add(
                    {"k": "p", "t0": t0, "t1": t1, "rows": [list(r) for r in zip(starts, ends, slots)], "dev": False}
                )
            self._report()
            return out
        out = g.prefill_forward(
            toks,
            page_table=tables,
            prompt_lens=lens,
            start_pos=starts,
            slots=slots,
            sample_on_device=True,
            sampling_params=self._params(sampling_params, B),
            seed_positions=[e - 1 for e in ends],
            enable_trace=enable_trace,
        )
        t1 = time.perf_counter()
        self.prefill_call_ms.append([B, sum(lens), (t1 - t0) * 1000])
        if self.timeline is not None:  # the return follows the blocking read of the sampled first tokens
            self.timeline.add(
                {"k": "p", "t0": t0, "t1": t1, "rows": [list(r) for r in zip(starts, ends, slots)], "dev": True}
            )
        self._report()
        return out.reshape(B, 1)

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        page_tables_per_layer=None,
        slot_remap=None,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        **kwargs,
    ):
        """One decode step of the padded batch (``B = tokens.shape[0]``, a captured bucket; position -1 = padding)."""
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1: the legacy reset_batch keyword is not accepted")
        unsupported = [k for k in ("defer_device_sampling", "spec_mode", "num_valid_drafts") if kwargs.get(k)]
        if unsupported or kwargs.get("prompt_tokens") is not None:
            raise NotImplementedError(f"unsupported decode inputs {unsupported or ['penalty history']}")
        assert not (reload_inputs and reload_page_table), "reload_inputs already includes the page tables"
        assert not reset_sampling_state or reload_inputs, "a sampling-state reset needs authoritative positions"
        t0 = time.perf_counter()
        self._check_cache(kv_cache)
        g = self.generator
        rows = int(tokens.shape[0])
        device_sampling = sampling_params is not None
        if slot_remap is not None and device_sampling:
            # every per-slot state of this adapter is rebuilt on the reset that accompanies a layout change: tokens,
            # positions and tables (reload_inputs), k / p / 1/T (reload_sampling_params) and seeds
            # (reset_sampling_state, aligned to the authoritative positions)
            identity = all(int(v) == i for i, v in enumerate(slot_remap))
            assert identity or reset_sampling_state, "a slot remap must come with a sampling-state reset"
        tables = None
        if reload_inputs or reload_page_table:
            tables = [t[:rows] for t in self._group_tables(page_tables_per_layer, page_table)]
        if reload_inputs:
            tok = torch.as_tensor(tokens).reshape(-1)[:rows]
            pos = torch.as_tensor(start_pos).reshape(-1)[:rows].to(torch.int32)
        else:  # steady device decode: tokens and positions stay resident (the plugin's are one step behind)
            assert device_sampling, "resident decode inputs exist only after an on-device sampled step"
            tok = None
            pos = g.resident_positions(rows)
        active = int((pos >= 0).sum())
        self.decode_signatures[(rows, active, reload_inputs, reload_page_table, reset_sampling_state)] += 1
        self.decode_steps += 1
        if self.decode_steps % 64 == 0:
            self._report()
        step = self.decode_steps
        if self.timeline is not None:
            active_pos = [int(v) for v in pos.tolist() if v >= 0]
        if not device_sampling:  # host sampling compatibility path: logits [B, 1, V]
            logits = g.decode_forward(tok, pos, page_table=tables, return_logits=True, enable_trace=enable_trace)
            if self.timeline is not None:  # logits read back: complete on return
                t1 = time.perf_counter()
                self.timeline.add({"k": "d", "i": step, "t0": t0, "t1": t1, "b": rows, "pos": active_pos, "host": True})
                self.timeline.add({"k": "c", "i": step, "t": t1})
            return logits[:, None, :]
        if reload_sampling_params:
            g.set_sampling_params(self._params(sampling_params, rows), rows, reseed=False)
        if reset_sampling_state:
            g.reset_sampling_state(list(sampling_params.seed[:rows]), pos.tolist(), range(rows))
        out = g.decode_forward(
            tok, pos, page_table=tables, read_from_device=read_from_device, enable_trace=enable_trace
        )
        if self.timeline is not None:
            t1 = time.perf_counter()
            self.timeline.add({"k": "d", "i": step, "t0": t0, "t1": t1, "b": rows, "pos": active_pos, "host": False})
            if read_from_device:  # tokens read back: complete on return
                self.timeline.add({"k": "c", "i": step, "t": t1})
        if read_from_device:
            return out.reshape(rows, 1)
        handle = _DecodeOutput(rows)
        handle.step = step
        return handle

    def read_decode_output(self, tt_out, async_read=True, **kwargs):
        """Register the copy of a submitted step's sampled tokens without enqueueing it: the generator enqueues it
        (with its event) at its next device call, before anything can change the token buffer, or when
        ``process_decode_output_host`` needs it. No event is returned, so the plugin's finalize goes straight to
        ``process_decode_output_host``, which waits for the copy. This keeps the copy's host dispatch (1-3 ms) off
        the path from a request's first decode submission to vLLM returning its first token."""
        if isinstance(tt_out, torch.Tensor):  # host output (logits / synchronous tokens): nothing to read
            return tt_out, []
        assert isinstance(tt_out, _DecodeOutput) and tt_out.read is None
        tt_out.read = self.generator.read_decode_output(async_read=async_read, defer=async_read)
        return tt_out, []

    def process_decode_output_host(self, tt_out, is_tokens=True, **kwargs):
        if isinstance(tt_out, torch.Tensor):
            return tt_out
        assert is_tokens, "a device-sampled decode step returns tokens"
        if tt_out.read is None:  # synchronous scheduling finalizes without read_decode_output: blocking read
            tt_out.read = self.generator.read_decode_output(async_read=False)
        out = self.generator.process_decode_output_host(tt_out.read, tt_out.rows).reshape(tt_out.rows, 1)
        if self.timeline is not None:  # the step's tokens are on the host: its completion boundary
            self.timeline.add({"k": "c", "i": tt_out.step, "t": time.perf_counter()})
        return out
