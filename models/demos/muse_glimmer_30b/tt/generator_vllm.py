# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for Muse-Glimmer-30B: one 1 x TP replica, vLLM-owned paged KV, traced decode, device sampling.

Selected with TT_MODEL_CLASS_OVERRIDES (no vllm_metadata.json here: an EXTRA_MODELS_DIR bundle would shadow
muse_glimmer_30b_qb2, which registers the same architecture from that directory).
"""

import os
from contextlib import contextmanager

import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingParams, format_sampling_params
from models.demos.muse_glimmer_30b.tt.model import DECODE_B_PAD, MuseGlimmerModel
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs

# Per-device KV budget when MUSE_VLLM_KV_TOKENS is unset: 52 layers x 1 KV head x 128 x 2 (K,V) x 2 B = 26.6 KB/token
# per device, so 262144 tokens take ~7 GB of each chip's DRAM next to ~8 GB (TP4) of BFP8 weights.
DEFAULT_KV_TOKENS = 262144

# Decode traces are captured per active-slot bucket so paged SDPA splits its cores over the slots in use only, as the
# native batch-1 path does (decode_forward slices SDPA inputs to args.max_batch_size rows).
SLOT_BUCKETS = (1, 2, 4, 8, 16, 32)
SLOT_TRACES = os.environ.get("MUSE_VLLM_SLOT_TRACES", "1") == "1"
# Decode reload contract v1 + async decode (vllm-tt-plugin docs/DECODE_RELOAD_CONTRACT.md); MUSE_VLLM_V1=0 keeps the
# legacy every-step reload for A/B runs.
CONTRACT_V1 = os.environ.get("MUSE_VLLM_V1", "1") == "1"


def prefill_buckets(max_len: int, step_div: int | None = None) -> list[int]:
    """Prefill lengths compiled at warmup.

    Above 256 the step is the largest power of two <= bucket / step_div (min 256, the prefill SDPA chunk), so a prompt
    just past a bucket pads by at most 1/step_div: 33 shapes and 25 % at the default 4 (up to 128K), about 57 shapes
    and 12.5 % at 8. step_div=1 gives power-of-two buckets.
    """
    step_div = step_div or int(os.environ.get("MUSE_VLLM_PREFILL_STEP_DIV", 4))
    buckets, n = [128, 256], 256
    while n < max_len:
        step = 256
        while step * 2 <= n // step_div:
            step *= 2
        n += step
        buckets.append(min(n, max_len))
    return [b for b in buckets if b <= max_len] or [max_len]


def _checkpoint_dir(hf_config) -> str:
    if os.environ.get("HF_MODEL"):
        return os.environ["HF_MODEL"]
    path = getattr(hf_config, "_name_or_path", "")
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(path, revision=getattr(hf_config, "_commit_hash", None))


def _row(value, i):
    return value[i] if isinstance(value, (list, tuple, torch.Tensor)) else value


class MuseGlimmerForConditionalGeneration:
    # v1: decode_and_sample writes the sampled token into the resident token buffer and advances the resident position
    # inside the trace, so steady device-sampled steps reload nothing. Unseeded per-core PRNG state cannot be moved by a
    # slot remap; it is reinitialized on a commanded sampling-state reset.
    decode_input_update_contract = 1 if CONTRACT_V1 else 0
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_sample_on_device": True,
        "supports_async_decode": CONTRACT_V1,
        "max_device_top_k": 32,
    }

    # vLLM's registry checks that __init__ takes vllm_config / prefix; the plugin builds the adapter through
    # initialize_vllm_model instead.
    def __init__(self, model: MuseGlimmerModel = None, *, vllm_config=None, prefix=""):
        self.model = model
        self.args = model.args if model is not None else None
        self.mesh_device = model.mesh_device if model is not None else None
        self.batch_size = DECODE_B_PAD
        self.n_blocks = self.args.max_num_blocks_per_user if model is not None else None
        self.buckets = prefill_buckets(self.args.max_prefill_len) if model is not None else None
        self.decode_inputs = None
        self.trace_ids = {}
        self.trace_outputs = {}
        self._sampling_key = None
        self._prefill_warm = False
        self._slots = DECODE_B_PAD
        self._active = []

    # vLLM's registry inspects these; execution only goes through prefill_forward / decode_forward.
    def embed_input_ids(self, input_ids):
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def forward(self, input_ids, positions):
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def compute_logits(self, hidden_states):
        raise RuntimeError("The TT model owns the LM head")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=131072, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError("Muse-Glimmer serving runs one 1 x TP replica per server (tt_data_parallel=1)")
        if optimizations is not None:
            raise ValueError("Muse-Glimmer serving has no optimization presets")
        if not 1 <= max_batch_size <= DECODE_B_PAD:
            raise ValueError(f"max_num_seqs must be 1..{DECODE_B_PAD}, got {max_batch_size}")
        shape = tuple(mesh_device.shape)
        if shape[0] != 1:
            raise ValueError(f"Muse-Glimmer serving needs a (1, TP) mesh, got {shape}")
        max_seq_len = -(-max_seq_len // 256) * 256
        args = MuseGlimmerModelArgs.from_checkpoint(
            _checkpoint_dir(hf_config),
            mesh_device=mesh_device,
            max_batch_size=DECODE_B_PAD,
            max_seq_len=max_seq_len,
            max_prefill_len=max_seq_len,
        )
        model = MuseGlimmerModel(args, mesh_device, allocate_kv=False)
        logger.info(f"Muse-Glimmer vLLM adapter: tp={args.tp} max_seq_len={max_seq_len} max_num_seqs={max_batch_size}")
        return cls(model)

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=131072, max_num_seqs=32, **kwargs):
        return min(int(os.environ.get("MUSE_VLLM_KV_TOKENS", DEFAULT_KV_TOKENS)), max_num_seqs * max_model_len)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """vLLM's cache shape [num_blocks, kv_heads_local, block_size, head_dim] is the model's paged layout."""
        num_blocks, kv_heads, block_size, head_dim = kv_cache_shape
        if (kv_heads, head_dim) != (self.args.kv_heads_local, self.args.head_dim):
            raise ValueError(f"Unexpected KV cache shape {kv_cache_shape}")
        if num_layers != self.args.n_layers:
            raise ValueError(f"vLLM asks for {num_layers} KV layers, model has {self.args.n_layers}")
        self.args.block_size = block_size
        self.n_blocks = self.args.max_num_blocks_per_user
        cache = [
            tuple(
                ttnn.zeros(
                    list(kv_cache_shape),
                    device=self.mesh_device,
                    dtype=self.args.kv_cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
            for _ in range(num_layers)
        ]
        self.model.kv_caches = cache
        self.decode_inputs = self.model.allocate_decode_inputs(self.batch_size, self.n_blocks)
        return cache

    def _device_params(self, sampling_params, rows):
        """vLLM per-row sampling params -> device SamplingParams for lanes 0..len(rows)-1."""
        temperature, top_k, top_p = [], [], []
        for i in rows:
            t = float(_row(sampling_params.temperature, i))
            greedy = t <= 0
            temperature.append(0.0 if greedy else t)
            top_k.append(1 if greedy else int(_row(sampling_params.top_k, i)) or DECODE_B_PAD)
            top_p.append(1.0 if greedy else float(_row(sampling_params.top_p, i)))
        return format_sampling_params(SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p), DECODE_B_PAD)

    def _load_sampling(self, sampling_params, rows, force=False):
        params = self._device_params(sampling_params, rows)
        key = (tuple(params.temperature), tuple(params.top_k), tuple(params.top_p))
        if force or key != self._sampling_key:
            self.model.sampling.reset_sampling_params(params)
            self._sampling_key = key

    def _page_table_tt(self, row: torch.Tensor, width: int):
        pt = torch.zeros(1, width, dtype=torch.int32)
        n = min(width, row.shape[0])
        pt[0, :n] = row[:n].to(torch.int32)
        return ttnn.from_torch(
            pt,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _bucket(self, length: int) -> int:
        for b in self.buckets:
            if b >= length:
                return b
        raise ValueError(f"Prompt of {length} tokens exceeds max_prefill_len {self.args.max_prefill_len}")

    def _prefill_one(self, tokens: torch.Tensor, page_row: torch.Tensor):
        pad_len = self._bucket(tokens.shape[0])
        # Padded rows past the request's blocks map to block 0, vLLM's never-allocated null block.
        pt = self._page_table_tt(page_row, -(-pad_len // self.args.block_size))
        logits = self.model.prefill_forward_logits_tt(tokens, 0, pt, pad_len=pad_len)
        ttnn.deallocate(pt)
        return logits

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        enable_trace=False,
        **kwargs,
    ):
        if start_pos is not None and any(int(s) != 0 for s in torch.as_tensor(start_pos).flatten().tolist()):
            raise ValueError("Muse-Glimmer prefill starts at position 0 (no chunked prefill or prefix caching)")
        n = len(prompt_lens)
        lens = [int(x) for x in torch.as_tensor(prompt_lens).flatten().tolist()]
        out = []
        for i in range(n):
            logits = self._prefill_one(tokens[i, : lens[i]], page_table[i])
            if sampling_params is not None:
                self._load_sampling(sampling_params, [i])
                tok = self.model.sample_on_device(logits, None)
                out.append(self.model.tokens_to_torch(tok, 1))
                ttnn.deallocate(tok)
            else:
                out.append(self.model.logits_to_torch(logits, 1))
            ttnn.deallocate(logits)
        if sampling_params is not None:
            # Decode reloads the whole batch's parameters on its next step.
            self._sampling_key = None
            return torch.cat(out).reshape(n, 1), None
        return torch.cat(out).reshape(n, -1), None

    @staticmethod
    def _slot_bucket(start_pos: torch.Tensor) -> int:
        if not SLOT_TRACES:
            return DECODE_B_PAD
        live = (start_pos.reshape(-1) >= 0).nonzero()
        hi = int(live.max()) + 1 if live.numel() else 1
        return next(b for b in SLOT_BUCKETS if b >= hi)

    @contextmanager
    def _sdpa_slots(self, slots: int):
        old = self.args.max_batch_size
        self.args.max_batch_size = slots
        try:
            yield
        finally:
            self.args.max_batch_size = old

    def _run(self, sample: bool, slots: int, enable_trace: bool):
        fn = self.model.decode_and_sample if sample else self.model.decode_forward
        if not enable_trace:
            with self._sdpa_slots(slots):
                return fn(*self.decode_inputs)
        key = ("sample" if sample else "logits", slots)
        if key not in self.trace_ids:
            # Eager compile run, then capture. Both runs advance the resident token/position like a real step, so
            # capture only happens at warmup or on a step whose inputs are reloaded right after.
            with self._sdpa_slots(slots):
                ttnn.deallocate(fn(*self.decode_inputs))
                ttnn.synchronize_device(self.mesh_device)
                self.trace_ids[key] = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                self.trace_outputs[key] = fn(*self.decode_inputs)
                ttnn.end_trace_capture(self.mesh_device, self.trace_ids[key], cq_id=0)
            return None
        ttnn.execute_trace(self.mesh_device, self.trace_ids[key], cq_id=0, blocking=False)
        return self.trace_outputs[key]

    def _page_table_host(self, page_table: torch.Tensor, rows: int) -> torch.Tensor:
        pt = torch.zeros(rows, self.n_blocks, dtype=torch.int32)
        r = min(rows, page_table.shape[0])
        w = min(self.n_blocks, page_table.shape[1])
        pt[:r, :w] = page_table[:r, :w].to(torch.int32)
        return pt

    def _copy_decode_inputs(self, tokens, start_pos, page_table):
        b = min(tokens.shape[0], self.batch_size)
        pt = self._page_table_host(page_table, b)
        host = self.model.prepare_decode_inputs_host(tokens.reshape(-1)[:b], start_pos.reshape(-1)[:b], pt)
        self.model.copy_decode_inputs(host, self.decode_inputs)

    def _copy_page_table(self, page_table):
        host = ttnn.from_torch(
            self._page_table_host(page_table, self.batch_size),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self.decode_inputs[3])

    def _sampling_state(self, sampling_params, start_pos, reload_inputs, reload_sampling_params, reset_sampling_state):
        """Device sampler commands, in the order tt_transformers' generator applies them."""
        s = self.model.sampling
        rows = range(torch.as_tensor(start_pos).numel())
        if not self.decode_input_update_contract:
            self._load_sampling(sampling_params, rows)
            s.seed_manager.get_new_values()
            return
        s.validate_decode_state_commands(
            reload_sampling_params=reload_sampling_params, reset_sampling_state=reset_sampling_state
        )
        if reload_sampling_params:
            self._load_sampling(sampling_params, rows, force=True)
        seeds = format_sampling_params(sampling_params, s.seed_manager.max_batch_size).seed
        start_values = torch.as_tensor(start_pos).reshape(-1).tolist()
        if reload_inputs:
            s.seed_manager.deactivate_slots_except(self._active)
        if reset_sampling_state:
            # Unconditional even for seed=None, so decode-only sampling gets a fresh device seed upload.
            s.seed_manager.reset_seed_from_slots(seeds, self._active)
            reseeded = list(self._active)
        elif reload_sampling_params:
            reseeded = s.seed_manager.reset_seed_from_slots_if_needed(seeds, self._active)
        else:
            reseeded = []
        align = self._active if reload_inputs else reseeded
        if align:
            s.seed_manager.align_seed_counters_to_positions(seeds, align, start_values)
        s.commit_decode_state_commands(
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
            sampling_state_slots=None,
        )
        s.seed_manager.get_new_values(self._active)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        sampling_params=None,
        enable_trace=True,
        read_from_device=True,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        slot_remap=None,
        **kwargs,
    ):
        if self.decode_input_update_contract:
            if "reset_batch" in kwargs:
                raise TypeError("reset_batch is the legacy decode keyword; this adapter declares contract v1")
            if reload_inputs and reload_page_table:
                raise ValueError("reload_inputs and reload_page_table are exclusive")
            if reset_sampling_state and not reload_inputs:
                raise ValueError("reset_sampling_state requires reload_inputs")
        else:
            reload_inputs, reload_page_table = True, False
        sample = sampling_params is not None
        if not sample and not reload_inputs:
            raise ValueError("Host-sampled decode needs authoritative inputs (reload_inputs=True)")
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap).reshape(-1).tolist()
            remap += list(range(len(remap), DECODE_B_PAD))
            if remap != list(range(DECODE_B_PAD)):
                if not reload_inputs:
                    raise ValueError("A slot remap requires reload_inputs=True")
                # Model state is paged by page table, not slot-indexed; only the sampler holds slot state.
                if self.model.sampling is not None:
                    self.model.sampling.apply_slot_remap(remap)
                self._sampling_key = None
        if reload_inputs:
            start = torch.as_tensor(start_pos).reshape(-1)
            self._copy_decode_inputs(torch.as_tensor(tokens), start, torch.as_tensor(page_table))
            self._slots = self._slot_bucket(start)
            self._active = [i for i, p in enumerate(start.tolist()) if p >= 0]
        elif reload_page_table:
            self._copy_page_table(torch.as_tensor(page_table))
        if sample:
            self._sampling_state(
                sampling_params, start_pos, reload_inputs, reload_sampling_params, reset_sampling_state
            )
        out = self._run(sample, self._slots, enable_trace)
        if out is None:
            # First use of this trace outside warmup: capture advanced the resident inputs. Only host-authoritative
            # inputs can restore them.
            if not reload_inputs:
                raise RuntimeError(f"Decode trace {('sample' if sample else 'logits', self._slots)} was not captured")
            self._copy_decode_inputs(torch.as_tensor(tokens), torch.as_tensor(start_pos), torch.as_tensor(page_table))
            out = self._run(sample, self._slots, enable_trace)
        result = (out, sample, enable_trace)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(result), is_tokens=sample)
        return result

    def read_decode_output(self, tt_out, async_read=False, **kwargs):
        tensor, is_tokens, traced = tt_out
        # Sampled tokens are replicated: read one device's copy. Logits stay sharded over the vocab.
        src = ttnn.get_device_tensors(tensor)[0] if is_tokens else tensor
        host = src.cpu(blocking=not async_read, cq_id=0)
        if not traced:
            ttnn.deallocate(tensor)
        result = (host, is_tokens, traced)
        return (result, [ttnn.record_event(self.mesh_device, 0)]) if async_read else result

    def process_decode_output_host(self, tt_out, is_tokens=False, **kwargs):
        if ttnn.is_tensor_storage_on_device(tt_out[0]):
            # Synchronous decode hands over the device output unread.
            tt_out = self.read_decode_output(tt_out)
        host, tokens, _ = tt_out
        if tokens:
            out = ttnn.to_torch(host).reshape(-1)[: self.batch_size].to(torch.int64)
            return out.reshape(-1, 1), None
        logits = ttnn.to_torch(host, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        return logits[0, 0, : self.batch_size, : self.args.vocab_size].float().reshape(self.batch_size, 1, -1), None

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, **kwargs):
        """Compile every prefill bucket once (prefill is not traced, so no shape may first compile while serving)."""
        if self._prefill_warm:
            return
        cap = int(os.environ.get("MUSE_VLLM_PREFILL_WARMUP_MAX", self.buckets[-1]))
        null_row = torch.zeros(1, dtype=torch.int32)
        greedy = SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0])
        for b in self.buckets:
            if b > cap:
                logger.warning(f"Prefill bucket {b} not warmed up (MUSE_VLLM_PREFILL_WARMUP_MAX={cap})")
                continue
            logits = self._prefill_one(torch.zeros(b, dtype=torch.int64), null_row)
            if can_sample_on_device:
                self._load_sampling(greedy, [0])
                ttnn.deallocate(self.model.sample_on_device(logits, None))
            ttnn.deallocate(logits)
            logger.info(f"Prefill bucket {b} warmed up")
        self._sampling_key = None
        self._prefill_warm = True

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        """Compile (eager) or capture (trace) every slot bucket for host and device sampling."""
        # Position -1 marks every slot inactive: the cache update and SDPA skip them, so warmup writes no KV.
        tokens = torch.zeros(self.batch_size, dtype=torch.int64)
        positions = torch.full((self.batch_size,), -1, dtype=torch.int64)
        page_table = torch.zeros(self.batch_size, 1, dtype=torch.int32)
        greedy = SamplingParams(
            temperature=[0.0] * self.batch_size, top_k=[1] * self.batch_size, top_p=[1.0] * self.batch_size
        )
        self._copy_decode_inputs(tokens, positions, page_table)
        if can_sample_on_device:
            self._load_sampling(greedy, range(self.batch_size), force=True)
        slot_sets = SLOT_BUCKETS if SLOT_TRACES else (DECODE_B_PAD,)
        for sample in [False, True] if can_sample_on_device else [False]:
            for slots in slot_sets:
                out = self._run(sample, slots, enable_trace)
                if out is not None and not enable_trace:
                    ttnn.deallocate(out)
            logger.info(f"Decode warmed up (trace={enable_trace}, device_sampling={sample}, slots={slot_sets})")
        ttnn.synchronize_device(self.mesh_device)
        self._sampling_key = None

    def release_persistent_capture(self):
        for tid in self.trace_ids.values():
            ttnn.release_trace(self.mesh_device, tid)
        self.trace_ids.clear()
        self.trace_outputs.clear()
