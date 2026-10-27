# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for Muse-Glimmer-30B: one 1 x TP replica, vLLM-owned paged KV, traced decode, device sampling.

Selected with TT_MODEL_CLASS_OVERRIDES (no vllm_metadata.json here: an EXTRA_MODELS_DIR bundle would shadow
muse_glimmer_30b_qb2, which registers the same architecture from that directory).
"""

import os

import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingParams, format_sampling_params
from models.demos.muse_glimmer_30b.tt.model import DECODE_B_PAD, MuseGlimmerModel
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs

# Per-device KV budget when MUSE_VLLM_KV_TOKENS is unset: 52 layers x 1 KV head x 128 x 2 (K,V) x 2 B = 26.6 KB/token
# per device, so 262144 tokens take ~7 GB of each chip's DRAM next to ~8 GB (TP4) of BFP8 weights.
DEFAULT_KV_TOKENS = 262144


def prefill_buckets(max_len: int) -> list[int]:
    """Prefill lengths compiled at warmup: 128, powers of two to 32K, then 32K steps (chunked SDPA above 32K)."""
    buckets, n = [128], 256
    while n <= min(max_len, 32768):
        buckets.append(n)
        n *= 2
    n = 65536
    while n <= max_len:
        buckets.append(n)
        n += 32768
    if buckets[-1] < max_len:
        buckets.append(max_len)
    return buckets


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
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_sample_on_device": True,
        "supports_async_decode": False,
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

    def _load_sampling(self, sampling_params, rows):
        params = self._device_params(sampling_params, rows)
        key = (tuple(params.temperature), tuple(params.top_k), tuple(params.top_p))
        if key != self._sampling_key:
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

    def _capture(self, key, fn):
        """Eager compile run, then capture; restores the decode inputs the eager run advanced."""
        ttnn.deallocate(fn(*self.decode_inputs))
        ttnn.synchronize_device(self.mesh_device)
        self.trace_ids[key] = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.trace_outputs[key] = fn(*self.decode_inputs)
        ttnn.end_trace_capture(self.mesh_device, self.trace_ids[key], cq_id=0)

    def _copy_decode_inputs(self, tokens, start_pos, page_table):
        b = min(tokens.shape[0], self.batch_size)
        pt = torch.zeros(b, self.n_blocks, dtype=torch.int32)
        w = min(self.n_blocks, page_table.shape[1])
        pt[:, :w] = page_table[:b, :w].to(torch.int32)
        host = self.model.prepare_decode_inputs_host(tokens.reshape(-1)[:b], start_pos.reshape(-1)[:b], pt)
        self.model.copy_decode_inputs(host, self.decode_inputs)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        sampling_params=None,
        enable_trace=True,
        read_from_device=True,
        **kwargs,
    ):
        sample = sampling_params is not None
        start_pos = torch.as_tensor(start_pos)
        self._copy_decode_inputs(torch.as_tensor(tokens), start_pos, torch.as_tensor(page_table))
        if sample:
            self._load_sampling(sampling_params, range(start_pos.numel()))
            self.model.sampling.seed_manager.get_new_values()
        key = "sample" if sample else "logits"
        fn = self.model.decode_and_sample if sample else self.model.decode_forward
        if not enable_trace:
            out = fn(*self.decode_inputs)
        else:
            if key not in self.trace_ids:
                self._capture(key, fn)
                self._copy_decode_inputs(torch.as_tensor(tokens), start_pos, torch.as_tensor(page_table))
            ttnn.execute_trace(self.mesh_device, self.trace_ids[key], cq_id=0, blocking=False)
            out = self.trace_outputs[key]
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
        # Position -1 marks every slot inactive: the cache update and SDPA skip them, so warmup writes no KV.
        tokens = torch.zeros(self.batch_size, dtype=torch.int64)
        positions = torch.full((self.batch_size,), -1, dtype=torch.int64)
        page_table = torch.zeros(self.batch_size, 1, dtype=torch.int32)
        greedy = SamplingParams(
            temperature=[0.0] * self.batch_size, top_k=[1] * self.batch_size, top_p=[1.0] * self.batch_size
        )
        modes = [None, greedy] if can_sample_on_device else [None]
        for params in modes:
            self.decode_forward(
                tokens, positions, page_table, kv_cache, sampling_params=params, enable_trace=enable_trace
            )
            logger.info(f"Decode warmed up (trace={enable_trace}, device_sampling={params is not None})")

    def release_persistent_capture(self):
        for tid in self.trace_ids.values():
            ttnn.release_trace(self.mesh_device, tid)
        self.trace_ids.clear()
        self.trace_outputs.clear()
