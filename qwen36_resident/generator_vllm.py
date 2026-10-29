# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter (vllm-tt-plugin) for the resident Qwen3.6-27B on a 1 x 4 Blackhole mesh.

One user: the prompt but its last token goes through the pipeline-parallel prefill, the state is handed to
the resident decode model, and the last prompt token runs one decode step for the first logits. The model
owns its KV cache, GDN state and conv history, so the plugin's paged KV cache is not used.

Decode steps with temperature 0 or top-k <= 32 (and no penalties, log-probs or other host-only parameters)
are sampled on device and run asynchronously (decode input update contract 1): each step's hub draws the next
token from the chips' top 32 logits and writes the next step's token state, so on a step without reload_inputs
the adapter only replays the trace, and the runner reads the (token, position) word one step late. Any other
step samples on the host from the full logits, with the host writing the token state.

Several users (max_num_seqs > 1): initialize_vllm_model returns Qwen36ResidentRowsForCausalLM instead, the
rows-mode build (launches of 8 independent users, paged KV pool) behind the host bookkeeping of vllm_rows.py;
max_num_seqs 1 keeps the single-user build above unchanged.

Qwen36ResidentMTPForCausalLM serves greedy MTP speculative decode (qwen36_resident.mtp) through the
plugin's adaptive block output: every decode step commits MTP_BLOCK tokens, the device running MTP_DEPTH
verify + draft steps ahead of the host so it keeps decoding during the runner's per-step work.
"""
import os

import torch
from loguru import logger
from vllm.transformers_utils.config import uses_mrope

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.common import HIDDEN
from qwen36_resident.model import TOP_K, Dims, ResidentModel, State
from qwen36_resident.mtp import SpecDecoder
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.vllm_rows import ResidentRowsAdapter, RowsEngine

PREFILL_CHUNKS = (128, 256, 512, 1024)


class Qwen36ResidentForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        # the hub samples from the top TOP_K logits; anything else samples on the host from the logits
        "supports_sample_on_device": True,
        "max_device_top_k": TOP_K,
        "supports_device_penalties": False,
        # several users (Qwen36ResidentRowsForCausalLM): the PP prefill resumes a split prompt only at multiples of
        # its largest chunk and holds one prompt at a time (the plugin's TT chunk core; off at max_num_seqs 1)
        "supports_chunked_prefill": True,
        "tt_prefill_chunk_tokens": max(PREFILL_CHUNKS),
        "tt_prefill_exclusive_chunks": True,
    }
    decode_input_update_contract = 1

    def __init__(self, mesh, ck, max_seq_len, mrope):
        self.mesh, self.mrope = mesh, mrope
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        self.vocab = cfg["vocab_size"]
        n = mesh.get_num_devices()
        self.d = Dims(n, mesh.dram_grid_size().x, vocab=self.vocab)
        self.emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-max_seq_len // C) * C
        n_attn = sum((l + 1) % interval == 0 for l in range(layers))
        st = State(self.d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True, kv=False)
        # the prefill's embedding table is also the one the decode hub feeds from
        self.pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        w = Q.load(ck, self.d, layers, interval)
        self.model = ResidentModel(mesh, self.d, w, st, layers, interval, lm_head=True, embed=self.pp.embed)
        del w
        self.max_seq_len = max_seq_len
        self.out_view = ttnn.get_device_tensors(self.model.out_t)[0]
        self.out_range = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))
        self.model.step(self.model.token_state(self.emb[0].float(), 0))
        self.model.reset()
        self.model.capture_trace()
        self.pp.capture()
        self.pp.handoff(self.model, 1)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, **kwargs):
        # the served revision: a cache filled for a pinned commit need not hold refs/main
        path = Q.checkpoint_dir(
            os.environ.get("HF_MODEL") or hf_config._name_or_path, getattr(hf_config, "_commit_hash", None)
        )
        ck = Q.Checkpoint(path)
        if max_batch_size > 1 and cls is Qwen36ResidentForCausalLM:
            logger.info(f"Building the resident Qwen3.6 model for {max_batch_size} users of up to {max_seq_len} tokens")
            return Qwen36ResidentRowsForCausalLM(mesh_device, ck, max_seq_len, max_batch_size, uses_mrope(hf_config))
        assert max_batch_size == 1, f"{cls.__name__} serves one user: max_num_seqs=1"
        logger.info(f"Building the resident Qwen3.6 model for up to {max_seq_len} tokens")
        return cls(mesh_device, ck, max_seq_len, mrope=uses_mrope(hf_config))

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, max_num_seqs=None, **kwargs):
        """the KV pool in tokens: max_model_len per user unless RESIDENT_MAX_TOKENS_ALL_USERS sets it (several users;
        8.7 KB per token per chip)"""
        users = int(max_num_seqs or 1)
        if users > 1 and os.environ.get("RESIDENT_MAX_TOKENS_ALL_USERS"):
            return int(os.environ["RESIDENT_MAX_TOKENS_ALL_USERS"])
        return int(max_model_len) * users

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return []

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass

    def _write_token(self, token, pos):
        assert pos < self.max_seq_len, f"position {pos} is past max_model_len {self.max_seq_len}"
        return self.model.token_state(self.emb[int(token)].float(), int(pos))

    def _step(self, token, pos):
        self.model.step(self._write_token(token, pos))
        ttnn.synchronize_device(self.mesh)
        # per chip [n, padded vocab slice]: drop the padding, then concatenate in vocab order
        logits = self.model.logits()[:, : self.d.vocab_chip].reshape(-1)[: self.vocab]
        return logits.view(1, 1, -1)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, **kwargs):
        assert tokens.shape[0] == 1, "one user per prefill"
        if start_pos is not None and int(start_pos[0]) > 0:
            # the chunk core of vllm-tt-plugin resident-mu keeps one user's prompts whole; an older plugin splits
            # prompts for a model that declares supports_chunked_prefill
            raise RuntimeError("one user: a prompt is prefilled whole (serve with vllm-tt-plugin resident-mu)")
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        ids = tokens[0, :T]
        self.model.reset()
        self.pp.reset()
        if T > 1:
            self.pp.run(ids[: T - 1])
            self.pp.handoff(self.model, T - 1)
        logits = self._step(ids[T - 1], T - 1)
        # the runner keeps a per-request M-RoPE delta when the config has one; text only, so it is 0
        return (logits, torch.zeros(1, dtype=torch.long)) if self.mrope else logits

    def decode_forward(
        self,
        tokens,
        start_pos,
        *args,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        read_from_device=False,
        sampling_params=None,
        **kwargs,
    ):
        assert "reset_batch" not in kwargs, "contract 1 adapters take the explicit reload commands"
        token, pos = tokens.reshape(-1)[0], start_pos.reshape(-1)[0]
        # sampling_params arrive only on the steps the runner samples on device (top-k <= TOP_K or greedy)
        if sampling_params is None:
            return self._step(token, pos)
        if reload_sampling_params:
            self.model.set_sampling(
                float(sampling_params.temperature[0]),
                int(sampling_params.top_k[0]),
                float(sampling_params.top_p[0]),
                sampling_params.seed[0],
            )
        # without reload_inputs, tokens and start_pos are one step behind: the hub wrote the token state
        self.model.step(self._write_token(token, pos) if reload_inputs else None)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(self.model.out_t))
        return self.model.out_t

    def read_decode_output(self, out, async_read=False, **kwargs):
        """enqueue the read of the fed (token, position) word on chip 0, ahead of the next replay"""
        if isinstance(out, torch.Tensor):
            return (out, []) if async_read else out
        host = ttnn.from_device(self.out_view, blocking=not async_read)
        return (host, [ttnn.record_event(self.mesh, 0, device_range=self.out_range)]) if async_read else host

    def process_decode_output_host(self, out, is_tokens=True, **kwargs):
        if isinstance(out, torch.Tensor):
            return out
        if ttnn.is_tensor_storage_on_device(out):
            out = ttnn.from_device(ttnn.get_device_tensors(out)[0])
        return ttnn.to_torch(out).reshape(-1)[:1].to(torch.int32).view(1, 1)


class Qwen36ResidentRowsForCausalLM(ResidentRowsAdapter):
    """Several users: a rows-mode ResidentModel (launches of 8 rows, paged KV pool of the plugin's block count, one
    trace per launch), built in allocate_kv_cache once the plugin knows its block count, and the PP prefill. Device
    sampling needs the rows-mode hub feed (WP7): without it, launch without sample_on_device_mode (host sampling)."""

    model_capabilities = dict(Qwen36ResidentForCausalLM.model_capabilities)

    def __init__(self, mesh, ck, max_seq_len, max_batch, mrope):
        cfg = ck.config
        self.mesh, self.ck = mesh, ck
        self.layers, self.interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        n = mesh.get_num_devices()
        self.d = Dims(n, mesh.dram_grid_size().x, vocab=cfg["vocab_size"])
        self.emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        self.max_len = -(-max_seq_len // C) * C
        self.pp = PPPrefill(mesh, ck, self.layers, self.interval, chunk=list(PREFILL_CHUNKS), max_len=self.max_len)
        super().__init__(None, cfg["vocab_size"], max_seq_len, max_batch, C, mrope)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """build the rows-mode model on a KV pool of the plugin's block count (its block ids index the pool)"""
        nb = int(kv_cache_shape[0])
        n_attn = sum((l + 1) % self.interval == 0 for l in range(self.layers))
        sts = [
            State(self.d, self.layers - n_attn, n_attn, 0, max_pos=self.max_len + 64, zero=True, kv=False)
            for _ in range(self.n_slots)
        ]
        logger.info(f"Resident rows model: {self.n_slots} slots, KV pool {nb} blocks")
        w = Q.load(self.ck, self.d, self.layers, self.interval)
        # every slot starts on block 0 (vLLM's null block): the page tables come with each request
        model = ResidentModel(
            self.mesh,
            self.d,
            w,
            sts,
            self.layers,
            self.interval,
            lm_head=True,
            kv_pool_blocks=nb,
            block_tables=[[0]] * self.n_slots,
            rows=True,
        )
        del w
        model.capture_trace()
        self.pp.capture()
        self.pp.handoff(model, 1, block_table=[0], slot=0)  # compiles the handoff kernels
        self.engine = RowsEngine(
            model,
            self.pp,
            self.emb,
            self.vocab,
            self.d.vocab_chip,
            HIDDEN,
            self.n_slots,
            self.chunk,
            sync=lambda: ttnn.synchronize_device(self.mesh),
        )
        return []

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass


MTP_BLOCK = int(os.environ.get("QWEN36_MTP_BLOCK", "4"))
MTP_DEPTH = int(os.environ.get("QWEN36_MTP_DEPTH", "4"))


class Qwen36ResidentMTPForCausalLM(Qwen36ResidentForCausalLM):
    """Greedy only: the device decodes from its own greedy tokens, so the runner's first token (sampled from
    the prefill's logits) must be their argmax; anything else raises. One user (max_num_seqs=1), so every
    request owns the scheduler's block session and every decode step is a block step."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        # decode returns the committed tokens
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "output_tokens_per_step": MTP_BLOCK,
        "tt_adaptive_block_output": True,
        # positions one block step may run ahead: the block, then MTP_DEPTH steps of up to 2 each
        "tt_block_kv_extent_tokens": MTP_BLOCK + 2 * MTP_DEPTH + 2,
    }

    def __init__(self, mesh, ck, max_seq_len, mrope):
        self.mesh, self.mrope = mesh, mrope
        cfg = ck.config
        self.vocab = cfg["vocab_size"]
        self.max_seq_len = max_seq_len
        self.eos = int(cfg["eos_token_id"])
        self.spec = SpecDecoder(mesh, ck, cfg["num_hidden_layers"], max_seq_len, prefill=True)
        self._first = None

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        assert tokens.shape[0] == 1, "one user per prefill"
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        logits = self.spec.start(tokens[0, :T].tolist())
        self._first = int(logits.argmax())
        # the plugin samples a block-output model's prefill logits at temperature 1, whatever the request
        # asks: only the argmax the device decodes from may be sampled
        logits = torch.full_like(logits, float("-inf"))
        logits[self._first] = 0
        logits = logits.view(1, 1, -1)
        return (logits, torch.zeros(1, dtype=torch.long)) if self.mrope else logits

    def decode_forward(self, tokens, start_pos, *args, **kwargs):
        if self._first is not None:
            token = int(tokens.reshape(-1)[0])
            if token != self._first:
                raise RuntimeError(
                    f"MTP serving is greedy only: the runner's first token {token} is not the argmax "
                    f"{self._first} the device decodes from"
                )
            self._first = None
        block = self.spec.next_block(MTP_BLOCK, MTP_DEPTH, self.max_seq_len)
        # short only at the position limit: the scheduler trims at the first stop token
        out = torch.full((1, MTP_BLOCK), self.eos, dtype=torch.int32)
        out[0, : len(block)] = torch.tensor(block, dtype=torch.int32)
        return out

    def note_state_slots_moved(self, moves):
        pass

    def release_request(self, row):
        self.spec.stop()
        self._first = None

    def release_persistent_capture(self):
        self.spec.stop()
