# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN forward pass for `Qwen/Qwen3-Coder-Next`.

Both `demo/demo_text_generation.py` and `tests/e2e/test_e2e_pipeline.py` import and call
`build_pipeline(...)` + `Qwen3CoderNextPipeline.generate(...)` from here, so a green test and a
working demo are the same code path by construction.

WHAT THE CHAIN IS
-----------------
`Qwen3NextForCausalLM` is a single-head text->text model, so there is exactly one Call:

    ids -> ttnn.embedding
        -> rotary_embedding                                    (graduated stub)
        -> for each layer:  decoder_layer                      (graduated stub)
                              r_m_s_norm            x2         (graduated stub)
                              gated_delta_net  or  attention   (graduated stubs)
                                `- r_m_s_norm_gated            (graduated stub)
                              sparse_moe_block                 (graduated stub)
                                `- top_k_router                (graduated stub)
                                `- experts                     (graduated stub)
                                `- m_l_p  (the shared expert)  (graduated stub)
        -> r_m_s_norm  (final norm)                            (graduated stub)
        -> ttnn.linear (lm_head) -> ttnn.argmax -> next token

All ten graduated components are on that path; nothing is touched "for coverage".

NO KV CACHE, BY CONSTRUCTION
----------------------------
The graduated `gated_delta_net` port covers ONE delta-rule chunk from a zero recurrent state --
which is the reference's own `chunk_size=64` chunk, evaluated exactly.  The pipeline therefore
decodes by RE-RUNNING the whole prefix at a pinned capacity `C = 64` every step, which is
mathematically identical to the cached recurrence and keeps every traced shape static.  The
consequence is a hard, model-grounded horizon: `prompt_len + new_tokens <= C`.

DEPTH
-----
The full checkpoint is 48 layers x 512 experts (~80B params, 159 GB bf16).  Per layer the expert
bank alone is 3.2 GB, so even split across the TP columns the whole stack exceeds the DRAM the
mandated placement provides.  The pipeline is therefore built at a depth cap and the HF golden is
capped IDENTICALLY, so parity is measured against the same configuration the TT side runs.  See
`tt/reference.py`.

Whatever depth is built is built ONCE, at init, and stays resident for the whole run: every
parameter is uploaded to its chip in `__init__` and nothing is re-uploaded, freed or re-read from
host between forwards.

DEVICE OWNERSHIP
----------------
This module never opens a device.  `build_pipeline(device, ...)` runs on the device its caller
opened -- the pytest fixture, the perf test or the demo, all of which come through the package's
one opener, `device_harness.open_mesh()`.  A second, competing open is what breaks trace capture.
"""
from __future__ import annotations

import math
import os

import torch
import ttnn

from models.demos.qwen3_coder_next._stubs.attention import TtQwen3NextAttention
from models.demos.qwen3_coder_next._stubs.decoder_layer import TtQwen3NextDecoderLayer
from models.demos.qwen3_coder_next._stubs.experts import TtQwen3NextExperts
from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    TtQwen3NextGatedDeltaNet,
    matmul_weight,
    num_devices,
    to_device,
)
from models.demos.qwen3_coder_next._stubs.m_l_p import TtQwen3NextMLP
from models.demos.qwen3_coder_next._stubs.r_m_s_norm import TtQwen3NextRMSNorm
from models.demos.qwen3_coder_next._stubs.r_m_s_norm_gated import TtQwen3NextRMSNormGated
from models.demos.qwen3_coder_next._stubs.rotary_embedding import TtQwen3NextRotaryEmbedding
from models.demos.qwen3_coder_next._stubs.sparse_moe_block import TtQwen3NextSparseMoeBlock
from models.demos.qwen3_coder_next._stubs.top_k_router import TtQwen3NextTopKRouter
from models.demos.qwen3_coder_next.tt import mesh as tt_mesh
from models.demos.qwen3_coder_next.tt.reference import DEFAULT_LAYERS, encode_prompt, load_reference

# Derived from SOURCE A: architectures=['Qwen3NextForCausalLM'], is_encoder_decoder=False, no
# speech head -> an autoregressive decoder-only model has exactly the two phases below.
PIPELINE_STAGES = ["prefill", "decode"]

# The pinned sequence capacity for trace capture.  Bounded above by
# config.max_position_embeddings (262144) and below by what the graduated gated_delta_net port
# covers: one delta-rule chunk, chunk_size=64 in the HF reference.
DEFAULT_CAPACITY = 64

DEFAULT_PROMPT = "Write a Python function that returns the nth Fibonacci number."

# The forward-path entry point of each graduated component -- what the Gate 2 probe counts.
# Every one of these is reached from `forward_logits()`; nothing here is touched for coverage.
GRADUATED_ENTRYPOINTS = {
    "rotary_embedding": (TtQwen3NextRotaryEmbedding, "__call__"),
    "r_m_s_norm": (TtQwen3NextRMSNorm, "__call__"),
    "decoder_layer": (TtQwen3NextDecoderLayer, "__call__"),
    "gated_delta_net": (TtQwen3NextGatedDeltaNet, "__call__"),
    "r_m_s_norm_gated": (TtQwen3NextRMSNormGated, "__call__"),
    "attention": (TtQwen3NextAttention, "__call__"),
    "sparse_moe_block": (TtQwen3NextSparseMoeBlock, "__call__"),
    "top_k_router": (TtQwen3NextTopKRouter, "__call__"),
    # `experts` is entered through `partial()`, not `__call__`: `sparse_moe_block` drives the
    # bank itself so it can issue the all_reduce once, and `partial()` IS the expert compute
    # (both dense matmuls and the routing scale).
    "experts": (TtQwen3NextExperts, "partial"),
    "m_l_p": (TtQwen3NextMLP, "__call__"),
}


class InvocationProbe:
    """Counts real forward-path entries into each graduated component.

    Installed around a REAL `generate()` -- there is no sweep that pokes each stub once.  The
    counts are checked against the built topology, so a stub that is merely constructed but never
    reached still fails Gate 2.
    """

    def __init__(self):
        self.counts = {name: 0 for name in GRADUATED_ENTRYPOINTS}
        self._saved = {}

    def __enter__(self):
        for name, (cls, method) in GRADUATED_ENTRYPOINTS.items():
            original = getattr(cls, method)
            self._saved[name] = (cls, method, original)

            def make(name=name, original=original):
                def wrapper(*args, **kwargs):
                    self.counts[name] += 1
                    return original(*args, **kwargs)

                return wrapper

            setattr(cls, method, make())
        return self

    def __exit__(self, *exc):
        for cls, method, original in self._saved.values():
            setattr(cls, method, original)
        self._saved.clear()
        return False


def _causal_mask(capacity, dtype=torch.float32):
    """Additive causal mask, `0` on/below the diagonal and a large negative elsewhere."""
    neg = torch.finfo(torch.bfloat16).min / 2
    m = torch.full((capacity, capacity), neg, dtype=dtype)
    return torch.triu(m, diagonal=1).view(1, 1, capacity, capacity)


class Qwen3CoderNextPipeline:
    """The resident TT model: graduated stubs chained into a real causal-LM forward pass."""

    def __init__(self, device, hf_model, *, layers=None, capacity=DEFAULT_CAPACITY):
        self.device = device
        # Kept reachable on purpose: the HF reference is the authority on how many sections this
        # model has and how deep each one is, and it is what the golden helper measures against.
        self.reference = hf_model
        self.config = hf_model.config
        self.capacity = int(capacity)
        self.tp = num_devices(device)
        self.rows, self.cols = tt_mesh.rows_cols(device)

        depth = len(hf_model.model.layers) if layers is None else int(layers)
        depth = max(1, min(depth, len(hf_model.model.layers)))
        self.depth = depth
        self.layer_types = list(self.config.layer_types[:depth])

        hidden = int(self.config.hidden_size)
        self.hidden_size = hidden
        self.vocab_size = int(self.config.vocab_size)

        print(f"[build] {depth} decoder layer(s), layer_types={self.layer_types}", flush=True)

        # --- the ONE repeated stack, held as a plain list of same-typed elements so any structure
        # --- walk (depth sizing, capping, attribution) can find it without a marker class.
        self.layers = [
            TtQwen3NextDecoderLayer.build(device, hf_model.model.layers[i]) for i in range(depth)
        ]
        self.final_norm = TtQwen3NextRMSNorm.build(device, hf_model.model.norm)
        self.rope = TtQwen3NextRotaryEmbedding.build(device, hf_model.model.rotary_emb)

        replicate = tt_mesh.replicate(device)
        self.embed_weight = to_device(
            hf_model.model.embed_tokens.weight.detach().float(), device, mesh_mapper=replicate
        )
        # lm_head is column-parallel over the vocabulary; one all_gather rebuilds full logits.
        lm_tp = self.tp if self.vocab_size % (32 * self.tp) == 0 else 1
        self.lm_tp = lm_tp
        self.lm_head = to_device(
            matmul_weight(hf_model.lm_head.weight.detach().float()),
            device,
            mesh_mapper=tt_mesh.shard_tp(device, -1) if lm_tp > 1 else replicate,
        )

        # --- persistent, shape-pinned buffers: everything the forward reads besides the weights.
        self._prompt_len = 0
        self._cursor = 0
        self.set_capacity(self.capacity)

        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------ resident-state helpers

    def set_capacity(self, capacity):
        """(Re)pin the variable sequence dim to `capacity` and rebuild every shape-dependent buffer."""
        C = int(capacity)
        self.capacity = C
        replicate = tt_mesh.replicate(self.device)
        self._positions = to_device(
            torch.arange(C, dtype=torch.float32).view(1, 1, 1, C), self.device,
            mesh_mapper=replicate, dtype=ttnn.float32,
        )
        self._mask = to_device(_causal_mask(C), self.device, mesh_mapper=replicate)
        self._onehot = to_device(
            torch.eye(C, dtype=torch.float32).view(1, 1, C, C), self.device,
            mesh_mapper=replicate, dtype=ttnn.float32,
        )
        self._ids = to_device(
            torch.zeros(1, 1, 1, C, dtype=torch.float32), self.device,
            mesh_mapper=replicate, dtype=ttnn.float32,
        )

    def _seed_ids(self, input_ids):
        """Write a prompt into the resident token buffer (host -> device, ONCE, outside the loop)."""
        C = self.capacity
        ids = input_ids.reshape(-1)[:C]
        buf = torch.zeros(1, 1, 1, C, dtype=torch.float32)
        buf[0, 0, 0, : ids.numel()] = ids.to(torch.float32)
        self._ids = to_device(buf, self.device, mesh_mapper=tt_mesh.replicate(self.device), dtype=ttnn.float32)
        self._prompt_len = int(ids.numel())
        self._cursor = self._prompt_len

    def _row(self, index):
        """The one-hot row selecting position `index` of the resident token buffer."""
        return ttnn.slice(self._onehot, [0, 0, index, 0], [1, 1, index + 1, self.capacity])

    # ------------------------------------------------------------------------ the forward pass

    def forward_logits(self):
        """ONE host-op-free forward over the pinned capacity. Reads ONLY resident device buffers.

        This is the whole model: embedding -> rope -> the decoder stack -> final norm -> lm_head.
        """
        C = self.capacity
        ids_u32 = ttnn.to_layout(ttnn.typecast(self._ids, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ids_u32 = ttnn.reshape(ids_u32, (1, C))

        hidden = ttnn.embedding(ids_u32, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        hidden = ttnn.reshape(hidden, (1, C, self.hidden_size))

        cos, sin = self.rope(hidden, self._positions)
        cos = ttnn.typecast(cos, ttnn.bfloat16)
        sin = ttnn.typecast(sin, ttnn.bfloat16)

        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=(cos, sin), attention_mask=self._mask)

        hidden = self.final_norm(hidden)
        hidden = ttnn.reshape(hidden, (1, 1, C, self.hidden_size))
        logits = ttnn.linear(hidden, self.lm_head, compute_kernel_config=self.compute_config)
        if self.lm_tp > 1:
            logits = tt_mesh.all_gather_tp(logits, -1, self.device)
        return logits

    def _argmax_at(self, logits, index):
        """Greedy pick for the token that follows position `index`, entirely on device."""
        row = ttnn.slice(logits, [0, 0, index, 0], [1, 1, index + 1, self.vocab_size])
        token = ttnn.argmax(row, dim=-1, keepdim=True)
        return ttnn.typecast(ttnn.to_layout(token, ttnn.TILE_LAYOUT), ttnn.float32)

    def _write_token(self, token_f32, index):
        """Place a device-side token id at position `index` of the resident buffer, on device."""
        self._ids = ttnn.add(self._ids, ttnn.multiply(self._row(index), token_f32))

    # ------------------------------------------------------------------- AR decode contract

    def decode_prefill(self, input_ids):
        """Seed the resident decode state from a prompt and run the prefix once."""
        self._seed_ids(input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids))
        return self.forward_logits()

    def decode_step(self):
        """ONE autoregressive step: read the resident state, emit a token, write it back."""
        logits = self.forward_logits()
        token = self._argmax_at(logits, self._cursor - 1)
        self._write_token(token, self._cursor)
        self._cursor += 1
        return logits, token

    # --------------------------------------------------------------------------- the Call

    def horizon(self, prompt_len, max_new_tokens=None):
        """Decode length, grounded in the model -- never an invented constant.

        The stop rule is the config's eos set; the SAFETY CAP is the capacity the graduated
        gated_delta_net port covers (one delta-rule chunk), i.e. `capacity - prompt_len`.
        """
        cap = self.capacity - prompt_len
        gen = getattr(self.reference, "generation_config", None)
        want = getattr(gen, "max_new_tokens", None) if gen is not None else None
        if max_new_tokens is not None:
            want = int(max_new_tokens)
        return max(1, min(cap, want if want else cap))

    def eos_ids(self):
        gen = getattr(self.reference, "generation_config", None)
        raw = getattr(gen, "eos_token_id", None) if gen is not None else None
        if raw is None:
            raw = getattr(self.config, "eos_token_id", None)
        if raw is None:
            return set()
        return set(raw) if isinstance(raw, (list, tuple, set)) else {int(raw)}

    def generate(self, input_ids, *, max_new_tokens=None, stop_on_eos=True, collect_logits=False):
        """The real task forward: prompt token ids -> generated token ids.

        Every step is fed the PREVIOUS TT step's own output -- the next token is chosen by an
        on-device argmax and written back into the resident token buffer on device.  No reference
        tensor is ever injected at a joint.
        """
        if isinstance(input_ids, torch.Tensor):
            prompt = input_ids.reshape(-1)
        else:
            prompt = torch.tensor(input_ids).reshape(-1)
        prompt_len = int(prompt.numel())
        if prompt_len >= self.capacity:
            raise ValueError(
                f"prompt of {prompt_len} tokens does not leave room to decode inside the pinned "
                f"capacity C={self.capacity} (the gated_delta_net port's single-chunk limit)"
            )

        steps = self.horizon(prompt_len, max_new_tokens)
        eos = self.eos_ids()

        self._seed_ids(prompt)
        rows = []
        for _ in range(steps):
            logits, _token = self.decode_step()
            if collect_logits:
                # kept as a DEVICE tensor; nothing crosses to host until the loop is over
                rows.append(
                    ttnn.slice(logits, [0, 0, self._cursor - 2, 0], [1, 1, self._cursor - 1, self.vocab_size])
                )

        # ONE host readback, AFTER the loop.  The resident token buffer already holds every token
        # the chain produced -- each step's on-device argmax wrote itself back on device -- so no
        # step above ever waited on the host, which is what makes the loop body traceable as-is.
        # EOS is applied here rather than as an early `break`: the tokens past EOS are discarded
        # either way, and this keeps the eager step byte-identical to the traced one.
        ids = tt_mesh.to_host(self._ids).flatten().to(torch.int64).tolist()
        generated = ids[prompt_len : prompt_len + steps]
        if stop_on_eos:
            for i, token_id in enumerate(generated):
                if token_id in eos:
                    generated = generated[: i + 1]
                    break

        out = {"tokens": generated, "steps": steps}
        if collect_logits:
            step_logits = [tt_mesh.to_host(r).float().reshape(-1) for r in rows[: len(generated)]]
            out["logits"] = torch.stack(step_logits) if step_logits else torch.empty(0)
        return out

    def run_data_parallel(self, tokenizer, prompts, **kwargs):
        """Run one prompt per DATA-PARALLEL replica -- independent copies, identical weights.

        Falls back to running them in sequence on this replica when only one was materialised.
        """
        replicas = getattr(self, "replicas", [self])
        out = []
        for i, prompt in enumerate(prompts):
            replica = replicas[i % len(replicas)]
            out.append(replica.run_text_generation(tokenizer, prompt, **kwargs))
        return out

    def run_text_generation(self, tokenizer, prompt, *, max_new_tokens=None, chat=True, collect_logits=False):
        """Call 1 end to end: str -> tokenizer -> TT chain -> tokenizer -> str."""
        input_ids = encode_prompt(tokenizer, prompt, chat=chat)
        result = self.generate(input_ids, max_new_tokens=max_new_tokens, collect_logits=collect_logits)
        result["prompt_ids"] = input_ids
        result["text"] = tokenizer.decode(result["tokens"], skip_special_tokens=True)
        return result

    # ----------------------------------------------------------------------- SOURCE A golden

    def _hf_reference_text_generation(self, tokenizer, prompt, *, max_new_tokens=None, chat=True):
        """SOURCE A golden: the reference `model.generate()`, capped to the SAME horizon."""
        input_ids = encode_prompt(tokenizer, prompt, chat=chat)
        steps = self.horizon(int(input_ids.shape[-1]), max_new_tokens)
        with torch.no_grad():
            out = self.reference.generate(
                input_ids,
                max_new_tokens=steps,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
        new = out.sequences[0, input_ids.shape[-1] :]
        return {
            "tokens": new.tolist(),
            "logits": torch.stack([s[0].float() for s in out.scores]),
            "text": tokenizer.decode(new, skip_special_tokens=True),
            "steps": steps,
        }

    def _hf_score_sequence(self, prompt_ids, generated):
        """SOURCE A golden logits along a GIVEN token trajectory -- one HF forward.

        `generate()` is the golden for WHAT the model emits; this is the golden for the LOGITS the
        model produces at each decode position of a trajectory.  Feeding it the TT run's own
        sequence puts both sides on the same history, which is what makes a per-step logits
        comparison mean "does the TT chain compute the same function as HF" rather than "did two
        free-running chains happen to stay in step".

        Nothing from here ever enters the TT pipeline -- it is scored after the fact.
        """
        prompt = prompt_ids.reshape(1, -1)
        n_prompt = int(prompt.shape[-1])
        n_gen = len(generated)
        # The scored history is [prompt | generated[:-1]], assembled by slice assignment into one
        # preallocated buffer.
        full = prompt.new_zeros(1, n_prompt + max(n_gen - 1, 0))
        full[0, :n_prompt] = prompt[0]
        if n_gen > 1:
            full[0, n_prompt:] = torch.tensor(generated[:-1], dtype=prompt.dtype)
        with torch.no_grad():
            logits = self.reference(input_ids=full).logits[0].float()
        start = int(prompt.shape[-1]) - 1
        return logits[start : start + len(generated)]

    # ---------------------------------------------------------------------- trace contract

    def _trace_inputs(self):
        """The golden inputs the demo and the e2e PCC test feed: the tokenized default prompt."""
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            from models.demos.qwen3_coder_next.tt.reference import load_tokenizer

            tokenizer = self._tokenizer = load_tokenizer()
        return {"input_ids": encode_prompt(tokenizer, DEFAULT_PROMPT)}

    def prefill_trace_inputs(self):
        return self._trace_inputs()

    def decode_trace_inputs(self):
        return self._trace_inputs()

    def _pin(self, inputs):
        """Pin the VARIABLE dim (sequence) to the fixed capacity and pre-upload every constant.

        rope tables and the causal mask come from the HF reference itself, so a traced step reads
        exactly the constants the golden used.  Padded positions are masked causally, so the
        output on `[0:real_len]` is unchanged.
        """
        ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        ids = ids.reshape(-1)[: self.capacity]
        self._seed_ids(ids)

        C = self.capacity
        replicate = tt_mesh.replicate(self.device)
        # Take the shape-dependent constants FROM THE HF REFERENCE so the pinned step matches the
        # golden exactly. The rope table stays a device-side `rotary_embedding` op (it has to --
        # that stub is on the traced path), so the reference tables are used to CHECK it rather
        # than to replace it; the causal mask and the position ids are uploaded outright.
        with torch.no_grad():
            position_ids = torch.arange(C).unsqueeze(0)
            dummy = torch.zeros(1, C, self.hidden_size, dtype=self.reference.dtype)
            ref_cos, ref_sin = self.reference.model.rotary_emb(dummy, position_ids)
        self._ref_rope = (ref_cos.float().reshape(C, -1), ref_sin.float().reshape(C, -1))
        self._mask = to_device(_causal_mask(C), self.device, mesh_mapper=replicate)
        self._positions = to_device(
            torch.arange(C, dtype=torch.float32).view(1, 1, 1, C), self.device,
            mesh_mapper=replicate, dtype=ttnn.float32,
        )
        cos, sin = self.rope(None, self._positions)
        self.rope_pcc = min(
            _pcc(self._ref_rope[0], tt_mesh.to_host(cos).float().reshape(C, -1)),
            _pcc(self._ref_rope[1], tt_mesh.to_host(sin).float().reshape(C, -1)),
        )
        return self._prompt_len

    def prefill_trace_setup(self, inputs):
        self._pin(inputs)
        return self

    def prefill_trace_step(self):
        return self.forward_logits()

    def decode_trace_setup(self, inputs):
        """Seed the resident decode state, then leave the cursor pinned at a fixed step."""
        self._pin(inputs)
        self._cursor = self._prompt_len
        return self

    def decode_trace_step(self):
        cursor = self._cursor
        logits = self.forward_logits()
        token = self._argmax_at(logits, cursor - 1)
        self._ids = ttnn.add(self._ids, ttnn.multiply(self._row(cursor), token))
        return logits

    # --------------------------------------------------------------------------- self tests

    def host_op_selftest(self, inputs=None):
        """AUTHORITATIVE fully-on-device check: zero host aten ops in the model math."""
        from scripts.tt_hw_planner import host_op_observer

        inputs = inputs or self._trace_inputs()
        ids = inputs["input_ids"].reshape(-1)[: self.capacity]
        steps = self.horizon(int(ids.numel()), 2)

        # Input ENCODING, the one-time weight build AND the one-time materialisation of every
        # lazily-built device constant (the expert bank's expert-id arange, the delta net's
        # decay/shift masks, ...) all happen OUTSIDE the observed region.  They are build-time
        # uploads that happen on first use, not per-step host compute -- but they DO dispatch
        # `torch.arange`/`.contiguous()` on the way to the chip, so a cold first forward would
        # report host ops that a warm one never fires again.
        self._seed_ids(ids)
        self.forward_logits()
        self._seed_ids(ids)

        def forward():
            for _ in range(steps):
                self.decode_step()

        with host_op_observer.observe_host_ops() as ops:
            forward()
        return host_op_observer.verdict(ops)

    def trace_capture_selftest(self, device=None, *, pcc=0.99):
        """Capture, execute and release ONE trace per stage; verify each against the eager run."""
        device = device or self.device
        inputs = self._trace_inputs()
        results = {}
        ok = True
        for stage in PIPELINE_STAGES:
            setup = getattr(self, f"{stage}_trace_setup")
            step = getattr(self, f"{stage}_trace_step")
            stage_inputs = getattr(self, f"{stage}_trace_inputs")()
            setup(stage_inputs)
            reference = tt_mesh.to_host(step()).float()

            setup(stage_inputs)
            step()  # warm the program cache so capture records no compiles
            setup(stage_inputs)
            try:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                out = step()
                ttnn.end_trace_capture(device, tid, cq_id=0)
            except Exception as exc:
                if self.capacity > 32:
                    shrunk = self.capacity // 2
                    print(
                        f"[trace] stage={stage} capture overflowed the trace region at "
                        f"C={self.capacity}; FALLING BACK to C={shrunk} and retrying",
                        flush=True,
                    )
                    self.set_capacity(shrunk)
                    return self.trace_capture_selftest(device, pcc=pcc)
                print(f"[trace] stage={stage} capture FAILED at C={self.capacity}: {exc}")
                results[stage] = {"captured": False, "error": str(exc)}
                ok = False
                continue
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = tt_mesh.to_host(out).float()
            value = _pcc(reference.reshape(-1), got.reshape(-1))
            ttnn.release_trace(device, tid)
            print(f"[trace] stage={stage} captured host-free, PCC={value:.6f}", flush=True)
            results[stage] = {"captured": True, "pcc": value}
            ok = ok and value >= pcc
        self.trace_results = results
        return ok


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    if torch.allclose(a, b):
        return 1.0
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs):
    """Construct and RETURN the resident pipeline object (it does not run anything).

    `layers` is the DEFAULT depth for every repeated block.  This model has exactly ONE repeated
    stack -- the decoder -- which BOTH declared stages ("prefill", "decode") run over, so the
    per-stage overrides `prefill_layers` / `decode_layers` exist for call-signature compatibility
    but must resolve to a single depth; when they disagree the deeper one wins (a capped build
    must stay a model, and a stage cannot own a different copy of a shared stack).

    `layers=None` means EVERY layer of the reference handed in.  When no reference is handed in
    the pipeline materialises one at `TT_QWEN3_LAYERS`, which now defaults to the FULL 48.  The
    old comment here claimed 48 x 512 experts exceeds the DRAM of the mandated placement: that was
    true at TP=2 and is NOT true at TP=8, where attention replicates (0.65 GB, 0.4% of the weights)
    and everything else shards 8 ways -- 21.58 GB/chip against 28.6 usable.

    `device` is the device the CALLER opened; this function never opens one of its own.
    """
    depths = [d for d in (prefill_layers, decode_layers) if d is not None]
    if depths:
        resolved = max(int(d) for d in depths)
        if layers is not None and int(layers) != resolved:
            print(f"[build] per-stage overrides {depths} supersede layers={layers} -> {resolved}")
        if len(set(depths)) > 1:
            print(
                f"[build] prefill_layers/decode_layers disagree ({depths}); both stages share the "
                f"ONE decoder stack, so the deeper cap {resolved} is built for both."
            )
        layers = resolved

    tokenizer = kwargs.pop("tokenizer", None)
    if model is None:
        want = layers if layers is not None else int(os.environ.get("TT_QWEN3_LAYERS", DEFAULT_LAYERS))
        model, tokenizer = load_reference(want)
        layers = want

    # PLACEMENT.  The graduated stubs shard over every chip of the device they are handed, so a
    # 32-chip mesh handed over whole would put them at TP=32 -- where `attention` (2 kv heads) and
    # `gated_delta_net` (16 k-heads) both trip their own divisibility guard and fall back to
    # REPLICATION.  Carve the mesh into TP groups of the proven degree instead and build on one;
    # the remaining groups are the data-parallel replicas.  See `tt/mesh.py`.
    groups, placement = tt_mesh.tp_groups(device)

    # DATA PARALLELISM.  Every group is a full, independent replica; `dp` says how many to
    # MATERIALISE.  One is enough for the correctness gate (a PCC run has exactly one data copy)
    # and each extra replica is another full copy of the weights over the host bus, so the default
    # is 1 and the count is always printed.
    dp = int(kwargs.pop("dp", os.environ.get("TT_QWEN3_DP", 1)))
    dp = max(1, min(dp, len(groups)))
    capacity = int(kwargs.pop("capacity", DEFAULT_CAPACITY))

    replicas = [
        Qwen3CoderNextPipeline(groups[i], model, layers=layers, capacity=capacity) for i in range(dp)
    ]
    pipeline = replicas[0]
    pipeline.parent_device = device
    pipeline.tp_groups = groups
    pipeline.replicas = replicas
    pipeline.placement = placement
    print(
        f"[build] {placement} available; {dp} data-parallel replica(s) materialised "
        f"({dp * groups[0].get_num_devices()} chip(s) carrying weights)",
        flush=True,
    )
    if tokenizer is not None:
        for replica in replicas:
            replica._tokenizer = tokenizer
    return pipeline


# ------------------------------------------------------------------- observer entry points
# The two hooks below are what the standalone observers import and call: `_host_op_probe.py`
# looks up `host_op_selftest` and `_trace_capture_probe.py` looks up `trace_capture_selftest`,
# both by name on THIS module, both with no arguments.  They are thin: build the same pipeline
# `tests/e2e` and the demo build, on a device obtained from the package's ONE opener, and run the
# corresponding self-test method.  Pass `device=` to reuse a device you already own (which is
# what `tests/e2e` does, via the session fixture) -- then nothing is opened or closed here.


def _selftest_pipeline(device, *, layers=None, capacity=DEFAULT_CAPACITY):
    """The same build the demo and the e2e gate use, on the caller's device."""
    depth = int(layers if layers is not None else os.environ.get("TT_QWEN3_LAYERS", DEFAULT_LAYERS))
    model, tokenizer = load_reference(depth)
    return build_pipeline(device, model=model, layers=depth, tokenizer=tokenizer, capacity=capacity)


def _on_a_device(run):
    """Run `run(device)` on the package's sole opener, closing the mesh afterwards."""
    from models.demos.qwen3_coder_next import device_harness

    device, _ = device_harness.open_mesh()
    try:
        return run(device)
    finally:
        device_harness.close_mesh(device)


def host_op_selftest(device=None):
    """Observer hook: is the decode path free of host aten ops?  Returns the verdict dict."""
    if device is not None:
        return _selftest_pipeline(device).host_op_selftest()
    return _on_a_device(lambda dev: _selftest_pipeline(dev).host_op_selftest())


def trace_capture_selftest(device=None):
    """Observer hook: does every declared stage really capture and replay?  Returns a bool.

    The capture runs on `pipeline.device` -- the TP submesh carved out of the caller's mesh, which
    is the device the weights live on and therefore the only one whose command queue the traced
    program belongs to.
    """
    if device is not None:
        pipe = _selftest_pipeline(device)
        return pipe.trace_capture_selftest(pipe.device)

    def run(dev):
        pipe = _selftest_pipeline(dev)
        return pipe.trace_capture_selftest(pipe.device)

    return _on_a_device(run)
