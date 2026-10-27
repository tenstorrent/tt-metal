# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-Flash text generation on TTNN: the ONE chained forward over the graduated stubs.

Both demo/demo_text_generation.py and tests/e2e/ call `build_pipeline(...)` and the methods below,
so the demo and the gate run identical code.

Chain (per prefill / decode step, all on device):
    token_embed(ids) -> rotary_embedding(positions)
    -> layer 0      : decoder_layer                                   (the one dense layer)
    -> layers 1..46 : h += attention(r_m_s_norm(h)); h += mo_e(r_m_s_norm(h))
                      mo_e = topk_router -> on-device top-k -> 16 local experts/chip + m_l_p shared expert
    -> r_m_s_norm (final) -> decoder_head -> ttnn.argmax -> next ids -> token_embed ...

Placement: mesh 1x4 (TP=4 x DP=1). Embedding / norms / router replicated; attention heads, dense
MLP and shared expert TP-sharded; routed experts expert-parallel (64/4 per chip); lm_head
vocab-sharded + all_gather. The sharded bodies are the graduated `.last_good_sharded` stubs.
"""
from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn

try:
    import ttl  # noqa: F401
except ImportError:
    ttl = None
from models.demos.glm_4_7_flash.tt import inputs as glm_inputs

PIPELINE_STAGES = ["prefill", "decode"]

# Every graduated module of Source B (a .last_good_native / .last_good_sharded snapshot exists).
GRADUATED_STUBS = (
    "token_embed",
    "rotary_embedding",
    "decoder_layer",
    "r_m_s_norm",
    "attention",
    "mo_e",
    "topk_router",
    "m_l_p",
    "decoder_head",
)
_STUB_PKG = "models.demos.glm_4_7_flash._stubs"
STUB_DIR = Path(__file__).resolve().parent.parent / "_stubs"

_TILE = 32


def _round_up(x: int, m: int) -> int:
    return ((int(x) + m - 1) // m) * m


def load_stub(name: str):
    return importlib.import_module(f"{_STUB_PKG}.{name}")


class StubCall:
    """Counts every call into a graduated stub instance (Gate 2), forwarding everything else."""

    def __init__(self, name: str, obj, counts: dict):
        self._name, self._obj, self._counts = name, obj, counts

    def __call__(self, *args, **kwargs):
        self._counts[self._name] = self._counts.get(self._name, 0) + 1
        return self._obj(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._obj, item)


@dataclass
class StepCtx:
    rot: tuple
    mode: str
    batch: int
    seq_len: int
    cur_pos: object
    kv: list


class GlmLayer:
    """Common base of the 47 decoder layers (one stack): `layer(h, ctx) -> h`, h = [1,1,N,hidden]."""

    idx: int

    def kv_kwargs(self, ctx: StepCtx):
        return dict(
            position_embeddings=ctx.rot,
            kv_cache=ctx.kv[self.idx],
            mode=ctx.mode,
            batch=ctx.batch,
            seq_len=ctx.seq_len,
            cur_pos=ctx.cur_pos,
        )


class DenseLayer(GlmLayer):
    """mlp_layer_types[i] == 'dense': the graduated decoder_layer body, whole."""

    def __init__(self, idx, decoder_layer):
        self.idx = idx
        self.decoder_layer = decoder_layer

    def __call__(self, h, ctx: StepCtx):
        return self.decoder_layer(h, **self.kv_kwargs(ctx))


class MoeLayer(GlmLayer):
    """mlp_layer_types[i] == 'sparse': input norm -> MLA attention -> residual -> post norm -> MoE -> residual."""

    def __init__(self, idx, input_norm, attention, post_norm, moe):
        self.idx = idx
        self.input_norm = input_norm
        self.attention = attention
        self.post_norm = post_norm
        self.moe = moe

    def __call__(self, h, ctx: StepCtx):
        a = self.attention(self.input_norm(h), **self.kv_kwargs(ctx))
        h = ttnn.add(h, a)
        ttnn.deallocate(a)
        m = self.moe(self.post_norm(h))
        out = ttnn.add(h, m)
        ttnn.deallocate(m)
        return out


class FastArgmax:
    """Trace-safe 32-core argmax via generic_op, replacing ttnn.argmax for decode.
    One core per batch row, 32-byte aligned writes (no semaphore), cached descriptor + persistent buffers.
    Each core writes its result replicated 8x into a 32-byte page to avoid DRAM cache-line races.
    SPMD dispatch: a single ProgramDescriptor is auto-replicated across the mesh by generic_op.
    """

    def __init__(self, device, batch, is_mesh):
        self.device = device
        self.batch = batch
        self.is_mesh = is_mesh
        self._program = None
        self._out_wide = None
        self._out_narrow = None
        self._inp = None

    def _build(self, rm):
        V = rm.shape[-1]
        batch = self.batch
        chunk_elems = 16384
        cb_idx = 0
        chunk_bytes = ((chunk_elems * 2) + 31) & ~31
        cb_total = chunk_bytes + 32

        rep = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None

        self._out_wide = ttnn.from_torch(
            torch.zeros(1, 1, batch, 8, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )
        self._out_narrow = ttnn.from_torch(
            torch.zeros(1, 1, batch, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )
        self._inp = ttnn.from_torch(
            torch.zeros(1, 1, batch, V, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )

        grid_x = min(8, batch)
        grid_y = (batch + grid_x - 1) // grid_x
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])

        src_ta = list(ttnn.TensorAccessorArgs(self._inp).get_compile_time_args())
        dst_ta = list(ttnn.TensorAccessorArgs(self._out_wide).get_compile_time_args())
        ct = [cb_idx, V, chunk_elems] + src_ta + dst_ta

        src_addr = self._inp.buffer_address()
        dst_addr = self._out_wide.buffer_address()

        kernel_path = None
        for p in [
            os.path.join(os.getcwd(), "argmax32.cpp"),
            os.path.join(os.environ.get("TT_METAL_HOME", ""), "argmax32.cpp"),
        ]:
            if os.path.exists(p):
                kernel_path = p
                break
        if kernel_path is None:
            kernel_path = "argmax32.cpp"

        rt = ttnn.RuntimeArgs()
        cid = 0
        for y in range(grid_y):
            for x in range(grid_x):
                if cid < batch:
                    rt[x][y] = [src_addr, dst_addr, cid]
                    cid += 1
        k = ttnn.KernelDescriptor(
            kernel_source=kernel_path,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_idx,
                    data_format=ttnn.uint32,
                    page_size=cb_total,
                )
            ],
        )
        self._program = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])

    def __call__(self, rm):
        if self._program is None:
            self._build(rm)
        ttnn.copy(rm, self._inp)
        ttnn.generic_op([self._inp, self._out_wide], self._program)
        return ttnn.slice(self._out_wide, [0, 0, 0, 0], [1, 1, self.batch, 1])


def _weight_cache_root():
    return Path(os.environ.get("GLM47_WEIGHT_CACHE", Path.home() / ".cache" / "tt_glm_4_7_flash_weights"))


class GlmPipeline:
    """Resident GLM-4.7-Flash pipeline on a (mesh) device."""

    def __init__(self, device, hf_model, layers=None, batch=None, verbose=True):
        # The graduated attention / decoder_layer / m_l_p / mo_e / decoder_head bodies shard on the
        # TP axis when this is set (that is the form they graduated in, .last_good_sharded).
        os.environ.setdefault("TT_HW_PLANNER_SHARD_RUN", "1")
        self.device = device
        self.hf = hf_model  # ground truth for section structure (hf.model.layers); weights read at build only
        self.config = hf_model.config
        self.batch = int(batch or glm_inputs.batch_size())
        self.verbose = verbose
        self.counts: dict = {}
        self.is_mesh = isinstance(device, ttnn.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh else 1

        cfg = self.config
        n_total = int(cfg.num_hidden_layers)
        layer_types = list(getattr(cfg, "mlp_layer_types", None) or ["dense"] + ["sparse"] * (n_total - 1))
        if layers is None and os.environ.get("TT_PERF_LAYERS"):
            layers = int(os.environ["TT_PERF_LAYERS"])
        n_build = n_total if layers is None else max(1, min(int(layers), n_total))
        # A capped build stays a MODEL: it must still hold at least one layer of each block kind the
        # stack interleaves (dense layer 0 + one MoE layer), else a whole op family vanishes.
        need = max(layer_types.index(k) for k in set(layer_types)) + 1
        if n_build < need:
            print(f"[glm] layers={layers} would drop a block kind; building {need} (smallest depth with every kind)")
            n_build = need
        self.n_layers = n_build
        self.layer_types = layer_types[:n_build]

        t0 = time.time()
        reg = self._reg
        hf_m = hf_model.model
        self.token_embed = reg("token_embed", load_stub("token_embed").build(device, hf_m.embed_tokens))
        self.rotary = reg("rotary_embedding", load_stub("rotary_embedding").build(device, hf_m.rotary_emb))
        self.layers: list = []
        wc = _weight_cache_root()
        for i in range(n_build):
            hl = hf_m.layers[i]
            if self.layer_types[i] == "dense":
                self.layers.append(DenseLayer(i, reg("decoder_layer", load_stub("decoder_layer").build(device, hl))))
            else:
                router = reg("topk_router", load_stub("topk_router").build(device, hl.mlp.gate))
                shared = reg("m_l_p", load_stub("m_l_p").build(device, hl.mlp.shared_experts))
                moe = load_stub("mo_e").build(
                    device, hl.mlp, router=router, shared_expert=shared, weight_cache=str(wc / f"layer{i}")
                )
                self.layers.append(
                    MoeLayer(
                        i,
                        reg("r_m_s_norm", load_stub("r_m_s_norm").build(device, hl.input_layernorm)),
                        reg("attention", load_stub("attention").build(device, hl.self_attn)),
                        reg("r_m_s_norm", load_stub("r_m_s_norm").build(device, hl.post_attention_layernorm)),
                        reg("mo_e", moe),
                    )
                )
            if verbose:
                print(
                    f"[glm] built layer {i + 1}/{n_build} ({self.layer_types[i]}) t={time.time() - t0:.0f}s", flush=True
                )
        self.final_norm = reg("r_m_s_norm", load_stub("r_m_s_norm").build(device, hf_m.norm))
        self.lm_head = reg("decoder_head", load_stub("decoder_head").build(device, hf_model.lm_head))
        self.counts.clear()  # count only forward-path invocations, not builds

        self.hidden = int(cfg.hidden_size)
        self.vocab = int(cfg.vocab_size)
        self.local_heads = self.layers[0].decoder_layer._local_heads
        self.head_dim = int(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        self.pad_id = int(cfg.pad_token_id)
        self.kv = None
        self.kv_capacity = 0
        self._T = None
        self._fast_argmax = None
        if verbose:
            print(f"[glm] pipeline built: {n_build} layers, batch {self.batch}, {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ helpers

    def _reg(self, name, obj):
        return StubCall(name, obj, self.counts)

    def _rep(self):
        return ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None

    def _upload(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device, mesh_mapper=self._rep())

    def _write(self, name, host, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        """(Re)fill a persistent device input buffer, keeping its address stable for trace."""
        buf = getattr(self, name, None)
        if buf is not None and tuple(buf.shape) == tuple(host.shape):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=self._rep()), buf
            )
        else:
            setattr(self, name, self._upload(host, dtype, layout))

    def to_host(self, t):
        """Device -> host from ONE replica (all outputs read here are replicated / all-gathered)."""
        if self.is_mesh:
            t = ttnn.get_device_tensors(t)[0]
        return ttnn.to_torch(t)

    def ensure_kv(self, capacity: int):
        C = _round_up(capacity, 64)
        if self.kv is not None and self.kv_capacity >= C:
            return
        assert C <= self.rotary.cos_table.shape[0], "capacity exceeds the rotary_embedding table"
        self.kv = None
        shape = (self.batch, self.local_heads, C, self.head_dim)
        zeros = torch.zeros(shape, dtype=torch.bfloat16)
        self.kv = [
            (
                self._upload(zeros, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                self._upload(zeros, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            )
            for _ in range(self.n_layers)
        ]
        self.kv_capacity = C

    # ------------------------------------------------------------------ inputs (host encoding, outside the forward)

    def prepare(self, input_ids: torch.Tensor, horizon: int):
        """Upload one request: padded prompt ids/positions + decode-state buffers. Not part of the forward."""
        B, T = input_ids.shape
        assert B == self.batch, f"pipeline built for batch {self.batch}, got {B}"
        Tp = _round_up(T, _TILE)
        self.ensure_kv(max(Tp, T + horizon))
        ids = torch.full((B, Tp), self.pad_id, dtype=torch.int32)
        ids[:, :T] = input_ids.to(torch.int32)
        pos = torch.arange(Tp, dtype=torch.int32).unsqueeze(0).repeat(B, 1)
        self._write("_pf_ids", ids, ttnn.uint32)
        self._write("_pf_pos", pos, ttnn.uint32)
        self._write("_dec_ids", torch.zeros((B, 1), dtype=torch.int32), ttnn.uint32)
        self._write("_next_ids", torch.zeros((B, 1), dtype=torch.int32), ttnn.uint32)
        self._write("_cur_pos", torch.full((B,), T, dtype=torch.int32), ttnn.int32)
        self._write("_rot_pos", torch.full((B, 1), T, dtype=torch.int32), ttnn.uint32)
        self._T, self._Tp = T, Tp

    # ------------------------------------------------------------------ the forward (pure ttnn)

    def _stack(self, h, ctx: StepCtx):
        for layer in self.layers:
            h = layer(h, ctx)
        return h

    def prefill_forward(self):
        """Prompt -> (last-position logits [1,1,B,V], final-norm hidden [1,1,B*Tp,H]); seeds the KV cache."""
        B, T, Tp = self.batch, self._T, self._Tp
        cos_sin = self.rotary(None, position_ids=self._pf_pos)  # ([B,Tp,64], [B,Tp,64])
        h = ttnn.reshape(self.token_embed(self._pf_ids), [1, 1, B * Tp, self.hidden])
        h = self._stack(h, StepCtx(cos_sin, "prefill", B, Tp, None, self.kv))
        hn = self.final_norm(h)
        last = ttnn.to_layout(ttnn.reshape(hn, [B, Tp, self.hidden]), ttnn.ROW_MAJOR_LAYOUT)
        last = ttnn.slice(last, [0, T - 1, 0], [B, T, self.hidden])
        last = ttnn.to_layout(ttnn.reshape(last, [1, 1, B, self.hidden]), ttnn.TILE_LAYOUT)
        logits = self.lm_head(last)
        return logits, hn

    def decode_forward(self):
        """One token per row at position _cur_pos -> logits [1,1,B,V]; appends to the KV cache."""
        B = self.batch
        cos_sin = self.rotary(None, position_ids=self._rot_pos)  # ([B,1,64], [B,1,64])
        h = ttnn.reshape(self.token_embed(self._dec_ids), [1, 1, B, self.hidden])
        h = self._stack(h, StepCtx(cos_sin, "decode", B, 1, self._cur_pos, self.kv))
        return self.lm_head(self.final_norm(h))

    def select_next(self, logits, feed=True):
        """Greedy (the example's do_sample=False) on device into the next-ids buffer; `feed` also makes
        it the next decode input (free running). Teacher forcing feeds a given token instead."""
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        if self._fast_argmax is None:
            self._fast_argmax = FastArgmax(self.device, self.batch, self.is_mesh)
        nxt_raw = self._fast_argmax(rm)
        nxt = ttnn.reshape(nxt_raw, [self.batch, 1])
        ttnn.deallocate(rm)
        ttnn.copy(nxt, self._next_ids)
        if feed:
            ttnn.copy(nxt, self._dec_ids)
        return self._next_ids

    def advance(self):
        ttnn.plus_one(self._cur_pos)
        ttnn.plus_one(self._rot_pos)

    # ------------------------------------------------------------------ AR decode contract

    def decode_prefill(self, feed=True):
        logits, hn = self.prefill_forward()
        self.select_next(logits, feed=feed)
        return logits, hn

    def decode_step(self):
        logits = self.decode_forward()
        self.select_next(logits)
        self.advance()
        return logits

    def feed_tokens(self, ids_dev):
        """Teacher forcing: the next decode input is `ids_dev` ([B,1] uint32 on device) instead of argmax."""
        ttnn.copy(ids_dev, self._dec_ids)

    # ------------------------------------------------------------------ generate (demo + tests)

    def generate(self, input_ids, rule, teacher_tokens=None, keep_logits=False, log=print):
        """Greedy generation under the shared StopRule (generate()'s own rule).

        teacher_tokens: optional [B, n] host tokens to feed instead of the TT argmax (gate test).
        Returns dict(tokens [B, n] int64, logits [B, n, V] bf16 or None, prefill_hidden [B, T, H],
        ended_by, steps)."""
        B, T = input_ids.shape
        horizon = rule.horizon
        self.prepare(input_ids, horizon)
        teach = None
        if teacher_tokens is not None:
            teach = [
                self._upload(teacher_tokens[:, i : i + 1].to(torch.int32), ttnn.uint32)
                for i in range(teacher_tokens.shape[1])
            ]
        out = self.run_request(rule, teach=teach, keep_logits=keep_logits, log=log)
        return out

    def run_request(self, rule, teach=None, keep_logits=False, log=print):
        """The request loop over prepared buffers: prefill, then one decode step per new token until the
        stop rule. Device work only, plus reading the chosen ids back for the stop check."""
        B, T = self.batch, self._T
        eos = set(rule.eos_ids)
        t0 = time.time()
        logits, hn = self.decode_prefill(feed=teach is None)
        hidden = self.to_host(hn).reshape(B, self._Tp, self.hidden)[:, :T]
        steps_tok, steps_logits = [], []
        done = [False] * B
        ended_by = "incomplete"
        for step in range(rule.horizon):  # bounded by max_new_tokens / safety cap; breaks on eos
            ids = self.to_host(self._next_ids).reshape(-1).tolist()
            if keep_logits:
                steps_logits.append(self.to_host(logits).reshape(B, -1)[:, : self.vocab])
            row = [self.pad_id if done[b] else int(ids[b]) for b in range(B)]
            steps_tok.append(row)
            done = [done[b] or row[b] in eos for b in range(B)]
            log(f"[glm] step {step + 1}/{rule.horizon} tok0={row[0]} t={time.time() - t0:.1f}s")
            n = len(steps_tok)
            if all(done):
                ended_by = "eos"
                break
            if n >= rule.horizon:
                ended_by = rule.ended_by(n, False)
                break
            if teach is not None:
                self.feed_tokens(teach[step])
            logits = self.decode_forward()
            self.select_next(logits, feed=teach is None)
            self.advance()
            ttnn.synchronize_device(self.device)
        tokens = torch.tensor(steps_tok, dtype=torch.int64).t().contiguous()
        lg = torch.stack(steps_logits, dim=1) if keep_logits else None
        return dict(tokens=tokens, logits=lg, prefill_hidden=hidden, ended_by=ended_by, steps=len(steps_tok))

    # ------------------------------------------------------------------ trace contract

    def _trace_request(self):
        from models.demos.glm_4_7_flash.tt.reference import StopRule

        tok = glm_inputs.load_tokenizer()
        ids, _ = glm_inputs.encode_example(tok, self.batch)
        rule = StopRule.for_prompt(ids.shape[1], self.config, _gen_config())
        return ids, rule

    def prefill_trace_inputs(self):
        ids, rule = self._trace_request()
        return dict(input_ids=ids, horizon=rule.horizon)

    def prefill_trace_setup(self, inputs):
        # Pins the sequence axis to Tp = round_up(T, 32) and pre-uploads ids / positions; the
        # rotary cos/sin come from the HF rotary_emb's inv_freq (rotary_embedding stub table) and the
        # causal mask is the SDPA is_causal mask, so padded positions [T, Tp) never reach [0, T).
        self.prepare(inputs["input_ids"], int(inputs["horizon"]))

    def prefill_trace_step(self):
        logits, _ = self.prefill_forward()
        return logits

    def prefill_trace_items(self):
        return self.batch * _round_up(self._T or self._trace_request()[0].shape[1], _TILE)

    def decode_trace_inputs(self):
        return self.prefill_trace_inputs()

    def decode_trace_setup(self, inputs):
        # decode_prefill seeds the resident KV cache and the first decode id; the step reads them.
        self.prepare(inputs["input_ids"], int(inputs["horizon"]))
        self.decode_prefill()
        ttnn.synchronize_device(self.device)

    def decode_trace_step(self):
        logits = self.decode_forward()
        self.select_next(logits)
        self.advance()
        return logits

    def decode_trace_items(self):
        return self.batch

    def decode_trace_repeats(self):
        # one request = max_new_tokens new tokens; the first comes from prefill, the rest from decode
        return max(1, self._trace_request()[1].horizon - 1)

    # ------------------------------------------------------------------ self-tests

    def trace_capture_selftest(self, device=None, pcc_min=0.99):
        """Capture each stage once (released before the next), replay, compare to the eager step."""
        device = device or self.device
        inputs = self.prefill_trace_inputs()
        ok = True
        for stage in PIPELINE_STAGES:
            setup = getattr(self, f"{stage}_trace_setup")
            step = getattr(self, f"{stage}_trace_step")
            setup(inputs)
            ref = self.to_host(step()).float()
            setup(inputs)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            setup(inputs)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = self.to_host(out).float()
            ttnn.release_trace(device, tid)
            p = pcc(got, ref)
            print(f"[glm] trace {stage}: PCC vs eager={p:.6f}", flush=True)
            ok = ok and p >= pcc_min
        return ok

    def host_op_selftest(self, decode_steps=2):
        """Authoritative on-device check: the forward (prefill + decode steps) under observe_host_ops.
        Encoding (tokenize + upload) and the weight build happen outside the observed region."""
        from scripts.tt_hw_planner import host_op_observer

        inputs = self.prefill_trace_inputs()
        self.prepare(inputs["input_ids"], int(inputs["horizon"]))
        with host_op_observer.observe_host_ops() as ops:
            self.decode_prefill()
            for _ in range(decode_steps):
                self.decode_step()
            ttnn.synchronize_device(self.device)
        return host_op_observer.verdict(ops)


def _gen_config():
    from models.demos.glm_4_7_flash.tt.reference import load_hf_config

    return load_hf_config()[1]


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def build_pipeline(device, model=None, layers=None, **kwargs):
    """Construct and RETURN the resident pipeline (not run it). `layers` caps the one decoder stack
    (None = all 47). Extra demo kwargs (prompt, text, ...) are accepted and ignored."""
    from models.demos.glm_4_7_flash.tt.reference import load_hf_model

    hf = model if model is not None else load_hf_model()
    return GlmPipeline(device, hf, layers=layers, batch=kwargs.get("batch"), verbose=kwargs.get("verbose", True))


def host_op_selftest():
    """Module-level hook for scripts/tt_hw_planner/_host_op_probe.py."""
    from models.demos.glm_4_7_flash._host_op_hook import run_selftest

    return run_selftest()


def trace_capture_selftest():
    """Module-level hook for scripts/tt_hw_planner/_trace_capture_probe.py."""
    from models.demos.glm_4_7_flash._host_op_hook import run_trace_selftest

    return run_trace_selftest()
