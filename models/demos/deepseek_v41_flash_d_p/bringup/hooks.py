# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for deepseek_v41_flash_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, hf_model, hf_layers, state_sources.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch

ORACLE_MAX_SEQ = 65536  # positions the oracle's caches hold (parity and sanity prompts are far shorter)


def _config(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config

    return Config.from_dir(hf_path(spec))


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/ds41_ref.py), the requested layers resident (fp32)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import DS41Reference

    ref = DS41Reference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)
    for i in ref.layer_ids:  # the state each layer returns is what the spec's block type declares
        assert ref.cfg.state_names(i) == spec.state_names(i), (i, ref.cfg.state_names(i), spec.state_names(i))
    return ref


def hf_model(spec, num_layers):
    """DeepSeek's inference/model.py on the checkpoint, on the CPU (reference/hf_oracle.py). num_layers=None: the
    whole text model in bf16 (the GPU configuration; intake sanity); else the first num_layers blocks in fp32."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.hf_oracle import build_oracle

    dtype = torch.bfloat16 if num_layers is None else torch.float32
    return build_oracle(hf_path(spec), num_layers, dtype, max_seq_len=ORACLE_MAX_SEQ)


def hf_layers(model):
    """Per-block taps: the block output packed like the reference's boundary, [S, 4 * H + 4] (streams, pending pre)."""
    return model.taps


def state_sources(spec, layer):
    """Earlier layers whose state `layer` reads: its compressed-KV source, its top-k source, layer 20's candidates."""
    return _config(spec).sources(layer)


# Device steps of the hybrid model, per block type: every step listed passed its component gate on the device.
# Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "swa": {
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "q_a",
        "kv",
        "attention",
        "attn_residual",
        "ffn_hc",
        "ffn_collapse",
        "ffn_norm",
        "router",
        "experts",
        "shared_expert",
        "moe_add",
        "ffn_residual",
    },
    "swa_engram": {"engram_rows"},
}

# mHC coefficient steps (tt/mhc.py:TtHcMixes): packed boundary [S, 4H + 4] (or streams [S, 4H]) in -> [S, 24].
_HC_STEPS = {"attn_hc", "ffn_hc"}


def _checkpoint(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint

    return Checkpoint(hf_path(spec))


def _hc_host_fn(mesh, module, hidden):
    """fn(ctx, x_host [S, 4H + 4] or [S, 4H]) -> mixes host [S, 24] fp32 (harness boundary: the streams in, the
    row-split, TP-replicated coefficients back from mesh column 0)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import row_split_to_host, streams_to_device

    def fn(ctx, x):
        xd = streams_to_device(mesh, x, hidden)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _collapse_host_fn(mesh, module, hidden):
    """fn(ctx, x_host [S, 4H + 4]) -> attn_in host [S, H] fp32 (harness boundary: the streams and the pending pre in,
    the row- and hidden-split output back)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import (
        HC,
        hidden_split_to_host,
        row_split_to_device,
        streams_to_device,
    )

    def fn(ctx, x):
        assert x.shape[-1] == HC * hidden + HC, x.shape
        xd = streams_to_device(mesh, x, hidden)
        pd = row_split_to_device(mesh, x[:, HC * hidden :])
        od = module(xd, pd)
        out = hidden_split_to_host(mesh, od).float()
        for t in (xd, pd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _ffn_collapse_host_fn(mesh, module, hidden):
    """fn(ctx, h_mid_host [S, 4H], attn_mix_host [S, 24]) -> ffn_in host [S, H] fp32 (harness boundary: the streams
    in, attn_hc's mixes row-split and TP-replicated (the module reads columns 0..3, the pre; DS41-3), the row- and
    hidden-split output back)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import (
        HC,
        hidden_split_to_host,
        row_split_to_device,
        streams_to_device,
    )

    def fn(ctx, h, mix):
        assert h.shape[-1] == HC * hidden and mix.shape[-1] >= HC, (h.shape, mix.shape)
        hd = streams_to_device(mesh, h, hidden)
        md = row_split_to_device(mesh, mix)
        od = module(hd, md)
        out = hidden_split_to_host(mesh, od).float()
        for t in (hd, md, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _residual_host_fn(mesh, module, hidden):
    """fn(ctx, x_host [S, 4H + 4], mix_host [S, 24], y_host [S, H]) -> h_mid host [S, 4H] fp32 (harness boundary: the
    streams in (the pending pre dropped), the mixes row-split and TP-replicated, y rows over axis 0 and hidden columns
    over axis 1; the streams back)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import (
        hidden_split_to_device,
        row_split_to_device,
        streams_to_device,
        streams_to_host,
    )

    def fn(ctx, x, mix, y):
        xd = streams_to_device(mesh, x, hidden)
        md = row_split_to_device(mesh, mix)
        yd = hidden_split_to_device(mesh, y)
        od = module(xd, md, yd)
        out = streams_to_host(mesh, od, hidden).float()
        for t in (xd, md, yd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _ffn_residual_host_fn(mesh, module, hidden):
    """fn(ctx, h_mid_host [S, 4H], mix_host [S, 24], y_host [S, H]) -> out host [S, 4H + 4] fp32 (harness boundary:
    h_mid's streams in, ffn_hc's mixes row-split and TP-replicated, mlp_out rows over axis 0 and hidden columns over
    axis 1; back the streams and the module's own pending-pre tensor (ffn_hc's pre, DS41-3), packed here only)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import (
        HC,
        hidden_split_to_device,
        row_split_to_device,
        row_split_to_host,
        streams_to_device,
        streams_to_host,
    )

    def fn(ctx, h, mix, y):
        assert h.shape[-1] == HC * hidden, h.shape
        hd = streams_to_device(mesh, h, hidden)
        md = row_split_to_device(mesh, mix)
        yd = hidden_split_to_device(mesh, y)
        od, pd = module(hd, md, yd)
        out = torch.cat([streams_to_host(mesh, od, hidden).float(), row_split_to_host(mesh, pd).float()], dim=-1)
        for t in (hd, md, yd, od, pd):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _hidden_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] fp32 (harness boundary: rows split over axis 0, hidden columns over
    axis 1, in and out)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, hidden_split_to_host

    def fn(ctx, x):
        xd = hidden_split_to_device(mesh, x)
        od = module(xd)
        out = hidden_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _hidden_to_rows_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, W] fp32 (harness boundary: x rows split over axis 0 and hidden columns over
    axis 1 in; the output rows split over axis 0, replicated over axis 1, read back from mesh column 0)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = hidden_split_to_device(mesh, x)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _kv_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> window KV host [S, 512] fp32 (harness boundary: attn_norm rows split over axis 0 and
    hidden columns over axis 1 in; the output rows split over axis 0, replicated over axis 1, read back from mesh
    column 0). The positions are [ctx.start, ctx.start + S)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = hidden_split_to_device(mesh, x)
        od = module(xd, int(ctx.start))
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _attention_host_fn(mesh, module):
    """fn(ctx, q_resid_host [S, 1280], kv_host [S, 512]) -> attn_out host [S, H] fp32 (harness boundary: both inputs
    row-split over axis 0, replicated over axis 1; the output's rows over axis 0, hidden over axis 1). The served
    window cache is ctx.extra["window_cache"] (the hybrid model's state); without one (component tests) a cache for
    ctx.extra["max_seq"] positions is made for the call and loaded with the golden ring (state_prefix window_kv) of
    positions [start - 128, start). ctx.extra["end"] = actual_end (default: the whole chunk)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.attention import TtWindowCache
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_host, row_split_to_device

    def fn(ctx, qr, kv):
        start, s = int(ctx.start), qr.shape[0]
        end = ctx.extra.get("end")
        end = start + s if end is None else int(end)
        cache = ctx.extra.get("window_cache")
        own = cache is None
        if own:
            cache = TtWindowCache(mesh, module.tables, max(int(ctx.extra.get("max_seq") or 0), start + s))
            prefix = ctx.extra.get("state_prefix") or {}
            if start and "window_kv" in prefix:
                cache.load_ring(prefix["window_kv"], start)
        qd = row_split_to_device(mesh, qr)
        kd = row_split_to_device(mesh, kv, dtype=ttnn.bfloat16)
        od = module(qd, kd, start, end, cache)
        out = hidden_split_to_host(mesh, od).float()
        for t in (qd, kd, od):
            ttnn.deallocate(t)
        if own:
            cache.free()
        return out

    fn.module = module
    return fn


def _router_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> dense routing host [S, E] fp32 (harness boundary: ffn_norm's gathered rows split over
    axis 0, replicated over axis 1, in; the dense [S/SP, E] fp32 ROW_MAJOR routing back from mesh column 0)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import row_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = row_split_to_device(mesh, x)
        dense, idx, wts = module(xd)
        out = row_split_to_host(mesh, dense).float()
        for t in (xd, dense, idx, wts):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _experts_host_fn(mesh, module):
    """fn(ctx, x_host [S, H], routing_host [S, E]) -> experts_out host [S, H] fp32 (harness boundary: ffn_norm's
    gathered rows and the dense routing split over axis 0, replicated over axis 1, in; the rows over axis 0 and hidden
    columns over axis 1 back)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_host, row_split_to_device

    def fn(ctx, x, routing):
        xd = row_split_to_device(mesh, x)
        rd = row_split_to_device(mesh, routing)
        od = module(xd, rd)
        out = hidden_split_to_host(mesh, od).float()
        for t in (xd, rd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _rows_to_hidden_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] fp32 (harness boundary: ffn_norm's gathered rows split over axis 0,
    replicated over axis 1, in; the rows over axis 0 and hidden columns over axis 1 back)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_host, row_split_to_device

    def fn(ctx, x):
        xd = row_split_to_device(mesh, x)
        od = module(xd)
        out = hidden_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _two_hidden_host_fn(mesh, module):
    """fn(ctx, a_host [S, H], b_host [S, H]) -> host [S, H] fp32 (harness boundary: both inputs rows over axis 0 and
    hidden columns over axis 1, fp32, in and out)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, hidden_split_to_host

    def fn(ctx, a, b):
        ad = hidden_split_to_device(mesh, a)
        bd = hidden_split_to_device(mesh, b)
        od = module(ad, bd)
        out = hidden_split_to_host(mesh, od).float()
        for t in (ad, bd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _engram_rows_host_fn(mesh, module):
    """fn(ctx) -> engram rows host [S, 24 x 256] fp32 (the host hash + lookup, the one transfer to the device, then
    the harness boundary read-back from mesh column 0). The tail (3 compressed ids before ctx.start) is the reference
    state's when the context carries one (the hybrid model: it is updated there, host state per slot), else the
    golden state_prefix's (component tests); at start 0 there is none. ctx.extra["end"] = actual_end: rows past it
    are pad (zero, not hashed)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import row_split_to_host

    def fn(ctx):
        start, layer = int(ctx.start), module.layer
        st = ctx.state
        own = st is not None and layer in getattr(st, "tail", {})
        if start == 0:
            tail = module.new_tail()
        elif own:
            tail = st.tail[layer]
        else:
            tail = (ctx.extra.get("state_prefix") or {})["engram_tail"]
        end = ctx.extra.get("end")
        rd, new_tail = module(ctx.extra["tokens"], tail.long(), None if end is None else int(end) - start)
        if own:
            st.tail[layer] = new_tail
        out = row_split_to_host(mesh, rd).float()
        ttnn.deallocate(rd)
        return out

    fn.module = module
    return fn


def _chunk(spec):
    """The serving chunk the window tables are built for (the run's target chunk; every ladder rung uses it)."""
    return int(spec.get("target.chunk"))


def _window_tables(mesh, spec, shared):
    """One set of load-time window tables per model (tt/attention.WindowTables)."""
    from models.demos.deepseek_v41_flash_d_p.tt.attention import WindowTables

    if "window_tables" not in shared:
        shared["window_tables"] = WindowTables(mesh, _chunk(spec))
    return shared["window_tables"]


def _max_seq(spec):
    """Positions the load-time tables (RoPE) cover: the longest sequence this run's ladders reach."""
    seqs = [int(spec.get("target.seq", 0) or 0)]
    for key in ("ladder", "scaled.ladder"):
        seqs += [int(r.get("seq", 0) or 0) for r in (spec.get(key) or [])]
    return max(seqs)


def _device_step_fn(mesh, spec, layer, step, ck, cfg, shared=None):
    shared = {} if shared is None else shared
    if step == "engram_rows":
        from models.demos.deepseek_v41_flash_d_p.tt.engram import build_engram_rows

        return _engram_rows_host_fn(mesh, build_engram_rows(mesh, ck, cfg, layer, shared.get("engram_hasher")))
    if step == "attention":
        from models.demos.deepseek_v41_flash_d_p.tt.attention import build_attention

        tables = _window_tables(mesh, spec, shared)
        return _attention_host_fn(
            mesh, build_attention(mesh, ck, cfg, layer, _max_seq(spec), _chunk(spec), tables=tables)
        )
    if step == "kv":
        from models.demos.deepseek_v41_flash_d_p.tt.kv import build_kv

        return _kv_host_fn(mesh, build_kv(mesh, ck, cfg, layer, _max_seq(spec)))
    if step == "experts":
        from models.demos.deepseek_v41_flash_d_p.tt.experts import build_experts

        return _experts_host_fn(mesh, build_experts(mesh, ck, cfg, layer, _chunk(spec)))
    if step == "shared_expert":
        from models.demos.deepseek_v41_flash_d_p.tt.shared_expert import build_shared_expert

        return _rows_to_hidden_host_fn(mesh, build_shared_expert(mesh, ck, cfg, layer))
    if step == "moe_add":
        from models.demos.deepseek_v41_flash_d_p.tt.moe_add import build_moe_add

        return _two_hidden_host_fn(mesh, build_moe_add(cfg))
    if step == "router":
        from models.demos.deepseek_v41_flash_d_p.tt.router import build_router

        return _router_host_fn(mesh, build_router(mesh, ck, cfg, layer, _chunk(spec)))
    if step == "q_a":
        from models.demos.deepseek_v41_flash_d_p.tt.q_a import build_q_a

        return _hidden_to_rows_host_fn(mesh, build_q_a(mesh, ck, cfg, layer))
    if step == "attn_norm":
        from models.demos.deepseek_v41_flash_d_p.tt.norm import build_norm

        return _hidden_host_fn(mesh, build_norm(mesh, ck, cfg, layer, step))
    if step == "ffn_norm":
        from models.demos.deepseek_v41_flash_d_p.tt.norm import build_norm

        # output gathered over axis 1: whole rows, replicated within a mesh row (read back from mesh column 0)
        return _hidden_to_rows_host_fn(mesh, build_norm(mesh, ck, cfg, layer, step))
    if step == "attn_residual":
        from models.demos.deepseek_v41_flash_d_p.tt.residual import build_residual

        return _residual_host_fn(mesh, build_residual(cfg), cfg.dim)
    if step == "ffn_residual":
        from models.demos.deepseek_v41_flash_d_p.tt.residual import build_residual

        return _ffn_residual_host_fn(mesh, build_residual(cfg, step=step), cfg.dim)
    if step == "attn_collapse":
        from models.demos.deepseek_v41_flash_d_p.tt.collapse import build_collapse

        return _collapse_host_fn(mesh, build_collapse(cfg), cfg.dim)
    if step == "ffn_collapse":
        from models.demos.deepseek_v41_flash_d_p.tt.collapse import build_collapse

        return _ffn_collapse_host_fn(mesh, build_collapse(cfg), cfg.dim)
    if step in _HC_STEPS:
        from models.demos.deepseek_v41_flash_d_p.tt.mhc import build_hc

        return _hc_host_fn(mesh, build_hc(mesh, ck, cfg, layer, step), cfg.dim)
    return None


def device_component(mesh, spec, layer, step):
    fn = _device_step_fn(mesh, spec, layer, step, _checkpoint(spec), _config(spec))
    if fn is None:
        raise NotImplementedError(f"implement step: no device module for {step} yet")
    return fn


class _HybridState:
    """The CPU reference state, plus the served window cache (tt/attention.TtWindowCache, 0_window_kv) of every layer
    whose attention runs on the device: the reference's window ring is then a view of that cache (to_torch /
    load_prefix), and state.served reads it (serving_contract.md "KV cache")."""

    def __init__(self, ref, max_seq, mesh=None, tables=None, window_layers=()):
        from models.demos.deepseek_v41_flash_d_p.tt.attention import TtWindowCache

        self.ref, self.s = ref, ref.new_state(max_seq)
        self.window = {i: TtWindowCache(mesh, tables, max_seq) for i in window_layers}
        self.tokens = None

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)
        if layer in self.window:
            self.window[layer].load_ring(tensors["window_kv"], length)

    def to_torch(self, layer, length):
        out = self.ref.state_tensors(self.s, layer, length)
        if layer in self.window:
            out["window_kv"] = self.window[layer].ring_host(length)
        return out

    def served(self, layer, length):
        """{config name: natural-order rows of positions [0, ceil32(length))} (tests/bringup/contract/parts.py)."""
        if layer not in self.window:
            raise NotImplementedError(f"layer {layer}: its attention (and served caches) is not on the device yet")
        return {"0_window_kv": self.window[layer].to_host(length)}

    def load_served(self, layer, tensors, length):
        if layer not in self.window:
            raise NotImplementedError(f"layer {layer}: its attention (and served caches) is not on the device yet")
        self.window[layer].load_host(tensors["0_window_kv"][:length], 0)

    def load_tokens(self, ids):
        self.tokens = ids

    def formats(self):
        import ttnn

        return {"0_window_kv": ttnn.bfloat16}

    def free(self):
        for c in self.window.values():
            c.free()
        self.window = {}


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    The block boundary (the packed [S, 4H + 4] streams + pending pre) stays on the host. The device model until the
    assemble step builds the all-device one."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        t0 = time.time()
        self.spec, self.mesh = spec, mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        ck = _checkpoint(spec)
        self.overrides, self.shared = {}, {"engram_hasher": self.ref.hasher}
        for i in self.ref.layer_ids:
            steps = sorted(DEVICE_STEPS.get(spec.block_type_of(i), ()))
            self.overrides[i] = {st: _device_step_fn(mesh, spec, i, st, ck, self.cfg, self.shared) for st in steps}
            missing = [st for st, f in self.overrides[i].items() if f is None]
            assert not missing, f"layer {i}: no device module for {missing}"
        self.window_layers = [i for i in self.ref.layer_ids if "attention" in self.overrides[i]]
        self._tokens = None
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq, self.mesh, self.shared.get("window_tables"), self.window_layers)

    def begin_chunk(self, tokens, start):
        self._tokens = tokens.reshape(-1).long()

    def embed(self, tokens, start=0):
        """Natural-order ids of the chunk at ``start`` (PAD_ID past actual_end, clamped into the vocabulary, K-72:
        those rows are pad and never reach a cache)."""
        ids = tokens.reshape(-1).long().clamp(0, self.cfg.vocab_size - 1)
        return self.ref.first_block_input(self.ref.embed(ids))

    def from_host(self, h, start=0):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state, end=None):
        """``end`` = actual_end (rows at or past it are pad: they never reach a served cache)."""
        from models.demos.common.bringup.reference.interface import run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
        if self._tokens is not None:
            ctx.extra["tokens"] = self._tokens
        ctx.extra["end"] = end
        if i in state.window:
            ctx.extra["window_cache"] = state.window[i]
        return run_block(
            self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=self.overrides[i]
        )

    def final_norm(self, h):
        return self.ref.final_norm(h)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    """The hybrid model (CPU reference + the validated device steps); the all-device model comes with assemble."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
