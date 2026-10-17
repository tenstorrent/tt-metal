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


def official_taps(spec, layers, tokens, chunk, chunk_size):
    """F80: DeepSeek's inference/model.py at its shipped dtypes (bf16, the GPU configuration) over the first chunk + 1
    chunks, every block step of every layer tapped for chunk ``chunk`` (reference/official_taps.py has the map;
    official_taps.yaml the dtypes model.py holds each step's inputs and output in). Per layer also ``_state``: the
    official run's own state at the chunk's start and end in DS41Reference.state_tensors' format (the KV window,
    compressed caches, top-k and candidates from model.py's buffers; the engram tail from the token ids)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.official_taps import capture

    path = hf_path(spec)
    taps, states = capture(path, layers, tokens, chunk, chunk_size, states=True)
    cfg = _config(spec)
    tail = {}
    if any(cfg.is_engram(L) for L in taps):  # the engram tail depends only on the token ids (model.py: none kept)
        from transformers import AutoTokenizer

        from models.demos.deepseek_v41_flash_d_p.reference.ngram import EngramHasher

        hasher = EngramHasher.build(cfg.raw, AutoTokenizer.from_pretrained(path))
        toks = torch.as_tensor(tokens).long().reshape(-1)
        for tag, length in (("start", chunk * chunk_size), ("end", (chunk + 1) * chunk_size)):
            t = hasher.new_tail()
            if length:
                L0 = next(L for L in taps if cfg.is_engram(L))
                t = hasher.hashes(toks[:length], L0, t)[1]
            tail[tag] = t
    for L, st in states.items():
        for tag, d in st.items():
            if "engram_tail" in d:
                d["engram_tail"] = tail[tag].clone()
        taps[L]["_state"] = st
    return taps


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
    # the attention side reuses the swa modules (component gates C.swa_engram.attn_hc .. attention passed on them)
    "swa_engram": {
        "engram_rows",
        "engram_kv",
        "engram_add",
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "q_a",
        "kv",
        "attention",
    },
    # attn_hc .. kv: the swa modules (C.enc_source.* and S.enc_source.01-05 passed on them, DS41-28); the compressor
    # writes the served compressed caches and updates the reference state; the indexer reads its keys from the
    # served index-key cache and updates the reference state's topk_idx; the attention (tt/sparse_attention.py)
    # reads the served compressed cache and writes the served window cache
    "enc_source": {
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "q_a",
        "kv",
        "compressor",
        "indexer",
        "attention",
    },
    # enc_source + engram: the engram steps are the swa_engram modules, the rest the enc_source ones (every step
    # listed passed its C.enc_source_engram.* component gate on them, DS41-28; the attention is
    # tt/sparse_attention.py, which writes layer 14's served window cache and reads its served compressed cache)
    "enc_source_engram": {
        "engram_rows",
        "engram_kv",
        "engram_add",
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "q_a",
        "kv",
        "compressor",
        "indexer",
        "attention",
    },
    # attn_hc .. kv: the swa modules (C.enc_reuse.attn_hc .. kv and S.enc_reuse.01-05 passed on them, DS41-28); the
    # attention (tt/sparse_attention.py, C.enc_reuse.attention) reads its source layer's served compressed cache and
    # device top-k, and writes its own served window cache
    "enc_reuse": {
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "q_a",
        "kv",
        "attention",
    },
    # attn_hc .. compressor passed C.dec_candidate.* (and S.dec_candidate.01-06) on the enc_source modules at ratio 1
    # (DS41-28); the indexer is tt/indexer.py:TtCandidateIndexer (top-k + the candidate pool kept on the device)
    "dec_candidate": {
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "q_a",
        "kv",
        "compressor",
        "indexer",
    },
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


def _sparse_attention_host_fn(mesh, module, cfg, layer):
    """fn(ctx, q_resid_host [S, 1280], kv_host [S, 512][, compressed_host [S / r, head_dim + index_dim]][, topk_host
    [S, topk] int32]) -> attn_out host [S, H] fp32, for a compressed layer (harness boundary: q_resid, kv and the top-k
    row-split over axis 0, replicated over axis 1; the output's rows over axis 0, hidden over axis 1). ``compressed``
    is an input only of a layer that is its own KV source, ``topk`` only of its own index source (enc_source); a
    reader (enc_reuse) reads both from its sources (ds41_ref.Config.kv_source_of / index_source_of). The window cache
    as _attention_host_fn.

    Compressed rows: read on the device from the KV source's served compressed cache ctx.extra["compress_cache"].ckv
    (the hybrid model's: the layer's own, or for a reader its source's, which the source's device compressor wrote
    this chunk into); without one (component tests) a cache for the call is made and loaded with the golden: an own
    source's prefix (state_prefix compress_kv) + this chunk's rows (``compressed``'s first head_dim columns), a
    reader's source state as of the end of the chunk (source_state[kv source] compress_kv, F65).
    Top-k of a reader: the index source's device top-k ctx.extra["topk_device"] (the hybrid model's: the [1, 1, S/SP,
    K] uint32 output the source's device indexer kept this chunk, read, not consumed); without one the source state's
    rows [start, start + S) (source_state[index source] topk_idx, F65; the hybrid model's reference state)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.attention import TtWindowCache
    from models.demos.deepseek_v41_flash_d_p.tt.compressor import TtServedRows
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_host, row_split_to_device

    r, hd = cfg.ratio(layer), cfg.head_dim
    kv_src, idx_src = cfg.kv_source_of(layer), cfg.index_source_of(layer)
    own_kv, own_idx = kv_src == layer, idx_src == layer
    assert own_kv or not own_idx, f"layer {layer}: its own top-k over another layer's KV is not built yet"

    def _source(ctx, j, name, a, b):
        """Rows [a, b) of source layer j's state tensor `name` (F65 source_state, else the reference state)."""
        src = (ctx.extra.get("source_state") or {}).get(j)
        if src is not None:
            return src[name][a:b]
        st = ctx.state
        assert st is not None, f"layer {layer}: no state of its source layer {j}"
        return {"compress_kv": st.ckv, "topk_idx": st.topk}[name][j][a:b]

    def fn(ctx, qr, kv, *rest):
        rest = list(rest)
        compressed = rest.pop(0) if own_kv else None
        topk = rest.pop(0) if own_idx else None
        assert not rest, f"layer {layer}: unexpected attention inputs"
        start, s = int(ctx.start), qr.shape[0]
        end = ctx.extra.get("end")
        end = start + s if end is None else int(end)
        max_seq = max(int(ctx.extra.get("max_seq") or 0), start + s)
        prefix = ctx.extra.get("state_prefix") or {}
        cache = ctx.extra.get("window_cache")
        own = cache is None
        if own:
            cache = TtWindowCache(mesh, module.tables, max_seq)
            if start and "window_kv" in prefix:
                cache.load_ring(prefix["window_kv"], start)
        caches = ctx.extra.get("compress_cache")
        own_c = caches is None
        if own_c:
            ckv = TtServedRows(mesh, module.tables, max_seq, hd, r, dedup=True)
            if own_kv:
                rows = [prefix["compress_kv"][: start // r].float()] if start else []
                rows.append(compressed[:, :hd].float())
                rows = torch.cat(rows)
            else:
                rows = _source(ctx, kv_src, "compress_kv", 0, (start + s) // r).float()
            ckv.load_host(rows, 0)
        else:
            ckv = caches.ckv
        qd = row_split_to_device(mesh, qr)
        kd = row_split_to_device(mesh, kv, dtype=ttnn.bfloat16)
        sp = mesh.shape[0]
        td = None if own_idx else ctx.extra.get("topk_device")
        own_t = td is None
        if own_t:
            if topk is None:
                topk = _source(ctx, idx_src, "topk_idx", start, start + s)
            td = ttnn.from_torch(
                topk.to(torch.int32).reshape(sp, 1, s // sp, -1).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
            )
        od = module(qd, kd, start, end, cache, ckv, td)
        out = hidden_split_to_host(mesh, od).float()
        for t in (qd, kd, od):
            ttnn.deallocate(t)
        if own_t:
            ttnn.deallocate(td)
        if own:
            cache.free()
        if own_c:
            ckv.free()
        return out

    fn.module = module
    return fn


def _compressor_host_fn(mesh, module, cfg, layer):
    """fn(ctx, x_host [S, H]) -> compressed host [S / r, head_dim + index_dim] fp32 = [ckv | k] (harness boundary:
    attn_norm rows over axis 0 and hidden columns over axis 1 in; both halves row-split, TP-replicated, read back from
    mesh column 0). The chunk's complete groups (by ctx.extra["end"] = actual_end, default the whole chunk) are
    written into the served caches ctx.extra["compress_cache"] (tt/compressor.TtCompressCaches, the hybrid model's
    state); without one (component tests) a pair is made for the call and freed. With a reference state carrying
    the layer (the hybrid model), its compress_kv / index_k rows of the chunk and its partial group are updated as
    the reference step does, for the CPU steps that read them (indexer, attention, reader layers)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.compressor import TtCompressCaches
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, row_split_to_host

    r, hd = module.ratio, cfg.head_dim

    def fn(ctx, x):
        start, s = int(ctx.start), x.shape[0]
        assert start % r == 0 and s % r == 0, (start, s, r)
        end = ctx.extra.get("end")
        end = start + s if end is None else int(end)
        caches = ctx.extra.get("compress_cache")
        own = caches is None
        if own:
            max_seq = max(int(ctx.extra.get("max_seq") or 0), start + s)
            caches = TtCompressCaches(mesh, module.tables, max_seq, r, hd, cfg.index_dim)
        xd = hidden_split_to_device(mesh, x)
        ckv, k = module(xd, start)
        module.write(ckv, k, start, end, caches)
        out = torch.cat([row_split_to_host(mesh, ckv).float(), row_split_to_host(mesh, k).float()], dim=-1)
        for t in (xd, ckv, k):
            ttnn.deallocate(t)
        if own:
            caches.free()
        st = ctx.state
        if st is not None and layer in getattr(st, "ckv", {}):
            g0, n = start // r, out.shape[0]
            st.ckv[layer][g0 : g0 + n] = out[:, :hd]
            st.index_k[layer][g0 : g0 + n] = out[:, hd:]
            if layer in st.partial:  # start and S even: no group pending after the chunk (reference _compressor)
                st.partial[layer].zero_()
        return out

    fn.module = module
    return fn


def _indexer_host_fn(mesh, module, cfg, layer, tables):
    """fn(ctx, x_host [S, H], qr_host [S, q_lora], compressed_host [S / r, head_dim + index_dim]) -> top-k host
    [S, topk] int32 (key positions, -1 = none) (harness boundary: attn_norm rows over axis 0 and hidden columns over
    axis 1, q_resid row-split and TP-replicated, in; the row-split, TP-replicated top-k back from mesh column 0).
    The keys are read on the device from the layer's served index-key cache: ctx.extra["compress_cache"].index_k
    (the hybrid model's, which the device compressor wrote this chunk into); without one (component tests) a cache
    for the call is made and loaded with the golden prefix (state_prefix index_k) and this chunk's keys
    (``compressed``'s last index_dim columns). With a reference state carrying the layer (the hybrid model), its
    topk_idx rows of the chunk are updated as the reference step does (for the ladder state and reader layers). With
    ctx.extra["topk_store"] (the hybrid model's {layer: (start, device top-k)}) the device output [1, 1, S/SP, K] uint32 stays
    on the device there for this chunk's reader layers (the previous chunk's is freed)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.compressor import TtServedRows
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, row_split_to_device

    r, hd = module.ratio, cfg.head_dim

    def fn(ctx, x, qr, compressed=None):
        start, s = int(ctx.start), x.shape[0]
        caches = ctx.extra.get("compress_cache")
        own = caches is None
        if own:
            max_seq = max(int(ctx.extra.get("max_seq") or 0), start + s)
            cache = TtServedRows(mesh, tables, max_seq, cfg.index_dim, r, dedup=False)
            prefix = (ctx.extra.get("state_prefix") or {}).get("index_k")
            keys = [prefix[: start // r].float()] if start else []
            keys.append(compressed[:, hd:].float())
            cache.load_host(torch.cat(keys), 0)
        else:
            cache = caches.index_k
        module.setup(cache)
        xd = hidden_split_to_device(mesh, x)
        qd = row_split_to_device(mesh, qr)
        cands = None
        if hasattr(module, "candidates_of"):  # the candidate source (layer 20): top-k + the candidate pool
            od, cands = module.forward(xd, qd, start, cache)
        else:
            od = module(xd, qd, start, cache)
        full = ttnn.to_torch(
            od, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3))
        )
        k = od.shape[-1]
        idx = full[..., :k].reshape(-1, k).to(torch.int64) & 0xFFFFFFFF  # mesh column 0's copy, as uint32
        out = torch.where(idx >= 0xFFFFFFFF, torch.full_like(idx, -1), idx).to(torch.int32)
        for t in (xd, qd):
            ttnn.deallocate(t)
        store = ctx.extra.get("topk_store")
        if store is None:
            ttnn.deallocate(od)
        else:
            old = store.pop(layer, None)
            if old is not None:
                ttnn.deallocate(old[1])
            store[layer] = (start, od)
        if own:
            cache.free()
        st = ctx.state
        if st is not None and layer in getattr(st, "topk", {}):
            st.topk[layer][start : start + s] = out
        if cands is not None:
            _keep_candidates(mesh, ctx, st, layer, start, s, cands)
        return out

    fn.module = module
    return fn


def _keep_candidates(mesh, ctx, st, layer, start, s, cands):
    """The candidate source's device pool [1, 1, Sq, n] uint32 per chip (the chip's own query rows, device order:
    chip (r, c) holds rows (r TP + c) Sq): with ctx.extra["cand_store"] (the hybrid model's {layer: (start, tensor)})
    it stays on the device for this chunk's dec_reindex layers (the previous chunk's freed), else it is freed. With a
    reference state carrying the layer, its candidates rows [start, start + S) are written in the reference's format
    (ascending block ids, -1 = none, last): harness boundary, read back per chip."""
    import ttnn

    if st is not None and layer in getattr(st, "cand", {}):
        rows = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(cands)], dim=-2)
        n = rows.shape[-1]
        ids = rows.reshape(-1, n).to(torch.int64) & 0xFFFFFFFF
        none = ids >= 0xFFFFFFFF
        ids = torch.sort(torch.where(none, torch.full_like(ids, 1 << 40), ids), dim=-1).values
        st.cand[layer][start : start + s] = torch.where(ids >= (1 << 40), -1, ids).to(torch.int32)[:s]
    store = ctx.extra.get("cand_store")
    if store is None:
        ttnn.deallocate(cands)
        return
    old = store.pop(layer, None)
    if old is not None:
        ttnn.deallocate(old[1])
    store[layer] = (start, cands)


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


def _engram_kv_host_fn(mesh, module, hidden):
    """fn(ctx, rows_host [S, 6144]) -> engram_kv host [S, 5H] fp32 (harness boundary: the rows split over axis 0,
    replicated over axis 1, in; each chip's hidden slice of the 5 H-wide blocks gathered over both axes and put back
    in block order)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.engram_kv import BLOCKS
    from models.demos.deepseek_v41_flash_d_p.tt.layout import blocks_to_host, row_split_to_device

    def fn(ctx, rows):
        rd = row_split_to_device(mesh, rows)
        od = module(rd)
        out = blocks_to_host(mesh, od, hidden, BLOCKS).float()
        ttnn.deallocate(rd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _engram_add_host_fn(mesh, module, hidden):
    """fn(ctx, x_host [S, 4H + 4], kv_host [S, 5H]) -> x host [S, 4H + 4] fp32 (harness boundary: the streams in, the
    pending pre row-split and TP-replicated, engram_kv as each chip's hidden slice of its 5 blocks; back the new
    streams and the module's pending pre, packed here only)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.engram_kv import BLOCKS
    from models.demos.deepseek_v41_flash_d_p.tt.layout import (
        HC,
        blocks_to_device,
        row_split_to_device,
        row_split_to_host,
        streams_to_device,
        streams_to_host,
    )

    def fn(ctx, x, kv):
        assert x.shape[-1] == HC * hidden + HC and kv.shape[-1] == BLOCKS * hidden, (x.shape, kv.shape)
        xd = streams_to_device(mesh, x, hidden)
        pd = row_split_to_device(mesh, x[:, HC * hidden :])
        kd = blocks_to_device(mesh, kv, hidden, BLOCKS)
        od, po = module(xd, kd, pd)
        out = torch.cat([streams_to_host(mesh, od, hidden).float(), row_split_to_host(mesh, po).float()], dim=-1)
        for t in (xd, pd, kd, od):
            ttnn.deallocate(t)
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
    if step == "engram_kv":
        from models.demos.deepseek_v41_flash_d_p.tt.engram_kv import build_engram_kv

        return _engram_kv_host_fn(mesh, build_engram_kv(mesh, ck, cfg, layer), cfg.dim)
    if step == "engram_add":
        from models.demos.deepseek_v41_flash_d_p.tt.engram_add import build_engram_add

        return _engram_add_host_fn(mesh, build_engram_add(mesh, ck, cfg, layer), cfg.dim)
    if step == "attention" and cfg.ratio(layer):
        from models.demos.deepseek_v41_flash_d_p.tt.sparse_attention import SparseIndexTables, build_sparse_attention

        if cfg.uses_candidates(layer) or cfg.is_cand_source(layer):
            return None  # the decoder layers (candidate pool) are not built yet
        if cfg.is_index_source(layer) and not cfg.is_kv_source(layer):
            return None  # its own top-k over another layer's KV (dec_reindex) is not built yet
        tables = _window_tables(mesh, spec, shared)
        if "sparse_index_tables" not in shared:
            shared["sparse_index_tables"] = SparseIndexTables(mesh, tables, cfg.index_topk)
        module = build_sparse_attention(
            mesh, ck, cfg, layer, _max_seq(spec), _chunk(spec), tables=tables, sparse=shared["sparse_index_tables"]
        )
        return _sparse_attention_host_fn(mesh, module, cfg, layer)
    if step == "attention":
        from models.demos.deepseek_v41_flash_d_p.tt.attention import build_attention

        tables = _window_tables(mesh, spec, shared)
        return _attention_host_fn(
            mesh, build_attention(mesh, ck, cfg, layer, _max_seq(spec), _chunk(spec), tables=tables)
        )
    if step == "compressor":
        from models.demos.deepseek_v41_flash_d_p.tt.compressor import build_compressor

        tables = _window_tables(mesh, spec, shared)
        return _compressor_host_fn(mesh, build_compressor(mesh, ck, cfg, layer, _max_seq(spec), tables), cfg, layer)
    if step == "indexer":
        from models.demos.deepseek_v41_flash_d_p.tt.indexer import build_indexer

        own_keys = cfg.is_kv_source(layer) and cfg.is_index_source(layer)
        if not own_keys or cfg.uses_candidates(layer):
            return None  # the indexers over their own keys (enc_source; dec_candidate with its pool) so far
        tables = _window_tables(mesh, spec, shared)
        module = build_indexer(mesh, ck, cfg, layer, _max_seq(spec), _chunk(spec))
        return _indexer_host_fn(mesh, module, cfg, layer, tables)
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

    def __init__(self, ref, max_seq, mesh=None, tables=None, window_layers=(), compress_layers=()):
        from models.demos.deepseek_v41_flash_d_p.tt.attention import TtWindowCache
        from models.demos.deepseek_v41_flash_d_p.tt.compressor import TtCompressCaches

        cfg = ref.cfg
        self.ref, self.s = ref, ref.new_state(max_seq)
        self.window = {i: TtWindowCache(mesh, tables, max_seq) for i in window_layers}
        # served 1_compress_kv_r2 / 2_index_k_r2 (3_ / 4_ for ratio 1) of every layer whose compressor is on device
        self.compress = {
            i: TtCompressCaches(mesh, tables, max_seq, cfg.ratio(i), cfg.head_dim, cfg.index_dim)
            for i in compress_layers
        }
        self.topk = {}  # {index-source layer: (chunk start, its device indexer's top-k of that chunk)} (readers)
        self.cands = {}  # {candidate source: (chunk start, its device candidate pool [1, 1, Sq, 2048] per chip)}
        self.tokens = None

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)
        if layer in self.window:
            self.window[layer].load_ring(tensors["window_kv"], length)
        if layer in self.compress and length:
            c = self.compress[layer]
            c.ckv.load_host(tensors["compress_kv"][: length // c.ratio], 0)
            c.index_k.load_host(tensors["index_k"][: length // c.ratio], 0)

    def to_torch(self, layer, length):
        out = self.ref.state_tensors(self.s, layer, length)
        if layer in self.window:
            out["window_kv"] = self.window[layer].ring_host(length)
        return out

    def served(self, layer, length):
        """{config name: natural-order rows of positions [0, ceil32(length))} (tests/bringup/contract/parts.py)."""
        if layer not in self.window:
            raise NotImplementedError(f"layer {layer}: its attention (and served caches) is not on the device yet")
        out = {"0_window_kv": self.window[layer].to_host(length)}
        if layer in self.compress:
            out.update({n: t.to_host(length) for n, t in self.compress[layer].by_config().items()})
        return out

    def load_served(self, layer, tensors, length):
        if layer not in self.window:
            raise NotImplementedError(f"layer {layer}: its attention (and served caches) is not on the device yet")
        self.window[layer].load_host(tensors["0_window_kv"][:length], 0)

    def load_tokens(self, ids):
        self.tokens = ids

    def formats(self):
        import ttnn

        out = {"0_window_kv": ttnn.bfloat16}
        for c in self.compress.values():
            out.update({n: ttnn.bfloat16 for n in c.names})
        return out

    def free(self):
        import ttnn

        for c in [*self.window.values(), *self.compress.values()]:
            c.free()
        for _, t in [*self.topk.values(), *self.cands.values()]:
            ttnn.deallocate(t)
        self.window, self.compress, self.topk, self.cands = {}, {}, {}, {}


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
        self.compress_layers = [i for i in self.ref.layer_ids if "compressor" in self.overrides[i]]
        self._tokens = None
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(
            self.ref, max_seq, self.mesh, self.shared.get("window_tables"), self.window_layers, self.compress_layers
        )

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
        if i in state.compress:
            ctx.extra["compress_cache"] = state.compress[i]
        if "indexer" in self.overrides[i]:
            ctx.extra["topk_store"] = state.topk
            ctx.extra["cand_store"] = state.cands
        if self.cfg.ratio(i):
            # a reader layer (enc_reuse) reads its sources' device state of this chunk: the KV source's served
            # compressed cache and the index source's device top-k (else the reference state's rows)
            ks, js = self.cfg.kv_source_of(i), self.cfg.index_source_of(i)
            if ks != i and ks in state.compress:
                ctx.extra["compress_cache"] = state.compress[ks]
            if js != i and js in state.topk and state.topk[js][0] == start:
                ctx.extra["topk_device"] = state.topk[js][1]
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
