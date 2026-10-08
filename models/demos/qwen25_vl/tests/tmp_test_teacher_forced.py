# TEMPORARY investigation test (not for commit): teacher-forced per-step decode comparison TT vs HF (CPU).
# Feeds HF's greedy continuation token-by-token into the TT text decoder and records, per decode position,
# TT argmax agreement with HF and logits PCC. Text-only prompt (no vision), batch 1 per DP lane.
import json
import os

import pytest
import torch
from loguru import logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from models.common.utility_functions import comp_pcc
from models.demos.qwen25_vl.demo.demo import _qwen25_vl_device_params, prepare_generator_args
from models.demos.qwen25_vl.tt.common import multimodal_rope_from_hf, preprocess_inputs_prefill
from models.demos.qwen25_vl.tt.generator import Generator
from models.demos.qwen25_vl.tt.model_config import default_data_parallel, qwen25_vl_mesh_shape
from models.tt_transformers.tt.model_config import DecodersPrecision

PROMPT = os.environ.get("TF_PROMPT", "/home/cirrascale/ashai/rebase_oct26/ref/text_100.json")
GEN = int(os.environ.get("TF_GEN", 120))


@pytest.mark.parametrize("mesh_device", [qwen25_vl_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [_qwen25_vl_device_params()], indirect=True)
def test_teacher_forced(mesh_device, reset_seeds):
    model_id = os.environ["HF_MODEL"]
    num_devices = mesh_device.get_num_devices()
    data_parallel = default_data_parallel(model_id, num_devices)
    conv = json.load(open(PROMPT))
    conv = conv if isinstance(conv[0], dict) else conv[0]

    processor = AutoProcessor.from_pretrained(model_id)
    ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt")
    ids = inputs.input_ids
    L = ids.shape[1]

    # HF greedy continuation on CPU (reference tokens) + per-step reference logits via a forward over the full sequence
    with torch.no_grad():
        gen = ref.generate(**inputs, max_new_tokens=GEN, do_sample=False)
        full = gen[:, : L + GEN]
        ref_logits = ref(input_ids=full).logits.float()  # [1, L+GEN, V]; logits at t predict token t+1
    ref_tokens = full[0, L:].tolist()
    logger.info(f"prompt_len={L} ref_tokens[:12]={ref_tokens[:12]}")

    page_params = {"page_block_size": 32, "page_max_num_blocks": 1024}
    model_args_list, model_list, paged_cfg, kv_list, page_table = prepare_generator_args(
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        instruct=True,
        batch_size=1,
        optimizations=lambda a: DecodersPrecision.performance(a.n_layers, a.model_name),
        max_seq_len=4096,
        page_params=page_params,
        use_paged_kv_cache=True,
    )
    for a in model_args_list:
        a.use_qk_fused = False
    model_args = model_args_list[0]
    generator = Generator(model_list, model_args_list, mesh_device, processor=processor, tokenizer=model_args.tokenizer)
    global_batch = data_parallel
    pad_id = processor.tokenizer.pad_token_id

    # Prefill with the same prompt in every lane
    embeds = ref.model.language_model.embed_tokens(ids).repeat(global_batch, 1, 1)
    input_prefill, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        list(embeds),
        model_args,
        inputs.attention_mask.repeat(global_batch, 1),
        pad_embedding=ref.model.language_model.embed_tokens(torch.tensor(pad_id)),
    )
    inputs_b = type(inputs)(
        {"input_ids": ids.repeat(global_batch, 1), "attention_mask": inputs.attention_mask.repeat(global_batch, 1)}
    )
    cos, sin, rope_deltas = multimodal_rope_from_hf(inputs_b, embeds, ref, model_args, pad_token_id=pad_id)
    logits = generator.prefill_forward_text(
        input_prefill, rot_mats=(cos, sin), page_table=page_table, kv_cache=kv_list, prompt_lens=decoding_pos
    )
    generator.update_rope_deltas([int(d) for d in rope_deltas.flatten()])
    tt_first = int(torch.argmax(logits[0, -1]))
    _, pcc0 = comp_pcc(ref_logits[0, L - 1], logits[0, -1].float(), 0.99)
    logger.info(f"[TF] prefill: tt_argmax={tt_first} ref={ref_tokens[0]} pcc={pcc0}")

    current_pos = torch.tensor([L] * global_batch)
    rows = []
    for step in range(GEN - 1):
        tok = torch.tensor([[ref_tokens[step]]] * global_batch)  # teacher forcing: feed HF's token
        out, _ = generator.decode_forward(
            tok,
            current_pos,
            enable_trace=True,
            page_table=page_table,
            kv_cache=kv_list,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        lg = out[0].reshape(-1)[: model_args.vocab_size].float()
        pos = L + step  # position of the token just fed; logits predict token at pos+1
        r = ref_logits[0, pos]
        _, msg = comp_pcc(r, lg, 0.99)
        try:
            pcc = float(msg)
        except (TypeError, ValueError):
            pcc = float(str(msg).split()[-1])
        top2 = torch.topk(r, 2).values
        margin = float(top2[0] - top2[1])
        tt_top = int(torch.argmax(lg))
        agree = tt_top == ref_tokens[step + 1]
        rows.append((pos + 1, agree, pcc, margin, tt_top, ref_tokens[step + 1]))
        current_pos += 1
    bad = [(p, round(c, 3), round(m, 2), t, rr) for p, a, c, m, t, rr in rows if not a]
    logger.info(
        f"[TF] steps={len(rows)} argmax_agree={sum(a for _, a, *_ in rows)}/{len(rows)} disagreements(pos,pcc,hf_margin,tt_tok,hf_tok)={bad[:30]}"
    )
    logger.info("[TF] pcc by pos: " + " ".join(f"{p}:{c:.3f}" for p, _, c, *_ in rows if p % 4 == 0 or 124 <= p <= 134))
    logger.info("[TF] low-pcc positions (<0.97): " + str([(p, round(c, 3)) for p, _, c, *_ in rows if c < 0.97][:40]))
    logger.info(
        f"[TF] mean pcc={sum(c for _, _, c, *_ in rows)/len(rows):.4f} min pcc={min(c for _, _, c, *_ in rows):.4f}"
    )

    # ---- Layer-by-layer bisect of one decode step at position L+DEC_STEP (teacher forced), untraced, vs HF hidden_states
    import ttnn as _ttnn

    DEC_STEP = int(os.environ.get("TF_BISECT_STEP", 60))
    model = model_list[0]
    captured = {}
    orig_forwards = []
    for li, layer in enumerate(model.layers):
        of = layer.forward

        def mk(li, of):
            def wrapped(x, *a, **kw):
                out = of(x, *a, **kw)
                captured[li] = out
                return out

            return wrapped

        orig_forwards.append((layer, of))
        layer.forward = mk(li, of)
    with torch.no_grad():
        hf_h = ref(input_ids=full[:, : L + DEC_STEP + 1], output_hidden_states=True).hidden_states
    pos = L + DEC_STEP
    tok = torch.tensor([[ref_tokens[DEC_STEP]]] * global_batch)
    cp = torch.tensor([pos] * global_batch)
    generator.decode_forward(
        tok,
        cp,
        enable_trace=False,
        page_table=page_table,
        kv_cache=kv_list,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    for layer, of in orig_forwards:
        layer.forward = of
    res = []
    for li in sorted(captured):
        t = captured[li]
        try:
            tt_h = _ttnn.to_torch(t, mesh_composer=_ttnn.ConcatMeshToTensor(model.mesh_device, dim=-1)).float()
        except Exception:
            tt_h = torch.cat([_ttnn.to_torch(d) for d in _ttnn.get_device_tensors(t)], dim=-1).float()
        tt_vec = tt_h.reshape(-1, tt_h.shape[-1])[0, : model_args.dim]
        hf_vec = hf_h[li + 1][0, pos].float()
        _, pcc = comp_pcc(hf_vec, tt_vec, 0.99)
        res.append((li, float(pcc)))
    logger.info(
        f"[TF-LAYERS] decode step at pos {pos} per-layer output PCC vs HF: "
        + " ".join(f"L{li}:{p:.4f}" for li, p in res)
    )

    # ---- Intra-layer bisect at the same decode step: norm -> attention -> ffn-norm -> MLP for layers 0 and 3 vs HF hooks
    cap = {}

    def to_vec(t):
        try:
            h = _ttnn.to_torch(t, mesh_composer=_ttnn.ConcatMeshToTensor(model.mesh_device, dim=-1)).float()
        except Exception:
            h = torch.cat([_ttnn.to_torch(d) for d in _ttnn.get_device_tensors(t)], dim=-1).float()
        return h.reshape(-1, h.shape[-1])[0, : model_args.dim].clone()

    def wrap(obj, attr, key):
        orig = getattr(obj, attr)

        def w(*a, **kw):
            out = orig(*a, **kw)
            try:
                cap[key] = to_vec(out)  # read back immediately: the layer deallocates intermediates after use
            except Exception as e:
                cap[key] = None
                logger.warning(f"capture {key} failed: {e}")
            return out

        setattr(obj, attr, w)
        return (obj, attr, orig)

    restores = []
    for li in (0, 3):
        lay = model.layers[li]
        restores += [
            wrap(lay.attention_norm, "forward", f"L{li}.attn_norm"),
            wrap(lay.attention, "forward", f"L{li}.attn"),
            wrap(lay.ff_norm, "forward", f"L{li}.ff_norm"),
            wrap(lay.feed_forward, "forward", f"L{li}.mlp"),
        ]
    hf_cap = {}
    hooks = []
    for li in (0, 3):
        hl = ref.model.language_model.layers[li]
        for name, mod in (
            ("attn_norm", hl.input_layernorm),
            ("attn", hl.self_attn),
            ("ff_norm", hl.post_attention_layernorm),
            ("mlp", hl.mlp),
        ):
            hooks.append(
                mod.register_forward_hook(
                    lambda m, i, o, k=f"L{li}.{name}": hf_cap.__setitem__(
                        k, (o[0] if isinstance(o, tuple) else o).detach()
                    )
                )
            )
    with torch.no_grad():
        ref(input_ids=full[:, : L + DEC_STEP + 1])
    for h in hooks:
        h.remove()
    tok = torch.tensor([[ref_tokens[DEC_STEP]]] * global_batch)
    cp = torch.tensor([pos] * global_batch)
    generator.decode_forward(
        tok,
        cp,
        enable_trace=False,
        page_table=page_table,
        kv_cache=kv_list,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    for obj, attr, orig in restores:
        setattr(obj, attr, orig)
    parts = []
    for k in sorted(cap):
        if cap[k] is None:
            parts.append(f"{k}:n/a")
            continue
        tv = cap[k]
        hv = hf_cap[k][0, pos].float()
        _, pcc = comp_pcc(hv, tv, 0.99)
        parts.append(f"{k}:{float(pcc):.4f}")
    logger.info("[TF-INTRA] pos %d: %s" % (pos, "  ".join(parts)))

    # ---- Capture the real paged SDPA decode call at layers 0..3 and recompute attention in fp32 from its exact inputs
    sdpa_cap = []
    _orig_sdpa = _ttnn.transformer.paged_scaled_dot_product_attention_decode

    def _sdpa_spy(q, k, v, *a, **kw):
        out = _orig_sdpa(q, k, v, *a, **kw)
        if len(sdpa_cap) < 4:
            qd = [_ttnn.to_torch(t).float() for t in _ttnn.get_device_tensors(q)]
            kd = [_ttnn.to_torch(t).float() for t in _ttnn.get_device_tensors(k)]
            vd = [_ttnn.to_torch(t).float() for t in _ttnn.get_device_tensors(v)]
            od = [_ttnn.to_torch(t).float() for t in _ttnn.get_device_tensors(out)]
            ptd = _ttnn.to_torch(_ttnn.get_device_tensors(kw["page_table_tensor"])[0])
            cpd = _ttnn.to_torch(_ttnn.get_device_tensors(kw["cur_pos_tensor"])[0])
            pc_ = kw.get("program_config")
            cc_ = kw.get("compute_kernel_config")
            meta = dict(
                q_dtype=str(q.dtype),
                k_dtype=str(k.dtype),
                v_dtype=str(v.dtype),
                q_mem=str(q.memory_config()),
                q_shape=list(q.shape),
                k_shape=list(k.shape),
                pt_dtype=str(kw["page_table_tensor"].dtype),
                cp_dtype=str(kw["cur_pos_tensor"].dtype),
                q_chunk=pc_.q_chunk_size,
                k_chunk=pc_.k_chunk_size,
                exp_approx=pc_.exp_approx_mode,
                grid=str(pc_.compute_with_storage_grid_size),
                mcphb=pc_.max_cores_per_head_batch,
                fidelity=str(cc_.math_fidelity),
                fp32_acc=cc_.fp32_dest_acc_en,
                l1_acc=cc_.packer_l1_acc,
                approx=cc_.math_approx_mode,
                scale=kw.get("scale"),
                sliding=kw.get("sliding_window_size"),
            )
            # raw (un-floated) device tensors for exact replay
            raw = dict(
                q=[_ttnn.to_torch(t) for t in _ttnn.get_device_tensors(q)],
                k=[_ttnn.to_torch(t) for t in _ttnn.get_device_tensors(k)],
                v=[_ttnn.to_torch(t) for t in _ttnn.get_device_tensors(v)],
                pt=[_ttnn.to_torch(t) for t in _ttnn.get_device_tensors(kw["page_table_tensor"])],
                cp=[_ttnn.to_torch(t) for t in _ttnn.get_device_tensors(kw["cur_pos_tensor"])],
                out=[_ttnn.to_torch(t) for t in _ttnn.get_device_tensors(out)],
                meta=meta,
            )
            torch.save(raw, os.environ.get("TF_CAP_DIR", "/tmp") + f"/sdpa_cap_L{len(sdpa_cap)}.pt")
            sdpa_cap.append(
                dict(q=qd, k=kd, v=vd, out=od, pt=ptd, cp=cpd, scale=kw.get("scale"), pc=str(meta), cc=str(cc_))
            )
        return out

    _ttnn.transformer.paged_scaled_dot_product_attention_decode = _sdpa_spy
    try:
        generator.decode_forward(
            tok,
            cp,
            enable_trace=False,
            page_table=page_table,
            kv_cache=kv_list,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
    finally:
        _ttnn.transformer.paged_scaled_dot_product_attention_decode = _orig_sdpa
    for li, c in enumerate(sdpa_cap):
        q0 = c["q"][0]
        k0 = c["k"][0]
        v0 = c["v"][0]
        o0 = c["out"][0]
        cur = int(c["cp"].reshape(-1)[0])
        pt = c["pt"].reshape(c["pt"].shape[-2], -1)[0].long()
        nblk = cur // 32 + 1
        K = k0[pt[:nblk]].permute(1, 0, 2, 3).reshape(k0.shape[1], -1, k0.shape[-1])[:, : cur + 1]  # [nkv_local, S, D]
        V = v0[pt[:nblk]].permute(1, 0, 2, 3).reshape(v0.shape[1], -1, v0.shape[-1])[:, : cur + 1]
        Q = q0.reshape(-1, q0.shape[-2], q0.shape[-1])[0]  # [nh_local(padded rows?), D] -> take row block for user 0
        nh = model_args.n_heads // model_args.num_devices
        nkv = K.shape[0]
        Q = Q[:nh]
        rep = nh // nkv
        scores = (
            Q.unsqueeze(1)
            @ K.repeat_interleave(rep, 0).transpose(-1, -2).unsqueeze(0).squeeze(0).reshape(nh, K.shape[-1], -1)
            if False
            else None
        )
        ref_heads = []
        for h in range(nh):
            kh = K[h // rep]
            vh = V[h // rep]
            sc = (Q[h] @ kh.T) * float(c["scale"])
            w = torch.softmax(sc.double(), dim=-1).float()
            ref_heads.append(w @ vh)
        ref_out = torch.stack(ref_heads)  # [nh, D]
        tt_out = o0.reshape(-1, o0.shape[-2], o0.shape[-1])[0][:nh]
        _, pcc = comp_pcc(ref_out, tt_out, 0.99)
        per_head = [float(comp_pcc(ref_out[h], tt_out[h], 0.99)[1]) for h in range(nh)]
        qn = float(Q.abs().max())
        kn = float(K.abs().max())
        smax = float(((Q[0] @ K[0].T) * float(c["scale"])).abs().max())
        logger.info(
            f"[TF-SDPA] layer {li} cur_pos={cur} S={cur+1} nh={nh} nkv={nkv} dev0 PCC(fp32 ref vs TT)={float(pcc):.4f} per_head={[round(x,3) for x in per_head]} |Q|max={qn:.1f} |K|max={kn:.1f} |score|max={smax:.1f} pc={c['pc']}"
        )

    # ---- Context vectors (SDPA output, all devices, head-rearranged) vs HF o_proj input; and wo check in torch
    hf_ctx = {}
    hks = []
    for li in range(len(sdpa_cap)):
        hl = ref.model.language_model.layers[li]
        hks.append(
            hl.self_attn.o_proj.register_forward_hook(
                lambda m, i, o, k=li: hf_ctx.__setitem__(k, (i[0].detach(), o.detach()))
            )
        )
    with torch.no_grad():
        ref(input_ids=full[:, : L + DEC_STEP + 1])
    for h in hks:
        h.remove()
    hd = model_args.head_dim
    for li, c in enumerate(sdpa_cap):
        att = model.layers[li].attention
        q_order = (
            att._build_q_head_order()
            if hasattr(att, "_build_q_head_order") and att._needs_head_rearrangement()
            else list(range(model_args.n_heads))
        )
        nh_local = model_args.n_heads // model_args.num_devices
        tt_ctx = torch.zeros(len([o for o in q_order if o is not None]), hd)
        for d, od in enumerate(c["out"]):
            rows = od.reshape(-1, od.shape[-2], od.shape[-1])[0][:nh_local]  # [nh_local, D] user 0
            for hl_, slot in enumerate(range(d * nh_local, (d + 1) * nh_local)):
                orig = q_order[slot]
                if orig is not None:
                    tt_ctx[orig] = rows[hl_]
        ctx_in, attn_out_hf = hf_ctx[li]
        hf_c = ctx_in[0, pos].float().reshape(-1, hd)  # [28, D]
        per_head = [float(comp_pcc(hf_c[h], tt_ctx[h], 0.99)[1]) for h in range(hf_c.shape[0])]
        _, pcc_all = comp_pcc(hf_c, tt_ctx, 0.99)
        # wo in torch from TT context vs HF attention output
        Wo = ref.model.language_model.layers[li].self_attn.o_proj.weight.float()
        tt_wo = tt_ctx.reshape(-1) @ Wo.T
        _, pcc_wo = comp_pcc(attn_out_hf[0, pos].float(), tt_wo, 0.99)
        logger.info(
            f"[TF-CTX] layer {li}: ctx PCC(HF o_proj in vs TT sdpa out)={float(pcc_all):.4f} worst heads={sorted([(round(p,3),h) for h,p in enumerate(per_head)])[:4]} | torch-wo(TT ctx) vs HF attn out={float(pcc_wo):.4f}"
            + (
                f" | TT attn out vs HF={float(comp_pcc(attn_out_hf[0,pos].float(), cap.get(f'L{li}.attn'), 0.99)[1]):.4f}"
                if cap.get(f"L{li}.attn") is not None
                else ""
            )
        )

    # ---- Swap-in experiment: fp32 attention from TT inputs with one of Q/K/V replaced by HF's, per device/head, vs HF ctx
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb as _hf_rope

    hf_attn_in = {}
    hks = []
    for li in range(len(sdpa_cap)):
        hl = ref.model.language_model.layers[li]
        hks.append(
            hl.self_attn.register_forward_hook(
                lambda m, a, kw, o, k=li: hf_attn_in.__setitem__(
                    k,
                    (
                        a[0] if a else kw["hidden_states"],
                        kw.get("position_embeddings")
                        if kw.get("position_embeddings") is not None
                        else (a[-1] if len(a) > 6 else None),
                    ),
                ),
                with_kwargs=True,
            )
        )
    with torch.no_grad():
        hf_out2 = ref(input_ids=full[:, : pos + 1], use_cache=True)
    for h in hks:
        h.remove()
    pkv2 = hf_out2.past_key_values
    rs = ref.config.text_config.rope_scaling if hasattr(ref.config, "text_config") else ref.config.rope_scaling
    mrope_section = rs["mrope_section"]
    nh_local = model_args.n_heads // model_args.num_devices

    def _lay(x, name):
        D = x.shape[-1]
        if name == "interleave_halves":
            return x.view(*x.shape[:-1], 2, D // 2).transpose(-1, -2).reshape(x.shape)
        if name == "deinterleave":
            return x.view(*x.shape[:-1], D // 2, 2).transpose(-1, -2).reshape(x.shape)
        return x

    for li, c in enumerate(sdpa_cap):
        att = model.layers[li].attention
        hl = ref.model.language_model.layers[li]
        hs, pe = hf_attn_in[li]
        cos, sin = pe
        bs_, S_ = hs.shape[:2]
        q_hf = hl.self_attn.q_proj(hs).view(bs_, S_, -1, hd).transpose(1, 2)
        k_hf0 = hl.self_attn.k_proj(hs).view(bs_, S_, -1, hd).transpose(1, 2)
        q_hf, _ = _hf_rope(q_hf, k_hf0, cos, sin, mrope_section)
        q_hf = q_hf[0, :, pos].float()  # [28, D] post-rope at pos
        K_hf = pkv2.layers[li].keys[0].float() if hasattr(pkv2, "layers") else pkv2[li][0][0].float()  # [4, S, D]
        V_hf = pkv2.layers[li].values[0].float() if hasattr(pkv2, "layers") else pkv2[li][1][0].float()
        q_order = att._build_q_head_order() if att._needs_head_rearrangement() else list(range(model_args.n_heads))
        kv_order = att._build_kv_head_order() if att._needs_head_rearrangement() else list(range(model_args.n_kv_heads))
        hf_c = hf_ctx[li][0][0, pos].float().reshape(-1, hd)
        cur = int(c["cp"].reshape(-1)[0])
        pt = c["pt"].reshape(c["pt"].shape[-2], -1)[0].long()
        nblk = cur // 32 + 1
        lines = []
        for d in range(len(c["q"])):
            q0, k0, v0 = c["q"][d], c["k"][d], c["v"][d]
            K = k0[pt[:nblk]].permute(1, 0, 2, 3).reshape(k0.shape[1], -1, k0.shape[-1])[:, : cur + 1]
            V = v0[pt[:nblk]].permute(1, 0, 2, 3).reshape(v0.shape[1], -1, v0.shape[-1])[:, : cur + 1]
            Q = q0.reshape(-1, q0.shape[-2], q0.shape[-1])[0][:nh_local]
            nkv_local = K.shape[0]
            rep = nh_local // nkv_local
            for hl_ in range(nh_local):
                orig = q_order[d * nh_local + hl_]
                if orig is None:
                    continue
                kvg = kv_order[d * nkv_local + hl_ // rep]
                Kt, Vt, Qt = K[hl_ // rep], V[hl_ // rep], Q[hl_]
                # pick layout mapping HF->TT using prefill K rows
                lay = max(
                    ("hf", "interleave_halves", "deinterleave"),
                    key=lambda n: float(comp_pcc(_lay(K_hf[kvg, :L], n), Kt[:L], 0.99)[1]),
                )
                Kh, Qh, Vh = _lay(K_hf[kvg, : cur + 1], lay), _lay(q_hf[orig], lay), V_hf[kvg, : cur + 1]

                def attn(q, k, v):
                    w = torch.softmax(((q @ k.T) * float(c["scale"])).double(), dim=-1).float()
                    return w @ v

                tgt = hf_c[orig]
                r = {
                    "TT": attn(Qt, Kt, Vt),
                    "Qhf": attn(Qh, Kt, Vt),
                    "Khf": attn(Qt, Kh, Vt),
                    "Vhf": attn(Qt, Kt, Vh),
                    "allhf": attn(Qh, Kh, Vh),
                }
                pc = {k: float(comp_pcc(tgt, v, 0.99)[1]) for k, v in r.items()}
                qp = float(comp_pcc(Qh, Qt, 0.99)[1])
                kp_pre = float(comp_pcc(Kh[:L], Kt[:L], 0.99)[1])
                kp_dec = float(comp_pcc(Kh[L:], Kt[L:], 0.99)[1])
                vp_pre = float(comp_pcc(Vh[:L], Vt[:L], 0.99)[1])
                vp_dec = float(comp_pcc(Vh[L:], Vt[L:], 0.99)[1])
                # bias-free K compare: subtract HF k bias (rope'd) is hard; instead compare centered rows
                kc_dec = float(comp_pcc(Kh[L:] - Kh[L:].mean(0), Kt[L:] - Kt[L:].mean(0), 0.99)[1])
                lines.append(
                    f"d{d}h{hl_}->q{orig}/kv{kvg} lay={lay[:3]} ctx: TT={pc['TT']:.3f} Qhf={pc['Qhf']:.3f} Khf={pc['Khf']:.3f} Vhf={pc['Vhf']:.3f} all={pc['allhf']:.3f} | in: Q={qp:.3f} Kpre={kp_pre:.3f} Kdec={kp_dec:.3f} Kdec_c={kc_dec:.3f} Vpre={vp_pre:.3f} Vdec={vp_dec:.3f}"
                )
        logger.info(f"[TF-SWAP] layer {li} pos={pos} L={L}\n" + "\n".join(lines))

    # ---- Kernel-vs-recompute on ALL devices/heads, with alternative "what did the kernel attend over" hypotheses
    for li, c in enumerate(sdpa_cap):
        cur = int(c["cp"].reshape(-1)[0])
        pt = c["pt"].reshape(c["pt"].shape[-2], -1)[0].long()
        nblk_all = int(pt.numel())
        lines = []
        for d in range(len(c["q"])):
            q0, k0, v0, o0 = c["q"][d], c["k"][d], c["v"][d], c["out"][d]
            Kall = k0[pt[:nblk_all]].permute(1, 0, 2, 3).reshape(k0.shape[1], -1, k0.shape[-1])  # [nkv_local, max_S, D]
            Vall = v0[pt[:nblk_all]].permute(1, 0, 2, 3).reshape(v0.shape[1], -1, v0.shape[-1])
            Q = q0.reshape(-1, q0.shape[-2], q0.shape[-1])[0][:nh_local]
            O = o0.reshape(-1, o0.shape[-2], o0.shape[-1])[0][:nh_local]
            nkv_local = Kall.shape[0]
            rep = nh_local // nkv_local

            def attn(q, k, v):
                w = torch.softmax(((q @ k.T) * float(c["scale"])).double(), dim=-1).float()
                return w @ v

            for hl_ in range(nh_local):
                kh, vh = Kall[hl_ // rep], Vall[hl_ // rep]
                res = {}
                for tag, S_ in (
                    ("S", cur + 1),
                    ("128", 128),
                    ("160", 160),
                    ("192", 192),
                    ("256", 256),
                    ("all", kh.shape[0]),
                ):
                    res[tag] = float(comp_pcc(O[hl_], attn(Q[hl_], kh[:S_], vh[:S_]), 0.99)[1])
                # chunk-dropped hypotheses: only first 128, only last chunk, skip positions 128..cur
                res["skip128+"] = res["128"]
                res["only128+"] = float(comp_pcc(O[hl_], attn(Q[hl_], kh[128 : cur + 1], vh[128 : cur + 1]), 0.99)[1])
                best = max(res.items(), key=lambda kv: kv[1])
                lines.append(
                    f"d{d}h{hl_}: vsS={res['S']:.3f} vs128={res['128']:.3f} vs160={res['160']:.3f} vs192={res['192']:.3f} vsAll={res['all']:.3f} only128+={res['only128+']:.3f} best={best[0]}({best[1]:.3f})"
                )
        logger.info(f"[TF-KERN] layer {li} cur={cur}\n" + "\n".join(lines))

    # ---- KV-cache content check: prefill-written (pos < L) vs decode-written (pos >= L) entries vs HF past_key_values
    import ttnn
    from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta

    with torch.no_grad():
        hf_out = ref(input_ids=full[:, : L + GEN - 1], use_cache=True)
        pkv = hf_out.past_key_values
    n_kv = model_args.unpadded_n_kv_heads if hasattr(model_args, "unpadded_n_kv_heads") else model_args.n_kv_heads
    kv_pad = model_args.n_kv_heads
    dup = kv_pad // n_kv  # each real KV head duplicated over dup devices on padded layouts
    pt = page_table[0]  # user 0 of lane 0: virtual block -> physical block
    S = L + GEN - 1
    for layer_idx in (0, 13, 27):
        layer = model_list[0].layers[layer_idx].attention
        k_tt, v_tt = layer.layer_past

        def gather(cache):
            dev = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(cache)]  # each [max_blocks, local_kv, 32, D]
            full_cache = torch.cat(dev, dim=1)  # [max_blocks, kv_pad, 32, D]
            full_cache = (
                full_cache[:, ::dup] if dup > 1 else full_cache
            )  # drop duplicated heads -> [max_blocks, n_kv, 32, D]
            blocks = full_cache[pt[: (S + 31) // 32].long()]  # [nblocks, n_kv, 32, D]
            return blocks.permute(1, 0, 2, 3).reshape(n_kv, -1, full_cache.shape[-1])[:, :S].float()  # [n_kv, S, D]

        K_tt, V_tt = gather(k_tt), gather(v_tt)
        K_hf = (
            pkv.layers[layer_idx].keys[0].float() if hasattr(pkv, "layers") else pkv[layer_idx][0][0].float()
        )  # [n_kv, S, D], post-rope, HF order
        V_hf = pkv.layers[layer_idx].values[0].float() if hasattr(pkv, "layers") else pkv[layer_idx][1][0].float()
        # HF K is in HF rope order; TT K is Meta-interleaved. Convert HF K to Meta order (same op as cos/sin conversion).
        K_hf_meta, _ = convert_rope_style_hf_to_meta(K_hf, K_hf)
        D = K_hf.shape[-1]
        K_cands = {
            "hf": K_hf,
            "meta(cos-conv)": K_hf_meta,
            "interleave_halves": K_hf.view(*K_hf.shape[:-1], 2, D // 2).transpose(-1, -2).reshape(K_hf.shape),
            "deinterleave": K_hf.view(*K_hf.shape[:-1], D // 2, 2).transpose(-1, -2).reshape(K_hf.shape),
        }
        best = max(K_cands.items(), key=lambda kv: float(comp_pcc(kv[1][:, :L], K_tt[:, :L], 0.99)[1]))
        logger.info(f"[TF-KV] layer {layer_idx} K layout best={best[0]}")
        K_hf_meta = best[1]
        if getattr(layer, "k_bias_shift_prefill", None) is not None:
            # cache holds post-rope keys minus the K bias (softmax-invariant shift); undo it for the comparison
            shift_dev = [
                ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(layer.k_bias_shift_prefill)
            ]  # each [1, local_kv, 1, D]
            shift_all = torch.cat(shift_dev, dim=1)[0, :, 0, :]  # [kv_pad, D]
            shift_all = shift_all[::dup] if dup > 1 else shift_all
            K_tt = K_tt + shift_all[:, None, :]
        for name, A, B in (("K", K_hf_meta, K_tt), ("V", V_hf, V_tt)):
            seg = []
            for lo, hi, tag in (
                (0, L, "prefill"),
                (L, L + 30, "dec0-30"),
                (L + 30, L + 60, "dec30-60"),
                (L + 60, S, "dec60+"),
            ):
                _, pcc = comp_pcc(A[:, lo:hi], B[:, lo:hi], 0.99)
                seg.append(f"{tag}:{float(pcc):.4f}")
            logger.info(f"[TF-KV] layer {layer_idx} {name}: " + "  ".join(seg))
