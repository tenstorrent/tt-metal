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
