# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""TT pipeline for one user: vision -> merge -> prefill -> teacher-forced decode, mirroring demo/demo.py."""
import torch

import ttnn
from models.experimental.ops.quasar.qwen3_vl.tt.common import (
    PagedAttentionConfig,
    get_hf_visual,
    get_pad_embedding,
    merge_vision_tokens_single_user_ttnn,
    multimodal_rope_single_user_from_hf,
    preprocess_inputs_prefill_single_user_ttnn,
)
from models.experimental.ops.quasar.qwen3_vl.tt.generator import Generator
from models.experimental.ops.quasar.qwen3_vl.tt.model import DropInVisionTransformer, Transformer
from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import model_args_classes


def _page_table(num_blocks, batch):
    perm = torch.randperm(num_blocks)
    return torch.argsort(perm).reshape(batch, num_blocks // batch)


def _logits_vec(t, vocab):
    t = t if isinstance(t, torch.Tensor) else ttnn.to_torch(t)
    return t.float().reshape(-1, t.shape[-1])[0, :vocab]


def _is_prefill(args, kwargs):
    return any(str(v).upper().endswith("PREFILL") for v in (*args, *kwargs.values()) if not hasattr(v, "shape"))


def run_tt(cfg, preset, inputs, hf_model, goldens, mesh_device, recorder, monkeypatch):
    text_cls, vision_cls = model_args_classes(force=cfg.quasar_config)
    hf_cfg = hf_model.config
    out_dim = hf_cfg.vision_config.out_hidden_size
    n_patches, n_img, L = goldens.num_patches, goldens.num_patches // 4, goldens.prefill_len
    vocab = hf_cfg.text_config.vocab_size
    kv_blocks = cfg.kv_blocks or preset.kv_blocks

    # --- vision ---
    vargs = vision_cls(mesh_device, max_batch_size=1, max_seq_len=preset.max_seq_len)
    vargs.hf_config.vision_config.depth = hf_cfg.vision_config.depth
    vargs.hf_config.vision_config.deepstack_visual_indexes = list(hf_cfg.vision_config.deepstack_visual_indexes)
    visual = DropInVisionTransformer(get_hf_visual(hf_model), vargs)
    tv = visual.tt_model
    for i, blk in enumerate(tv.blocks):
        rows = lambda t: t.reshape(-1, t.shape[-1])[:n_patches]
        recorder.wrap(monkeypatch, blk, f"vision.block{i}", rows)
        recorder.wrap(monkeypatch, blk.attention, f"vision.block{i}.attn", rows)
        recorder.wrap(monkeypatch, blk.feed_forward, f"vision.block{i}.mlp", rows)
    taps = [i for i in tv.deepstack_visual_indices if i < len(tv.blocks)]
    for j, _ in enumerate(taps):
        recorder.wrap(
            monkeypatch,
            tv.deepstack_merger_list[j],
            f"vision.deepstack{j}",
            lambda t: t.reshape(-1, t.shape[-1])[:n_img, :out_dim],
        )
    recorder.wrap(monkeypatch, tv.patch_merger, "vision.merger", lambda t: t.reshape(-1, t.shape[-1])[:n_img, :out_dim])

    # --- text ---
    args = text_cls(mesh_device, instruct=True, max_batch_size=1, max_seq_len=preset.max_seq_len)
    args.n_layers = cfg.text_layers
    state_dict = args.load_state_dict()
    dtype = ttnn.bfloat16 if cfg.quasar_config else ttnn.bfloat8_b
    paged = PagedAttentionConfig(block_size=preset.block_size, max_num_blocks=kv_blocks)
    model = Transformer(
        args=args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=args.weight_cache_path(dtype),
        paged_attention_config=paged,
    )
    kv_cache = [layer.attention.layer_past for layer in model.layers]
    flat = lambda t: t.reshape(-1, t.shape[-1])
    for i, layer in enumerate(model.layers):
        recorder.wrap(monkeypatch, layer, f"text.layer{i}", flat, when=_is_prefill, append_dim=0)
        recorder.wrap(monkeypatch, layer.attention, f"text.layer{i}.attn", flat, when=_is_prefill, append_dim=0)
        recorder.wrap(monkeypatch, layer.feed_forward, f"text.layer{i}.mlp", flat, when=_is_prefill, append_dim=0)
    row = (L - 1) % 32  # the prefill output norm sees only the 32-row tile holding the last real token
    recorder.wrap(monkeypatch, model.norm, "text.norm", lambda t: t.reshape(-1, t.shape[-1])[row], when=_is_prefill)
    args.use_qk_fused = False
    gen = Generator(model, args, mesh_device, processor=args.processor, tokenizer=args.tokenizer)
    page_table = _page_table(kv_blocks, 1)

    # --- one user, as in demo/demo.py ---
    ids, mask, thw = inputs["input_ids"][0], inputs["attention_mask"][0], inputs["image_grid_thw"][0]
    image_embeds, deepstack = visual.forward_single_user(inputs["pixel_values"], grid_thw=thw)
    text_embeds = hf_model.model.language_model.embed_tokens(ids.unsqueeze(0))
    text_tt = ttnn.from_torch(
        text_embeds.squeeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=args.cluster_shape),
    )
    embeds, deepstack = merge_vision_tokens_single_user_ttnn(ids, text_tt, image_embeds, hf_cfg, deepstack, args)
    pad_id = args.tokenizer.pad_token_id
    pad = get_pad_embedding(hf_model, pad_id, args)
    x, deepstack, decoding_pos, _ = preprocess_inputs_prefill_single_user_ttnn(
        embeds, args, mask, pad_embedding=pad, deepstack_visual_embeds=deepstack
    )
    for j, d in enumerate(deepstack or []):  # debug only: what the text model will add after layer j
        recorder.tensors[f"debug.deepstack_in{j}"] = recorder.to_host(d)
    cos, sin, rope_deltas = multimodal_rope_single_user_from_hf(
        ids, thw.unsqueeze(0), hf_model, args, pad_token_id=pad_id
    )
    pt_user = gen._ttt_generator._get_prefill_user_page_table(page_table, kv_cache, decoding_pos)
    logits = gen.prefill_forward_single_user_text(
        ttnn.unsqueeze(x, 0),
        page_table=pt_user,
        user_id=0,
        last_token_idx=decoding_pos - 1,
        rot_mats=(cos, sin),
        kv_cache=kv_cache,
        deepstack_visual_embeds=deepstack,
    )
    recorder.tensors["text.logits.prefill"] = _logits_vec(logits, vocab)
    for name in [k for k in recorder.tensors if k.startswith("text.layer")]:  # prefill pads; keep real tokens
        recorder.tensors[name] = recorder.tensors[name][:L]

    gen.update_rope_deltas([rope_deltas.squeeze(0).item()])
    pos = torch.tensor([decoding_pos])
    for k, tok in enumerate(goldens.teacher_tokens):
        recorder.progress.stage = f"text.decode{k}"
        out = gen.decode_forward(
            torch.tensor([[tok]]),
            pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,  # host logits only, as demo.py with sampling_params=None
            reset_sampling_state=False,
        )
        recorder.tensors[f"text.logits.decode{k}"] = _logits_vec(out[0] if isinstance(out, tuple) else out, vocab)
        pos = pos + 1
