# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional vision attention through cached-prefix and chunked prefill, on the model (device).

One prompt (text, an image placeholder run, text) is prefilled three ways through the generator's
multimodal path (``pixel_values`` = the merged host embeddings, as the vLLM bridge stages them):

* ``whole``: one call, the whole prompt (the single-chunk masked SDPA, the reference);
* ``cached``: the text prefix is prefilled first, then a continuation from a cached prefix that starts
  before the image (the sliding layers attend the ``[tail | chunk]`` square with the explicit mask, the
  full layers recompute the image rows against the paged cache);
* ``chunked``: the generator chunks the prompt at a boundary that would cut the image, so the planner
  moves the boundary below it (first chunk masked, second chunk a continuation).

The last-token logits of ``cached`` and ``chunked`` must agree with ``whole`` within bf16 noise, and
with the overlay off (``GEMMA4_MM_BIDIR_MASK=0``) the continuation must move further away than that.

    HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/tests/unit/test_mm_prefill_vision_device.py -k 1x4 -s
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint

BLOCK = 64
TEXT_BEFORE, IMAGE_ROWS, TEXT_AFTER = 2000, 260, 16
PROMPT = TEXT_BEFORE + IMAGE_ROWS + TEXT_AFTER  # 2276: the image [2000, 2260) crosses a 2048 chunk boundary
PADDED = 4096
CACHED = 1920  # a 128-aligned cached prefix below the image (what the bridge's realignment produces)


def _host_embeds(generator, ids):
    model = generator.model[0]
    weight = getattr(model, "_embed_weight_cpu", None)
    if weight is not None:
        return (torch.nn.functional.embedding(ids, weight).float() * model.embed_scale).to(torch.bfloat16)
    mesh_device = generator.mesh_device
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    ids_tt = ttnn.from_torch(
        ids.view(1, -1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    emb = model.embed_tokens(ids_tt)
    first = ttnn.get_device_tensors(emb)[0] if is_mesh else emb
    return ttnn.to_torch(first).reshape(-1, model.hidden_size)[: ids.numel()].to(torch.bfloat16)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)], device_params_extra={"l1_small_size": 65536})
def test_vision_attention_survives_cached_prefix_and_chunking(mesh_device, monkeypatch):
    skip_if_config_only_checkpoint()
    for k, v in {"GEMMA4_PAGE_BLOCK_SIZE": str(BLOCK), "GEMMA4_CP_PREFILL": "0"}.items():
        monkeypatch.setenv(k, os.environ.get(k, v))
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-12B-it")
    blocks_per_user = PADDED // BLOCK
    slots = 4
    paged_cfg = PagedAttentionConfig(block_size=BLOCK, max_num_blocks=1 + slots * blocks_per_user)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device, model_path, max_batch_size=1, max_seq_len=PADDED, paged_attention_config=paged_cfg
    )
    model = generator.model[0]
    image_id = int(generator.image_token_id)
    vocab = model.vocab_size
    gen = torch.Generator().manual_seed(7)
    # natural, non-repeating text (the repository README): a random-token prompt amplifies ulp-level
    # differences into large next-token moves (the staged host embeddings alone differ from the device
    # embedding by rounding), a repeated paragraph makes the next token a copy that no image can move;
    # the 16 text tokens after the image keep its rows within the last token's reach
    import pathlib

    readme = pathlib.Path(__file__).resolve().parents[5] / "README.md"
    prose = tokenizer(readme.read_text(), add_special_tokens=False, return_tensors="pt").input_ids[0]
    assert prose.numel() >= PROMPT, prose.numel()
    ids = prose[:PROMPT].clone()
    ids[TEXT_BEFORE : TEXT_BEFORE + IMAGE_ROWS] = image_id
    merged = _host_embeds(generator, ids)
    # the "image": random soft tokens at the embedding scale, deterministic
    scale = float(merged[:TEXT_BEFORE].float().std())
    merged[TEXT_BEFORE : TEXT_BEFORE + IMAGE_ROWS] = (
        torch.randn(IMAGE_ROWS, merged.shape[1], generator=gen) * scale
    ).to(torch.bfloat16)
    tok = torch.zeros(1, PADDED, dtype=torch.long)
    tok[0, :PROMPT] = ids

    def table(slot):
        return torch.arange(1 + slot * blocks_per_user, 1 + (slot + 1) * blocks_per_user, dtype=torch.int32).unsqueeze(
            0
        )

    def prefill(slot, *, start_pos=None, prompt_len=PROMPT, with_image=True, label="", tokens=None, embeds=None):
        out = generator.prefill_forward_text(
            tok if tokens is None else tokens,
            page_table=table(slot),
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([prompt_len]),
            empty_slots=[0],
            enable_trace=False,
            sampling_params=None,
            warmup_prefill=False,
            start_pos=start_pos,
            pixel_values=[merged if embeds is None else embeds] if with_image else None,
        )
        lg = out[0] if isinstance(out, (list, tuple)) else out
        lg = lg.reshape(-1)[:vocab].float()
        assert torch.isfinite(lg).all(), f"{label}: non-finite logits"
        logger.info(f"{label}: argmax {int(lg.argmax())}")
        return lg

    def distance(a, b):
        pa, pb = torch.softmax(a, -1), torch.softmax(b, -1)
        return float((pa - pb).abs().sum() / 2)  # total variation of the next-token distribution

    whole = prefill(0, label="whole prompt")

    # cached prefix: the text before the image, then the continuation from 1920 (the image whole in it)
    prefill(1, prompt_len=CACHED, with_image=False, label="text prefix")
    cached = prefill(1, start_pos=[CACHED], label="continuation after the cached prefix")

    # the noise floor of a cached-prefix continuation itself: the same prompt as plain text (placeholder ids
    # embedded as tokens, no spans, causal everywhere), whole vs continuation
    whole_text = prefill(3, with_image=False, label="whole prompt, text only")
    prefill(3, prompt_len=CACHED, with_image=False, label="text prefix (text only)")
    cached_text = prefill(3, start_pos=[CACHED], with_image=False, label="continuation, text only")

    # the staged-embeddings path on its own: a text prompt without placeholders driven through pixel_values (host
    # text embeddings, no spans): override vs token path, and whole vs continuation through the override
    ids_plain = prose[:PROMPT].clone()
    tok_plain = torch.zeros(1, PADDED, dtype=torch.long)
    tok_plain[0, :PROMPT] = ids_plain
    emb_plain = _host_embeds(generator, ids_plain)
    whole_plain_tok = prefill(3, with_image=False, tokens=tok_plain, label="plain prompt, token path")
    whole_plain_ovr = prefill(3, tokens=tok_plain, embeds=emb_plain, label="plain prompt, embeddings override")
    prefill(3, prompt_len=CACHED, with_image=False, tokens=tok_plain, label="plain prefix")
    cached_plain_ovr = prefill(
        3, start_pos=[CACHED], tokens=tok_plain, embeds=emb_plain, label="plain continuation, override"
    )

    # generator chunking at 2048: the image crosses the boundary, the planner moves it to 1920
    model_args = generator.model_args[0]
    prev_chunk = model_args.max_prefill_chunk_size
    model_args.max_prefill_chunk_size = 2048
    try:
        chunked = prefill(2, label="two generator chunks")
        chunked_text = prefill(2, with_image=False, label="two generator chunks, text only")
    finally:
        model_args.max_prefill_chunk_size = prev_chunk

    # the same continuation with the overlay off: causal attention over the image rows
    monkeypatch.setenv("GEMMA4_MM_BIDIR_MASK", "0")
    prefill(3, prompt_len=CACHED, with_image=False, label="text prefix (causal control)")
    causal = prefill(3, start_pos=[CACHED], label="continuation, overlay off")
    monkeypatch.delenv("GEMMA4_MM_BIDIR_MASK")

    d_cached, d_chunked, d_causal = distance(cached, whole), distance(chunked, whole), distance(causal, whole)
    floor = distance(cached_text, whole_text)
    floor_chunked = distance(chunked_text, whole_text)
    floor_ovr = distance(whole_plain_ovr, whole_plain_tok)
    floor_ovr_cached = distance(cached_plain_ovr, whole_plain_ovr)
    logger.info(
        f"RESULT mm_vision_device TV(cached, whole)={d_cached:.4f} TV(chunked, whole)={d_chunked:.4f} "
        f"TV(causal, whole)={d_causal:.4f} floor TV(cached_text, whole_text)={floor:.4f} "
        f"TV(chunked_text, whole_text)={floor_chunked:.4f} TV(override, token)={floor_ovr:.4f} "
        f"TV(cached_override, whole_override)={floor_ovr_cached:.4f} "
        f"argmax whole={int(whole.argmax())} cached={int(cached.argmax())} chunked={int(chunked.argmax())} "
        f"causal={int(causal.argmax())} text whole={int(whole_text.argmax())} cached={int(cached_text.argmax())}"
    )
    # the paths under test must stay within the perturbation envelope the staged embeddings themselves set
    # (host vs device rounding), and the overlay must matter far more than that envelope
    floors = (floor, floor_chunked, floor_ovr, floor_ovr_cached)
    bound = max(0.05, 3 * max(floors))
    assert d_cached <= bound and d_chunked <= bound, (d_cached, d_chunked, floors)
    assert d_causal > 2 * max(d_cached, d_chunked, *floors, 1e-3), "the control must move the distribution more"
    ttnn.synchronize_device(mesh_device)
