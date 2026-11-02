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

The generator reads the lm_head on the 32-row tile that holds the last token; the prompt ends so that
this tile holds the last 20 image rows and the 12 text tokens after the image. SDPA promises no
bit-identity across chunkings, since the accumulation order follows the chunking (tt-metal #63857:
the precision recipes add about 0.1 of total variation on this prompt), so the paths under test are
judged with tolerances: the last token's next-token distribution within three times the floors the
same prompt measures on the device (a text-only continuation and chunking, the staged host embeddings
against the device embedding), and the mean over the 12 text rows after the image within 0.35 (the
vision rows of a continuation are recomputed against the bfloat8_b paged cache, which moves the rows
right after the image by 0.13-0.15 on an exact-SDPA build). With the overlay off
(``GEMMA4_MM_BIDIR_MASK=0``) the image rows attend causally, which moves the text rows after the image
by 0.55 against the same continuation: the control is judged there, against the matched continuation
(the same chunking, so only the overlay differs), and must exceed 0.4 and 1.5 times the largest
distance any supported path shows on those rows. The image-row distances are logged, not judged: the
image rows' own distributions are random-token rows and amplify any numeric difference.

    HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/tests/unit/test_mm_prefill_vision_device.py -k 1x4 -s
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint

BLOCK = 64
TEXT_BEFORE, IMAGE_ROWS, TEXT_AFTER = 2000, 260, 12
PROMPT = TEXT_BEFORE + IMAGE_ROWS + TEXT_AFTER  # 2272: the image [2000, 2260) crosses a 2048 chunk boundary
PADDED = 4096
CACHED = 1920  # a 128-aligned cached prefix below the image (what the bridge's realignment produces)
TILE = 32  # the lm_head runs on the 32-row tile holding the last token; the generator hands the model that tile
TILE_START = PROMPT - TILE  # 2240: 20 image rows, then the 12 text rows after the image
IMAGE_TILE_ROWS = list(range(0, TEXT_BEFORE + IMAGE_ROWS - TILE_START))
TEXT_TILE_ROWS = list(range(TEXT_BEFORE + IMAGE_ROWS - TILE_START, TILE))
assert PROMPT % TILE == 0 and CACHED % TILE == 0, "the tile must cover the same absolute rows on every path"
assert len(IMAGE_TILE_ROWS) == 20 and len(TEXT_TILE_ROWS) == TEXT_AFTER


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

    # The eager single-user prefill queues the whole [1, 1, 32, vocab] lm_head tile and hands it to the model's
    # ``process_output_prefill`` with the in-tile row; keep the tile so every row's distribution can be compared.
    tiles = []
    read_tile = model.process_output_prefill

    def process_output_prefill(tt_out, last_token_idx):
        try:
            first = ttnn.get_device_tensors(tt_out)[0]
        except Exception:  # noqa: BLE001 - a single-shard host tensor
            first = tt_out
        rows = ttnn.to_torch(first).float()
        rows = rows.reshape(-1, rows.shape[-1])
        tiles.append(rows[:TILE, :vocab].clone() if rows.shape[0] >= TILE else None)
        return read_tile(tt_out, last_token_idx)

    monkeypatch.setattr(model, "process_output_prefill", process_output_prefill)

    def prefill(slot, *, start_pos=None, prompt_len=PROMPT, with_image=True, label="", tokens=None, embeds=None):
        tiles.clear()
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
        if prompt_len != PROMPT:
            logger.info(f"{label}: argmax {int(lg.argmax())}")
            return None  # a prefix fill: its tile is not the prompt's last tile
        tile = tiles[-1] if tiles else None
        assert tile is not None and tile.shape == (TILE, vocab), f"{label}: no 32-row lm_head tile captured"
        assert torch.isfinite(tile).all(), f"{label}: non-finite tile logits"
        assert torch.equal(tile[-1], lg), f"{label}: the tile's last row is not the returned last-token logits"
        logger.info(f"{label}: argmax {int(lg.argmax())}")
        return tile

    def distance(a, b, rows=None):
        """Mean (and max) total variation of the next-token distributions over ``rows`` of two tiles."""
        rows = list(range(TILE)) if rows is None else rows
        pa, pb = torch.softmax(a[rows], -1), torch.softmax(b[rows], -1)
        tv = (pa - pb).abs().sum(-1) / 2
        return float(tv.mean()), float(tv.max())

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

    text, image = TEXT_TILE_ROWS, IMAGE_TILE_ROWS
    last = lambda a, b: distance(a, b, [TILE - 1])[0]  # noqa: E731 - the last token alone, the earlier gate's number
    d_cached, d_chunked, d_causal = last(cached, whole), last(chunked, whole), last(causal, whole)
    floors_last = (
        last(cached_text, whole_text),
        last(chunked_text, whole_text),
        last(whole_plain_ovr, whole_plain_tok),
        last(cached_plain_ovr, whole_plain_ovr),
    )
    # the 12 text rows after the image: the supported paths against the whole prompt and each other, the floors,
    # and the overlay-off control against the matched continuation (same chunking: only the overlay differs)
    t_cached, t_chunked, t_mutual = (
        distance(cached, whole, text),
        distance(chunked, whole, text),
        distance(chunked, cached, text),
    )
    floors_text = (
        distance(cached_text, whole_text, text)[0],
        distance(chunked_text, whole_text, text)[0],
        distance(whole_plain_ovr, whole_plain_tok, text)[0],
        distance(cached_plain_ovr, whole_plain_ovr, text)[0],
    )
    control = distance(causal, cached, text)
    control_image = distance(causal, cached, image)
    i_cached, i_chunked = distance(cached, whole, image), distance(chunked, whole, image)
    logger.info(
        f"RESULT mm_vision_device last token TV: cached={d_cached:.4f} chunked={d_chunked:.4f} causal={d_causal:.4f} "
        f"floors={tuple(round(f, 4) for f in floors_last)} | text rows mean/max TV: cached={t_cached[0]:.4f}/{t_cached[1]:.4f} "
        f"chunked={t_chunked[0]:.4f}/{t_chunked[1]:.4f} chunked_vs_cached={t_mutual[0]:.4f}/{t_mutual[1]:.4f} "
        f"floors={tuple(round(f, 4) for f in floors_text)} control(causal vs cached)={control[0]:.4f}/{control[1]:.4f} | "
        f"image rows mean TV: cached={i_cached[0]:.4f} chunked={i_chunked[0]:.4f} control={control_image[0]:.4f} | "
        f"argmax whole={int(whole[-1].argmax())} cached={int(cached[-1].argmax())} chunked={int(chunked[-1].argmax())} "
        f"causal={int(causal[-1].argmax())}"
    )

    def per_row(a, b):
        return " ".join(f"{v:.3f}" for v in ((torch.softmax(a, -1) - torch.softmax(b, -1)).abs().sum(-1) / 2).tolist())

    for name, a, b in (
        ("causal vs cached", causal, cached),
        ("causal vs whole", causal, whole),
        ("cached vs whole", cached, whole),
        ("chunked vs whole", chunked, whole),
        ("chunked vs cached", chunked, cached),
        ("override vs token", whole_plain_ovr, whole_plain_tok),
        ("cached_override vs whole_override", cached_plain_ovr, whole_plain_ovr),
    ):
        logger.info(f"RESULT mm_vision_device per-row TV {name}: {per_row(a, b)}")

    # The supported paths: the last token within the perturbation envelope the staged embeddings set (host vs device
    # rounding, three times the floors), and the rows after the image within the documented tolerance.
    bound = max(0.05, 3 * max(floors_last))
    assert d_cached <= bound and d_chunked <= bound, (d_cached, d_chunked, floors_last)
    TEXT_ROWS_BOUND = 0.35
    assert t_cached[0] <= TEXT_ROWS_BOUND and t_chunked[0] <= TEXT_ROWS_BOUND, (t_cached, t_chunked, floors_text)
    # The control: the overlay alone must move the rows after the image well past anything a supported path does.
    envelope = max(t_cached[0], t_chunked[0], t_mutual[0], *floors_text)
    assert control[0] > max(0.4, 1.5 * envelope), (
        "the overlay-off control must move the text rows after the image more than any supported path",
        control,
        envelope,
    )
    ttnn.synchronize_device(mesh_device)
