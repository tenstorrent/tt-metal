# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Image-conditioned Qwen3-VL-8B text encoder (TTNN) vs the goldens.

    python -m pytest models/experimental/qwen_image_2_1/tests/test_text_encoder_vl.py -v -s

Env: QWEN_TE_LAYERS=<n> (default 36), QWEN_TE_WDTYPE=bf16|bfp8 (default bfp8).
`-k vision` exercises only the tower; the whole model still loads, which costs nothing measurable
(loading the tower alone is also ~25 s, since the decoder's bfp8 conversion is the cheap part).

TWO references, and the distinction matters:

* `goldens/edit/text_encoder_edit.pt` -- the reference pipeline in bfloat16 on a GPU. This is what the
  shipped model actually computes, so the vision tower is measured against it.
* `goldens/edit/text_encoder_edit_fp32.pt` -- the same forward in fp32 on CPU (built by
  `reference/make_goldens_edit_fp32.py`). For a ~1k-token image prompt the bf16 golden's FINAL hidden
  states are noise-dominated: its vision tower is itself only PCC 0.994 against fp32, and the 36 decoder
  layers amplify that ~12x, so the bf16 golden scores 0.930 against fp32 at the last layer. No
  implementation reaches 0.99 against it -- an exact fp32 forward does not. The text-only prompt does not
  show this: at 20 tokens it stays at 0.997.

So the end-to-end assertions are "no worse than exact arithmetic is", and `test_decoder_given_exact_vision`
measures the decoder on its own against fp32, where a real threshold is meaningful.
"""
import os
import time

import pytest
import torch

import ttnn
from models.experimental.qwen_image_2_1.common.config import DROP_IDX, GOLDENS_DIR, IMAGE_PAD_TOKEN_ID, TE
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.common.weights import text_encoder_ckpt
from models.experimental.qwen_image_2_1.tt.text_encoder import TEPrecision
from models.experimental.qwen_image_2_1.tt.text_encoder_vl import Qwen3VLEncoderVL, mm_token_type_ids

N_LAYERS = int(os.environ.get("QWEN_TE_LAYERS", str(TE.num_layers)))
GOLDEN = os.path.join(GOLDENS_DIR, "edit", "text_encoder_edit.pt")
GOLDEN_FP32 = os.path.join(GOLDENS_DIR, "edit", "text_encoder_edit_fp32.pt")
# Our bf16 tower may not trail the reference's own bf16 tower by more than this, and our end-to-end
# result may not trail an exact forward's score against the bf16 golden by more than this.
SLACK = 0.03


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def dram_used(dev):
    """Allocated DRAM in GiB, or nan if this build does not expose it."""
    try:
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return float(v.total_bytes_allocated_per_bank * v.num_banks) / 2**30
    except Exception:
        return float("nan")


@pytest.fixture(scope="module")
def dev():
    d = open_device(trace_region_size=64 * 1024 * 1024)
    yield d
    close_device(d)


@pytest.fixture(scope="module")
def golden():
    assert os.path.exists(GOLDEN), f"missing {GOLDEN}; run reference/make_goldens_edit.py"
    return torch.load(GOLDEN, weights_only=False, map_location="cpu")


@pytest.fixture(scope="module")
def fp32ref():
    if not os.path.exists(GOLDEN_FP32):
        pytest.skip(f"missing {GOLDEN_FP32}; run reference/make_goldens_edit_fp32.py")
    return torch.load(GOLDEN_FP32, weights_only=False, map_location="cpu")


@pytest.fixture(scope="module")
def model(dev):
    prec = TEPrecision()
    if os.environ.get("QWEN_TE_WDTYPE", "bfp8") == "bf16":
        prec.weight_dtype = ttnn.bfloat16
    base = dram_used(dev)
    t0 = time.time()
    m = Qwen3VLEncoderVL(dev, text_encoder_ckpt(), prec=prec, layers=N_LAYERS)
    print(f"\nloaded {N_LAYERS} TE layers ({prec.weight_dtype}) + the vision tower in {time.time()-t0:.1f}s")
    print(f"DRAM after load: {dram_used(dev):.2f} GiB (+{dram_used(dev)-base:.2f} since open)")
    return m


def test_host_side_inputs_match_reference(golden):
    """Modality ids derived from input_ids agree with the processor's, and the placeholders line up."""
    ids = golden["input_ids"]
    assert torch.equal(mm_token_type_ids(ids), golden["mm_token_type_ids"])
    n = int((ids == IMAGE_PAD_TOKEN_ID).sum())
    assert n == golden["visual_out"]["pooler_output"][0].shape[0] == int(golden["image_pad_mask"].sum())
    print(f"\n{tuple(ids.shape)} tokens, {n} image placeholders, grid {golden['image_grid_thw'].tolist()}")


@pytest.mark.parametrize("grids", [[[1, 8, 8]], [[1, 8, 8], [1, 4, 6]], [[1, 64, 64], [1, 48, 84], [1, 4, 4]]])
def test_multi_image_position_grid(grids):
    """Host side for 1-3 condition images, against transformers' own `get_rope_index`. No device.

    The goldens only cover one image, and a second image is where the mRoPE clock offset and the
    ordering of the merged tokens become observable, so this is checked against the reference directly.
    """
    import transformers

    from models.experimental.qwen_image_2_1.tt.text_encoder_vl import VISION, mrope_position_ids

    merge = VISION.spatial_merge_size
    ids, grid = [151644, 8948, 198], []  # a few text tokens, then alternating image runs and text
    for t, h, w in grids:
        ids += [151652] + [IMAGE_PAD_TOKEN_ID] * (t * h * w // merge**2) + [151653, 198, 198]
        grid.append([t, h, w])
    ids += [151645, 198]
    ids = torch.tensor([ids])
    grid = torch.tensor(grid, dtype=torch.long)
    tt = mm_token_type_ids(ids)
    assert int((tt == 1).sum()) == sum(t * h * w for t, h, w in grids) // merge**2

    cfg = transformers.Qwen3VLConfig(
        text_config={
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "num_heads": 2,
            "out_hidden_size": 64,
            "spatial_merge_size": merge,
        },
    )
    ref = transformers.AutoModel.from_config(cfg).eval()
    expected, _ = ref.get_rope_index(ids, mm_token_type_ids=tt, image_grid_thw=grid.clone(), video_grid_thw=None)
    actual = mrope_position_ids(tt, image_grid_thw=grid, spatial_merge_size=merge)
    assert torch.equal(actual, expected), f"mRoPE grid differs at {(actual != expected).nonzero()[:5].tolist()}"
    print(
        f"\n{len(grids)} image(s), L={ids.shape[1]}, {int((tt == 1).sum())} vision tokens, "
        f"max position {int(actual.max())}: mRoPE grid exact"
    )


def test_vision_tower_matches_reference(model, golden, fp32ref):
    """The tower alone: merged tokens and the three deepstack features."""
    t0 = time.time()
    merged, feats = model.encode_image(golden["pixel_values"], golden["image_grid_thw"])
    ttnn.synchronize_device(model.dev)
    print(f"\nvision tower forward: {time.time()-t0:.2f}s for {golden['pixel_values'].shape[0]} patches")

    g_merged = golden["visual_out"]["pooler_output"][0]
    ref_merged = fp32ref["pooler_output"]
    p_gold, p_fp32 = pcc(merged, g_merged), pcc(merged, ref_merged)
    ref_own = pcc(g_merged, ref_merged)
    print(
        f"merged tokens {tuple(merged.shape)}: vs bf16 golden {p_gold:.5f}, vs fp32 {p_fp32:.5f} "
        f"(the golden's own score vs fp32 is {ref_own:.5f})"
    )
    assert p_gold > 0.99
    assert p_fp32 > ref_own - SLACK, "our bf16 tower is materially worse than the reference's own"

    for i, (f, gf) in enumerate(zip(feats, golden["visual_out"]["deepstack_features"])):
        pf = pcc(f, gf)
        print(f"deepstack {i}: vs bf16 golden {pf:.5f}, vs fp32 {pcc(f, fp32ref['deepstack_features'][i]):.5f}")
        assert pf > 0.99
    # a tower tapping one block three times would still score well on each feature
    for i in range(len(feats) - 1):
        assert not torch.allclose(feats[i].float(), feats[i + 1].float(), atol=1e-2)


def test_decoder_given_exact_vision(model, golden, fp32ref):
    """The decoder on its own: fp32 vision features in, measured against the fp32 reference.

    This is the assertion with real teeth. It removes the tower's bf16 error, which otherwise dominates
    and is not ours -- the reference tower has the same error.
    """
    if N_LAYERS != TE.num_layers:
        pytest.skip("needs all 36 layers")
    ref_hs = fp32ref["hidden_states"]
    vision = (fp32ref["pooler_output"], list(fp32ref["deepstack_features"]))
    taps = list(range(N_LAYERS))
    out, extra = model.encode(
        golden["input_ids"], None, golden["image_grid_thw"], taps=taps, vision=vision, return_vision=True
    )
    for li in (0, 11, 23, 29, 35):
        print(f"layer {li:2d} vs fp32 = {pcc(extra['taps'][li], ref_hs[li + 1][0]):.5f}")
    p_final = pcc(out, ref_hs[-1][0])
    p_pe = pcc(out[DROP_IDX:], ref_hs[-1][0][DROP_IDX:])
    print(f"final vs fp32 = {p_final:.5f}; prompt_embeds slice vs fp32 = {p_pe:.5f}")
    assert p_final > 0.98
    assert p_pe > 0.98


def test_encoder_vl_matches_reference(model, golden, fp32ref):
    """Full encode, tower included, against both references."""
    ids, hs, ref_hs = golden["input_ids"], golden["hidden_states"], fp32ref["hidden_states"]
    taps = list(range(N_LAYERS))
    t0 = time.time()
    out, extra = model.encode(ids, golden["pixel_values"], golden["image_grid_thw"], taps=taps, return_vision=True)
    ttnn.synchronize_device(model.dev)
    print(f"\nencode (eager, with taps): {time.time()-t0:.2f}s")

    # text tokens start at position 0 on all three mRoPE axes; the image run breaks that
    assert torch.equal(extra["position_ids"][:, 0, :3], torch.arange(3).expand(3, -1))
    assert not torch.equal(extra["position_ids"][1, 0], extra["position_ids"][2, 0])
    print(f"merged tokens vs bf16 golden = {pcc(extra['merged'], golden['visual_out']['pooler_output'][0]):.5f}")

    tp = extra["taps"]
    print(f"{'layer':>6} {'vs bf16 golden':>15} {'vs fp32':>9} {'golden vs fp32':>15}")
    worst = (1.0, -1)
    for li in taps:
        p = pcc(tp[li], hs[li + 1][0])
        worst = min(worst, (p, li))
        if li in (0, 5, 11, 17, 23, 29, 32, 35):
            print(
                f"{li:>6} {p:>15.5f} {pcc(tp[li], ref_hs[li + 1][0]):>9.5f} "
                f"{pcc(hs[li + 1][0], ref_hs[li + 1][0]):>15.5f}"
            )
    print(f"worst layer vs bf16 golden = {worst[0]:.5f} (layer {worst[1]})")
    assert pcc(tp[0], hs[1][0]) > 0.99, "layer 0 already diverges: check the vision injection or mRoPE"

    if N_LAYERS == TE.num_layers:
        p_gold = pcc(out, hs[-1][0])
        gold_vs_fp32 = pcc(hs[-1][0], ref_hs[-1][0])
        p_fp32 = pcc(out, ref_hs[-1][0])
        p_pe = pcc(out[DROP_IDX:], golden["prompt_embeds"][0])
        pe_gold_vs_fp32 = pcc(golden["prompt_embeds"][0], ref_hs[-1][0][DROP_IDX:])
        print(
            f"final:         ours vs bf16 golden {p_gold:.5f}, ours vs fp32 {p_fp32:.5f}, "
            f"bf16 golden vs fp32 {gold_vs_fp32:.5f}"
        )
        print(f"prompt_embeds: ours vs bf16 golden {p_pe:.5f}, bf16 golden vs fp32 {pe_gold_vs_fp32:.5f}")
        assert out[DROP_IDX:].shape == golden["prompt_embeds"][0].shape
        # An exact forward scores `gold_vs_fp32` against the bf16 golden; we may not trail that by much.
        assert (
            p_gold > gold_vs_fp32 - SLACK
        ), f"ours-vs-golden {p_gold:.5f} trails exact-vs-golden {gold_vs_fp32:.5f} by more than {SLACK}"
        assert p_fp32 > gold_vs_fp32 - SLACK, "further from fp32 than the bf16 golden is"

    t0 = time.time()
    model.encode(ids, golden["pixel_values"], golden["image_grid_thw"])
    ttnn.synchronize_device(model.dev)
    print(f"encode (eager, warm, no taps): {time.time()-t0:.3f}s")
