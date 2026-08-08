"""What an autoregressive step streams: the decoder stack and the output projection. Nothing else.

Two model-agnostic mistakes in the read set, which happened to partly cancel on gemma-3-12b-it and
so went unnoticed -- it reported 11.18 B where the streamed set is 11.77 B:

  TIED EMBEDDINGS. _LOOKUP_ONLY skips embed_tokens because a token reads ONE row of the table, not
  the table. Correct -- while a separate lm_head tensor exists to carry the output projection. When
  tie_word_embeddings is set there is no lm_head in the checkpoint: the embedding table IS the output
  projection, read in full every token. Skipping it deletes a real streamed tensor (1.007 B params on
  gemma3). The comment above _LOOKUP_ONLY already says "the output projection is read in full and is
  deliberately absent from this list" -- tying is the case where that intent silently inverts.

  TOWERS. A vision or audio encoder runs once per image or clip, never per generated token. gemma3
  ships 437 vision tensors, 0.411 B params, all counted against every token.

Neither is about gemma3: tying is standard in Gemma, Qwen, Phi and most small Llamas, and every
multimodal wrapper carries a tower. The fixes key on tensor names and the tie flag, never on a model.
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import model_bytes as mb  # noqa: E402


def _shard(path: Path, tensors: dict):
    """A minimal real safetensors file: 8-byte header length, then the JSON header."""
    hdr = {n: {"dtype": dt, "shape": list(sh), "data_offsets": [0, 0]} for n, (dt, sh) in tensors.items()}
    blob = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def _mk(tmp_path, tensors) -> dict:
    _shard(tmp_path / "model.safetensors", tensors)
    return mb.weight_bytes(tmp_path, unit="token") or {}


EMB = ("BF16", (262208, 3840))  # 1.007 B params
LAYER = ("BF16", (3840, 15360))  # 0.059 B params, stands in for a decoder weight
VIS = ("BF16", (1152, 4304))


# ---------------------------------------------------------------- tied embeddings


def test_a_tied_checkpoint_counts_the_embedding_table_once(tmp_path):
    """No lm_head tensor -> the table is the output projection and must be counted."""
    r = _mk(tmp_path, {"model.embed_tokens.weight": EMB, "model.layers.0.mlp.up_proj.weight": LAYER})
    assert r["params"] == 262208 * 3840 + 3840 * 15360, r["params"]
    assert r["skipped_lookup_bytes"] == 0


def test_an_untied_checkpoint_still_skips_the_table(tmp_path):
    """lm_head present -> embed_tokens really is lookup-only. The original behaviour, preserved."""
    r = _mk(
        tmp_path,
        {
            "model.embed_tokens.weight": EMB,
            "lm_head.weight": EMB,
            "model.layers.0.mlp.up_proj.weight": LAYER,
        },
    )
    assert r["params"] == 262208 * 3840 + 3840 * 15360, r["params"]
    assert r["skipped_lookup_bytes"] > 0


def test_the_table_is_never_counted_twice(tmp_path):
    """Tied or not, the output projection is read once per token. Counting both tensors would
    overstate the read set and drag the ceiling down."""
    tied = _mk(tmp_path / "a", {"model.embed_tokens.weight": EMB, "model.layers.0.w": LAYER}) if False else None
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir(), d2.mkdir()
    tied = _mk(d1, {"model.embed_tokens.weight": EMB, "model.layers.0.w": LAYER})
    untied = _mk(d2, {"model.embed_tokens.weight": EMB, "lm_head.weight": EMB, "model.layers.0.w": LAYER})
    assert tied["params"] == untied["params"], (tied["params"], untied["params"])


def test_a_prefixed_tied_name_is_recognised(tmp_path):
    """Multimodal wrappers nest the decoder: language_model.model.embed_tokens.weight."""
    r = _mk(tmp_path, {"language_model.model.embed_tokens.weight": EMB, "language_model.model.layers.0.w": LAYER})
    assert r["params"] == 262208 * 3840 + 3840 * 15360


def test_an_output_projection_under_another_name_still_counts_as_untied(tmp_path):
    """Some checkpoints call it output.weight or embed_out.weight."""
    for head in ("output.weight", "embed_out.weight"):
        d = tmp_path / head.split(".")[0]
        d.mkdir()
        r = _mk(d, {"model.embed_tokens.weight": EMB, head: EMB, "model.layers.0.w": LAYER})
        assert r["skipped_lookup_bytes"] > 0, head


# ---------------------------------------------------------------- towers


def test_a_vision_tower_is_not_streamed_per_token(tmp_path):
    r = _mk(tmp_path, {"vision_tower.encoder.layers.0.w": VIS, "model.layers.0.w": LAYER, "lm_head.weight": EMB})
    assert r["params"] == 3840 * 15360 + 262208 * 3840, r["params"]


def test_every_tower_spelling_is_excluded(tmp_path):
    for i, name in enumerate(
        (
            "vision_tower.x.w",
            "vision_model.x.w",
            "visual.blocks.0.w",
            "audio_tower.x.w",
            "audio_encoder.x.w",
            "speech_encoder.x.w",
            "image_encoder.x.w",
        )
    ):
        d = tmp_path / ("t%d" % i)
        d.mkdir()
        r = _mk(d, {name: VIS, "model.layers.0.w": LAYER, "lm_head.weight": EMB})
        assert r["params"] == 3840 * 15360 + 262208 * 3840, (name, r["params"])


def test_a_decoder_weight_is_never_mistaken_for_a_tower(tmp_path):
    """'vision' must match a component, not a substring anywhere in a name -- a decoder tensor with
    an unlucky name being dropped is the failure mode that would silently shrink the read set."""
    r = _mk(tmp_path, {"model.layers.0.self_attn.q_proj.weight": LAYER, "lm_head.weight": EMB})
    assert r["params"] == 3840 * 15360 + 262208 * 3840


def test_a_tower_only_checkpoint_does_not_return_zero(tmp_path):
    """An encoder-only model has no decoder stack; excluding everything would make the ceiling
    infinite. Nothing to stream means no analytic answer, not a divide-by-zero."""
    r = _mk(tmp_path, {"vision_tower.x.w": VIS})
    assert not r or r.get("params", 0) == 0 or r.get("bytes", 0) == 0


# ---------------------------------------------------------------- the reported case


def test_the_gemma3_read_set(tmp_path):
    """48 layers + the tied table, no tower: 11.77 B, which rounds to the 12 GB the xB->xGB rule
    wants. The old count -- layers + tower, no table -- gave 11.18 B, which rounds to 11."""
    t = {"language_model.model.embed_tokens.weight": EMB}
    for i in range(48):
        t["language_model.model.layers.%d.mlp.gate_proj.weight" % i] = ("BF16", (3840, 15360))
        t["language_model.model.layers.%d.mlp.up_proj.weight" % i] = ("BF16", (3840, 15360))
        t["language_model.model.layers.%d.mlp.down_proj.weight" % i] = ("BF16", (15360, 3840))
        t["language_model.model.layers.%d.self_attn.q_proj.weight" % i] = ("BF16", (3840, 4096))
        t["language_model.model.layers.%d.self_attn.k_proj.weight" % i] = ("BF16", (3840, 2048))
        t["language_model.model.layers.%d.self_attn.v_proj.weight" % i] = ("BF16", (3840, 2048))
        t["language_model.model.layers.%d.self_attn.o_proj.weight" % i] = ("BF16", (4096, 3840))
    for i in range(27):
        t["vision_tower.vision_model.encoder.layers.%d.w" % i] = VIS
    r = _mk(tmp_path, t)
    b = r["params"] / 1e9
    assert 11.5 <= b <= 12.0, b
    assert round(b) == 12, b
