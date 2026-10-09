"""Weight remap contract: keys, shapes, tied head, wqkv split, codec params. CPU only, needs the snapshot."""
import torch

from models.autoports.fishaudio_s2_pro.config import S2Config
from models.autoports.fishaudio_s2_pro.tt import weights as W


def test_config_matches_snapshot(snapshot):
    cfg = S2Config.from_snapshot(snapshot)
    d = S2Config.default()
    assert cfg.slow == d.slow and cfg.fast == d.fast and cfg.num_codebooks == 10 and cfg.codebook_size == 4096
    assert (cfg.semantic_begin_id, cfg.semantic_end_id, cfg.im_end_id) == (151678, 155773, 151645)


def test_slow_remap(snapshot):
    cfg = S2Config.from_snapshot(snapshot)
    sd = W.load_fish_state_dict(snapshot)
    assert len(sd) == 358
    slow = W.slow_tower_state_dict(sd, cfg)
    assert slow["tok_embeddings.weight"].data_ptr() == slow["output.weight"].data_ptr(), "head must be tied"
    assert tuple(slow["tok_embeddings.weight"].shape) == (155776, 2560)
    for i in (0, 35):
        p = f"layers.{i}."
        assert tuple(slow[p + "attention.wq.weight"].shape) == (4096, 2560)
        assert tuple(slow[p + "attention.wk.weight"].shape) == (1024, 2560)
        assert tuple(slow[p + "attention.wv.weight"].shape) == (1024, 2560)
        assert tuple(slow[p + "attention.wo.weight"].shape) == (2560, 4096)
        assert tuple(slow[p + "attention.q_norm.weight"].shape) == (128,)
        assert tuple(slow[p + "feed_forward.w1.weight"].shape) == (9728, 2560)
        fused = sd[f"text_model.model.layers.{i}.attention.wqkv.weight"]
        assert torch.equal(
            torch.cat(
                [slow[p + "attention.wq.weight"], slow[p + "attention.wk.weight"], slow[p + "attention.wv.weight"]]
            ),
            fused,
        )
    assert "norm.weight" in slow and not any(k.startswith("fast_") or "codebook" in k for k in slow)
    fast = W.fast_tower_state_dict(sd, cfg)
    assert tuple(fast["embeddings.weight"].shape) == (4096, 2560) and tuple(fast["output.weight"].shape) == (4096, 2560)
    assert "layers.3.attention.wq.weight" in fast and "layers.0.attention.q_norm.weight" not in fast
    cb = W.codebook_table(sd, cfg)
    assert tuple(cb.shape) == (40960, 2560)
    total = sum(v.numel() for v in sd.values())
    assert total == 4_561_852_416


def test_codec_state(snapshot):
    sd = W.load_codec_state(snapshot)
    n = sum(v.numel() for v in sd.values())
    assert 391_000_000 < n < 392_000_000, n
    assert not any(k.endswith(("causal_mask", "freqs_cis")) for k in sd)
    assert tuple(sd["quantizer.semantic_quantizer.quantizers.0.codebook.weight"].shape) == (4096, 8)
    assert tuple(sd["quantizer.quantizer.quantizers.0.codebook.weight"].shape) == (1024, 8)


def test_hf_view(snapshot, tmp_path):
    view = W.ensure_hf_view(snapshot, tmp_path / "view")
    import json

    cfg = json.load(open(view / "config.json"))
    assert cfg["model_type"] == "qwen3" and cfg["hidden_size"] == 2560 and (view / "tokenizer.json").exists()
