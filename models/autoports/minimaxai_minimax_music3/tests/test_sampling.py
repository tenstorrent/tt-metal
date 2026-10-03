import torch

from models.autoports.minimaxai_minimax_music3.reference import sampling as S


def test_sample_top_k_matches_diffusers(diffusers_mm3):
    for width in (1024, 200000):
        logits = torch.randn(1, width) * 3
        g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        for _ in range(5):
            assert int(S.sample_top_k(logits, g1)) == int(diffusers_mm3._sample_top_k(logits, g2))


def test_sliced_guidance_equals_full():
    V = 200000
    mask = S.full_vocab_mask(V)
    for _ in range(20):
        logits = torch.randn(2, V) * 4
        gf = S.guided_c0_logits_full(logits, mask).reshape(-1)
        gs = S.guided_c0_logits_sliced(S.slice_logits(logits)).reshape(-1)
        fin_f, fin_s = torch.isfinite(gf), torch.isfinite(gs)
        assert fin_f.sum() == fin_s.sum() == 50
        assert torch.allclose(gf[fin_f], gs[fin_s])
        assert S.sliced_to_token(int(gs.argmax())) == int(gf.argmax())
