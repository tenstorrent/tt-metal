"""Host sampler parity with fish-speech sample()/RAS (reference venv) and greedy semantics."""
import torch

from models.autoports.fishaudio_s2_pro.tt.sampling import S2Sampler, logits_to_probs, sample


def test_greedy_is_constrained_argmax():
    s = S2Sampler(155776, 151678, 155773, 151645)
    logits = torch.randn(155776)
    logits[5] = 100.0  # a non-semantic token must never be chosen
    tok = s.sample_slow(logits, 0.8, 0.8, greedy=True)
    assert 151678 <= tok <= 155773 or tok == 151645
    assert tok == int((logits + s.bias).argmax())


def test_sample_matches_upstream(upstream):
    from fish_speech.models.text2semantic import inference as fsi

    g = torch.Generator().manual_seed(3)
    logits = torch.randn(4096, generator=g) * 3
    for temp, top_p in [(0.8, 0.8), (1.0, 0.9), (0.3, 0.5)]:
        torch.manual_seed(7)
        theirs = int(
            fsi.sample(logits.view(1, 1, -1), temperature=torch.tensor(temp), top_p=torch.tensor(top_p), top_k=30)[
                0
            ].item()
        )
        gen = torch.Generator().manual_seed(7)
        ours = sample(logits, temp, top_p, 30, gen)
        assert ours == theirs, (temp, top_p, ours, theirs)
        p_ours = logits_to_probs(logits, temp, top_p, 30)
        p_theirs = fsi.logits_to_probs(
            logits.view(1, 1, -1), temperature=torch.tensor(temp), top_p=torch.tensor(top_p), top_k=30
        ).view(-1)
        assert torch.allclose(p_ours, p_theirs, atol=1e-6)
