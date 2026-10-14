"""t498 pytest plugin: save the first prompt embeddings Gemma4TokenizerEncoderPair.encode returns to $T498_EMB_DIR."""
import os

import torch


def pytest_configure(config):
    from models.tt_dit.encoders.gemma4 import encoder_pair as ep

    out, orig, calls = os.environ["T498_EMB_DIR"], ep.Gemma4TokenizerEncoderPair.encode, [0]

    def encode(self, prompts):
        res = orig(self, prompts)
        if calls[0] < 4:
            torch.save({"prompts": prompts, "embeds": res}, os.path.join(out, f"emb_{calls[0]}.pt"))
        calls[0] += 1
        return res

    ep.Gemma4TokenizerEncoderPair.encode = encode
