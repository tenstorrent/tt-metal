# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Simple word-based tokenizer for PI0 task instructions."""

import torch
from typing import Tuple


class SimpleTokenizer:
    """Maps common robotics words to token IDs for PI0."""

    VOCAB = {
        "<pad>": 0, "<bos>": 2, "<eos>": 3,
        "pick": 100, "place": 101, "grasp": 102, "release": 103,
        "move": 104, "reach": 105, "push": 106, "pull": 107,
        "lift": 108, "drop": 109,
        "cube": 200, "block": 201, "object": 202, "ball": 203, "box": 204,
        "up": 300, "down": 301, "left": 302, "right": 303,
        "forward": 304, "backward": 305,
        "the": 400, "a": 401, "to": 403, "and": 405, "red": 500, "blue": 501,
    }
    VOCAB_SIZE = 256000

    def encode(self, text: str, max_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = [self.VOCAB["<bos>"]]
        for w in text.lower().split():
            tokens.append(self.VOCAB.get(w, hash(w) % (self.VOCAB_SIZE - 1000) + 1000))
        tokens.append(self.VOCAB["<eos>"])
        mask = [True] * len(tokens)
        pad_len = max_length - len(tokens)
        if pad_len > 0:
            tokens += [0] * pad_len
            mask += [False] * pad_len
        else:
            tokens = tokens[:max_length]
            mask = mask[:max_length]
        return (torch.tensor([tokens], dtype=torch.long),
                torch.tensor([mask], dtype=torch.bool))
