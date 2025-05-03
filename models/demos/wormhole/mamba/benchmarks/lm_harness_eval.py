# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple
from tqdm import tqdm

import torch

from transformers import AutoTokenizer

from models.demos.wormhole.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.wormhole.mamba.benchmarks.loglikelihood import compute_loglikelihood_given_prompt_and_target

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("mamba-cpu-reference")
class MambaEvalWrapper(LM):
    def __init__(
        self,
        pretrained: MambaPretrainedModelName = "state-spaces/mamba-370m",
        max_length=2048,
        batch_size=1,
        device="cpu",
    ):
        LM.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

        self.model = MambaDecode.from_pretrained(pretrained, batch_size=int(batch_size))
        self.model.eval()

        self.device = torch.device(device)

    def loglikelihood(self, requests: List[Instance]):
        results = []
        with torch.no_grad():
            for instance in tqdm(requests):
                context, target = instance.arguments

                context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(
                    device=self.device
                )  # (1 x CONTEXT_LEN)
                if context == "":
                    context_ids = torch.Tensor([self.tokenizer.eos_token_id])
                assert len(context_ids.shape) == 2 and context_ids.shape[1] > 0, "Expected at least one context token"

                target_ids = self.tokenizer(target, return_tensors="pt").input_ids.to(
                    device=self.device
                )  # (1 x TARGET_LEN)
                assert len(target_ids.shape) == 2 and target_ids.shape[1] > 0, "Expected at least one target token"

                loglikelihood, is_greedy = compute_loglikelihood_given_prompt_and_target(
                    context_ids,
                    target_ids,
                    self.model,
                    self.vocab_size,
                )
                results.append((loglikelihood, is_greedy))
        return results

    def generate_until(self, requests):
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
