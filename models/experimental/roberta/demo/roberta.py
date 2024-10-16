# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer, RobertaModel
from transformers import (
    RobertaForMaskedLM,
    RobertaForQuestionAnswering,
    RobertaForMultipleChoice,
)
import torch

"""
These functions are used for demonstrating the use of Huggingface models,
as well as testing various inputs, outputs and configs of the modules.
This file will be removed when whole RoBERTa model is ported.
"""


def roberta_for_masked_lm():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")

    state_dict = model.state_dict()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    outputs = model(**inputs)


def roberta_for_qa():
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    inputs = tokenizer(question, text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


def roberta_for_multiple_choice():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaForMultipleChoice.from_pretrained("roberta-base")

    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."
    # labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()})  # batch size is 1
    predicted_class = outputs.logits.argmax().item()


if __name__ == "__main__":
    roberta_for_qa()
    roberta_for_masked_lm()
    roberta_for_multiple_choice()
