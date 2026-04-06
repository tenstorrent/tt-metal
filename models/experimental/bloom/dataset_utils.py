# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json


class InputExample(object):
    def __init__(self, input_sentence, endings=None, label=None):
        self.input_sentence = input_sentence
        self.endings = endings
        self.label = label


def get_input(item):
    if "ctx_a" not in item:
        return item["ctx"]
    if "ctx" not in item:
        return item["ctx_a"]
    if len(item["ctx"]) == len(item["ctx_a"]):
        return item["ctx"]
    return item["ctx_a"]


def get_endings(item):
    if ("ctx_b" not in item) or len(item["ctx_b"]) == 0:
        return item["endings"]
    return ["{} {}".format(item["ctx_b"], x) for x in item["endings"]]


def get_data(input_loc):
    examples = []
    with open(input_loc, "r") as file:
        for data in file:
            item = json.loads(data)
            examples.append(
                InputExample(
                    input_sentence=get_input(item),
                    endings=get_endings(item),
                    label=item["label"],
                )
            )

    return examples
