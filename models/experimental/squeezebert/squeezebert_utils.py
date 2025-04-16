# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Utility methods for squeezebert


def get_answer(inputs, output, tokenizer):
    answer_start_index = output.start_logits.argmax()
    answer_end_index = output.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    return answer
