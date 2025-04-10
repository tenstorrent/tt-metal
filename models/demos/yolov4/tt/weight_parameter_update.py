# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import re
from collections import OrderedDict


def update_weigth_keys(key):
    key = key.replace("downsample", "down")
    key = key.replace("neck", "neek")
    if ".res" in key:

        def res_name_update(match):
            chr = match.group(1)
            num = int(match.group(2))
            if num == 0 or num == 1:
                return f".{chr}.0.conv.{num}."
            if num == 3 or num == 4:
                return f".{chr}.1.conv.{num-3}."

        key = re.sub(r"\.res\.", r".resblock.", key)
        key = re.sub(r"\.(\d+)\.(\d+)\.", res_name_update, key)
        return key
    if "neek" in key:

        def neek_underscore_update_rule(match):
            chr = match.group(1)
            num1 = int(match.group(2))
            num2 = int(match.group(3))
            dict = {
                (7, 2): 8,
                (7, 3): 9,
                (7, 4): 11,
                (8, 2): 12,
                (7, 5): 13,
                (9, 2): 15,
                (9, 3): 16,
                (9, 4): 18,
                (10, 2): 19,
                (9, 5): 20,
            }
            if chr == "b":
                return f".conv{dict[(num1, num2)]}.conv.1."
            return f".conv{dict[(num1, num2)]}.conv.0."

        def neck_rename_update(match):
            chr = match.group(1)
            num = int(match.group(2))
            if num <= 7:
                return f".conv{num}.conv.1." if chr == "b" else f".conv{num}.conv.0."
            dict = {8: 10, 9: 14, 10: 17}
            return f".conv{dict[num]}.conv.1." if chr == "b" else f".conv{dict[num]}.conv.0."

        updated_name = re.sub(r"\.([a-z])(\d+)_(\d+)\.", neek_underscore_update_rule, key)
        if key != updated_name:  # chk if name got updated
            return updated_name
        updated_name = re.sub(r"\.([a-z])(\d+)\.", neck_rename_update, key)
        if key != updated_name:
            return updated_name
    key = re.sub(r"\.c(\d+)\.", r".conv\1.conv.0.", key)
    key = re.sub(r"\.b(\d+)\.", r".conv\1.conv.1.", key)
    return key


def update_weight_parameters(model_weight):
    ttnn_model_random_weight = OrderedDict()
    for key, weight in model_weight.items():
        updated_key = update_weigth_keys(key)
        ttnn_model_random_weight[updated_key] = weight
    return ttnn_model_random_weight
