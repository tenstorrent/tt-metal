# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.functional_yolox_m.reference.csp_darknet import CSPDarknet
import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_d2 as D2
import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_d3 as D3
import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_d4 as D4
import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_d5 as D5


def custom_preprocessor(device, model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, CSPDarknet):
        parameters["stem"] = D2.custom_preprocessor_focus(device, model.stem, name, ttnn_module_args["stem"])
        parameters["dark2"] = D2.custom_preprocessor(device, model.dark2, name, ttnn_module_args["dark2"])
        parameters["dark3"] = D3.custom_preprocessor(device, model.dark3, name, ttnn_module_args["dark3"])
        parameters["dark4"] = D4.custom_preprocessor(device, model.dark4, name, ttnn_module_args["dark4"])
        parameters["dark5"] = D5.custom_preprocessor(device, model.dark5, name, ttnn_module_args["dark5"])

    return parameters
