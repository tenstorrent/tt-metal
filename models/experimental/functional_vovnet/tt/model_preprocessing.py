import torch
import ttnn

from ttnn.model_preprocessing import infer_ttnn_module_args


def create_vovnet_model_parameters(model, input_tensor, device):
    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None

    for key in parameters:
        parameters[key].module = getattr(model, key)

    parameters["stem"][0].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stem"][1].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stem"][1].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stem"][2].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stem"][2].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_reduction"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_mid"][0].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_mid"][0].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_mid"][1].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_mid"][1].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_mid"][2].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_mid"][2].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["conv_concat"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][0]["blocks"][0]["attn"].fc["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_reduction"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_mid"][0].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_mid"][0].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_mid"][1].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_mid"][1].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_mid"][2].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_mid"][2].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["conv_concat"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][1]["blocks"][0]["attn"].fc["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_reduction"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_mid"][0].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_mid"][0].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_mid"][1].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_mid"][1].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_mid"][2].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_mid"][2].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["conv_concat"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][2]["blocks"][0]["attn"].fc["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_reduction"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_mid"][0].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_mid"][0].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_mid"][1].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_mid"][1].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_mid"][2].conv_dw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_mid"][2].conv_pw["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["conv_concat"].conv["math_fidelity"] = ttnn.MathFidelity.LoFi
    parameters["stages"][3]["blocks"][0]["attn"].fc["math_fidelity"] = ttnn.MathFidelity.LoFi

    parameters["stem"][0].conv["dtype"] = ttnn.bfloat8_b
    parameters["stem"][1].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stem"][1].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stem"][2].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stem"][2].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_reduction"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][0].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][0].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][1].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][1].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][2].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][2].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_concat"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["attn"].fc["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_reduction"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][0].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][0].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][1].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][1].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][2].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][2].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_concat"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["attn"].fc["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_reduction"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][0].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][0].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][1].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][1].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][2].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][2].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_concat"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["attn"].fc["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_reduction"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][0].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][0].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][1].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][1].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][2].conv_dw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][2].conv_pw["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_concat"].conv["dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["attn"].fc["dtype"] = ttnn.bfloat8_b

    parameters["stem"][0].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stem"][1].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stem"][1].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stem"][2].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stem"][2].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_reduction"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][0].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][0].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][1].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][1].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][2].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_mid"][2].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["conv_concat"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][0]["blocks"][0]["attn"].fc["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_reduction"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][0].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][0].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][1].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][1].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][2].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_mid"][2].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["conv_concat"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][1]["blocks"][0]["attn"].fc["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_reduction"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][0].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][0].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][1].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][1].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][2].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_mid"][2].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["conv_concat"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][2]["blocks"][0]["attn"].fc["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_reduction"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][0].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][0].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][1].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][1].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][2].conv_dw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_mid"][2].conv_pw["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["conv_concat"].conv["weights_dtype"] = ttnn.bfloat8_b
    parameters["stages"][3]["blocks"][0]["attn"].fc["weights_dtype"] = ttnn.bfloat8_b

    parameters["stages"][2]["blocks"][0]["conv_mid"][0].conv_dw["use_1d_systolic_array"] = False
    parameters["stages"][2]["blocks"][0]["conv_mid"][1].conv_dw["use_1d_systolic_array"] = False
    parameters["stages"][2]["blocks"][0]["conv_mid"][2].conv_dw["use_1d_systolic_array"] = False
    parameters["stages"][3]["blocks"][0]["conv_mid"][0].conv_dw["use_1d_systolic_array"] = False
    parameters["stages"][3]["blocks"][0]["conv_mid"][1].conv_dw["use_1d_systolic_array"] = False
    parameters["stages"][3]["blocks"][0]["conv_mid"][2].conv_dw["use_1d_systolic_array"] = False
    return parameters
