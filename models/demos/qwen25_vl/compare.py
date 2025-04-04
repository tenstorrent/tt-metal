import ttnn
import torch
from models.utility_functions import comp_pcc


def compare(base_name):
    model = torch.load("our_" + base_name + ".pt", weights_only=False)
    test = torch.load("ref_" + base_name + ".pt", weights_only=False)
    model = model[: test.shape[0], : test.shape[1]]

    print(f"{base_name}: PCC: {comp_pcc(model, test)[1]}")

    # print(f"{base_name}: Row-wise max absolute difference:")
    # for i in range(model.shape[0]):
    #     print(base_name, i, i % 448, (model[i] - test[i]).abs().max().item())


# for i in range(32):
#     compare(f"x_{i}")

compare("1_attn_norm")
compare("2_attn")
compare("3_residual_add")
compare("4_ff_norm")
compare("5_ff")
compare("6_residual_add")
