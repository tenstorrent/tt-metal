import torch
from models.utility_functions import comp_pcc


def compare(base_name):
    model = torch.load("model_" + base_name + ".pt")
    test = torch.load("test_" + base_name + ".pt")

    print(f"{base_name}: PCC: {comp_pcc(model, test)[1]}")

    print(f"{base_name}: Row-wise max absolute difference:")
    for i in range(model.shape[0]):
        print(base_name, i, i % 448, (model[i] - test[i]).abs().max().item())


compare("1_post_norm")
compare("1a_untilized")
compare("1b_reshaped")
compare("2_tilized")
compare("3_post_linear")
compare("4_post_gelu")
compare("5_post_linear")

# print('Column-wise max absolute difference:')
# for i in range(model.shape[1]):
#     print(i, (model[:, i] - test[:, i]).abs().max())

# print('Row 3191:')
# for i in range(model.shape[1]):
#     print(i, (model[3191, i] - test[3191, i]).abs().max())
