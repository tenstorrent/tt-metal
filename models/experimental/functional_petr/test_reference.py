import torch
from models.experimental.functional_petr.reference.petr import PETR


def test_reference():
    inputs = torch.load("models/experimental/functional_petr/reference/golden_input_inputs_sample1.pt")
    modified_batch_img_metas = torch.load(
        "models/experimental/functional_petr/reference/modified_input_batch_img_metas_sample1.pt"
    )
    model = PETR(use_grid_mask=True)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/reference/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )["state_dict"]
    model.load_state_dict(weights_state_dict)
    model.eval()
    output = model.predict(inputs, modified_batch_img_metas)
    print("output", output)
