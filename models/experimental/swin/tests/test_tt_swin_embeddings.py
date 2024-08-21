# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger


from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import ttnn
from models.experimental.swin.tt.swin_embeddings import TtSwinEmbeddings
from models.experimental.swin.swin_utils import get_shape
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_embeddings_inference(device, imagenet_sample_input, pcc, reset_seeds):
    image = imagenet_sample_input
    base_address = f"embeddings"

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinembedding
    torch_model = model.embeddings

    # Tt swinembedding
    tt_model = TtSwinEmbeddings(
        config=model.config, state_dict=model.state_dict(), base_address=base_address, device=device
    )

    # Run torch model
    torch_output = torch_model(image)

    # Run tt model
    tt_image = ttnn.Tensor(
        image.reshape(-1).tolist(),
        get_shape(image.shape),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_output = tt_model(tt_image)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)
    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SwinEmbedding Passed!")
    else:
        logger.warning("SwinEmbedding Failed!")

    assert does_pass
