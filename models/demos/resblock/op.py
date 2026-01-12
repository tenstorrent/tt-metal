import torch
from loguru import logger


class FusedResblock:
    @staticmethod
    def golden(input_a, weight0, weight1):
        x = input_a @ weight0
        x = torch.nn.functional.relu(x)
        x = x @ weight1
        x = x + input_a
        return x

    @staticmethod
    def op(
        input_a,
        weight0,
        weight1,
        output_tensor,
    ):
        logger.info(f"Running ResBlock operation with shape {input_a.shape} x {weight0.shape} x {weight1.shape}")
        return output_tensor
