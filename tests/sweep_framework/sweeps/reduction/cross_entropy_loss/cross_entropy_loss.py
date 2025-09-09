# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

parameters = {
    "suite_1": {
        "predictions_shape": [
            (3, 5),
        ],
    }
}


def cross_entropy_loss_ttnn(predictions, labels, reduction_constant):
    reduction_const_reshaped = ttnn.reshape(reduction_constant, (1, 1))
    softmax_predictions = ttnn.softmax(predictions, dim=-1)
    log_softmax = ttnn.log(softmax_predictions)
    weighted_log_probs = ttnn.multiply(labels, log_softmax)
    sum_per_sample = ttnn.sum(weighted_log_probs, dim=-1, keepdim=True)
    scaled_loss = ttnn.multiply(sum_per_sample, reduction_const_reshaped)
    final_loss = ttnn.mean(scaled_loss, dim=0, keepdim=True)
    return final_loss


def run(predictions_shape, device):
    torch.manual_seed(0)
    predictions = torch.randn(predictions_shape)
    target_indices = torch.empty(3, dtype=torch.long).random_(5)
    target_onehot = torch.nn.functional.one_hot(target_indices, num_classes=5).float()

    pytorch_result = torch.nn.CrossEntropyLoss()(predictions, target_indices)

    predictions_ttnn = ttnn.from_torch(predictions, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    labels_ttnn = ttnn.from_torch(target_onehot, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    reduction_const_ttnn = ttnn.from_torch(torch.tensor([-1.0]), dtype=ttnn.float32, device=device)
    ttnn_result = ttnn.to_torch(cross_entropy_loss_ttnn(predictions_ttnn, labels_ttnn, reduction_const_ttnn))
    err = abs(pytorch_result.item() - ttnn_result.item()) / pytorch_result.item()

    print(f"PyTorch: {pytorch_result.item():.6f}")
    print(f"TTNN: {ttnn_result.item():.6f}")
    print(f"Difference: {abs(pytorch_result.item() - ttnn_result.item()):.6f}, Percentage: {err * 100:.6f}%")

    assert err < 0.009
