# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SFace (MobileFaceNet) Reference Model in PyTorch.

Architecture based on OpenCV Zoo's face_recognition_sface_2021dec.onnx
Input: 112x112x3 RGB face image (normalized)
Output: 128-D embedding vector for face recognition

Structure (from ONNX analysis):
- Conv1: 3->32, 3x3, stride=1, pad=1
- Block2: DW 32->32 s=1, PW 32->64
- Block3: DW 64->64 s=2, PW 64->128  (112->56)
- Block4: DW 128->128 s=1, PW 128->128
- Block5: DW 128->128 s=2, PW 128->256 (56->28)
- Block6: DW 256->256 s=1, PW 256->256
- Block7: DW 256->256 s=2, PW 256->512 (28->14)
- Blocks 8-12: DW 512->512 s=1, PW 512->512 (5 blocks)
- Block13: DW 512->512 s=2, PW 512->1024 (14->7)
- Block14: DW 1024->1024 s=1, PW 1024->1024
- Global BN + Dropout + Flatten + FC(50176->128) + BN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Module):
    """Conv2d + BatchNorm + PReLU block."""

    # ONNX model uses eps=0.001 (not PyTorch default 1e-5)
    ONNX_BN_EPS = 0.001

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=self.ONNX_BN_EPS)  # Match ONNX epsilon
        self.prelu = nn.PReLU(out_ch)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise Separable Convolution block: DW 3x3 + PW 1x1.

    Note: Stride is applied on the DEPTHWISE conv, not pointwise.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # Depthwise 3x3 with stride
        self.dw = ConvBNPReLU(in_ch, in_ch, kernel=3, stride=stride, padding=1, groups=in_ch)
        # Pointwise 1x1 (always stride=1)
        self.pw = ConvBNPReLU(in_ch, out_ch, kernel=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class SFace(nn.Module):
    """
    SFace (MobileFaceNet) for face recognition.

    Input: [B, 3, 112, 112] - normalized RGB face image
    Output: [B, 128] - L2-normalized embedding vector

    Spatial dimensions:
    - Input: 112x112
    - After conv1: 112x112 (stride=1)
    - After block3: 56x56 (stride=2 in DW)
    - After block5: 28x28 (stride=2 in DW)
    - After block7: 14x14 (stride=2 in DW)
    - After block13: 7x7 (stride=2 in DW)
    - FC input: 1024 * 7 * 7 = 50176
    """

    def __init__(self, embedding_size: int = 128):
        super().__init__()
        self.embedding_size = embedding_size

        # Conv1: 3 -> 32, stride 1, 112x112 -> 112x112
        self.conv1 = ConvBNPReLU(3, 32, kernel=3, stride=1, padding=1)

        # Block 2: 32 -> 64, DW stride=1, 112x112 -> 112x112
        self.block2 = DepthwiseSeparableBlock(32, 64, stride=1)

        # Block 3: 64 -> 128, DW stride=2, 112x112 -> 56x56
        self.block3 = DepthwiseSeparableBlock(64, 128, stride=2)

        # Block 4: 128 -> 128, DW stride=1, 56x56 -> 56x56
        self.block4 = DepthwiseSeparableBlock(128, 128, stride=1)

        # Block 5: 128 -> 256, DW stride=2, 56x56 -> 28x28
        self.block5 = DepthwiseSeparableBlock(128, 256, stride=2)

        # Block 6: 256 -> 256, DW stride=1, 28x28 -> 28x28
        self.block6 = DepthwiseSeparableBlock(256, 256, stride=1)

        # Block 7: 256 -> 512, DW stride=2, 28x28 -> 14x14
        self.block7 = DepthwiseSeparableBlock(256, 512, stride=2)

        # Blocks 8-12: 512 -> 512, DW stride=1, 14x14 -> 14x14 (5 blocks)
        self.block8 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.block9 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.block10 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.block11 = DepthwiseSeparableBlock(512, 512, stride=1)
        self.block12 = DepthwiseSeparableBlock(512, 512, stride=1)

        # Block 13: 512 -> 1024, DW stride=2, 14x14 -> 7x7
        self.block13 = DepthwiseSeparableBlock(512, 1024, stride=2)

        # Block 14: 1024 -> 1024, DW stride=1, 7x7 -> 7x7
        self.block14 = DepthwiseSeparableBlock(1024, 1024, stride=1)

        # Global BN (ONNX uses eps=0.001)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001)

        # Dropout
        self.dropout = nn.Dropout(p=0.0)  # Set to 0 for inference

        # FC layer: 1024 * 7 * 7 = 50176 -> 128
        self.fc = nn.Linear(1024 * 7 * 7, embedding_size)

        # Final BN on embedding (ONNX uses eps=0.001)
        self.bn2 = nn.BatchNorm1d(embedding_size, eps=0.001)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, 112, 112] (raw 0-255 pixel values)

        Returns:
            embedding: [B, 128] face embedding (L2-normalized)
        """
        # Preprocessing (matches ONNX: (x - 127.5) / 128.0)
        x = (x - 127.5) * 0.0078125  # 0.0078125 = 1/128

        # Backbone
        x = self.conv1(x)  # 112 -> 112
        x = self.block2(x)  # 112 -> 112
        x = self.block3(x)  # 112 -> 56
        x = self.block4(x)  # 56 -> 56
        x = self.block5(x)  # 56 -> 28
        x = self.block6(x)  # 28 -> 28
        x = self.block7(x)  # 28 -> 14
        x = self.block8(x)  # 14 -> 14
        x = self.block9(x)  # 14 -> 14
        x = self.block10(x)  # 14 -> 14
        x = self.block11(x)  # 14 -> 14
        x = self.block12(x)  # 14 -> 14
        x = self.block13(x)  # 14 -> 7
        x = self.block14(x)  # 7 -> 7

        # Head
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.flatten(1)  # [B, 1024, 7, 7] -> [B, 50176]
        x = self.fc(x)  # [B, 50176] -> [B, 128]
        x = self.bn2(x)

        # L2 normalize embedding
        x = F.normalize(x, p=2, dim=1)

        return x


def load_sface_from_onnx(onnx_path: str) -> SFace:
    """
    Load SFace model weights from ONNX file.

    Args:
        onnx_path: Path to face_recognition_sface_2021dec.onnx

    Returns:
        SFace model with loaded weights
    """
    import onnx
    import numpy as np

    onnx_model = onnx.load(onnx_path)

    # Create weight dict from ONNX initializers and inputs
    weights = {}
    for tensor in list(onnx_model.graph.initializer) + list(onnx_model.graph.input):
        if hasattr(tensor, "raw_data") or hasattr(tensor, "float_data"):
            if hasattr(tensor, "raw_data") and tensor.raw_data:
                arr = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(tensor.dims)
            elif hasattr(tensor, "float_data") and tensor.float_data:
                arr = np.array(tensor.float_data, dtype=np.float32).reshape(tensor.dims)
            else:
                continue
            weights[tensor.name] = torch.from_numpy(arr)

    model = SFace()

    # Map ONNX weights to PyTorch model
    def load_conv_bn_prelu(module, conv_prefix, bn_prefix, prelu_prefix):
        """Load weights for ConvBNPReLU block."""
        # Conv weight
        if f"{conv_prefix}_weight" in weights:
            module.conv.weight.data = weights[f"{conv_prefix}_weight"]
        # BN params
        if f"{bn_prefix}_gamma" in weights:
            module.bn.weight.data = weights[f"{bn_prefix}_gamma"]
            module.bn.bias.data = weights[f"{bn_prefix}_beta"]
            module.bn.running_mean.data = weights[f"{bn_prefix}_moving_mean"]
            module.bn.running_var.data = weights[f"{bn_prefix}_moving_var"]
        # PReLU
        if f"{prelu_prefix}_gamma" in weights:
            module.prelu.weight.data = weights[f"{prelu_prefix}_gamma"].squeeze()

    def load_dw_sep_block(module, conv_idx):
        """Load weights for DepthwiseSeparableBlock."""
        # DW conv: conv_{idx}_dw_conv2d, conv_{idx}_dw_batchnorm, conv_{idx}_dw_relu
        load_conv_bn_prelu(
            module.dw,
            f"conv_{conv_idx}_dw_conv2d",
            f"conv_{conv_idx}_dw_batchnorm",
            f"conv_{conv_idx}_dw_relu",
        )
        # PW conv: conv_{idx}_conv2d, conv_{idx}_batchnorm, conv_{idx}_relu
        load_conv_bn_prelu(module.pw, f"conv_{conv_idx}_conv2d", f"conv_{conv_idx}_batchnorm", f"conv_{conv_idx}_relu")

    # Load conv1
    load_conv_bn_prelu(model.conv1, "conv_1_conv2d", "conv_1_batchnorm", "conv_1_relu")

    # Load blocks 2-14
    load_dw_sep_block(model.block2, 2)
    load_dw_sep_block(model.block3, 3)
    load_dw_sep_block(model.block4, 4)
    load_dw_sep_block(model.block5, 5)
    load_dw_sep_block(model.block6, 6)
    load_dw_sep_block(model.block7, 7)
    load_dw_sep_block(model.block8, 8)
    load_dw_sep_block(model.block9, 9)
    load_dw_sep_block(model.block10, 10)
    load_dw_sep_block(model.block11, 11)
    load_dw_sep_block(model.block12, 12)
    load_dw_sep_block(model.block13, 13)
    load_dw_sep_block(model.block14, 14)

    # Load global BN
    if "bn1_gamma" in weights:
        model.bn1.weight.data = weights["bn1_gamma"]
        model.bn1.bias.data = weights["bn1_beta"]
        model.bn1.running_mean.data = weights["bn1_moving_mean"]
        model.bn1.running_var.data = weights["bn1_moving_var"]

    # Load FC
    if "pre_fc1_weight" in weights:
        model.fc.weight.data = weights["pre_fc1_weight"]
        model.fc.bias.data = weights["pre_fc1_bias"]

    # Load final BN
    if "fc1_gamma" in weights:
        model.bn2.weight.data = weights["fc1_gamma"]
        model.bn2.bias.data = weights["fc1_beta"]
        model.bn2.running_mean.data = weights["fc1_moving_mean"]
        model.bn2.running_var.data = weights["fc1_moving_var"]

    model.eval()
    return model


def preprocess_face(image, target_size=(112, 112)):
    """
    Preprocess face image for SFace model.

    Args:
        image: numpy array [H, W, 3] BGR or RGB, uint8
        target_size: (height, width) tuple

    Returns:
        tensor: [1, 3, 112, 112] normalized tensor
    """
    import cv2
    import numpy as np

    # Resize
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)

    # Convert to float and normalize to [-1, 1]
    # ONNX model uses: (x - 127.5) / 128.0
    image = image.astype(np.float32)
    image = (image - 127.5) / 128.0

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    # Add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0)

    return tensor


def cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: [128] or [B, 128] tensor
        embedding2: [128] or [B, 128] tensor

    Returns:
        similarity: float or [B] tensor, range [-1, 1]
    """
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)

    return F.cosine_similarity(embedding1, embedding2)


if __name__ == "__main__":
    # Test model
    model = SFace()
    model.eval()

    # Test forward pass
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        embedding = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embedding.shape}")
    print(f"Embedding norm: {torch.norm(embedding, dim=1).item():.4f} (should be ~1.0)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
