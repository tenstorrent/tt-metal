"""Configuration for Inworld TTS codec decoder TTNN implementation."""

import ttnn

# VocosBackbone configuration (from ARCHITECTURE.md)
VOCOS_DIM = 1024
VOCOS_DEPTH = 12
VOCOS_HEADS = 16
VOCOS_HEAD_DIM = VOCOS_DIM // VOCOS_HEADS  # 64
VOCOS_POS_EMB_DIM = 64
VOCOS_MLP_DIM = 4096

# FSQ configuration
FSQ_LEVELS = [4, 4, 4, 4, 4, 4, 4, 4]
FSQ_CODEBOOK_SIZE = 65536  # 4^8
FSQ_VQ_DIM = 2048

# ISTFT configuration
ISTFT_N_FFT = 1280
ISTFT_HOP_LENGTH = 320
ISTFT_WIN_LENGTH = 1280

# ResnetBlock configuration
RESNET_NUM_GROUPS = 32

# Encoder configuration
ENCODER_CHANNELS = [48, 96, 192, 384, 768, 1536]
ENCODER_STRIDES = [2, 2, 4, 4, 5]
ENCODER_TOTAL_STRIDE = 320  # product of strides: 2*2*4*4*5
ACOUSTIC_OUTPUT_DIM = 1024
SEMANTIC_DIM = 1024
FUSION_DIM = 2048

# Wav2Vec2-BERT (facebook/w2v-bert-2.0) configuration
W2V_DIM = 1024
W2V_HEADS = 16
W2V_HEAD_DIM = W2V_DIM // W2V_HEADS  # 64
W2V_FFN_DIM = 4096
W2V_INPUT_DIM = 160
W2V_NUM_LAYERS = 16
W2V_DEPTHWISE_KERNEL = 31
W2V_LEFT_MAX = 64
W2V_RIGHT_MAX = 8


# Compute kernel configs
def get_compute_kernel_config_hifi2():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def get_compute_kernel_config_hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
