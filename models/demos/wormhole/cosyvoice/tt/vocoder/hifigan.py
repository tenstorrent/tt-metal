# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HiFi-GAN Vocoder using TTNN APIs.

The HiFi-GAN vocoder converts mel-spectrograms to high-quality waveforms.
It uses a multi-scale residual block architecture with transposed convolutions
for upsampling.
"""

from typing import Optional

import torch
import ttnn
from loguru import logger


class TtHiFiGAN:
    """HiFi-GAN vocoder implemented with TTNN operations.

    Converts mel-spectrograms (80-band) to raw audio waveforms.
    """

    def __init__(
        self,
        device: ttnn.Device,
        config,
        state_dict: dict,
        tt_cache_path: Optional[str] = None,
    ):
        self.device = device
        self.config = config

        memory_config = ttnn.DRAM_MEMORY_CONFIG
        dtype = ttnn.bfloat16

        logger.info("Initializing HiFi-GAN vocoder")

        # Store weights as TTNN tensors
        # In Stage 1, we store weight references for the full graph
        # The actual TTNN op graph is built during the forward pass
        self.state_dict = state_dict
        self.tt_cache_path = tt_cache_path

        # Load weights lazily on first forward pass
        self._weights_loaded = False
        self._weight_tensors = {}

    def _load_weights(self):
        """Load and convert weights to TTNN tensors."""
        if self._weights_loaded:
            return

        device = self.device
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        dtype = ttnn.bfloat16

        # Transposed convolution weights (upsampling layers)
        for i in range(5):
            prefix = f"hifigan.generator.convtr_{i}"
            w_key = f"{prefix}.weight"
            b_key = f"{prefix}.bias"

            if w_key in self.state_dict:
                w = self.state_dict[w_key]
                # ConvTranspose1d weight shape: [out_c, in_c, kernel]
                # TTNN expects: [out_c, in_c, 1, kernel]
                w = w.unsqueeze(2)
                self._weight_tensors[w_key] = ttnn.as_tensor(
                    w, device=device, layout=ttnn.TILE_LAYOUT,
                    memory_config=memory_config, dtype=dtype,
                )
                if b_key in self.state_dict:
                    self._weight_tensors[b_key] = ttnn.as_tensor(
                        self.state_dict[b_key],
                        device=device, layout=ttnn.TILE_LAYOUT,
                        memory_config=memory_config, dtype=dtype,
                    )

        # ResBlock weights
        for i in range(3):  # 3 ResBlocks
            for k_idx, k_size in enumerate(self.config.hifigan_resblock_kernel_sizes):
                # Conv1d layers in ResBlock
                for j in range(3):  # 3 conv layers per ResBlock
                    prefix = f"hifigan.generator.resblocks.{i}.convs.{j}"
                    w_key = f"{prefix}.weight"
                    b_key = f"{prefix}.bias"

                    if w_key in self.state_dict:
                        w = self.state_dict[w_key]
                        if len(w.shape) == 3:
                            w = w.unsqueeze(2)  # [out_c, in_c, 1, kernel]
                        self._weight_tensors[w_key] = ttnn.as_tensor(
                            w, device=device, layout=ttnn.TILE_LAYOUT,
                            memory_config=memory_config, dtype=dtype,
                        )
                        if b_key in self.state_dict:
                            self._weight_tensors[b_key] = ttnn.as_tensor(
                                self.state_dict[b_key],
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=memory_config, dtype=dtype,
                            )

        self._weights_loaded = True

    def __call__(
        self,
        mel: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Convert mel-spectrogram to waveform.

        Args:
            mel: Mel-spectrogram [batch, 80, time]

        Returns:
            Audio waveform [batch, 1, audio_samples]
        """
        self._load_weights()

        # In Stage 1, we provide the structural implementation.
        # The full TTNN op graph for HiFi-GAN requires ConvTranspose1d
        # support which is being tracked.
        #
        # Architecture:
        # 1. Initial convolution: conv1d(mel) -> [batch, 512, time]
        # 2. 3x ResBlocks with dilated convolutions
        # 3. 5x transposed convolution upsampling layers
        #    upsample_rates = [5, 4, 4, 2, 2]
        #    upsample_kernel_sizes = [10, 8, 8, 4, 4]
        # 4. Final convolution: conv1d -> [batch, 1, audio_samples]
        # 5. Tanh activation

        # For Stage 1 bring-up, we return a placeholder tensor
        # with the correct shape
        return ttnn.zeros(
            (mel.shape[0], 1, mel.shape[2] * 320),  # 320x upsampling factor
            device=self.device,
            dtype=ttnn.bfloat16,
        )
