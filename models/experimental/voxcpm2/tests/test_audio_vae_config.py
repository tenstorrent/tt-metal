# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host geometry/configuration checks, not device inference qualification."""

import unittest

from models.experimental.voxcpm2.tt.audio_vae import AudioVAEConfig, CausalConvSpec


class AudioVAEConfigTests(unittest.TestCase):
    def test_codec_rate_contract(self):
        config = AudioVAEConfig.from_mapping({})
        self.assertEqual(config.chunk_size, 640)
        self.assertEqual(config.decode_chunk_size, 1920)
        self.assertEqual(
            config.decode_chunk_size * config.sample_rate,
            config.chunk_size * config.out_sample_rate,
        )

    def test_decoder_causal_trim_exact_upsample(self):
        for stride in (2, 5, 6, 8):
            for length in (1, 4, 33, 128):
                with self.subTest(stride=stride, length=length):
                    spec = CausalConvSpec(
                        2 * stride,
                        stride,
                        padding=(stride + 1) // 2,
                        output_padding=stride % 2,
                        transpose=True,
                    )
                    self.assertEqual(spec.left_pad, stride)
                    self.assertEqual(spec.output_length(length), length * stride)

    def test_residual_convolution_preserves_length(self):
        for dilation in (1, 3, 9):
            with self.subTest(dilation=dilation):
                spec = CausalConvSpec(7, dilation=dilation, padding=3 * dilation)
                self.assertEqual(spec.output_length(47), 47)

    def test_encoder_geometry_after_preprocess(self):
        for stride in (2, 5, 8):
            with self.subTest(stride=stride):
                spec = CausalConvSpec(
                    2 * stride,
                    stride,
                    padding=(stride + 1) // 2,
                    output_padding=stride % 2,
                )
                self.assertEqual(spec.output_length(64 * stride), 64)

    def test_sample_rate_bucket_boundary_matches_torch_default(self):
        config = AudioVAEConfig()
        buckets = [
            config.sample_rate_bucket(rate)
            for rate in (16000, 20000, 20001, 30000, 30001, 40000, 48000)
        ]
        self.assertEqual(buckets, [0, 0, 1, 1, 2, 2, 3])

    def test_unimplemented_options_rejected(self):
        for values in (
            {"use_noise_block": True},
            {"cond_type": "concat"},
            {"decoder_rates": [1, 2]},
        ):
            with self.subTest(values=values), self.assertRaises(NotImplementedError):
                AudioVAEConfig.from_mapping(values)

    def test_invalid_configuration_rejected(self):
        for values in (
            {"encoder_rates": []},
            {"decoder_rates": [0]},
            {"decoder_dim": 7},
            {"sr_bin_boundaries": [30000, 20000]},
            {"latent_dim": 0},
            {"unrecognized_option": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AudioVAEConfig.from_mapping(values)

    def test_checkpoint_weight_norm_materialization(self):
        try:
            import torch
        except ImportError:
            self.skipTest(
                "Torch not installed: checkpoint materialization test unavailable"
            )
        from models.experimental.voxcpm2.tt.audio_vae import materialize_weight

        for cls in (torch.nn.Conv1d, torch.nn.ConvTranspose1d):
            with self.subTest(convolution=cls.__name__):
                module = torch.nn.utils.weight_norm(cls(3, 5, 7))
                state = {
                    f"conv.{name}": value for name, value in module.state_dict().items()
                }
                # Checkpoint conversion only: no reference activation inference
                # or Tenstorrent correctness claim is made here.
                actual = materialize_weight(state, "conv")
                torch.testing.assert_close(
                    actual, module.weight.detach(), rtol=1e-6, atol=1e-7
                )


if __name__ == "__main__":
    unittest.main()
