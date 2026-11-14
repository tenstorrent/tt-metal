# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.sampling.tt_sampling import TTSampling

OPS = ("argmax_decode",)
_SHARDS = {128256: (109, 1184)}


def _width_sharded(device, width):
    cores, shard_w = _SHARDS[width]
    grid = device.compute_with_storage_grid_size()
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, True), [32, shard_w], ttnn.ShardOrientation.ROW_MAJOR
        ),
    )


class Llama31_8BBhTTSampling(TTSampling):
    def _greedy_here(self, x, apply_grammar, log_prob_logits):
        return (
            self._force_argmax_sampling
            and not apply_grammar
            and log_prob_logits is None
            and not self.log_probs_calculator.enable_log_probs
            and self.mesh_device.get_num_devices() == 1
            and self._sampling_dp == 1
            and self._force_argmax_sub_core_grids is None
            and x.dtype in (ttnn.bfloat16, ttnn.bfloat8_b)
            and x.layout == ttnn.TILE_LAYOUT
            and len(x.shape) == 4
            and x.shape[0] == 1
            and x.shape[1] == 1
            and x.shape[2] == 32
            and self._argmax_width(x) in _SHARDS
        )

    def _argmax_width(self, x):
        if self._can_slice_valid_vocab_for_argmax() and x.shape[-1] == self.padded_vocab_size:
            return self.vocab_size
        return x.shape[-1]

    def forward(self, x, tt_out_tok=None, apply_grammar=False, log_prob_logits=None):
        if not self._greedy_here(x, apply_grammar, log_prob_logits):
            return super().forward(x, tt_out_tok, apply_grammar, log_prob_logits)
        slice_valid_vocab = self._can_slice_valid_vocab_for_argmax()
        if not slice_valid_vocab:
            x = self._mask_invalid_vocab_logits(x)
        else:
            x = self._slice_valid_vocab_for_argmax(x)
        x_sh = ttnn.to_memory_config(x, _width_sharded(self.mesh_device, x.shape[-1]), dtype=ttnn.bfloat16)
        tokens = ttnn.experimental.llama31_8b.argmax_decode(x_sh)
        ttnn.deallocate(x_sh)
        if tt_out_tok is not None and self._decode_live_rows is not None:
            live_rows = self._decode_live_rows
            live = ttnn.slice(tokens, [0, 0, 0], [1, 1, live_rows])
            ttnn.deallocate(tokens)
            live = ttnn.reshape(live, [1, 1, 1, live_rows])
            padded = ttnn.pad(live, [(0, 0), (0, 0), (0, 0), (0, tt_out_tok.shape[-1] - live_rows)], 0)
            ttnn.deallocate(live)
            ttnn.copy(padded, tt_out_tok)
            ttnn.deallocate(padded)
        elif tt_out_tok is not None:
            ttnn.copy(ttnn.reshape(tokens, tt_out_tok.shape), tt_out_tok)
            ttnn.deallocate(tokens)
        else:
            tt_out_tok = tokens
        self.tt_log_probs = None
        return tt_out_tok, self.tt_log_probs


def use_for(sampling):
    if sampling is not None and type(sampling.tt_sampling) is TTSampling:
        sampling.tt_sampling.__class__ = Llama31_8BBhTTSampling
    return sampling
