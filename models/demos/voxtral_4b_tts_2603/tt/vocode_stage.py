# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The `vocode` section: integer audio codes -> a 24 kHz waveform, on the device.

This is the neural audio CODEC DECODER of `mistralai/Voxtral-4B-TTS-2603`
(`hf_model.audio_tokenizer`). The chain the reference runs, and the chain this stage runs:

    codes [B, 37, T]
      -> quantizer.decode        semantic row 0: an 8192 x 256 Euclidean table (built as
                                 embedding_sum / cluster_usage, both BUFFERS);
                                 acoustic rows 1..36: 21 FSQ levels rescaled to [-1, 1]
      -> latent [B, 292, T]
      -> CausalConv1d(292 -> 1024, k3, s1, pad_mode="replicate")
      -> CodecTransformer(window 2)    x n_layers
      -> CausalConvTranspose1d(k4, s2) -> CodecTransformer(window 4)
      -> CausalConvTranspose1d(k4, s2) -> CodecTransformer(window 8)
      -> CausalConvTranspose1d(k4, s2) -> CodecTransformer(window 16)
      -> output_proj CausalConv1d(1024 -> 240, k7, pad_mode="reflect", WEIGHT-NORMED)
      -> de-patch "b (c h) t -> b c (t h)", h = 240
      -> waveform [B, 1, T * 1920]

The windows DOUBLE because each transposed convolution doubles the frame rate, so each group's
ALiBi mask is built from its OWN window. The two padding modes are different and both matter: the
first convolution replicates the boundary sample, `output_proj` REFLECTS -- a mirror about index 0
that excludes index 0 itself.

The whole chain is `tt/modules/voxtral_t_t_s_audio_tokenizer.py`, which decodes every batch row.
This stage adds the two things around it: the code offset (the acoustic transformer emits codes
shifted up by the audio special tokens; the codec wants them unshifted), and the audio-token
embedding the text decode loop feeds back as its next input (`audio_token_embedding`).

FLOAT32 ACTIVATIONS, BFLOAT16 WEIGHTS. All-bfloat16 put the full chain at PCC 0.9873 -- every stage
is fine alone, but eight residual blocks plus five convolutions compound.

No device is ever opened here.
"""
from __future__ import annotations

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.modules import multi_vocab_embeddings, voxtral_t_t_s_audio_tokenizer

# fp32 accumulation in DEST for the reduction this module owns (the audio-token embedding sum).
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


class CodecBlock:
    """ONE codec transformer block position: its group, layer id, sliding window and callable.

    `VocodeStage.blocks` lists these in forward order, so anything sizing the codec's depth can
    read it. The codec body runs the blocks itself; calling one here runs it on `[B, 1, S, dim]`.
    """

    def __init__(self, group, layer_id, window, run):
        self.group = group
        self.layer_id = layer_id
        self.window = window
        self.run = run

    def __call__(self, hidden, **kwargs):
        return self.run(hidden)

    def __repr__(self):
        return f"CodecBlock(group={self.group}, layer_id={self.layer_id}, window={self.window})"


class VocodeStage:
    """The resident codec decoder. `decode` is the section's forward; nothing here is host compute."""

    def __init__(
        self,
        device,
        codec,
        blocks,
        n_layers,
        codec_body,
        multi_vocab,
        code_offset,
        max_frames,
        max_frames_provenance,
    ):
        self.device = device
        self.reference_module = codec
        self.blocks = blocks
        self.n_layers = int(n_layers)
        self.group_windows = tuple(sorted({b.window for b in blocks})) if blocks else ()
        self.frame_rate = float(codec.frame_rate)
        self.sampling_rate = int(codec.sampling_rate)
        self.samples_per_frame = int(codec.downsample_factor)
        self.n_codebooks = int(codec.num_codebooks)
        self.code_offset = int(code_offset)
        self.max_frames = int(max_frames)
        self.max_frames_provenance = max_frames_provenance
        self._codec_body = codec_body
        self._multi_vocab = multi_vocab

    # -- the section's two entry points -----------------------------------------------------

    def audio_token_embedding(self, codes):
        """`[B, 37, T]` SHIFTED codes -> `[B, 1, T, 3072]`, the SUM over the 37 codebooks.

        This is what the text decode loop feeds back as its next input. The codes stay in the
        SHIFTED space here -- `MultiVocabEmbeddings`'s 37 packed codebooks are indexed with the two
        audio special tokens included; only the quantizer wants them unshifted.

        The time axis is required, not optional: the reference broadcasts its per-codebook offsets
        as `[1, 37, 1]`, so a rank-2 frame would add them along the wrong axis. A rank-2 input is
        given the axis here rather than silently mis-indexed.
        """
        shape = [int(v) for v in codes.shape]
        if len(shape) == 2:
            codes = ttnn.reshape(codes, [shape[0], shape[1], 1])
            shape = shape + [1]
        if len(shape) != 3:
            raise ValueError(f"audio_token_embedding wants [B, {self.n_codebooks}, T], got {shape}")
        if shape[1] != self.n_codebooks:
            raise ValueError(f"audio_token_embedding wants {self.n_codebooks} codebook rows, got {shape[1]}")
        # The lookup table is bfloat16 (`ttnn.embedding` requires it), so the 37-term sum is done
        # widened: `input_embedding_concat_type: "sum"` adds 37 of them and the text stack consumes
        # the result in float32.
        if shape[2] == 1:
            # One frame (the decode feedback): the 37 lookups come back as the ROWS of one
            # `[B, 1, 37, D]` tile tensor and are summed over rows, 1/16 the tiles of `[B, 37, 1, D]`.
            emb = self._multi_vocab(codes, codebooks_as_rows=True)
            return ttnn.sum(ttnn.typecast(emb, ttnn.float32), dim=2, keepdim=True, compute_kernel_config=_COMPUTE)
        emb = self._multi_vocab(codes)
        return ttnn.sum(ttnn.typecast(emb, ttnn.float32), dim=1, keepdim=True)

    def decode(self, codes):
        """`[B, 37, T]` SHIFTED codes -> `[B, 1, T * 1920]` float32 waveform.

        The offset between the acoustic transformer's output space and the codec's own code space
        is subtracted ON DEVICE, then every row goes through the codec body.
        """
        shape = [int(v) for v in codes.shape]
        if len(shape) != 3:
            raise ValueError(f"decode wants [B, {self.n_codebooks}, T], got {shape}")
        _, rows, frames = shape
        if rows != self.n_codebooks:
            raise ValueError(f"decode wants {self.n_codebooks} codebook rows, got {rows}")
        if frames > self.max_frames:
            raise NotImplementedError(
                f"{frames} frames upsample to {frames * 8} rows, past this stage's ceiling of "
                f"{self.max_frames} frames ({self.max_frames_provenance}). The mask is a "
                f"build-time constant, so a longer input needs a LARGER `_MASK_MAX_SEQ`; it cannot "
                f"be rebuilt inside the forward, which must stay torch-free."
            )
        return self._codec_body(self._unshift(codes))

    __call__ = decode

    # -- internals ------------------------------------------------------------------------

    def _unshift(self, codes):
        """Subtract the audio special-token offset, on the device.

        `ttnn.subtract` on a uint32 tensor WRAPS instead of going negative, and an untilized
        integer input cannot be typecast at all, so the order is to_layout -> typecast -> subtract.
        Verified exact: the round trip reproduces `codes - offset` bit for bit over the code range.

        CLAMPED AT 0, because a batch renders rows that have already ENDED: a row's `end_audio`
        frame (semantic id 1) and anything after it are not that row's speech, but they are still in
        the `[B, 37, T]` block, and 1 - offset = -1 is not a codebook index -- torch's embedding
        raises on it and a uint32 gather reads out of range. `reference/golden.py` applies the same
        clamp, so both sides render identical codes, and `pipeline.trim_to_end` cuts each row before
        its end frame.
        """
        if not self.code_offset:
            return codes
        wide = ttnn.typecast(ttnn.to_layout(codes, ttnn.TILE_LAYOUT), ttnn.float32)
        shifted = ttnn.relu(ttnn.subtract(wide, float(self.code_offset)))
        return ttnn.to_layout(ttnn.typecast(shifted, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)


def build_vocode_stage(device, hf_model, layers=None):
    """Build the resident codec decoder. Opens no device and runs no forward.

    `layers` caps BLOCKS PER GROUP (full 2, at least 1). All four groups survive any cap because
    each carries a different sliding window. `None` means every block; it is never read as 0.
    """
    codec = hf_model.audio_tokenizer
    args = codec.args
    upsample = 1
    for stride in args.decoder_convs_strides:
        upsample *= int(stride)

    groups = [b for b in codec.decoder_blocks if type(b).__name__ == "CodecTransformer"]
    full_depth = max(len(g.layers) for g in groups)
    depth = full_depth if layers is None else int(layers)
    if depth < 1:
        print(
            f"[voxtral_4b_tts_2603] vocode: layers={layers} would leave a group with zero blocks; "
            f"clamping UP to 1 block per group ({len(groups)} groups x 1)"
        )
        depth = 1
    depth = min(depth, full_depth)

    codec_body = voxtral_t_t_s_audio_tokenizer.build(device, codec, layers=None if depth == full_depth else depth)
    blocks = [CodecBlock(group, layer_id, window, run) for group, layer_id, window, run in codec_body.blocks]

    # The frame ceiling: the codec prebuilds its ALiBi masks `_MASK_MAX_SEQ` rows long, and the
    # decoder upsamples `upsample`x, so the widest input it takes is `_MASK_MAX_SEQ / upsample` frames.
    mask_rows = int(voxtral_t_t_s_audio_tokenizer._MASK_MAX_SEQ)
    max_frames = mask_rows // upsample
    provenance = f"_MASK_MAX_SEQ = {mask_rows} rows in voxtral_t_t_s_audio_tokenizer.py / {upsample}x upsampling"

    if depth != full_depth:
        print(
            f"[voxtral_4b_tts_2603] vocode: {len(groups)} groups x {depth} of {full_depth} blocks "
            f"= {len(blocks)} blocks (windows {tuple(int(g.args.attn_sliding_window_size) for g in groups)})"
        )

    return VocodeStage(
        device,
        codec,
        blocks,
        depth,
        codec_body,
        multi_vocab_embeddings.build(device, codec.audio_token_embedding),
        common.n_audio_special_tokens(hf_model),
        max_frames,
        provenance,
    )
