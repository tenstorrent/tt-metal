# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral-TTS BACKBONE (Block 1): 3.4B AR transformer, embeddings -> hidden.

Built on `models/tt_transformers` rather than hand-written, because that gives us the 26-layer
stack, KV cache, RoPE and paged attention for free. What is bespoke here is the *interface*: this
model is not a text LM.

    embeddings [1,T,3072] --26 layers--> final RMSNorm --> hidden [1,T,3072] --> Block 2

Three things make it unlike every other tt_transformers model:
  * INPUT IS EMBEDDINGS, not token ids. Audio frames are a SUM of 37 codebook lookups
    (`embed_frames`), text comes from the tied text table. `Transformer.forward(x=...)` already
    takes a tensor (model.py:852), so `embedding.py` is simply bypassed.
  * NO lm_head. Block 2 consumes the post-final-norm hidden state
    (`voxtral_backbone_ref.py:157`); we never emit text tokens. The export therefore omits the
    402M-parameter tied head entirely.
  * `n_heads * head_dim` (4096) != `dim` (3072), so wq/wo are not square. tt_transformers handles
    this (`model_config.py:1983` uses k_dim = n_heads*head_dim) provided `head_dim` is explicit in
    config.json, which the export sets.

Requires an HF-format export first -- tt_transformers raises unless HF_MODEL is set
(`model_config.py:588`):

    python models/experimental/voxtral_tts/scripts/export_backbone_hf.py --out <dir>
    HF_MODEL=<dir> python models/experimental/voxtral_tts/tt/ttnn_voxtral_backbone.py

=== PRECISION: MEASURE IT ON REAL PROMPTS, NOT ON RANDOM INPUTS ===
`base_model_name` is whatever the export directory is called, so it matches none of
tt_transformers' known prefixes (Llama-3*, Mistral-7B*, Phi-3-mini*, ...) and its "accuracy" preset
falls through to a path that keeps BFP8 MLP weights. How much that costs depends entirely on what
you feed it. Against the fp32 CPU reference, comparing the LAST prefill position (the only one that
conditions Block 2):

    weights                        random N(0,0.02)     real prompt      real prompt
                                       T=128, 26L       P=200 (->256)    P=312 (->512)
    all-BFP8 MLP (tt_tf default)          0.89244          0.999379        0.999070
    FF1_FF3 BFP8, FF2 BF16                     --          0.999564        0.999579   <- USED
    BF16 everywhere                       0.96938          0.999698        L1 OOM

The random-input column is a MEASUREMENT ARTEFACT, and an expensive one -- it drove a precision
hunt that the real-prompt numbers made moot. Random embeddings are off-manifold, so activations do
not sit in the range the weights were trained for; block-float shares one exponent per block, so
an inflated dynamic range costs far more there than on real text. Every variant clears 0.999 on
real prompts. Never quote a bf16-vs-fp32 PCC from synthetic activations.

WHY FF1_FF3 IS BFP8 AND FF2 IS NOT -- an L1 capacity limit, not a numerics choice. At padded
prefill length 512 with BF16 FF1_FF3, the MLP's circular buffers overrun L1 by ~52 KB:
"Statically allocated circular buffers ... clash with L1 buffers ... allocated at 1428352 and
static circular buffer region ends at 1481920" (program.cpp:1764, reached via mlp.py:145).
FF1_FF3 holds BOTH 3072x9216 matrices, so it is the dominant buffer; halving it fits, and costs
0.000134 PCC. Probed and ruled out: l1_small_size 64K->16K (no effect -- separate region),
HIFI4 without fp32_dest_acc (no effect -- weight dtype is the driver), FF2-only BFP8 (still OOM).
tt_transformers' own chunked prefill cannot help: MAX_PREFILL_CHUNK_SIZE is in units of 1024, so
it never engages below 512.

fp32 ACTIVATIONS are a trap: 0.99989 at one layer and 0.644 at 26. Something in the residual or
KV path mismatches once layers chain, and it looks perfect if you only test shallow. Left as bf16.

PCC is not the acceptance test anyway -- Block 2 emits 37 INTEGER codes, so what matters is whether
the device hidden state yields the same codes, and ultimately the WER of the decoded audio. See
tests/ and STATUS.md.
"""

import os

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT

BACKBONE_DIM = 3072
N_LAYERS = 26
# tt_transformers' prefill asserts seqlen % 128 == 0 (attention.py:889), but that is NOT
# sufficient: prefill also shards the sequence across the core grid, and each shard must be
# tile-aligned. With 8 cores, a padded length of 384 gives 48 rows per shard and fails with
# "Physical shard shape (48, 128) must be tile {32, 32} sized!". 256 is the smallest multiple that
# always divides into 32-row shards (256/8=32, 512/8=64, 768/8=96, ...), so pad to that.
PREFILL_MULTIPLE = 256
# Decode pads the batch dimension to 32 (model.py:prepare_decode_inputs_host).
DECODE_BATCH_PAD = 32


def precision_override(args):
    """BF16 weights except FF1_FF3, plus HIFI4_FP32 math.

    FF1_FF3 stays BFP8 to fit L1 at padded prefill length 512; see the module docstring's
    precision table for the measured cost (0.000134 PCC) and the alternatives that were ruled out.
    """
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    precision = {
        g: PrecisionSetting.BF16
        for g in (TensorGroup.FF2, TensorGroup.WQKV, TensorGroup.WO, TensorGroup.KV_CACHE)
    }
    precision[TensorGroup.FF1_FF3] = PrecisionSetting.BFP8
    conf = ModelOptimizations(
        {
            "TensorPrecision": precision,
            "OpFidelity": {g: MathFidelitySetting.HIFI4_FP32 for g in OpGroup},
        }
    )
    inst = DecodersPrecision(args.n_layers, args.model_name)
    for i in range(args.n_layers):
        inst.set_decoder_conf(i, conf)
    return inst


class TtVoxtralBackbone:
    """26-layer AR backbone. prefill(embeds) -> hidden; step(embed) -> hidden, sharing a KV cache."""

    def __init__(self, mesh_device, hf_dir=None, max_seq_len=1024, dtype=ttnn.bfloat16):
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        hf_dir = hf_dir or os.environ.get("HF_MODEL")
        if not hf_dir:
            raise ValueError(
                "Block 1 needs an HF-format export; pass hf_dir= or set HF_MODEL. "
                "Build one with scripts/export_backbone_hf.py."
            )
        os.environ["HF_MODEL"] = os.path.abspath(hf_dir)

        self.mesh_device = mesh_device
        self.dtype = dtype
        self.pos = 0  # next KV-cache slot; prefill sets it, step advances it

        # use_hf_rope stays FALSE (the default): the export applies the meta->HF permute, so
        # tt_transformers' convert_hf_to_meta un-permute returns the original meta weights, which
        # its interleaved RotarySetup then matches. Verified: permuted+False scores 0.892 where
        # unpermuted scores 0.228. Flipping this without re-exporting is silently wrong.
        #
        # ModelArgs.__init__ builds a tokenizer unconditionally (model_config.py:673) and raises for
        # any model it does not recognise. We have no HF tokenizer and need none -- the real one is
        # tekken and it runs on the HOST, upstream of this model, because our input is embeddings.
        # Stub it rather than ship a wrong tokenizer.json a future reader might trust.
        # (dummy_weights=True would also skip it, but loads RANDOM weights.)
        orig_create = ModelArgs.create_tokenizer
        ModelArgs.create_tokenizer = lambda _self: None
        try:
            self.args = ModelArgs(
                mesh_device,
                instruct=False,
                max_seq_len=max_seq_len,
                max_batch_size=1,
                cache_hf=True,
                optimizations=precision_override,
            )
        finally:
            ModelArgs.create_tokenizer = orig_create

        self.model = Transformer(
            args=self.args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=self.args.load_state_dict(),
            weight_cache_path=self.args.weight_cache_path(dtype),
        )

        # DECODE mode's forward() unconditionally runs `norm` then `lm_head` (model.py:84-94) --
        # only the PREFILL + get_last_token=-1 path returns early. Our export has no lm_head, so
        # replace it with identity: forward() then returns the POST-NORM hidden state, which is
        # exactly what Block 2 wants. Consequence: `step()` must NOT apply _final_norm again,
        # while `prefill()` still must, because it returns before the norm. The two paths differ.
        self.model.lm_head = lambda x: x

    def _final_norm(self, out, mode):
        """The layer stack returns pre-norm hidden states; apply the model's final RMSNorm but NOT
        lm_head, which the export omits. Block 2 wants the post-norm state
        (voxtral_backbone_ref.py:157)."""
        return self.model.norm(
            out, mode=mode, norm_config=self.args.get_norm_config("lm_head", mode, None)
        )

    # ----------------------------------------------------------------------------------
    # Prefill: whole prompt at once, and it populates the KV cache for decode
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, embeds):
        """embeds torch [1,T,3072] -> hidden torch [1,T,3072] (post-final-norm)."""
        from models.tt_transformers.tt.common import Mode

        T = embeds.shape[1]
        # Prefill asserts seqlen % 128 == 0 (attention.py:889), so pad and trim here rather than
        # push it onto callers. Zeros are safe: attention is causal, so real positions never attend
        # to the padding, and the padded rows are discarded below.
        pad = (-T) % PREFILL_MULTIPLE
        if pad:
            embeds = torch.cat([embeds, embeds.new_zeros(1, pad, BACKBONE_DIM)], dim=1)
        Tp = T + pad
        x = ttnn.from_torch(
            embeds.reshape(1, 1, Tp, BACKBONE_DIM).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        # Prefill RoPE is a slice of the precomputed tables, as prepare_inputs_prefill does
        # (model.py:384). We cannot call that helper: it takes TOKEN IDS and runs the embedding we
        # are deliberately bypassing.
        rs = self.model.rope_setup
        rot_mats = [rs.cos_matrix_prefill[:, :, 0:Tp, :], rs.sin_matrix_prefill[:, :, 0:Tp, :]]
        out = self.model.forward(
            x, current_pos=None, rot_mats_global=rot_mats, mode=Mode.PREFILL, get_last_token=-1
        )
        out = self._final_norm(out, Mode.PREFILL)
        # The KV cache now holds the PADDED length; decode must continue from the real one, or the
        # first generated frame would attend to zero padding.
        self.pos = T
        return ttnn.to_torch(out).float().reshape(1, Tp, BACKBONE_DIM)[:, :T]

    def prefill_last(self, embeds):
        """[1,P,3072] -> hidden of the LAST position [1,1,3072]. Same name as `ttnn_voxtral_gpt`'s,
        so the pipeline can run either backbone without branching."""
        return self.prefill(embeds)[:, -1:]

    # ----------------------------------------------------------------------------------
    # Decode: one frame at a time, against the KV cache prefill built
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, embed):
        """embed torch [1,1,3072] (one frame) -> hidden torch [1,3072]. Advances self.pos."""
        from models.tt_transformers.tt.common import Mode

        # `use_qk_fused` is on (it is `not multimodal and not use_hf_rope`, model_config.py:679),
        # and get_rot_idxs internally does position_idxs.repeat(2) so Q and K can share the indices
        # (rope.py:886). batch_size_per_device_group is therefore 2 while max_batch_size is 1, and
        # the positions tensor must be length 1 -- passing 2 or 32 asserts.
        x_t = torch.zeros(1, 1, DECODE_BATCH_PAD, BACKBONE_DIM)
        x_t[0, 0, 0] = embed.reshape(-1)
        x = ttnn.from_torch(
            x_t.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )
        pos = torch.tensor([self.pos], dtype=torch.int64)
        cur = ttnn.from_torch(pos, dtype=ttnn.int32, device=self.mesh_device)
        rot_idxs = self.model.rope_setup.get_rot_idxs(pos, on_host=False)
        rot_mats = self.model.rope_setup.get_rot_mats(rot_idxs)
        # No _final_norm here: with lm_head stubbed to identity, forward() has already applied it.
        out = self.model.forward(
            x, current_pos=cur, rot_mats_global=rot_mats, mode=Mode.DECODE
        )
        self.pos += 1
        return ttnn.to_torch(out).float().reshape(-1, BACKBONE_DIM)[:1]


def main():
    """PCC the stack against the CPU reference on REAL weights, prefill and decode."""
    from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    dev = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        gen = TtVoxtralBackbone(dev)
        w = ref.load_backbone_state(DEFAULT_CKPT)
        torch.manual_seed(0)
        T = 128
        embeds = torch.randn(1, T, BACKBONE_DIM) * 0.02
        exp = ref.reference_forward(embeds, w)
        got = gen.prefill(embeds)
        print(f"  [prefill T={T}] PCC {pcc(got, exp):.8f}  "
              f"worst {(got - exp).abs().max().item() / exp.abs().max().item() * 100:.2f}%")

        # decode: one more frame, compared against the reference's incremental path
        nxt = torch.randn(1, 1, BACKBONE_DIM) * 0.02
        inc = ref.IncrementalBackbone(w)
        inc.prefill(embeds)
        exp_step = inc.step(nxt).reshape(1, BACKBONE_DIM)   # ref.step wants [1,1,3072]
        got_step = gen.step(nxt)
        print(f"  [decode  step ] PCC {pcc(got_step, exp_step):.8f}  "
              f"worst {(got_step - exp_step).abs().max().item() / exp_step.abs().max().item() * 100:.2f}%")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
