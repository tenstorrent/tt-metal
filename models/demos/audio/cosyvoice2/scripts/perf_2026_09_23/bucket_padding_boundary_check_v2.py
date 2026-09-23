"""v2 of bucket_padding_boundary_check.py -- rebuilt after finding real upstream source
(fetched directly from github.com/FunAudioLLM/CosyVoice, network access confirmed
available; `cosyvoice2.yaml` fetched from the real HF repo for the real config values).
v1 is kept for the record, not deleted -- see its own docstring for the (still valid)
naive-vs-lookahead-aware padding-content finding it established, and why THIS script
supersedes its masking scheme specifically.

**What v1 got wrong**: v1's mask was a single (1,1,1,B) row, identical for every query
position -- "every query can see keys [0,true_len), nothing beyond." Real upstream
(`cosyvoice/utils/mask.py`'s `subsequent_chunk_mask`, transcribed verbatim below) is a
full (L,L) matrix where EACH query's valid-key window depends on WHICH CHUNK THAT QUERY
ITSELF belongs to: query i sees keys `[0, (i//chunk_size + 1) * chunk_size)` -- an EARLY
query does NOT get to see the CURRENT total prefix length, only up through the end of its
OWN chunk. This is what makes recompute-the-whole-growing-prefix-every-chunk valid in the
first place: an early position's final output becomes STABLE once its own chunk is
complete (block_value(i) doesn't depend on the total sequence length T at all), so
appending more tokens later never changes it retroactively. v1's simplified mask does NOT
have this property (an early query's visible context DOES depend on the current total
prefix length), so it was not a faithful chunk-causal reference, even though it correctly
isolated the padding-CONTENT question (naive zero vs. real lookahead tokens) since that
comparison used the SAME (flawed) mask in both variants -- see v1's own results, still
valid as a padding-content-only comparison, just not as a "real chunk-causal" one.

Real config, confirmed from the real checkpoint's own `cosyvoice2.yaml` (fetched from
`FunAudioLLM/CosyVoice2-0.5B` directly): `chunk_size: 25` (token-rate), `token_mel_ratio:
2` (so up-rate stage's chunk size is 50), `num_decoding_left_chunks: -1` (unlimited left
context -- "whole growing prefix"), `pre_lookahead_len: 3`. Matches this repo's own
`tt/flow/encoder.py` constants exactly (D_MODEL=512, ATTENTION_HEADS=8, NUM_BLOCKS=6,
macaron_style=False, use_cnn_module=False).

This script reproduces upstream's OWN self-consistency test (found at the bottom of the
real `cosyvoice/flow/flow.py`, a `__main__`-style block comparing a full `finalize=True`
run against incremental `finalize=False` chunk calls) rather than inventing a new
methodology: build ONE "final" F-token sequence, compute `full_out` with real
`streaming=True`/chunk-causal masking over the whole thing (matches `finalize=True`); then
compute `chunk_out` for a SHORTER T-token prefix using the SAME masking scheme plus the
real next `pre_lookahead_len` tokens as separate lookahead context (matches
`finalize=False`, `token, context = token[:, :-pre_lookahead_len], token[:,
-pre_lookahead_len:]`); compare `chunk_out` against `full_out`'s corresponding prefix.
Still does not modify `tt/flow/encoder.py` or add production bucketing/streaming code --
external script only, driving the encoder's existing public sub-modules directly.

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 600 /opt/venv/bin/python bucket_padding_boundary_check_v2.py
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/home/user/tt-metal")

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.flow.encoder import (
    D_MODEL,
    PRE_LOOKAHEAD_LEN,
    TtUpsampleConformerEncoder,
    UpsampleConformerEncoderRef,
)
from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtSmallEmbedding
from models.common.utility_functions import comp_pcc

VOCAB_SIZE = 6561
CHUNK_SIZE = 25  # real cosyvoice2.yaml chunk_size
UPSAMPLE_STRIDE = 2  # real cosyvoice2.yaml token_mel_ratio
CHUNK_SIZE_UP = CHUNK_SIZE * UPSAMPLE_STRIDE  # real cosyvoice2.yaml: static_chunk_size * token_mel_ratio
BATCH = 1
NEG = -30000.0


def subsequent_chunk_mask(size: int, chunk_size: int) -> torch.Tensor:
    """Verbatim transcription of the real `cosyvoice/utils/mask.py::subsequent_chunk_mask`
    (fetched from github.com/FunAudioLLM/CosyVoice), `num_left_chunks=-1` (unlimited left
    context) baked in -- matches the real checkpoint's own `num_decoding_left_chunks: -1`,
    and matches the only behavior the real (non-deprecated) upstream function supports."""
    pos_idx = torch.arange(size)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode="trunc") + 1) * chunk_size
    return pos_idx.unsqueeze(0) < block_value.unsqueeze(1)  # True = valid


def chunk_bias(size: int, chunk_size: int) -> torch.Tensor:
    valid = subsequent_chunk_mask(size, chunk_size)  # [size, size] bool
    bias = torch.zeros(1, 1, size, size)
    bias[:, :, ~valid] = NEG
    return bias


print("=== loading real flow checkpoint, building real encoder + input_embedding ===")
flow_sd = load_checkpoint_file("flow.pt")
encoder_sd = sub_state_dict(flow_sd, "encoder.")
encoder_ref = UpsampleConformerEncoderRef.from_checkpoint(encoder_sd)
encoder_ref.eval()
input_embedding_weight = flow_sd["input_embedding.weight"]

print("=== opening device ===")
device = ttnn.CreateDevice(0, l1_small_size=65536)

try:
    tt_encoder = TtUpsampleConformerEncoder(device, encoder_ref, dtype=ttnn.bfloat16, use_trace=False)
    tt_embedding = TtSmallEmbedding(device, input_embedding_weight, dtype=ttnn.bfloat16)

    torch.manual_seed(0)
    F_LEN = 256  # the "eventual final" sequence length
    T_TRUE = 200  # a chunk boundary partway through it -- multiple of CHUNK_SIZE=25 (8 chunks)
    assert T_TRUE % CHUNK_SIZE == 0
    tokens = torch.randint(0, VOCAB_SIZE, (1, F_LEN), dtype=torch.int64)

    def embed(token_slice: torch.Tensor):
        ids_dev = ttnn.from_torch(
            token_slice.reshape(1, 1, 1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        return tt_embedding(ids_dev)

    def run_encoder(xs_dev, length: int):
        """Replicates TtUpsampleConformerEncoder._call_eager's body (no production code
        touched), using the REAL chunk-causal (L,L) bias instead of `_call_eager`'s
        always-all-valid one."""
        bias1 = ttnn.from_torch(chunk_bias(length, CHUNK_SIZE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        h = tt_encoder.embed(xs_dev)
        pos_emb = tt_encoder._pos_emb(length)
        h = tt_encoder.pre_lookahead_layer(h, length, BATCH)
        for layer in tt_encoder.encoders:
            h = layer(h, pos_emb, bias1)

        h = tt_encoder.up_layer(h, length, BATCH)
        length2 = length * UPSAMPLE_STRIDE
        bias2 = ttnn.from_torch(chunk_bias(length2, CHUNK_SIZE_UP), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        h = tt_encoder.up_embed(h)
        pos_emb2 = tt_encoder._pos_emb(length2)
        for layer in tt_encoder.up_encoders:
            h = layer(h, pos_emb2, bias2)

        out = ttnn.layer_norm(h, weight=tt_encoder.after_norm_w, bias=tt_encoder.after_norm_b, epsilon=1e-5)
        return ttnn.to_torch(out).float().reshape(1, length2, D_MODEL)

    print(f"\n=== ground truth: real chunk-causal masked encoder, FULL F={F_LEN} (matches finalize=True) ===")
    xs_full = embed(tokens[:, :F_LEN])
    full_out = run_encoder(xs_full, F_LEN)
    T2 = T_TRUE * 2
    ref_out = full_out[:, :T2, :]

    print(f"=== variant A: chunk-causal mask, NAIVE zero-lookahead (matches finalize=True truncated at T={T_TRUE}) ===")
    xs_a = embed(tokens[:, :T_TRUE])
    a_out = run_encoder(xs_a, T_TRUE)

    print(f"=== variant B: chunk-causal mask, REAL {PRE_LOOKAHEAD_LEN}-token lookahead fed to pre_lookahead_layer only (matches finalize=False) ===")
    # Real upstream: token, context = token[:, :-pre_lookahead_len], token[:, -pre_lookahead_len:];
    # pre_lookahead_layer(inputs=embed(token), context=embed(context)) -- context concatenated
    # ONLY inside pre_lookahead_layer's own conv1 call, never part of `xs` for attention.
    # TtPreLookaheadLayer doesn't expose a separate context argument (matches this port's
    # "streaming=False only" scope) -- replicate the same effect by calling conv1 directly
    # with the extended (main+lookahead) tensor, matching PreLookaheadLayerRef.forward's own
    # unpadded-right-context branch (`context.size(2) != 0`) mathematically: conv1 sees
    # [main_tokens, lookahead_tokens] with NO extra right-zero-pad (lookahead already fills
    # the kernel's receptive field), conv2 stays causal as always, then only the first
    # main_len positions of pre_lookahead_layer's output feed the rest of the encoder.
    xs_main = embed(tokens[:, :T_TRUE])
    xs_lookahead = embed(tokens[:, T_TRUE : T_TRUE + PRE_LOOKAHEAD_LEN])
    xs_extended = ttnn.concat([xs_main, xs_lookahead], dim=1)  # [1, T_TRUE+3, D_MODEL]

    def pre_lookahead_with_real_context(xs_extended, main_len: int):
        pl = tt_encoder.pre_lookahead_layer
        h = pl.conv1(xs_extended, main_len + PRE_LOOKAHEAD_LEN, BATCH)  # kernel=4, no extra pad needed: exactly covers [main_len+3]
        h = ttnn.leaky_relu(h, negative_slope=0.01)
        h = pl.conv2(h, main_len + PRE_LOOKAHEAD_LEN, BATCH)
        out = ttnn.add(h, xs_extended)
        return out[:, :main_len, :]  # slice back to main_len -- context's own output position is discarded, matching upstream

    bias1_b = ttnn.from_torch(chunk_bias(T_TRUE, CHUNK_SIZE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    h_embed = tt_encoder.embed(xs_main)
    h_embed_ext = tt_encoder.embed(xs_extended)
    h = pre_lookahead_with_real_context(h_embed_ext, T_TRUE)
    pos_emb = tt_encoder._pos_emb(T_TRUE)
    for layer in tt_encoder.encoders:
        h = layer(h, pos_emb, bias1_b)
    h = tt_encoder.up_layer(h, T_TRUE, BATCH)
    T2_up = T_TRUE * UPSAMPLE_STRIDE
    bias2_b = ttnn.from_torch(chunk_bias(T2_up, CHUNK_SIZE_UP), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    h = tt_encoder.up_embed(h)
    pos_emb2 = tt_encoder._pos_emb(T2_up)
    for layer in tt_encoder.up_encoders:
        h = layer(h, pos_emb2, bias2_b)
    b_out = ttnn.to_torch(
        ttnn.layer_norm(h, weight=tt_encoder.after_norm_w, bias=tt_encoder.after_norm_b, epsilon=1e-5)
    ).float().reshape(1, T2_up, D_MODEL)

    def report(name: str, out: torch.Tensor, n_boundary: int = 6):
        p_all, r_all = comp_pcc(ref_out, out, 0.99)
        boundary_ref = ref_out[:, T2 - n_boundary : T2, :]
        boundary_out = out[:, T2 - n_boundary : T2, :]
        p_b, r_b = comp_pcc(boundary_ref, boundary_out, 0.99)
        max_diff_all = float((ref_out - out).abs().max())
        max_diff_boundary = float((boundary_ref - boundary_out).abs().max())
        max_diff_interior = float((ref_out[:, : T2 - n_boundary, :] - out[:, : T2 - n_boundary, :]).abs().max())
        print(
            f"{name}: overall PCC={r_all:.6f} max|diff|={max_diff_all:.4f}  |  "
            f"last {n_boundary} positions PCC={r_b:.6f} max|diff|={max_diff_boundary:.4f}  |  "
            f"interior (excl. last {n_boundary}) max|diff|={max_diff_interior:.4f}"
        )
        return r_all, r_b, max_diff_boundary, max_diff_interior

    print("\n=== RESULTS (real chunk-causal mask throughout, chunk_size=25/50) ===")
    r_a = report("variant A (chunk-causal, naive zero-lookahead)     ", a_out)
    r_b = report("variant B (chunk-causal, real lookahead context)   ", b_out)

    print("\n=== VERDICT ===")
    print(f"variant A interior max|diff|={r_a[3]:.4f}  (should be ~0: interior is identical construction to v1's, chunk-causal mask should make interior EXACT if the mask itself is correctly applied)")
    print(f"variant B interior max|diff|={r_b[3]:.4f}")
    print(f"variant A boundary PCC={r_a[1]:.6f}  variant B boundary PCC={r_b[1]:.6f}")
    if r_a[1] < 0.99 and r_b[1] > r_a[1] and r_b[3] < 0.01:
        print(
            "CONFIRMED under the REAL chunk-causal mask: naive zero-lookahead still corrupts the "
            "boundary, real lookahead context recovers it, AND (unlike v1) the INTERIOR now matches "
            "near-exactly (not just closer) -- confirming the real subsequent_chunk_mask makes early "
            "positions' output stable regardless of total sequence length, exactly the property "
            "bucketing depends on."
        )
    else:
        print("Needs a closer look -- see the raw numbers above before drawing a conclusion.")
finally:
    ttnn.CloseDevice(device)
