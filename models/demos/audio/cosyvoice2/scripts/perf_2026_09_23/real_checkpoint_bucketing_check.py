"""Real-checkpoint validation of the actual `valid_length` bucketing code path (not a
synthetic/random-init structural check) -- requested explicitly before committing round 2,
given this bring-up's track record: STFT, the conv resolver, and the original
`subsequent_chunk_mask` mistake were all "structural test passes, real numeric path at a
real shape doesn't" bugs. Real `flow.pt` weights throughout, the SAME real chunk-causal
mask + real lookahead context this round already validated (round 1, v2), now run through
`TtUpsampleConformerEncoder`'s actual `valid_length=` bucketing parameter (not a hand-rolled
harness), at TWO real bucket boundaries from the decided linear-step=64 scheme -- not just
one arbitrary length.

For each (T_TRUE, B_BUCKET) pair: compare the encoder run at the EXACT length (no bucket
padding, `valid_length` defaulting to `length`) against the SAME real content padded out to
the bucket length via `valid_length=T_TRUE`, both with the real next-3-token lookahead
context. Same 0.99 PCC bar as the rest of this bring-up.

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 300 /opt/venv/bin/python real_checkpoint_bucketing_check.py
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
    bucket_length,
)
from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtSmallEmbedding
from models.common.utility_functions import comp_pcc

VOCAB_SIZE = 6561
GATE = 0.99

# Two real bucket boundaries from the decided linear-step=64 scheme, both multiples of
# CHUNK_SIZE=25 (required for the mask to correctly exclude the padding region), picked to
# land in DIFFERENT buckets -- not one arbitrary length.
CASES = [
    (150, bucket_length(150)),  # -> 192
    (325, bucket_length(325)),  # -> 384
]

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

    def embed(token_slice: torch.Tensor):
        ids_dev = ttnn.from_torch(
            token_slice.reshape(1, 1, 1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        return tt_embedding(ids_dev)

    all_passed = True
    for T_TRUE, B_BUCKET in CASES:
        assert T_TRUE % 25 == 0
        print(f"\n{'=' * 70}\nT_TRUE={T_TRUE}  B_BUCKET={B_BUCKET} (bucket_length step=64)\n{'=' * 70}")
        torch.manual_seed(T_TRUE)
        tokens = torch.randint(0, VOCAB_SIZE, (1, T_TRUE + PRE_LOOKAHEAD_LEN), dtype=torch.int64)

        context_dev = embed(tokens[:, T_TRUE : T_TRUE + PRE_LOOKAHEAD_LEN])

        # Exact-length reference: real content, no bucket padding, valid_length defaults to length.
        xs_exact = embed(tokens[:, :T_TRUE])
        exact_out = ttnn.to_torch(
            tt_encoder(xs_exact, T_TRUE, 1, context=context_dev, streaming=True)
        ).float()

        # Bucketed: same real content, zero-padded to B_BUCKET at the tensor level, real
        # valid_length=T_TRUE tells the encoder where the true content actually ends --
        # the ACTUAL production bucketing code path, not a hand-rolled harness.
        pad = torch.zeros(1, B_BUCKET - T_TRUE, D_MODEL)
        xs_bucketed = ttnn.from_torch(
            torch.cat([ttnn.to_torch(xs_exact).float(), pad], dim=1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        context_dev2 = embed(tokens[:, T_TRUE : T_TRUE + PRE_LOOKAHEAD_LEN])
        bucketed_out = ttnn.to_torch(
            tt_encoder(xs_bucketed, B_BUCKET, 1, context=context_dev2, streaming=True, valid_length=T_TRUE)
        ).float()

        T2 = T_TRUE * 2
        passed, pcc = comp_pcc(exact_out[:, :T2, :], bucketed_out[:, :T2, :], GATE)
        max_diff = float((exact_out[:, :T2, :] - bucketed_out[:, :T2, :]).abs().max())
        print(f"exact T={T_TRUE} vs bucketed B={B_BUCKET} (valid_length={T_TRUE}), REAL checkpoint weights:")
        print(f"  PCC={pcc}  max|diff|={max_diff:.6f}  {'PASS' if passed else 'FAIL'} (gate {GATE})")
        all_passed = all_passed and passed

    print(f"\n{'=' * 70}\nOVERALL: {'ALL CASES PASS' if all_passed else 'AT LEAST ONE CASE FAILED'} (gate {GATE})\n{'=' * 70}")
finally:
    ttnn.CloseDevice(device)
