# SPDX-License-Identifier: Apache-2.0
"""Host-only useful work model for the pinned, dense Gemma4 TP4 implementation.

FLOPs count useful projection/attention arithmetic (multiply-add = 2), not tile
padding or pointwise transcendental operations. DRAM is an estimate of weights,
KV and materialized tensors, not a measured memory-controller counter.
"""
H, I, V, TP = 5376, 21504, 262144, 4
BFP4_BYTES, BFP8_BYTES = 576 / 1024, 1088 / 1024
LAYERS = ((50, 256, 16, 1024, False), (10, 512, 4, None, True))


def projection_elements(d, kv, shared, *, physical=False):
    hidden_out = 5632 if physical else H
    intermediate = 22528 if physical else I
    return (
        H * (32 * d + kv * d * (1 if shared else 2))
        + hidden_out * 32 * d
        + 2 * H * intermediate
        + hidden_out * intermediate
    )


def causal_pairs(start, end, window=None):
    # Each query at absolute position p attends min(p+1, window) keys.
    def prefix(n):
        if window is None or n <= window:
            return n * (n + 1) // 2
        return window * (window + 1) // 2 + (n - window) * window

    return prefix(end) - prefix(start)


def prefill_work(starts, ends):
    rows = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s >= 0]
    tokens = sum(e - s for s, e in rows)
    linear = sum(n * 2 * tokens * projection_elements(d, kv, shared) for n, d, kv, _, shared in LAYERS)
    attention = sum(n * 4 * 32 * d * sum(causal_pairs(s, e, window) for s, e in rows) for n, d, _, window, _ in LAYERS)
    # Serving produces one head row per submitted nonempty continuation, including
    # intermediate chunks whose logits are host-sampled/discarded by stock plugin.
    head = len(rows) * 2 * H * V
    return {
        "tokens": tokens,
        "rows": len(rows),
        "linear_flops": linear,
        "attention_flops": attention,
        "head_flops": head,
        "flops": linear + attention + head,
    }


def decode_work(positions, batch_size):
    positions = [int(p) for p in positions if int(p) >= 0]
    # Weights are TP-sharded: sum full-model elements once, not four copies.
    weights = BFP4_BYTES * (
        sum(n * projection_elements(d, kv, shared, physical=True) for n, d, kv, _, shared in LAYERS) + H * V
    )
    kv_read = sum(
        n * 2 * kv * d * BFP8_BYTES * sum(min(p + 1, window) if window else p + 1 for p in positions)
        for n, d, kv, window, _ in LAYERS
    )
    kv_write = sum(n * 2 * kv * d * BFP8_BYTES * len(positions) for n, d, kv, _, _ in LAYERS)
    # Norm weights are BF16 replicated on four chips. Q/K norms are headwise.
    norms = TP * 2 * sum(n * (4 * H + 2 * d) for n, d, _, _, _ in LAYERS)
    # Logical batches <=32 execute one 32-row tile. Residual DRAM staging and
    # collective workspaces: estimated two read/write BF16 residuals per layer;
    # output logits: BF16 write/read plus head partial writes (three transfers).
    physical_rows = ((int(batch_size) + 31) // 32) * 32
    residual = 60 * TP * 4 * physical_rows * H * 2
    logits = 3 * physical_rows * V * 2
    embedding = len(positions) * H * 2
    parts = {
        "weight_bytes": weights,
        "kv_read_bytes": kv_read,
        "kv_write_bytes": kv_write,
        "norm_bytes": norms,
        "residual_staging_bytes": residual,
        "logit_bytes": logits,
        "embedding_bytes": embedding,
    }
    return {"active_tokens": len(positions), "physical_rows": physical_rows, **parts, "dram_bytes": sum(parts.values())}


ASSUMPTIONS = {
    "architecture": "60 dense layers: 50 sliding_attention (32 Q,16 KV,dim256,window1024), 10 full_attention (32 Q,4 KV,dim512,K=V shared projection); hidden5376, intermediate21504, vocab262144; TP4",
    "prefill": "Useful dense projection and causal/sliding QK+AV multiply-adds only; one LM head row per nonempty adapter prefill row. Padding, softmax, norm, GELU and rotary arithmetic omitted from useful-FLOP numerator; all remain in wall-time denominator.",
    "weights": "BFP4_B 576bytes/1024elements including shared exponents. Physical MLP intermediate22528 and output5632 padding from multichip_decoder.py; full TP-sharded weights read once per decode replay, not once per user. LoFi all projection groups.",
    "kv": "BFP8_B 1088bytes/1024elements; estimated one K and V read per attended token and write per active row; global K=V projection shared but attention consumes both K and V. Cache reuse, SDPA tile rounding and repeated reads are not measured.",
    "other_dram": "Includes replicated BF16 norm weights, approximate residual/collective staging, embedding and logits; one physical32-row tile for all supported decode batches. L1 reuse and exact temporary/CCL traffic unknown. This is modeled DRAM traffic, not a hardware counter; no claim of exact utilization.",
    "sources": [
        "tt/model.py:Gemma4Model",
        "tt/multichip_decoder.py:from_state_dict,_finish,_decode_attention_forward",
        "tt/generator.py:prefill_forward",
        "doc/datatype_sweep/selected_precision_config.json",
        "doc/optimized_decoder/peak_basis.json",
    ],
}
