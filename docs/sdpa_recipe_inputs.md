# Recommended SDPA recipe inputs

SDPA recipes consume the Q/K/V tensors they are given. The op does not prepare,
round or inspect inputs, and it cannot know how earlier ops produced them.
Producing suitable inputs is the calling library's responsibility.

| Recipe | Q | K / V | Rounding before SDPA |
| --- | --- | --- | --- |
| A FAST | BF16 | BF16 | none |
| B COMPENSATED | BF16 | BF16 | none |
| C BALANCED | BF16 | BF16 | none |
| D ACCURATE | BF16 | BF16 | none |
| E LOW_PRECISION | BF16, rounded to 7 significant bits (RNE) | BF16 or BFP8 rounded to 5 significant bits (RNE); or BFP4 on its shared-exponent grid (RNE, saturating) | required for the qualified accuracy |

Significant-bit counts include the leading bit. E's LoFi matmuls consume only
those bits; unrounded inputs are truncated, which biases the result.

`ttnn.transformer.prepare_sdpa_input` implements the E rounding on device as a
standalone op. Run it where it is cheapest: after norm/RoPE and before KV
communication or caching, and once for static KV. Any producer that implements
the same rounding is equivalent. With E, `inputs_prepared=True` is the caller's
statement that this was done; SDPA does not check it.

Value domain: finite normal BF16 or zero, no RNE5/RNE7 overflow; for BFP4,
nonzero groups need a maximum exponent in [-124, 106]. Large common components
and outliers remain stress cases for low-precision KV.
