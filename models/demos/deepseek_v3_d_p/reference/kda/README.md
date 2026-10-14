# Kimi Delta Attention reference

This package is the device-independent semantic oracle for Kimi Delta
Attention (KDA). It depends only on PyTorch and does not import TTNN, model
device code, mesh fixtures, checkpoints, or profilers.

## Provenance and formula correspondence

The formulas were transcribed from the `moonshotai/Kimi-K3` Hugging Face model
configuration and the Flash Linear Attention KDA/Gated DeltaNet formulation as
captured by tt-metal PR #51910 at prototype source revision
`ac9ebafccd9eeb1492451ee6265a9d927429834c`.

| Reference function | Semantic responsibility |
|---|---|
| `causal_depthwise_conv_reference` | Four-tap causal depthwise convolution, caller-owned three-token history, then SiLU |
| `kda_gate_reference` | Per-key negative log decay, including Kimi-K3's optional bounded sigmoid form |
| `l2_norm_reference` | FLA query/key L2 normalization with epsilon `1e-6` |
| `kda_recurrent_reference` | Token-ordered normalized delta-rule recurrence and replacement final state |
| `sigmoid_gated_rms_norm_reference` | RMS normalization followed by sigmoid output gating |
| `kda_forward_reference` | Stateless full-layer composition and replacement logical state |

Supported configurations require a four-tap convolution, positive dimensions
and RMS epsilon, equal per-head query/key dimensions, and an optional gate
lower bound in `[-5, 0)`. Reference arithmetic is accumulated in FP32.
Operation tests choose and record their own PCC threshold; full-layer reference
tests use `0.99999`, while exact state/metadata and determinism checks use
value or bit identity.
