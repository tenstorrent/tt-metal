# Kimi Delta Attention API specification

Status: design gate. This is the contract to implement; it contains no
checkpoint-specific dependency and supports random initialization in tests.

## Public surface

```python
@dataclass(frozen=True)
class KDAConfig:
    hidden_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    recurrent_state_dtype: ttnn.DataType = ttnn.float32
    chunk_size: int = 64

    @property
    def q_dim(self) -> int: ...

    @property
    def k_dim(self) -> int: ...

    @property
    def v_dim(self) -> int: ...

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> "KDAConfig": ...


class KimiDeltaAttention:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice | ttnn.Device,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor],
        tensor_cache_path: Path | None = None,
    ) -> None: ...

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        mode: Literal["recurrent", "chunk"] = "recurrent",
        chunk_size: int | None = None,
        valid_len: int | None = None,
    ) -> ttnn.Tensor: ...

    def reset_state(self, batch_size: int | None = None) -> None: ...

    def set_external_state(
        self,
        recurrent_state: ttnn.Tensor,
        convolution_state: ttnn.Tensor,
    ) -> None: ...
```

The package exports only `KDAConfig` and `KimiDeltaAttention`. Recurrence,
weight-loading, state-lifecycle, decode, prefill, and tensor-parallel helpers
remain private implementation details.

## Tensor contract

| Value | Logical shape | Ownership |
|---|---:|---|
| hidden input/output | `[B,T,2304]` | caller |
| q/k/v | `[B,T,H,128]` | layer, local heads on mesh |
| decay gate `g` | `[B,T,H,128]` | layer, log space |
| write gate `beta` | `[B,T,H]` | layer |
| recurrent state | `[B,H,128,128]` | persistent layer/cache |
| fused conv state | `[B,3,3*H*128]` | persistent layer/cache |

`H=32` globally and `H=32/tensor_parallel_size` locally. General dimensions
come from `KDAConfig`; target-shape assertions use the values above.

## Weight contract

Weights use the canonical Hugging Face names below. PyTorch source layout is
`[out_features, in_features]`; the loader owns any transpose and device/mesh
mapping.

- `q_proj.weight`, `k_proj.weight`, `v_proj.weight`
- `q_conv1d.weight`, `k_conv1d.weight`, `v_conv1d.weight`
- `A_log`, `f_a_proj.weight`, `f_b_proj.weight`, `dt_bias`
- `b_proj.weight`
- `g_a_proj.weight`, `g_b_proj.weight`
- `o_norm.weight`, `o_proj.weight`

The constructor accepts a layer-local mapping: callers strip the model/layer
prefix. Tests create this mapping from deterministic random tensors; production
checkpoint loading is out of scope for initial bringup.

## Forward semantics

1. Project hidden states independently to q, k, and v.
2. Apply an independent causal depthwise convolution plus SiLU to each stream.
3. Compute decay projection `f_b(f_a(x))`, then
   `g = -exp(A_log) * softplus(raw_g + dt_bias)`.
4. Compute `beta = sigmoid(b_proj(x))`.
5. L2-normalize q/k and apply KDA recurrence:
   `S = exp(g) * S + beta * k outer (v - k^T S)` and
   `o = q^T S / sqrt(K)`.
6. Compute output gate `z = g_b(g_a(x))`, apply sigmoid-gated RMSNorm, flatten
   heads, and project to hidden size.
7. Update convolution and recurrent state in place when external state is set;
   otherwise replace internal state after eager execution.

`recurrent` requires `T=1`. `chunk` requires `T>0`; padding to a supported
chunk boundary must not affect outputs through `valid_len` or the final state.

## State and trace invariants

- Decay precedes the state read; query observes the updated state.
- Splitting one sequence across calls is output/state equivalent to processing
  it in one call, within the declared PCC tolerance.
- `reset_state(B)` allocates zero state for B. `reset_state()` releases logical
  ownership and forces explicit reinitialization before the next forward.
- External state shapes are validated before any device work.
- Trace mode preserves buffer addresses and performs no lazy allocation.
- No `ttnn.to_torch`, torch operation, implicit fallback, or host
  synchronization exists in the production forward path.

## Distribution contract

- Whole heads are partitioned evenly across devices. Each device owns complete
  `[K,V]` states for its local heads; recurrence has no collective.
- Input q/k/v, decay, beta, and output-gate projections are column parallel.
- Output projection is row parallel and returns the caller's expected hidden
  sharding via reduce-scatter (or all-reduce when the caller requires
  replication).
- Collective topology/configuration belongs to the caller/model integration;
  this layer accepts the configured CCL handle rather than creating fabric.

## Errors

- Reject unsupported mode, nonpositive dimensions, head counts not divisible
  by mesh size, wrong hidden width, `T != 1` in recurrent mode, invalid
  `valid_len`, missing weights, and state-shape/dtype mismatch.
- Error messages include the offending logical shape and expected shape.

## Correctness gates

- Independent torch reference versus authoritative FLA recurrence.
- Scalar-over-K degeneration versus trusted GDN.
- Single-token output and final-state PCC >= 0.98.
- Short and multi-chunk prefill output/final-state PCC >= 0.98.
- Prefill-to-decode cache continuity PCC >= 0.98.
- Single-device and 8-device outputs agree with the same torch reference.
- Graph/trace inspection proves no host fallback in forward.

## Performance gates

- Report cold separately; optimize warm steady state only.
- Establish single-device recurrence and full-layer rooflines before tuning.
- Establish 8-device collective byte/time rooflines before claiming CCL
  utilization.
- Profile at target Kimi dimensions for decode and representative prefill
  lengths. The composed recurrence is an oracle, never the final perf path.
- Aspirational targets are approximately 60% measured compute roofline and 40%
  measured CCL roofline; misses are reported as measurements, not redefined.
