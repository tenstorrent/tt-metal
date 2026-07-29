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
        tt_ccl: TT_CCL | None = None,
        tensor_parallel_axis: int = 1,
        summary_group_chunks: int = 8,
    ) -> None: ...

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor: ...

    def reset_state(self, batch_size: int | None = None) -> None: ...

    def set_external_state(
        self,
        recurrent_state: ttnn.Tensor,
        convolution_state: ttnn.Tensor,
    ) -> None: ...
```

The package exports only `KDAConfig` and `KimiDeltaAttention`. Recurrence,
weight-loading, state-lifecycle, prefill, and tensor-parallel helpers
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

`forward` is prefill-only and accepts `T>0`. Padding to the internal 32-token
chunk boundary must not affect outputs or the final state.

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

- A 2D mesh is interpreted using `tensor_parallel_axis` (`0` or `1`); the
  sequence-parallel axis is the other axis. For a physical `(2,4)` mesh,
  `tensor_parallel_axis=1` means SP2×TP4 and `tensor_parallel_axis=0` means
  SP4×TP2. The physical mesh is not reshaped.
- Input and output are sequence-sharded over SP. Input hidden states are
  replicated over TP; output hidden states are sharded over TP.
- Weights are replicated over SP and sharded over TP. Persistent recurrent and
  convolution states are replicated over SP and head-sharded over TP.
- Whole heads are partitioned evenly over TP. Each device owns complete `[K,V]`
  states for its local heads.
- Input q/k/v, decay, beta, and output-gate projections are column parallel.
- Output projection is row parallel and reduce-scatters only along the TP axis.
- `summary_group_chunks` counts fixed 32-token KDA chunks. It must be positive,
  the local SP partition's chunk count must be divisible by it, and
  `B * local_heads * local_groups` must fit the available worker cores.
- SP requires `K == V` in this implementation. Cross-partition recurrence uses
  a logarithmic distributed affine prefix; no all-gather is used in production.
- SP supports prefill only.
- Collective topology/configuration belongs to the caller/model integration;
  this layer accepts the configured CCL handle rather than creating fabric.

## Errors

- Reject nonpositive dimensions, head counts not divisible by TP, wrong hidden
  width, `K != V` with SP>1, invalid local group divisibility, insufficient
  worker cores, missing weights, and state-shape/dtype mismatch.
- Error messages include the offending logical shape and expected shape.

## Correctness gates

- Independent torch reference versus authoritative FLA recurrence.
- Scalar-over-K degeneration versus trusted GDN.
- Single-token prefill output and final-state PCC >= 0.98.
- Short and multi-chunk prefill output/final-state PCC >= 0.98.
- Segmented-prefill cache continuity PCC >= 0.98.
- Single-device and 8-device outputs agree with the same torch reference.
- Graph/trace inspection proves no host fallback in forward.

## Performance gates

- Report cold separately; optimize warm steady state only.
- Establish single-device recurrence and full-layer rooflines before tuning.
- Establish 8-device collective byte/time rooflines before claiming CCL
  utilization.
- Profile at target Kimi dimensions for representative prefill lengths.
- Aspirational targets are approximately 60% measured compute roofline and 40%
  measured CCL roofline; misses are reported as measurements, not redefined.
