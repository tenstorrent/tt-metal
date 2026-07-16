# Stage 09 vLLM context capacity audit

## Verified limit

The final vLLM limit on the tested four-P150b `1x4` mesh is `113280` total
tokens. `readiness_vllm/max_context_prompt_check.json` records a successful
HTTP completion with `113279` actual input tokens and one generated token. The
response contains a real choice and usage total of `113280`; the request gate
rejects HTTP-200 error envelopes and missing choices.

This reduction applies only to the full-depth vLLM HMA serving allocation.
Standalone full-model capability remains `262144`.

## Physical model

Gemma 4 31B has fifty sliding-attention layers and ten global-attention layers.
Six logical HMA groups share ten physical K/V pairs (twenty buffers). For total
context `C`, the source-audited allocation model is:

```text
B(C) = 5 * (ceil(C / 64) + 1) + ceil(C / 128)
P(C) = 2178911936 - 174080 * B(C)
T(C) = 60 * 4 * ceil((4 * ceil(C / 64)) / 32) * 32
A(C) = 4096 * C + 4032 * 4096
M(C) = 3072 * C + 5376 * 4096
```

`B` is the shared physical pool-block count, `P` is the conservative post-KV
largest contiguous bytes per bank, `T` is the sixty physically aligned page
tables, `A` is the fused streamed-attention mandatory peak, and `M` is the
streamed-MLP peak. `A` is larger than `M` at the selected boundary.

| Candidate | Pool blocks | Post-KV bytes/bank | Mandatory attention peak | Page tables | Margin/shortfall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 113280 | 9740 | 483372736 | 480509952 | 1704960 | +1157824 |
| 113344 | 9746 | 482328256 | 480772096 | 1704960 | -148800 |

`113344` is therefore source-proven physically infeasible by `148800`
bytes/bank. `113280` is both the largest source-feasible 64-token-aligned
candidate and hardware verified with a maximum-total-length request.

## Failure and repair sequence

- `262144` (`22533` blocks) failed full-depth HMA KV allocation.
- `157696` (`13557` blocks) failed on physical K/V buffer 19 of 20.
- Real maximum-prompt attempts at `101888` and `100800` exposed prompt-sized
  head-concat and attention all-reduce lifetimes; the checker was also fixed to
  reject embedded error envelopes.
- Later `108672`/`111488` probes isolated sliding/full attention concat,
  post-attention norm/residual, MLP residual, and global HMA cache read-geometry
  constraints.
- Autofix streams full and sliding SDPA/head concat, attention output
  projection and BF16 all-reduce, post-attention norm/residual addition, and
  long-prompt MLP residual work. It releases dead normalization tensors and
  reads global HMA cache through a zero-copy layer-geometry view.
- `111488` passed as the conservative post-fix control. The source ceiling
  `113280` then passed; adjacent `113344` fails the source capacity inequality
  without requiring a hardware allocation attempt.

## Evidence

- `../context_contract.json`: final advertised and verified limit.
- `../../readiness_vllm/max_context_prompt_check.json`: direct `113279+1`
  completion.
- `evidence/full_113280_source_ceiling_max_context_passing_server.log` and
  `evidence/full_113280_source_ceiling_max_context_passing_response.json`:
  independent capacity-only pass.
- `evidence/final_full_113280_vllm_run.log`: final shared-run transcript.
- `evidence/full_111488_full_layer_streaming_max_context_passing_server.log`:
  conservative control.
- `evidence/full_262144_capacity_failed_server.log` and
  `evidence/full_157696_capacity_failed_server.log`: original physical HMA
  failures.

`evidence/context_capacity_audit.json` is the machine-readable form of the same
final derivation. The advertised limit is authoritative in
`doc/context_contract.json`.
