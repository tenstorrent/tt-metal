# Laguna-XS-2.1 p150x2 latency sweep

Result: **PASS, 9/9 requests**. Every row generated exactly 512 output tokens at concurrency 1. The
server had no request error, serving-time compilation/retrace, or critical fault, and final health was
HTTP 200.

Hardware: two P150 Blackhole ASICs on one physical dual-P150 card on the qualification host. Profile
`p150x2`, 131,072-token context, prefix caching off.

| Requested ISL | Actual prompt | Actual total | OSL | C | TTFT | TPOT | E2EL | Decode tok/s/user | Aggregate output tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 82 | 594 | 512 | 1 | 0.206 s | 50.09 ms | 25.801 s | 19.97 | 19.844 |
| 1,024 | 1,066 | 1,578 | 512 | 1 | 2.215 s | 50.46 ms | 28.002 s | 19.82 | 18.284 |
| 2,048 | 1,939 | 2,451 | 512 | 1 | 2.569 s | 50.50 ms | 28.372 s | 19.80 | 18.046 |
| 4,096 | 4,138 | 4,650 | 512 | 1 | 8.992 s | 50.57 ms | 34.835 s | 19.77 | 14.698 |
| 8,192 | 8,234 | 8,746 | 512 | 1 | 19.720 s | 50.72 ms | 45.638 s | 19.72 | 11.219 |
| 16,384 | 16,426 | 16,938 | 512 | 1 | 46.708 s | 51.02 ms | 72.780 s | 19.60 | 7.035 |
| 32,768 | 32,810 | 33,322 | 512 | 1 | 122.162 s | 51.61 ms | 148.532 s | 19.38 | 3.447 |
| 65,536 | 65,578 | 66,090 | 512 | 1 | 359.013 s | 52.78 ms | 385.982 s | 18.95 | 1.326 |
| 130,048 | 130,090 | 130,602 | 512 | 1 | 381.719 s | 55.11 ms | 409.878 s | 18.15 | 1.249 |

## Method

- OpenAI chat endpoint, random requested ISL, OSL 512, concurrency 1, request rate infinity,
  temperature 0, ignore EOS, and seed 1234.
- Each row is one measured request with no benchmark warmup. The model entered measurement with its
  complete power-of-two prefill bucket ladder compiled during boot and its persistent decode trace
  captured. No row caused runtime compilation or retracing.
- Requested ISL and actual server-counted prompt tokens are both reported. Random token IDs were
  decoded to text, wrapped in the chat template, and tokenized again, so the lengths need not match.
- `Decode tok/s/user = 1000 / mean_tpot_ms`. Aggregate output throughput includes cold-prefill time.
- The final row used 130,090 actual prompt tokens plus 512 output tokens, within the 131,072-token
  serving cap.
- This is one sample per point: a latency curve, not a variance study. The separate prefix-cache
  qualification uses exact raw-token prompts and three repetitions per cold/hit state.

Full-precision values: [`p150x2_latency_sweep_20260821.tsv`](p150x2_latency_sweep_20260821.tsv).
