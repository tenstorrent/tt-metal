# OLMo decode determinism debugging kit (2026-10-14)

`debug_probes.patch` (apply on top of 256d9b8 + the repeat-step demo flags) adds env-gated probes that were
used to localize run-to-run nondeterminism of batch-32 decode. Not merged into the model code: they read tensors
back inside forward (eager only) and are slow.

| env (needs OLMO_DECODE_TRACE=0 for the probes) | what |
|---|---|
| `OLMO_OP_PROBE=<path>` | md5 + per-row sums of every ttnn op output of the first eager decode step (4692 ops); diff two runs with `compare_op_probes.py` |
| `OLMO_LAYER_PROBE=1 OLMO_LAYER_PROBE_OUT=<path> [OLMO_LAYER_PROBE_STAGES=attn_out,ff_out,mlp.*] [OLMO_PROBE_SAVE_DEVS=22]` | same per decoder stage / MLP intermediate; `compare_layer_probes.py` |
| `OLMO_HAMMER_W1=K` | rerun the layer-0 w1 ring matmul K times on identical inputs and count per-device output flips |
| `OLMO_WEIGHT_READ_TEST=N` | read layer-0 w1/w3 back from devices N times |
| `OLMO_SYNC_AT=attn,ff,layer` | synchronize the device at those points of each decode layer |
| `OLMO_RING_SWAP="6,6:3,6;6,7:3,7"` | move ring slots to other cores (breaks the model: other code relies on the ring placement) |
| `OLMO_DUMP_KV=<path>` | md5 of the KV blocks each user owns after prefill |

Flags that stay in `text_olmo_demo.py`: `OLMO_DECODE_TRACE=0`, `OLMO_MAX_NEW=n`, `OLMO_HOST_SAMPLING=1`,
`OLMO_DUMP_LOGITS=<path>`, `OLMO_REPEAT_STEP=1` (replay one decode step N times; any logits change between replays
is nondeterminism), `OLMO_DUMP_TOKENS=<path>` (+ `scripts/compare_decode_tokens.py`).

Finding: see bringup/olmo3/LOG.md.
