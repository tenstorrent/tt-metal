# Prefill

The server prefills a prompt on the same device state the decode traces use. In every form the last prompt token is
teacher-forced through the decode traces, so decode, the MTP hand-off and the prompt-end snapshot all start from the
same state: the fp32 GDN recurrent state and conv history, the QSA KV and compressed caches, the staging tile and
raw-key ring, the PLE history and the device position.

## Forms

| form | flag | what runs |
|---|---|---|
| chunked (default) | `--prefill-mode chunked` | 128-row chunks where the prompt allows, then 32-row chunks and a padded tail; fewer than 16 rows after the alignment steps are teacher-forced |
| teacher-forced | `--prefill-mode teacher_forced` | every prompt token through the decode step |
| slab | `--prefill-slab ROWS` | slabs of ROWS rows (a multiple of 512 up to 2048) ahead of the 128-row chunks |

`--long-chunks` is accepted for compatibility: the 128-row chunks are the default. `--prefill-slab` implies it and
combines with `--mtp`.

## The slab

Inside a slab every layer runs all ROWS rows in one pass: the dense linears as one matmul over the rows, the GDN
recurrence as main's `ttnn.transformer.chunk_gated_delta_rule` over the whole slab, the routed experts on
`moe_compute`'s local output path, and the QSA layer's KV writes, block selection and sparse attention over all rows.
The PLE runs the 128-row pass per 128-row block, and so do the routed experts where `moe_compute` refuses the one
call.
With `--mtp` the MTP layer's rows run inside the slab body as ROWS / 128 slices of the 128-row form, so its KV rows
land where the 128-row chunks would write them.

At start, before any slab state:

- `moe_compute` is asked, on a probe instance of layer 0, whether it takes the slab's rows in one call. Where it
  does, each layer routes the slab's experts in one call; where it refuses, as on main, which caps one call at 128
  tokens for this model, each layer routes ROWS / 128 calls of 128 tokens (16 for a 2048-row slab) through a 128-row
  worker. The start-up log names the form and the op's reason. Packed routing
  ([#60469](https://github.com/tenstorrent/tt-metal/pull/60469), open) would allow the one call.
- The slab's resident dense-weight copies are allocated after the decode weights and the experts, and refused with
  the numbers when the free DRAM would not also hold the context state, the slab's working set and a margin. A build
  without a slab allocates no copies.
- A slab is a multiple of 512 rows up to 2048: the one-call form reduces in 512-row blocks and the probe instance
  is built in that form, so other row counts are refused before any slab work.

`/health` reports `prefill_slab_rows`.

## Numerics

The slab is tolerance class against the chunk bodies: its matmuls accumulate K in another block order than the
per-tile DRAM-sharded programs, so each linear's output can differ by one bf16 ULP of its scale. The acceptance
records are shorter than a slab except the slab record `document`; longer prompts (the reference corpus's long items)
exercise the slab body. The 128-row and 32-row chunk bodies are not bitwise alike on every tt-metal; compare their
streams on the build in use. See [NUMERICS.md](NUMERICS.md).

## Running it

```bash
models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile <profile> \
    --checkpoint <DIR> --cache-root <DIR> --prefill-slab 2048 [--acceptance --require-json-96]
```

The first start compiles the slab body's programs in the warm pass and captures its trace after the 128-row chunk
trace.
