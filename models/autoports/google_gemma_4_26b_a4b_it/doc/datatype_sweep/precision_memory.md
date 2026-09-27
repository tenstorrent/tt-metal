# Precision policy capacity accounting

`tests/precision_memory.py` computes per-device physical tile payloads at the unchanged **262144-token, batch-one** contract. It imports no accelerator library and opens no device. `memory_candidates.json` contains all generated policies, including every layer's local padded shapes and retained aliases.

The reconstructed baseline reconciles exactly with `doc/context_contract.json`:

| Component | Bytes/device |
| --- | ---: |
| Decoder weights and existing norm/router/position allowance | 10,986,081,280 |
| Embedding | 369,098,752 |
| LM head | 369,098,752 |
| BFP8 K/V at maximum context | 8,556,380,160 |
| Shared prefill/decode RoPE | 1,610,612,736 |
| Page table | 32,768 |
| Existing trace/activation/allocator reserve | 2,147,483,648 |
| Three long-prefill BF16 buffers | 4,429,185,024 |
| **Conservative peak bound** | **28,467,973,120** |

The shared persistent CCL payload is separately **2,690,688 bytes in L1**, excluding bank alignment/semaphores. The helper accounts for each unique role/dtype entry and does not add L1 payload to DRAM.

## Physical tensor geometry and ownership

A BF16/BFP8/BFP4 tile occupies **2048/1088/576 bytes**, including block exponents. Hidden width is 2816 (88 tiles), local TP expert width is 192 (six tiles), local shared width is 544 (17 tiles), and EP owns 32 complete experts of width 704 (22 tiles).

- TP gate/up/down contain `128*88*12` / `128*6*88` tiles per device. EP prefill gate/up/down contain `32*88*44` / `32*22*88` tiles. The model retains both expert layouts.
- Sliding QKV/output shapes are `[2816,2048]` / `[1024,2816]`; full shapes are `[2816,3072]` / `[2048,2816]`. Prefill weights remain BFP8. A decode dtype equal to BFP8 aliases that copy; another dtype retains an additional matrix.
- Sliding TP gate reload explicitly aliases the unused prefill gate to the new decode weight. Full TP gate recovery to BFP8/BF16 retains an extra unused BFP4 prefill gate. Every TP down recovery from BFP4 retains a separate BFP4 prefill down. These are counted even though the hybrid runtime sends prefill to EP.
- Shared MLP always retains BF16 prefill matrices and separately uploaded decode matrices, including when decode also chooses BF16. Terminal embeddings/head use distinct sharded representations.
- Current `ttnn.typecast` allocates even when source and destination dtype match (`ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp`). Explicit alias guards in the policy plumbing preserve baseline QKV/output ownership.

## Candidate implications

| Candidate | DRAM delta/device | Conservative maximum-context peak/device |
| --- | ---: | ---: |
| All decode groups BFP8 | +2,966,712,320 | 31,434,685,440 |
| Inner QKV BFP4 only | +77,856,768 | 28,545,829,888 |
| Inner output BFP4 only | +51,904,512 | 28,519,877,632 |
| Inner expert gate BFP4 only | -1,660,944,384 | 26,807,028,736 |
| Shared down BFP4 | -19,148,800 | 28,448,824,320 |
| Head BFP8 | -173,015,040 | 28,294,958,080 |
| Head BFP4 | -265,289,728 | 28,202,683,392 |
| BF16 K/V | +7,549,747,200 | 36,017,720,320 |

BF16 K/V alone uses **16,106,127,360 bytes/device**. The inherited conservative peak exceeds the decimal 32 GB comparison capacity by **4,017,720,320 bytes**. The resident bound before reserve and long-prefill buffers is **29,441,051,648 bytes**, so this is a failure of the established maximum-context budget, not proof that even short-context allocation is impossible. Selecting BF16 KV would require new maximum-context evidence or a separately proven memory reduction; the advertised context must not silently shrink.

Policies with no increase in weight/cache payload retain the existing conservative capacity envelope, subject to unchanged allocation and reserve assumptions. A payload calculation alone does not prove allocator fragmentation, dtype-dependent CB requirements or numerical correctness. Dtype-dependent temporaries remain covered only by the inherited 2 GiB reserve until runtime validation.

## Reproduction and checks

```bash
python3 -m models.autoports.google_gemma_4_26b_a4b_it.tests.precision_memory \
  models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/configs/*.json \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/memory_candidates.json
```

Pure-host assertions passed for exact baseline ledger reconciliation, BF16 KV totals, separate retained TP prefill copies on one full-layer upward recovery, and the head BFP8 delta. The baseline decoder subtotal and overall bound are also asserted inside the helper, so architecture drift fails loudly.
