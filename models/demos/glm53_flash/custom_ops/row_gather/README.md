# row_gather

`RowGather(device)(idx, table)`: `out[0, i] = table[idx[0, i]]` for a row-major bf16 table (one page per row), the
same bytes as `ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT)`.

- `table`: `[R, H]` (or `[1, R, H]`) bf16 ROW_MAJOR, DRAM interleaved. `idx`: `[1, N]` uint32 ROW_MAJOR, one page,
  every id `< R`. Output: `[1, N, H]` bf16 ROW_MAJOR, DRAM interleaved.
- Work: units of 16 output rows, contiguous unit ranges per core over the whole compute grid. Reader (NCRISC): index
  chunks of `idx_chunk` ids (64 B aligned reads of the one index page), then batches of `batch` row reads with one
  barrier per batch. Writer (BRISC): one write barrier per batch. The row CB holds two batches.
- Why: the prefill MoE regions path (tt-blaze G53F-0174 / 0186) spends ~35 ms per layer per 16 K chunk in its two
  row gathers; `ttnn.embedding`'s row-major reader keeps one read in flight per core.
- Test: `tests/custom_ops/test_row_gather.py` (bitwise vs torch and vs `ttnn.embedding`, plus timing).
