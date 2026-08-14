# matmul-in1-mcast-padding-host

- Kind: host integration
- Helper API: `MCAST_PIPE_API_VERSION=9`
- Bindings: all four `matmul-in1-mcast` legacy/descriptor 1D/2D bindings
- Status: migrated, fully end-to-end
- Code commit: `aeeb28ff007807c71b1f60842cca85e5c41efa7f`
  (post-rebase equivalent of `2d0280d3dacf8a2ba24882b35816c6a1fbffb7dd`, identical patch-id;
  remapped by reconcile 2026-08-03)
- Verified: **2026-08-03** (re-verified at tree state `eb05b3929a3`; first verified 2026-07-30)

## Verify-only re-run — 2026-08-03 (`apply-dm-helper --mode=halt`)

`../../archive/reconciliation/reconcile_2026-08-03.md` flagged all six rows of this unit `needs_recheck`: the two factories were
churned upstream (`54d8dfb7bef`→`4a1d6a97ca9`, +203/−13 and +93/−4, touching
`mm_in1_sender_writer_args`), then reworked again by `c946da17d29` + `eb05b3929a3`, which postdate the
ledger's last update `62f82dd4a64`. **No rewrite was performed** — this was a verify-only pass.

Static pre-check (from the reconcile): both kernel files byte-identical to the pre-rebase verified
state; the `McastArgs` wire confirmed intact on both factories (sender CT idx 10–14 with next = 15 =
`KtNt`, sender RT idx 2–5, receiver CT idx 4–8, receiver RT idx 0–3, `MCAST_ARGS` set at `2d:618` /
`1d:1512`, `SKIP_MCAST` coexistence coherent).

- `./build_metal.sh`: passed — **already current**; `_ttnn.so` (13:57) postdated both churned
  factories (13:47, 13:48) and its mtime did not change, so nothing needed recompiling.
- Exact compile-focused 2D node under `scripts/run_safe_pytest.sh --dev`: **PASSED**, no watcher or
  assert trips —
  `test_matmul_2d_multiple_output_blocks_per_core[transpose_mcast=False-num_out_block_w=1-num_out_block_h=1-out_sharded=False-in0_sharded=False-grid_size=(8, 4)-has_bias=False-n=1024-k=512-m=512-b=1]`
  - Device-verified that **both** kernels ran: JIT-built at 14:36:22 under the **new** cache root
    `tt-metal-cache12312614508320308860` — sender `6509650342639884602`, receiver
    `4791675444625965894` + `5078604005037224472`.
  - The 2026-07-30 hashes (`4616781822959825899` / `4167676435791909128`) live under the old root
    `tt-metal-cache15548382223525479139`. Hashes are **not comparable across the rebase** — the cache
    root and the CT args both moved — so equality is not the check; a green run of both kernels is.
- `MM-IN1-ALL`, re-run in 4 chunks (halt mode, `-x` per chunk, `--precompile` for the cold cache):

  | chunk | selection | result |
  |---|---|---|
  | A | `test_matmul_2d_multiple_output_blocks_per_core` (128) | 56 passed, 72 skipped |
  | B | `test_matmul_2d_tiny_tile` (96) | 46 passed, 50 skipped |
  | C | `test_matmul_1d_tiny_tile` (96) | 46 passed, 50 skipped |
  | D | remaining 16 test functions (170) | 154 passed, 16 skipped |
  | **total** | **490 selected** | **302 passed, 188 expected skips** |

  Chunked because the cache root changed and everything compiled cold; the reconstructed `-k`
  selection was confirmed against collection to select exactly 490, matching the recorded baseline.
- `McastHostFixture.*`: 19 passed.
- `test_mcast_pipe.py`: 68 passed.

**Result: PASS — `needs_recheck` cleared on all 6 rows** (2 kernels + 4 host bindings),
`last_verified` = 2026-08-03, `verified_at_commit` = `eb05b3929a3`. `commit` deliberately still points
at the migration commit `aeeb28ff007` (its documented role is the revert/bisect anchor).

The coverage gap below is **unchanged** by this pass — it is a property of who calls the legacy
constructors, not of the rebase.

## Atomic scope

- `matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- `matmul_multicore_reuse_mcast_2d_program_factory.cpp`
- `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
- `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`

The 1D factory's legacy and descriptor mcast-in1 paths now construct one host
`Mcast2D` over the actual offset worker bounding rectangle. The 2D legacy and
descriptor paths construct one `Mcast2D` per in1 line, preserving
`transpose_mcast`, sender placement, the preferred in1 NoC, subdevice offsets,
and active receiver acknowledgement counts. All four bindings adopt the
existing receiver/sender semaphore IDs.

Both kernels decode the helper's five-word CT/four-word RT wire through
`McastArgs`. Downstream compile-time and runtime argument offsets, tensor
bindings, optional fused bias arguments, and cached output-address overrides
were shifted to match. The `MCAST_IN0`, sparse, and non-multicast
`SKIP_MCAST` paths retain their previous wire and behavior.

## Validation

- `./build_metal.sh`: passed.
- Exact 1D mcast-in1 non-zero-subdevice node under
  `scripts/run_safe_pytest.sh --dev`: passed.
  - sender JIT hash `10580236968838213332`;
  - receiver JIT hash `6510408673418518324`.
- Exact 2D `transpose_mcast=false` and `transpose_mcast=true` descriptor nodes
  under `scripts/run_safe_pytest.sh --dev`: both passed.
  - sender JIT hash `4616781822959825899`;
  - receiver JIT hash `4167676435791909128`.
- `MM-IN1-ALL`: 302 passed, 188 expected skips, 490 selected.
- `McastHostFixture.*`: 19 passed.
- `test_mcast_pipe.py`: 68 passed.

The mapped matmul inventory exercises the descriptor constructors at runtime,
including both 2D multicast orientations and offset 1D subdevice placement.
The legacy constructors compile in the full host build, but their only current
callers are fused CCL factories, so no mapped single-chip pytest provides
legacy-constructor device-runtime proof.

## Diff and coverage

- Production diff: 196 insertions, 70 deletions.
- Coverage gap: yes; legacy factory constructors have build coverage but no
  mapped device-runtime coverage.
- Result: PASS; no rollback or quarantine required.
