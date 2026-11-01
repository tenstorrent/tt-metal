# Validation handoff: Kimi expanded-K MLA prefill on a full-torus Galaxy

Branch `pjosipovic/unrolled-v-tp-sharded-kv` (7 commits on main `f8f728b7627`):

| Commit | Change |
|---|---|
| `37d0d7142e6` | minimal_matmul: runtime-M rows, head-major output, K prefix, hardware multicast |
| `2b9c7d14f18` | CCL: rank-interleaved all-gather blocks and progressive slice readiness |
| `a6026ebbe24` | ring_joint SDPA: expanded MLA K, K split and interleaved ring steps |
| `e142f435bbc` | ring_joint SDPA: slot donation puts all 110 grid cores to work for expanded K |
| `39ba2d17a90` | deepseek_v3_d_p MLA: unrolled V and expanded K for Kimi chunked prefill (+ tests, CI) |
| `09961e716e9` | ring_joint SDPA: K-split bands follow the gather's readiness split (unaligned-prefix fix) |
| `a8241e87e17` | ring_joint SDPA: the reducer's K-split band always keeps the diagonal chunk (review fix) |

This scratch branch (`pjosipovic/unrolled-v-validation-scratch`, not for merge) adds only this directory.

## What is validated so far (bh-glx-120-b06u08, TP rings open, 2D torus Y)

- Through `39ba2d17a90`: expanded-K gate (K3 57.8%, K2.7 55.6%), determinism, linear accuracy, production accuracy
  (min per-head PCC K3 0.9980, K2.7 0.9979), cache-hit test, minimal_matmul module, high_bw_all_gather.
- `09961e716e9` + `a8241e87e17`: SDPA at K3 512,000 + {0 .. 4608} cached is 36.9-37.3 ms (was 41.4 ms at offsets 512-4096). K3
  rotation / partial / unaligned-prefix scenarios against the CPU reference: 7 of 9 passed when the move started;
  the rest of its validation runs here.
- Never measured on a closed TP ring: the gates in code (expanded-K util K3 58.21 / K2.7 55.95 %, LoudBox, K3 MLA
  realtime 5.83 ms) were measured or projected on torus Y.

## Setup on bh-glx-120-c02u08

```bash
git fetch origin pjosipovic/unrolled-v-validation-scratch
git checkout -B pjosipovic/unrolled-v-validation-scratch origin/pjosipovic/unrolled-v-validation-scratch
git submodule update --init --recursive
./create_venv.sh --force && ./build_metal.sh --release
source python_env/bin/activate      # /opt/venv on c02u0x hosts has no usable ttnn: check `python -c "import ttnn.device"`
```

If the firmware precompile fails with `ccache: error: Could not find compiler "sloppiness=..."`, `unset
TT_METAL_CCACHE_KERNEL_SUPPORT` and rebuild. Run every device test through `scripts/run_safe_pytest.sh`.

## Validation (full torus)

`validate.sh` in this directory runs all of it in order and prints one line per step; logs go to `$LOGS` (default
`./validation_logs`). The steps:

1. ring_joint SDPA nightly: expanded-K perf gate (re-centre if off), production accuracy K3 / K2.7, determinism, linear
   accuracy (2D line), cache-hit test (both cases; `slide` needs a closed SP=4 ring, which torus XY has).
2. MLA rotation edges and an unaligned deep prefix against the CPU reference (`rotacc.py`, K3 and K2.7), plus
   `test_mla_chunked_prefill` at `torus-xy-8x4` for the rotation scenarios and the TP-sharded layout.
3. K3 MLA realtime gate (`test_mla_chunked_perf_check`, 5.83 ms projected; re-centre on this host).
4. minimal_matmul nightly module and high_bw_all_gather.

## vs-main matrix

MLA layer forward time, traced replays, aligned prefixes only (51,200 and 512,000 are multiples of both 5k and 2k
chunks). Main needs its own build of the branch base:

```bash
git worktree add ../tt-metal-main f8f728b7627 && cd ../tt-metal-main && git submodule update --init --recursive
./create_venv.sh --force && ./build_metal.sh --release
cp ../tt-metal/models/demos/deepseek_v3_d_p/tests/scratch_validation/matrix_vs_main.py models/demos/deepseek_v3_d_p/tests/
```

Run `matrix_vs_main.py` in both trees (`-k "kv50k or kv500k"`; main runs `baseline` and `baselineTuned`, which gives
main a q32 / k512 config for 2k chunks since it has none), then `python gen_table.py <logs...>` prints the table. Start
with `-k "kv500k and kimi_k3 and c5k and hifi2"` for a 3-cell check.

## Open items

- Re-centre the expanded-K utilization gates, LoudBox projections and the K3 MLA realtime gate on this host.
- TP latent prefetch: failed on the earlier base on 6 column-11 cores; never re-checked on the current base (needs the
  TP-sharded layout, which this host can run).
- Slot donation reaches about half of its skip-a-slot ceiling; next idea is sizing each row group's donated slot by
  its band's real share of every slice.
