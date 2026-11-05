# Runbook: the tt-blaze#6224 filter-mask repro on the Blackhole RTL simulator

This branch is scratch work for [tt-blaze#6224](https://github.com/tenstorrent/tt-blaze/issues/6224). Do not merge it.
It is the October tt-metal pin (44d66500) plus:

- a single-core test, `tests/sources/glm_filter_pack_race_test.cpp` and
  `tests/python_tests/test_glm_filter_pack_race.py`;
- define-gated edits to the LLK, all off by default:
  - `RS_SEM_SWITCH`: tt-metal PR #57747's Tensix-semaphore `_llk_set_fp32_dest_acc_`, copied from the PR;
  - `RS_BLOCKING_MAILBOX`: #59202's blocking `mailbox_read`, in `ckernel.h`;
  - `RS_DRAIN_BEFORE_RELEASE`: #55807's MATH drain;
  - `RS_SAFE_FP32_SWITCH`: an extra drain on every thread.

## What the test does

One Tensix core runs the tt-blaze `glm_sparse_read_sdpa` tilizer index filter on one UInt32 tile. The kernel is compiled
for a 16-bit DEST with SyncFull, as in the fused decoder. One filter window is:

1. All threads switch DEST to fp32 (`_llk_set_fp32_dest_acc_`). PACK reconfigures for the UInt32 mask.
2. UNPACK unpacks the tile to DEST. MATH copies it, runs the SFPU bank filter (`_sparse_k_filter_tile_`, bank 6), runs
   the topk_xl K=1024 per-column sort, and commits.
3. PACK waits, re-runs the per-tile packer reconfig, and packs.
4. All threads switch DEST back.

Knobs, set by the test parameters:

| Knob | Meaning |
|---|---|
| `reps` | Number of back-to-back windows. |
| `tail` | Between the pack and the restore, add `tensix_sync` plus a `get_tile_address`-style mailbox exchange (UNPACK sends an address to MATH and PACK). |
| `remap` | MATH sets `DEST_ACCESS_CFG` remap_addrs and swizzle_32b at kernel start. |
| `sem_switch` | Use #57747's switch. |
| `pack_delay`, `unpack_delay`, `math_delay`, `drain`, `sort`, `per_tile_reconfig` | Further perturbations. None of them changed the outcome on silicon. |

The check: the packed mask must hold exactly the 80 nonzero words `(index & 0x3FFF) + 1`, one per index in bank 6.

## Result on p150-class silicon (bh-lb-23 loudbox, device 0, 2026-11-05)

Build with `TT_LLK_DISABLE_ASSERTS=1`, as tt-metal kernels are built.

| Switch | `reps` | `tail` | Mask (nonzero of 80) |
|---|---|---|---|
| mailbox (pin) | 1 | any | 80, pass |
| mailbox (pin) | 2 | off | **0, FAIL** (all zero, the decoder's signature) |
| mailbox (pin) | 2 | on | **55, FAIL** |
| #57747 semaphore switch | 1 or 2 | any | 80, pass |

- **`remap` makes no difference.** Both values give the rows above.
- **The failure is deterministic.** It reproduced in every run over 5 jobs.
- **Asserts on fixes it.** With `LLK_ASSERT` on, the switch's `mailbox_read` results are used, and every case passes.

## Working hypothesis (for the waveform to confirm or refute)

The pin's `_llk_set_fp32_dest_acc_` hands off through the TRISC mailboxes. With asserts off, the results of its
`mailbox_read`s (`UNPACK_READY`, `PACK_READY`, `MATH_DONE`) are unused, so the reads don't wait (see #59202). A handshake
message can be left unconsumed. The next mailbox read after the second switch is UNPACK's read of the
unpack-to-DEST address from MATH (`set_dst_write_addr` in `cunpack_common.h`). If it takes a stale message
(`MATH_DONE` = `0x46504110`), the tile is unpacked to the wrong DEST location. The SFPU then filters a cleared tile, and
PACK writes zeros.

Questions for the waveform of `reps=2, tail=False, sem_switch=False`:

1. What value does UNPACK's mailbox read return for the second window's unpack-to-DEST address, and where in DEST does
   that tile land?
2. Which `mailbox_read`s in the two switches return before their message arrives, or take a message meant for a
   different read?
3. When MATH's SFPU filter runs and when PACK's `PACR` runs: what DEST rows do they touch, and what do those rows hold?

The pair to compare is the same case with `sem_switch=True`, which passes.

## Caveat: this is not yet proven to be the decoder's failure

The decoder (DSv4.1 FC carry case, tt-blaze#6224) gives the same all-zero mask on two tilizer cores, and #57747 fixes it
too. But two of its properties differ from this test:

- the decoder failure needs `DEST_ACCESS_CFG` remap and swizzle set; this test fails without them;
- the #59202 backport (`RS_BLOCKING_MAILBOX`) did not fix the decoder in one run; this test is fixed by asserts-on,
  which has the same effect.

It is also not established that a previous fp32 switch runs on the culprit tilizers before the filter. See HANDOFF-6224.md on
the tt-blaze branch of the same name.

## Running it

```bash
cd tt_metal/tt-llk/tests
./setup_testing_env.sh            # or: python3 -m venv .venv && pip install -r requirements.txt, and link sfpi 7.85.0 to ./sfpi
source .venv/bin/activate && cd python_tests
export CHIP_ARCH=blackhole TT_LLK_DISABLE_ASSERTS=1
pytest -s test_glm_filter_pack_race.py
# Simulator: add --run-simulator (and the TT_UMD_SIMULATOR_PATH / NNG_* environment for vcs-blackhole).
# Run one case per simulator session, e.g.
#   -k "remap:False and sem_switch:False and reps:2 and tail:False"   (FAIL)
#   -k "remap:False and sem_switch:True and reps:2 and tail:False"    (PASS)
```
